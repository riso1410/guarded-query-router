"""Retrain the BC-thesis (GQR-Bench) classic classifiers with a 4th "background"
class — outlier exposure (Hendrycks et al. 2019) with auxiliary wikitext-103 +
dolly-15k passages, the same recipe as `scorer_oe` in the DP repo (safe-router).

Models (the "old" families from the bachelor thesis):
  xgb        XGBoost            x {baai, mini, tf_idf} embeddings
  svm        SVC (RBF)          x {baai, mini, tf_idf} embeddings
  fasttext   fastText supervised (own word embeddings, autotune)
  widemlp    WideMLP (Galke & Scherp) on ModernBERT tokenizer + idf
  bert       google-bert/bert-base-multilingual-cased  (fine-tuned end-to-end)
  modernbert answerdotai/ModernBERT-base               (fine-tuned end-to-end)

Variants:
  oe4    4-way (law / finance / healthcare / background) trained on GQR train +
         auxiliary outliers.  Rejection rules reported:
           argmax : predict ood iff background is the argmax
           tau    : predict ood iff p(background) > tau, tau = (1-ALPHA)-quantile of
                    p(background) on the ID-validation split (ALPHA = 0.02, as in DP)
  ctrl3  3-way control trained on GQR train only (no background class).  Rules:
           argmax : never rejects (pure ID accuracy, OOD acc = 0)
           msp    : reject iff max softmax prob < tau, tau = ALPHA-quantile of ID-val MSP

Evaluation = GQR-Bench protocol via the `gqr` package: ID accuracy on the ID test
set, OOD accuracy = mean of per-dataset accuracies on the OOD test set, GQR =
harmonic mean.  Per-query predictions are dumped for bootstrap CIs.

Apple-silicon friendly: torch MPS for encoders, CPU for XGBoost / SVM / fastText.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))  # widemlp.py from the BC repo

log = logging.getLogger("oe4")

ID_LABELS = (0, 1, 2)          # law, finance, healthcare (gqr.domain2label)
BG = 3                         # background class == gqr "ood" label 3
ALPHA = 0.02                   # a-priori ID false-rejection budget (DP convention)
ARTIFACTS = HERE / "artifacts"
CACHE = HERE / "cache"
MODELS = HERE / "models"
RESULTS = HERE / "results"
for d in (ARTIFACTS, CACHE, MODELS, RESULTS, RESULTS / "preds"):
    d.mkdir(parents=True, exist_ok=True)

EMBED_NAMES = {"baai": "BAAI/bge-small-en-v1.5",
               "mini": "sentence-transformers/all-MiniLM-L6-v2"}
ENCODER_NAMES = {"bert": "google-bert/bert-base-multilingual-cased",
                 "modernbert": "answerdotai/ModernBERT-base"}


def torch_device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _hash(texts):
    h = hashlib.sha1()
    for t in texts:
        h.update(t.encode("utf-8", "ignore"))
        h.update(b"\x00")
    return h.hexdigest()[:12]


# -----------------------------------------------------------------------------
# data

def aux_outliers(n, seed=22, min_len=40):
    """Identical recipe to safe-router routers._aux_outliers: half wikitext-103
    prose, half dolly-15k instructions. Disjoint from every GQR OOD test set."""
    path = ARTIFACTS / f"aux_mix_{n}_{seed}.json"
    if path.exists():
        return json.loads(path.read_text())
    from datasets import load_dataset

    out = []
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                      split="train", streaming=True)
    for row in ds.shuffle(seed=seed, buffer_size=50_000):
        t = row["text"].strip()
        if len(t) >= min_len and not t.startswith("="):
            out.append(t[:2000])
            if len(out) >= n // 2:
                break
    dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
    dolly = dolly.shuffle(seed=seed)
    for row in dolly:
        t = row["instruction"].strip()
        if len(t) >= 15:
            out.append(t[:2000])
            if len(out) >= n:
                break
    path.write_text(json.dumps(out))
    log.info("aux outliers: %d passages (wikitext+dolly) -> %s", len(out), path)
    return out


def load_all(seed, n_aux=None, aux_val_frac=0.2):
    import gqr
    train, val = gqr.load_train_dataset()
    id_test = gqr.load_id_test_dataset()
    ood_test = gqr.load_ood_test_dataset()
    n_aux = n_aux or max(len(train) // len(ID_LABELS), 1000)   # 9600 by default
    aux = aux_outliers(n_aux, seed=seed)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(aux))
    n_va = int(len(aux) * aux_val_frac)
    aux_val = [aux[i] for i in perm[:n_va]]
    aux_tr = [aux[i] for i in perm[n_va:]]
    log.info("train %d  val %d  id_test %d  ood_test %d  aux_train %d  aux_val %d",
             len(train), len(val), len(id_test), len(ood_test), len(aux_tr), len(aux_val))
    return dict(train=train.reset_index(drop=True), val=val.reset_index(drop=True),
                id_test=id_test.reset_index(drop=True),
                ood_test=ood_test.reset_index(drop=True),
                aux_train=aux_tr, aux_val=aux_val)


def training_set(D, variant):
    """Return (texts, labels) for the given variant."""
    texts = list(D["train"]["text"])
    labels = list(D["train"]["label"].astype(int))
    if variant == "oe4":
        texts += D["aux_train"]
        labels += [BG] * len(D["aux_train"])
    return texts, np.asarray(labels)


def validation_set(D, variant):
    texts = list(D["val"]["text"])
    labels = list(D["val"]["label"].astype(int))
    if variant == "oe4":
        texts += D["aux_val"]
        labels += [BG] * len(D["aux_val"])
    return texts, np.asarray(labels)


# -----------------------------------------------------------------------------
# embeddings (cached)

_ST = {}


def st_model(key):
    from sentence_transformers import SentenceTransformer
    if key not in _ST:
        _ST[key] = SentenceTransformer(EMBED_NAMES[key], device=torch_device())
    return _ST[key]


def embed(texts, key, batch_size=256):
    path = CACHE / f"emb_{key}_{_hash(texts)}.npy"
    if path.exists():
        return np.load(path)
    m = st_model(key)
    X = m.encode(list(texts), batch_size=batch_size, normalize_embeddings=True,
                 show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
    np.save(path, X)
    return X


class TfidfFeaturizer:
    def __init__(self, train_texts):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer().fit(train_texts)      # BC default params

    def __call__(self, texts):
        return self.vec.transform(texts)


# -----------------------------------------------------------------------------
# model wrappers: fit(texts, labels, n_classes) -> self ; proba(texts) -> (n, C)

class XGBModel:
    name = "XGBoost"

    def __init__(self, feat, feat_key, seed):
        self.feat, self.feat_key, self.seed = feat, feat_key, seed

    def fit(self, texts, labels, n_classes):
        from xgboost import XGBClassifier
        # BC: XGBClassifier(n_jobs=-1, tree_method="auto") — defaults otherwise
        self.clf = XGBClassifier(n_jobs=-1, tree_method="hist", device="cpu",
                                 objective="multi:softprob", num_class=n_classes,
                                 random_state=self.seed)
        self.clf.fit(self.feat(texts), labels)
        return self

    def proba(self, texts):
        return self.clf.predict_proba(self.feat(texts))

    def save(self, path):
        self.clf.save_model(str(path.with_suffix(".json")))


class SVMModel:
    name = "SVM"

    def __init__(self, feat, feat_key, seed):
        self.feat, self.feat_key, self.seed = feat, feat_key, seed

    def fit(self, texts, labels, n_classes):
        from sklearn.svm import SVC
        # BC: SVC(probability=True) — defaults (RBF, C=1)
        self.clf = SVC(probability=True, cache_size=4000, random_state=self.seed)
        self.clf.fit(self.feat(texts), labels)
        return self

    def proba(self, texts):
        return self.clf.predict_proba(self.feat(texts))

    def save(self, path):
        with open(path.with_suffix(".pkl"), "wb") as f:
            pickle.dump(self.clf, f)


class FastTextModel:
    name = "fastText"

    def __init__(self, seed, autotune_duration=300, val=None):
        self.seed, self.autotune_duration, self.val = seed, autotune_duration, val

    @staticmethod
    def _clean(t):
        return str(t).replace("\n", " ").strip().lower()

    def _write(self, path, texts, labels):
        with open(path, "w", encoding="utf-8") as f:
            for t, y in zip(texts, labels):
                f.write(f"__label__{int(y)} {self._clean(t)}\n")

    def fit(self, texts, labels, n_classes):
        import fasttext
        tr = CACHE / f"ft_train_{n_classes}.txt"
        va = CACHE / f"ft_valid_{n_classes}.txt"
        self._write(tr, texts, labels)
        kw = dict(input=str(tr), seed=self.seed, thread=max(os.cpu_count() - 2, 1))
        if self.val is not None and self.autotune_duration > 0:
            self._write(va, *self.val)
            kw.update(autotuneValidationFile=str(va),
                      autotuneDuration=self.autotune_duration)
        self.n_classes = n_classes
        self.clf = fasttext.train_supervised(**kw)
        return self

    def proba(self, texts):
        P = np.zeros((len(texts), self.n_classes), dtype=np.float32)
        labs, probs = self.clf.predict([self._clean(t) for t in texts], k=self.n_classes)
        for i, (ls, ps) in enumerate(zip(labs, probs)):
            for l, p in zip(ls, ps):
                P[i, int(l.replace("__label__", ""))] = p
        return P

    def save(self, path):
        self.clf.save_model(str(path.with_suffix(".bin")))


def smooth_idf(encoded_docs, vocab_size):
    """sklearn TfidfTransformer(smooth_idf=True) idf, vectorised:
    idf = log((1+n)/(1+df)) + 1.  (widemlp.inverse_document_frequency builds a
    dok_matrix token by token — slow for 36k docs and deadlock-prone next to
    xgboost's OpenMP runtime.)"""
    import torch
    n = len(encoded_docs)
    df = np.zeros(vocab_size, dtype=np.int64)
    for doc in encoded_docs:
        df[np.unique(np.asarray(doc, dtype=np.int64))] += 1
    return torch.FloatTensor(np.log((1 + n) / (1 + df)) + 1.0)


class WideMLPModel:
    name = "WideMLP"

    def __init__(self, seed, epochs=30, batch_size=128, lr=3e-4, val=None,
                 num_hidden_layers=1):
        self.seed, self.epochs, self.batch_size, self.lr = seed, epochs, batch_size, lr
        self.val, self.num_hidden_layers = val, num_hidden_layers

    def _encode(self, texts):
        return self.tok(list(texts), padding=False, truncation=True,
                        max_length=512)["input_ids"]

    def _forward(self, ids):
        import torch
        from widemlp import prepare_inputs_optimized
        flat, off = prepare_inputs_optimized(ids, device=self.device)
        return self.model(flat, off)

    def fit(self, texts, labels, n_classes):
        import torch
        from transformers import AutoTokenizer
        from widemlp import MLP
        torch.manual_seed(self.seed)
        self.tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        docs = self._encode(texts)
        idf = smooth_idf(docs, len(self.tok))
        self.device = torch_device()
        # softmax CE (K+1 classes) instead of the BC multi-label BCE head
        self.model = MLP(vocab_size=len(self.tok), num_hidden_layers=self.num_hidden_layers,
                         num_classes=n_classes, idf=idf, problem_type="classification")
        try:
            self.model.to(self.device)
            self.model.idf = self.model.idf.to(self.device)
            self._forward(docs[:4])
        except Exception as e:          # EmbeddingBag per_sample_weights on MPS
            log.warning("WideMLP: %s on %s failed (%s) -> cpu", "forward", self.device, e)
            self.device = "cpu"
            self.model.to("cpu")
            self.model.idf = self.model.idf.to("cpu")
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0)
        y = torch.tensor(labels, dtype=torch.long)
        rng = np.random.default_rng(self.seed)
        order = np.arange(len(docs))
        best, best_state, bad = float("inf"), None, 0
        for ep in range(self.epochs):
            self.model.train()
            rng.shuffle(order)
            tot, nb = 0.0, 0
            for i in range(0, len(order), self.batch_size):
                idx = order[i:i + self.batch_size]
                loss, _ = self.model(*self._prep([docs[j] for j in idx]), y[idx].to(self.device))
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss)
                nb += 1
            msg = f"widemlp epoch {ep}: train loss {tot / max(nb, 1):.4f}"
            if self.val is not None:
                vl = self._val_loss(*self.val)
                msg += f"  val loss {vl:.4f}"
                if vl < best - 1e-4:
                    best, bad = vl, 0
                    best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                else:
                    bad += 1
            log.info(msg)
            if self.val is not None and bad >= 5:
                log.info("widemlp: early stop (patience 5)")
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    def _prep(self, ids):
        from widemlp import prepare_inputs_optimized
        return prepare_inputs_optimized(ids, device=self.device)

    def _val_loss(self, texts, labels):
        import torch
        import torch.nn.functional as F
        ids = self._encode(texts)
        y = torch.tensor(labels, dtype=torch.long)
        self.model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(ids), 512):
                logits = self.model(*self._prep(ids[i:i + 512]))
                tot += float(F.cross_entropy(logits, y[i:i + 512].to(self.device), reduction="sum"))
                n += logits.shape[0]
        return tot / max(n, 1)

    def proba(self, texts):
        import torch
        ids = self._encode(texts)
        out = []
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(ids), 512):
                out.append(torch.softmax(self.model(*self._prep(ids[i:i + 512])), -1).float().cpu().numpy())
        return np.concatenate(out)

    def save(self, path):
        import torch
        torch.save({"model_state_dict": self.model.state_dict(), "idf": self.model.idf.cpu()},
                   path.with_suffix(".pt"))


class HFEncoderModel:
    """End-to-end fine-tuned transformer with a softmax (K or K+1)-way head."""

    def __init__(self, key, seed, epochs=3, batch_size=16, lr=2e-5, max_len=256, val=None,
                 eval_every=500):
        self.key, self.seed, self.epochs, self.batch_size = key, seed, epochs, batch_size
        self.lr, self.max_len, self.val, self.eval_every = lr, max_len, val, eval_every
        self.name = {"bert": "BERT-multilingual", "modernbert": "ModernBERT"}[key]
        self.model_name = ENCODER_NAMES[key]

    def fit(self, texts, labels, n_classes):
        import torch
        import torch.nn.functional as F
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
        torch.manual_seed(self.seed)
        self.device = torch_device()
        self.tok = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=n_classes)
        if hasattr(self.model.config, "reference_compile"):   # ModernBERT: no torch.compile on MPS
            self.model.config.reference_compile = False
        self.model = self.model.to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        steps = self.epochs * ((len(texts) + self.batch_size - 1) // self.batch_size)
        sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
        y = torch.tensor(labels, dtype=torch.long)
        rng = np.random.default_rng(self.seed)
        order = np.arange(len(texts))
        best, best_state, step = float("inf"), None, 0
        t0 = time.time()
        for ep in range(self.epochs):
            rng.shuffle(order)
            self.model.train()
            tot, nb = 0.0, 0
            for i in range(0, len(order), self.batch_size):
                idx = order[i:i + self.batch_size]
                bt = self.tok([texts[j] for j in idx], padding=True, truncation=True,
                              max_length=self.max_len, return_tensors="pt").to(self.device)
                logits = self.model(**bt).logits
                loss = F.cross_entropy(logits, y[idx].to(self.device))
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                sched.step()
                tot += float(loss)
                nb += 1
                step += 1
                if step % 100 == 0:
                    log.info("%s ep %d step %d/%d loss %.4f  (%.0fs)", self.key, ep, step, steps,
                             tot / nb, time.time() - t0)
                if self.val is not None and step % self.eval_every == 0:
                    vl = self._val_loss(*self.val)
                    log.info("%s step %d val loss %.4f", self.key, step, vl)
                    if vl < best:
                        best = vl
                        best_state = {k: v.detach().to("cpu", copy=True) for k, v in self.model.state_dict().items()}
                    self.model.train()
            log.info("%s epoch %d: train loss %.4f", self.key, ep, tot / max(nb, 1))
        if self.val is not None:
            vl = self._val_loss(*self.val)
            log.info("%s final val loss %.4f (best %.4f)", self.key, vl, best)
            if vl < best:
                best_state = None
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    @property
    def _infer_bs(self):
        return 128

    def _logits(self, texts):
        import torch
        out = []
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(texts), self._infer_bs):
                bt = self.tok(list(texts[i:i + self._infer_bs]), padding=True, truncation=True,
                              max_length=self.max_len, return_tensors="pt").to(self.device)
                out.append(self.model(**bt).logits.float().cpu())
        return torch.cat(out)

    def _val_loss(self, texts, labels):
        import torch
        import torch.nn.functional as F
        return float(F.cross_entropy(self._logits(texts), torch.tensor(labels, dtype=torch.long)))

    def proba(self, texts):
        import torch
        return torch.softmax(self._logits(texts), -1).numpy()

    def save(self, path):
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tok.save_pretrained(path)


# -----------------------------------------------------------------------------
# evaluation

def decide(P, rule, tau=None):
    """P: (n, C) probabilities. Returns gqr labels (0/1/2 ID, 3 ood)."""
    id_arg = P[:, :len(ID_LABELS)].argmax(1)
    if rule == "argmax":
        return P.argmax(1)                       # for ctrl3 this never returns 3
    if rule == "tau":                            # oe4: background prob threshold
        return np.where(P[:, BG] > tau, BG, id_arg)
    if rule == "msp":                            # ctrl3: max-softmax-prob threshold
        return np.where(P[:, :len(ID_LABELS)].max(1) < tau, BG, id_arg)
    raise ValueError(rule)


def score(preds_id, gold_id, preds_ood, ood_df):
    from gqr.core.evaluator import evaluate, evaluate_by_dataset
    id_acc = evaluate(predictions=list(preds_id), ground_truth=list(gold_id))["accuracy"]
    df = ood_df[["dataset", "label"]].copy()
    df["pred"] = preds_ood
    by = evaluate_by_dataset(df, pred_col="pred", true_col="label", dataset_col="dataset")
    ood_acc = float(by["accuracy"].mean())
    ds_acc = {k.strip(): round(float(v), 4) for k, v in zip(by["dataset"], by["accuracy"])}
    gqr = 2 * id_acc * ood_acc / (id_acc + ood_acc) if id_acc + ood_acc > 0 else 0.0
    return float(id_acc), ood_acc, float(gqr), ds_acc


def latency_probe(model, texts, n=200):
    """Single-query sequential latency (s/query) incl. featurisation."""
    probe = list(texts[:n])
    model.proba(probe[:2])  # warm-up
    t0 = time.perf_counter()
    for t in probe:
        model.proba([t])
    return (time.perf_counter() - t0) / len(probe)


def run_one(model, model_key, embed_key, variant, D, seed, results_csv, hparams):
    texts, labels = training_set(D, variant)
    n_classes = len(ID_LABELS) + (1 if variant == "oe4" else 0)
    tag = f"{model_key}_{embed_key or 'own'}_{variant}_s{seed}"
    log.info("=== %s: fit on %d texts (%d classes)", tag, len(texts), n_classes)
    t0 = time.time()
    model.fit(texts, labels, n_classes)
    train_time = time.time() - t0
    log.info("%s: trained in %.0fs", tag, train_time)
    try:
        model.save(MODELS / tag)
    except Exception as e:
        log.warning("%s: save failed: %s", tag, e)

    P_val = model.proba(list(D["val"]["text"]))        # ID-val only (threshold calibration)
    P_id = model.proba(list(D["id_test"]["text"]))
    P_ood = model.proba(list(D["ood_test"]["text"]))
    val_acc = float((P_val[:, :3].argmax(1) == D["val"]["label"].to_numpy()).mean())
    lat = latency_probe(model, list(D["id_test"]["text"][:100]) + list(D["ood_test"]["text"][:100]))

    if variant == "oe4":
        rules = {"argmax": None, "tau": float(np.quantile(P_val[:, BG], 1 - ALPHA))}
        # diagnostic: held-out aux rejection (NOT a benchmark number)
        P_aux = model.proba(D["aux_val"])
        aux_rej = {r: float((decide(P_aux, r, t) == BG).mean()) for r, t in rules.items()}
    else:
        rules = {"argmax": None, "msp": float(np.quantile(P_val[:, :3].max(1), ALPHA))}
        aux_rej = {}

    rows = []
    for rule, tau in rules.items():
        pid = decide(P_id, rule, tau)
        pood = decide(P_ood, rule, tau)
        id_acc, ood_acc, gqr, ds_acc = score(pid, D["id_test"]["label"], pood, D["ood_test"])
        log.info("%s [%s] ID %.4f  OOD %.4f  GQR %.4f  lat %.2f ms  tau=%s  aux_rej=%s",
                 tag, rule, id_acc, ood_acc, gqr, lat * 1e3,
                 None if tau is None else round(tau, 4), aux_rej.get(rule))
        rows.append(dict(timestamp=time.strftime("%Y%m%d_%H%M%S"), model=model.name,
                         model_key=model_key, embedding=embed_key or "own", variant=variant,
                         rule=rule, seed=seed, id_acc=round(id_acc, 4), ood_acc=round(ood_acc, 4),
                         gqr_score=round(gqr, 4), id_val_acc=round(val_acc, 4),
                         aux_val_reject=None if rule not in aux_rej else round(aux_rej[rule], 4),
                         tau=None if tau is None else round(tau, 5), avg_latency_s=round(lat, 6),
                         train_time_s=round(train_time, 1), n_train=len(texts),
                         dataset_acc=json.dumps(ds_acc), hyperparameters=json.dumps(hparams)))
        pd.concat([
            D["id_test"][["dataset" if "dataset" in D["id_test"] else "domain", "label"]].assign(split="id", pred=pid),
            D["ood_test"][["dataset", "label"]].assign(split="ood", pred=pood),
        ]).to_csv(RESULTS / "preds" / f"{tag}_{rule}.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(results_csv, mode="a", index=False, header=not results_csv.exists())
    return df


# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="xgb,svm,fasttext,widemlp,bert,modernbert")
    ap.add_argument("--embeds", default="baai,mini,tf_idf", help="for xgb/svm")
    ap.add_argument("--variants", default="oe4,ctrl3")
    ap.add_argument("--seed", type=int, default=22)
    ap.add_argument("--n-aux", type=int, default=None, help="aux outliers (default len(train)/3 = 9600)")
    ap.add_argument("--epochs", type=int, default=3, help="bert/modernbert epochs")
    ap.add_argument("--mlp-epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--ft-autotune", type=int, default=300, help="fastText autotune seconds (0 = defaults)")
    ap.add_argument("--results", default=str(RESULTS / "results.csv"))
    ap.add_argument("--dry-run", action="store_true", help="tiny subsets, smoke test")
    args = ap.parse_args()

    for _n in ("httpx", "urllib3", "sentence_transformers", "datasets", "transformers"):
        logging.getLogger(_n).setLevel(logging.WARNING)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(RESULTS / "run.log")])
    log.info("args: %s", vars(args))
    D = load_all(args.seed, n_aux=args.n_aux)
    if args.dry_run:
        for k in ("train", "val", "id_test", "ood_test"):
            D[k] = D[k].sample(n=min(300, len(D[k])), random_state=args.seed).reset_index(drop=True)
        D["aux_train"], D["aux_val"] = D["aux_train"][:100], D["aux_val"][:40]
    results_csv = Path(args.results)
    models = args.models.split(",")
    variants = args.variants.split(",")
    embeds = args.embeds.split(",")
    hp = dict(seed=args.seed, alpha=ALPHA, n_aux=len(D["aux_train"]) + len(D["aux_val"]),
              epochs=args.epochs, mlp_epochs=args.mlp_epochs, batch_size=args.batch_size,
              max_len=args.max_len, lr=args.lr, ft_autotune=args.ft_autotune, dry_run=args.dry_run)

    for variant in variants:
        val = validation_set(D, variant)
        for mk in models:
            try:
                if mk in ("xgb", "svm"):
                    for ek in embeds:
                        if ek == "tf_idf":
                            feat = TfidfFeaturizer(training_set(D, variant)[0])
                        else:
                            feat = (lambda k: (lambda texts: embed(texts, k)))(ek)
                        M = (XGBModel if mk == "xgb" else SVMModel)(feat, ek, args.seed)
                        run_one(M, mk, ek, variant, D, args.seed, results_csv, hp)
                elif mk == "fasttext":
                    run_one(FastTextModel(args.seed, autotune_duration=0 if args.dry_run else args.ft_autotune, val=val),
                            mk, None, variant, D, args.seed, results_csv, hp)
                elif mk == "widemlp":
                    run_one(WideMLPModel(args.seed, epochs=2 if args.dry_run else args.mlp_epochs, val=val),
                            mk, None, variant, D, args.seed, results_csv, hp)
                elif mk in ("bert", "modernbert"):
                    run_one(HFEncoderModel(mk, args.seed, epochs=1 if args.dry_run else args.epochs,
                                           batch_size=args.batch_size, lr=args.lr, max_len=args.max_len,
                                           val=val, eval_every=20 if args.dry_run else 500),
                            mk, None, variant, D, args.seed, results_csv, hp)
                else:
                    log.error("unknown model %s", mk)
            except Exception:
                log.exception("FAILED %s / %s", mk, variant)
    if results_csv.exists():
        df = pd.read_csv(results_csv)
        cols = ["model", "embedding", "variant", "rule", "id_acc", "ood_acc", "gqr_score", "avg_latency_s"]
        log.info("\n%s", df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
