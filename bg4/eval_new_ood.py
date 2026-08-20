"""Evaluate the saved bg4 / ctrl3 models on NEW out-of-distribution datasets that
were never used anywhere in the pipeline, with an explicit leakage screen.

New OOD panel (capped at --cap per set, seed --seed):
  trec            CogComp/trec                    factoid questions
  agnews_ws       fancyzhx/ag_news  (World, Sports only; Business/SciTech excluded)
  rotten_tomatoes cornell-movie-review-data/rotten_tomatoes   review sentences
  codealpaca      sahil2801/CodeAlpaca-20k        coding instructions
  gsm8k           openai/gsm8k                    math word problems (money vocabulary!)
  yahoo_se        community-datasets/yahoo_answers_topics (Sports, Entertainment & Music)
  tweets          mteb/tweet_sentiment_extraction  short informal tweets
ID-shift sanity check (NOT OOD — should be routed to finance, not rejected):
  banking77       mteb/banking77                  banking customer queries

Leakage screen — every candidate text is compared against EVERYTHING the models
saw: the auxiliary background corpus (wikitext-103 + dolly-15k passages), GQR
train / val / ID-test, and the GQR OOD test sets. A candidate is dropped when its
normalised text matches exactly OR shares any 8-word shingle with a seen text.
The same index is used to confirm the aux corpus is disjoint from the GQR test
sets (the assumption behind the main results).

Thresholds (tau) are taken from results/results.csv — the ID-val calibration of
the original run; nothing is re-calibrated on the new data.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import retrain_bg4 as R  # noqa: E402

log = logging.getLogger("new_ood")
SHINGLE = 8

# -----------------------------------------------------------------------------
# new datasets

def _hf(repo, split, **kw):
    from datasets import load_dataset
    return load_dataset(repo, split=split, **kw)


def load_new_sets(cap, seed):
    rng = np.random.default_rng(seed)

    def take(texts, n=cap):
        texts = [str(t).strip() for t in texts if str(t).strip()]
        if len(texts) > n:
            idx = rng.choice(len(texts), n, replace=False)
            texts = [texts[i] for i in idx]
        return texts

    out = {}
    ds = _hf("CogComp/trec", "test", revision="refs/convert/parquet")
    out["trec"] = take(ds["text"])

    ds = _hf("fancyzhx/ag_news", "test")
    ds = ds.filter(lambda x: x["label"] in (0, 1))           # 0 World, 1 Sports
    out["agnews_ws"] = take(ds["text"])

    ds = _hf("cornell-movie-review-data/rotten_tomatoes", "test")
    out["rotten_tomatoes"] = take(ds["text"])

    ds = _hf("sahil2801/CodeAlpaca-20k", "train")
    out["codealpaca"] = take([r["instruction"] + (("\n" + r["input"]) if r["input"] else "") for r in ds])

    ds = _hf("openai/gsm8k", "test", name="main")
    out["gsm8k"] = take(ds["question"])

    ds = _hf("community-datasets/yahoo_answers_topics", "test")
    ds = ds.filter(lambda x: x["topic"] in (5, 7))            # 5 Sports, 7 Entertainment & Music
    out["yahoo_se"] = take([(r["question_title"] + " " + r["question_content"]).strip() for r in ds])

    ds = _hf("mteb/tweet_sentiment_extraction", "test")
    out["tweets"] = take([t for t in ds["text"] if len(str(t).split()) >= 4])

    ds = _hf("mteb/banking77", "test")
    out["banking77"] = take(ds["text"])
    return out


# -----------------------------------------------------------------------------
# leakage screen

_ws = re.compile(r"\s+")
_punct = re.compile(r"[^\w\s]")


def norm(t):
    return _ws.sub(" ", _punct.sub(" ", str(t).lower())).strip()


def shingles(t, k=SHINGLE):
    w = norm(t).split()
    if len(w) < k:
        return {hash(" ".join(w))} if w else set()
    return {hash(" ".join(w[i:i + k])) for i in range(len(w) - k + 1)}


class SeenIndex:
    def __init__(self):
        self.exact = {}
        self.sh = {}

    def add(self, texts, tag):
        for t in texts:
            n = norm(t)
            if not n:
                continue
            self.exact.setdefault(n, tag)
            for h in shingles(t):
                self.sh.setdefault(h, tag)

    def hit(self, t):
        n = norm(t)
        if n in self.exact:
            return "exact:" + self.exact[n]
        for h in shingles(t):
            if h in self.sh:
                return "shingle:" + self.sh[h]
        return None


def screen(index, texts):
    keep, hits = [], {}
    for t in texts:
        h = index.hit(t)
        if h is None:
            keep.append(t)
        else:
            hits[h] = hits.get(h, 0) + 1
    return keep, hits


# -----------------------------------------------------------------------------
# model loading

def build(model_key, emb, variant, D, seed, hp):
    n_classes = 4 if variant == "bg4" else 3
    tag = f"{model_key}_{emb}_{variant}_s{seed}"
    path = R.MODELS / tag
    if model_key in ("xgb", "svm"):
        if emb == "tf_idf":
            feat = R.TfidfFeaturizer(R.training_set(D, variant)[0])   # deterministic refit
        else:
            feat = (lambda k: (lambda texts: R.embed(texts, k)))(emb)
        M = (R.XGBModel if model_key == "xgb" else R.SVMModel)(feat, emb, seed).load(path)
    elif model_key == "fasttext":
        M = R.FastTextModel(seed).load(path, n_classes)
    elif model_key == "widemlp":
        M = R.WideMLPModel(seed).load(path, n_classes)
    else:
        M = R.HFEncoderModel(model_key, seed, max_len=int(hp.get("max_len", 128))).load(path)
    return M, n_classes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="xgb,svm,fasttext,widemlp,bert,modernbert,bge")
    ap.add_argument("--variants", default="bg4,ctrl3")
    ap.add_argument("--seed", type=int, default=22)
    ap.add_argument("--cap", type=int, default=1000)
    ap.add_argument("--results", default=str(R.RESULTS / "results.csv"))
    ap.add_argument("--out", default=str(R.RESULTS / "new_ood.csv"))
    ap.add_argument("--screen-only", action="store_true")
    args = ap.parse_args()
    for _n in ("httpx", "urllib3", "sentence_transformers", "datasets", "transformers"):
        logging.getLogger(_n).setLevel(logging.WARNING)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(R.RESULTS / "new_ood.log")])

    D = R.load_all(args.seed)
    new = load_new_sets(args.cap, args.seed)

    # ---- leakage screen ------------------------------------------------------
    idx = SeenIndex()
    idx.add(D["aux_train"] + D["aux_val"], "aux(wikitext+dolly)")
    idx.add(D["train"]["text"], "gqr_train")
    idx.add(D["val"]["text"], "gqr_val")
    idx.add(D["id_test"]["text"], "gqr_id_test")
    idx.add(D["ood_test"]["text"], "gqr_ood_test")

    # sanity: is the aux (background) corpus disjoint from the benchmark test sets?
    test_idx = SeenIndex()
    test_idx.add(D["id_test"]["text"], "gqr_id_test")
    test_idx.add(D["ood_test"]["text"], "gqr_ood_test")
    _, aux_hits = screen(test_idx, D["aux_train"] + D["aux_val"])
    log.info("aux corpus vs GQR test sets: %s (of %d aux passages)", aux_hits or "no overlap",
             len(D["aux_train"]) + len(D["aux_val"]))

    report = {}
    for name, texts in list(new.items()):
        keep, hits = screen(idx, texts)
        report[name] = dict(n_raw=len(texts), n_kept=len(keep), dropped=hits)
        log.info("screen %-16s raw %4d  kept %4d  dropped %s", name, len(texts), len(keep), hits or "-")
        new[name] = keep
    screen_path = R.RESULTS / "new_ood_screen.json"
    screen_path.write_text(json.dumps(dict(aux_vs_gqr_test=aux_hits, datasets=report), indent=2))
    if args.screen_only:
        return

    # ---- evaluate saved models ------------------------------------------------
    res = pd.read_csv(args.results)
    res = res.sort_values("timestamp").drop_duplicates(["model_key", "embedding", "variant", "rule", "seed"], keep="last")
    rows = []
    out = Path(args.out)
    for variant in args.variants.split(","):
        for mk in args.models.split(","):
            embs = ["baai", "mini", "tf_idf"] if mk in ("xgb", "svm") else ["own"]
            for emb in embs:
                sub = res[(res.model_key == mk) & (res.embedding == emb) & (res.variant == variant) & (res.seed == args.seed)]
                if sub.empty:
                    log.warning("no results row for %s/%s/%s — skip", mk, emb, variant)
                    continue
                hp = json.loads(sub.iloc[0].hyperparameters)
                id_acc = float(sub.iloc[0].id_acc)   # same for all rules? no — per rule below
                try:
                    M, n_classes = build(mk, emb, variant, D, args.seed, hp)
                except Exception:
                    log.exception("load failed %s/%s/%s", mk, emb, variant)
                    continue
                P = {name: M.proba(texts) for name, texts in new.items() if texts}
                for _, r in sub.iterrows():
                    tau = None if pd.isna(r.tau) else float(r.tau)
                    accs = {}
                    for name, Pn in P.items():
                        pred = R.decide(Pn, r.rule, tau)
                        if name == "banking77":
                            accs[name] = dict(reject=float((pred == R.BG).mean()),
                                              finance=float((pred == 1).mean()),
                                              law=float((pred == 0).mean()),
                                              healthcare=float((pred == 2).mean()))
                        else:
                            accs[name] = float((pred == R.BG).mean())
                    ood_names = [n for n in accs if n != "banking77"]
                    mean_new = float(np.mean([accs[n] for n in ood_names]))
                    gqr_new = 2 * r.id_acc * mean_new / (r.id_acc + mean_new) if r.id_acc + mean_new > 0 else 0.0
                    log.info("%s/%s/%s [%s] new-OOD mean %.4f  GQR_new %.4f  %s  banking77 %s",
                             mk, emb, variant, r.rule, mean_new, gqr_new,
                             {k: round(v, 3) for k, v in accs.items() if k != "banking77"},
                             {k: round(v, 3) for k, v in accs.get("banking77", {}).items()})
                    rows.append(dict(timestamp=time.strftime("%Y%m%d_%H%M%S"), model=M.name, model_key=mk,
                                     embedding=emb, variant=variant, rule=r.rule, seed=args.seed, tau=tau,
                                     id_acc=r.id_acc, gqr_bench_ood_acc=r.ood_acc, gqr_bench=r.gqr_score,
                                     new_ood_acc=round(mean_new, 4), gqr_new=round(gqr_new, 4),
                                     **{f"acc_{n}": round(accs[n], 4) for n in ood_names},
                                     **{f"banking77_{k}": round(v, 4) for k, v in accs.get("banking77", {}).items()}))
                    pd.DataFrame(rows[-1:]).to_csv(out, mode="a", index=False, header=not out.exists())
                try:
                    M.model = None
                except Exception:
                    pass
    log.info("done: %d rows -> %s", len(rows), out)


if __name__ == "__main__":
    main()
