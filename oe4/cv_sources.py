"""Cross-validation for the 4-class routers — measures overfitting two ways.

A. Leave-one-SOURCE-out CV (the one that matters): every domain has several
   independent sources; each fold holds out one source per domain, trains on the
   remaining sources (+ background), and tests routing accuracy on the held-out
   sources and rejection on the new-OOD panel.  Fold 0 holds out the GQR sources
   themselves (train on the new sources only, test on the GQR ID test set).
     law        : gqr (law_stackexchange) | legal_reddit | legal_qa_v1 | legal_qa_ib
     finance    : gqr (finance_questions) | banking77    | fin_instruct   | reddit_finance
     healthcare : gqr (HealthCareMagic)   | icliniq      | medquad        | med_flashcards
B. Random stratified 5-fold CV on the pooled training set (all sources + bg):
   train acc vs CV acc gap — the classic overfitting number.

All source texts are leakage-screened against the GQR test sets, the background
corpus and each other (exact + 8-word shingle).  Background = --bg v1 (raw
wikitext+dolly) or v2 (ID-topic-filtered wikitext+dolly+yahoo).  Models: SVM and
XGBoost on frozen MiniLM / bge-small embeddings, argmax decision.

usage: .venv/bin/python cv_sources.py --bg v2 [--models svm,xgb] [--embeds mini,baai] [--cap 3000]
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import xgboost  # noqa: F401  (import before torch: avoids a libomp segfault on macOS)

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import retrain_oe4 as R  # noqa: E402
from eval_new_id import DOM  # noqa: E402
from eval_new_ood import SeenIndex, _hf, load_new_sets, screen  # noqa: E402

log = logging.getLogger("cv")

SOURCES = {  # domain -> ordered sources; index 0 = GQR source
    "law": ["gqr", "legal_reddit", "legal_qa_v1", "legal_qa_ib"],
    "finance": ["gqr", "banking77", "fin_instruct", "reddit_finance"],
    "healthcare": ["gqr", "icliniq", "medquad", "med_flashcards"],
}


def load_sources(D, cap, seed):
    rng = np.random.default_rng(seed)

    def take(texts, n=cap, min_words=3):
        texts = [str(t).strip() for t in texts if t is not None and len(str(t).split()) >= min_words]
        if len(texts) > n:
            idx = rng.choice(len(texts), n, replace=False)
            texts = [texts[i] for i in idx]
        return texts

    S = {}
    tr = D["train"]
    for dom, lab in DOM.items():
        S[(dom, "gqr")] = take(tr[tr.label == lab]["text"].tolist())
    ds = _hf("jonathanli/legal-advice-reddit", "train")
    S[("law", "legal_reddit")] = take([(r["title"] + "\n" + r["body"]).strip() for r in ds])
    ds = _hf("dzunggg/legal-qa-v1", "train")
    S[("law", "legal_qa_v1")] = take(ds["question"])
    ds = _hf("ibunescu/qa_legal_dataset_train", "train")
    S[("law", "legal_qa_ib")] = take(ds["Question"] if "Question" in ds.column_names else ds["question"])
    ds = _hf("mteb/banking77", "train")
    S[("finance", "banking77")] = take(ds["text"])
    ds = _hf("DeividasM/financial-instruction-aq22", "train")
    S[("finance", "fin_instruct")] = take(ds["instruction"])
    ds = _hf("winddude/reddit_finance_43_250k", "train")
    S[("finance", "reddit_finance")] = take([(r["title"] + "\n" + (r["selftext"] or "")).strip() for r in ds.select(range(40000, 80000))])
    ds = _hf("lavita/ChatDoctor-iCliniq", "train")
    S[("healthcare", "icliniq")] = take(ds["input"])
    ds = _hf("keivalya/MedQuad-MedicalQnADataset", "train")
    S[("healthcare", "medquad")] = take(ds["Question"])
    ds = _hf("medalpaca/medical_meadow_medical_flashcards", "train")
    S[("healthcare", "med_flashcards")] = take(ds["input"])

    # leakage screen: vs GQR val / ID-test / OOD-test and the background corpus, then across sources
    idx = SeenIndex()
    idx.add(D["aux_train"] + D["aux_val"], "aux")
    idx.add(D["val"]["text"], "gqr_val")
    idx.add(D["id_test"]["text"], "gqr_id_test")
    idx.add(D["ood_test"]["text"], "gqr_ood_test")
    report = {}
    for key in list(S):
        if key[1] == "gqr":
            continue
        keep, hits = screen(idx, S[key])
        report[f"{key[0]}/{key[1]}"] = dict(n_raw=len(S[key]), n_kept=len(keep), dropped=hits)
        S[key] = keep[:cap]
        idx.add(S[key], f"{key[0]}/{key[1]}")
    gq = SeenIndex()
    for key in S:
        if key[1] != "gqr":
            gq.add(S[key], key[1])
    for key in list(S):
        if key[1] == "gqr":
            keep, hits = screen(gq, S[key])
            report[f"{key[0]}/gqr"] = dict(n_raw=len(S[key]), n_kept=len(keep), dropped=hits)
            S[key] = keep
    for k, v in S.items():
        log.info("source %-10s %-15s n=%d", k[0], k[1], len(v))
    return S, report


def feats(emb):
    return lambda texts: R.embed(texts, emb, batch_size=256)


def fit_predict(model_key, Xtr, ytr, n_classes, seed):
    if model_key == "xgb":
        from xgboost import XGBClassifier
        clf = XGBClassifier(n_jobs=-1, tree_method="hist", device="cpu", objective="multi:softprob",
                            num_class=n_classes, random_state=seed)
    else:
        from sklearn.svm import SVC
        clf = SVC(cache_size=4000, random_state=seed)       # no Platt scaling needed for argmax
    clf.fit(Xtr, ytr)
    return clf


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bg", default="v2", choices=["v1", "v2"])
    ap.add_argument("--models", default="svm,xgb")
    ap.add_argument("--embeds", default="mini,baai")
    ap.add_argument("--cap", type=int, default=3000, help="max examples per source")
    ap.add_argument("--seed", type=int, default=22)
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--out", default=str(R.RESULTS / "cv_sources.csv"))
    args = ap.parse_args()
    for _n in ("httpx", "urllib3", "sentence_transformers", "datasets", "transformers"):
        logging.getLogger(_n).setLevel(logging.WARNING)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(R.RESULTS / f"cv_sources_{args.bg}.log")])

    D = R.load_all(args.seed, bg=args.bg)
    S, report = load_sources(D, args.cap, args.seed)
    (R.RESULTS / f"cv_sources_screen_{args.bg}.json").write_text(json.dumps(report, indent=2))
    bg_tr = D["aux_train"] + D["aux_val"]
    new_ood = load_new_sets(1000, args.seed)
    new_ood.pop("banking77", None)
    # screen the new-OOD panel against the background too (as in eval_new_ood)
    bi = SeenIndex()
    bi.add(bg_tr, "aux")
    for k in list(new_ood):
        new_ood[k], _ = screen(bi, new_ood[k])

    out = Path(args.out)
    rows = []
    folds = [0, 1, 2, 3]
    for emb in args.embeds.split(","):
        F = feats(emb)
        # embed every source once (cached)
        Xs = {k: F(v) for k, v in S.items() if v}
        Xbg = F(bg_tr)
        Xood = {k: F(v) for k, v in new_ood.items() if v}
        Xgqr_id = F(list(D["id_test"]["text"]))
        ygqr_id = D["id_test"]["label"].to_numpy()
        for mk in args.models.split(","):
            # ---- A. leave-one-source-out ----------------------------------------
            for f in folds:
                held = {dom: SOURCES[dom][f] for dom in DOM}
                tr_keys = [k for k in S if S[k] and k[1] != held[k[0]]]
                Xtr = np.vstack([Xs[k] for k in tr_keys] + [Xbg])
                ytr = np.concatenate([np.full(len(S[k]), DOM[k[0]]) for k in tr_keys] + [np.full(len(bg_tr), R.BG)])
                t0 = time.time()
                clf = fit_predict(mk, Xtr, ytr, 4, args.seed)
                row = dict(timestamp=time.strftime("%Y%m%d_%H%M%S"), bg=args.bg, model_key=mk, embedding=emb,
                           fold=f, held_out=json.dumps(held), n_train=len(ytr), train_time_s=round(time.time() - t0, 1))
                accs = []
                for dom, src in held.items():
                    if src is None:
                        continue
                    if src == "gqr":
                        m = ygqr_id == DOM[dom]
                        pred = clf.predict(Xgqr_id[m])
                    else:
                        pred = clf.predict(Xs[(dom, src)])
                    acc = float((pred == DOM[dom]).mean())
                    row[f"acc_{dom}"] = round(acc, 4)
                    row[f"rej_{dom}"] = round(float((pred == R.BG).mean()), 4)
                    accs.append(acc)
                row["heldout_id_acc"] = round(float(np.mean(accs)), 4)
                oods = {k: float((clf.predict(X) == R.BG).mean()) for k, X in Xood.items()}
                row["new_ood_acc"] = round(float(np.mean(list(oods.values()))), 4)
                row.update({f"ood_{k}": round(v, 4) for k, v in oods.items()})
                h = row["heldout_id_acc"]
                row["hmean"] = round(2 * h * row["new_ood_acc"] / (h + row["new_ood_acc"]), 4)
                if f == 0:   # GQR fold: also the benchmark OOD set
                    Xgo = F(list(D["ood_test"]["text"]))
                    row["gqr_ood_acc"] = round(float((clf.predict(Xgo) == R.BG).mean()), 4)
                log.info("bg=%s %s/%s fold %d held %s | held-out ID %.3f (%s) | new-OOD %.3f | hmean %.3f",
                         args.bg, mk, emb, f, held, row["heldout_id_acc"],
                         {d: row.get(f"acc_{d}") for d in DOM}, row["new_ood_acc"], row["hmean"])
                rows.append(row)
                pd.DataFrame([row]).to_csv(out, mode="a", index=False, header=not out.exists())
            # ---- B. random stratified k-fold on the pooled set ---------------------
            from sklearn.model_selection import StratifiedKFold
            keys = [k for k in S if S[k]]
            Xall = np.vstack([Xs[k] for k in keys] + [Xbg])
            yall = np.concatenate([np.full(len(S[k]), DOM[k[0]]) for k in keys] + [np.full(len(bg_tr), R.BG)])
            skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
            cv, trn = [], []
            for tri, tei in skf.split(Xall, yall):
                clf = fit_predict(mk, Xall[tri], yall[tri], 4, args.seed)
                cv.append(float((clf.predict(Xall[tei]) == yall[tei]).mean()))
                trn.append(float((clf.predict(Xall[tri]) == yall[tri]).mean()))
            row = dict(timestamp=time.strftime("%Y%m%d_%H%M%S"), bg=args.bg, model_key=mk, embedding=emb,
                       fold="random5", held_out="stratified 5-fold on pooled sources+bg", n_train=len(yall),
                       cv_acc_mean=round(float(np.mean(cv)), 4), cv_acc_std=round(float(np.std(cv)), 4),
                       train_acc_mean=round(float(np.mean(trn)), 4),
                       train_cv_gap=round(float(np.mean(trn) - np.mean(cv)), 4))
            log.info("bg=%s %s/%s random 5-fold: cv %.4f ± %.4f  train %.4f  gap %.4f",
                     args.bg, mk, emb, row["cv_acc_mean"], row["cv_acc_std"], row["train_acc_mean"], row["train_cv_gap"])
            rows.append(row)
            out5 = out.with_name(out.stem + "_random5.csv")
            pd.DataFrame([row]).to_csv(out5, mode="a", index=False, header=not out5.exists())
    log.info("done -> %s", out)


if __name__ == "__main__":
    main()
