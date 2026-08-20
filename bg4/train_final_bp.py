"""Final BP router: multi-source + cleaned background (bg v2), with an
overfitting audit and validation-only tau selection.  BP models only (frozen
sentence embeddings + classic heads: SVM / XGBoost).

Anti-overfitting recipe (validated by cv_sources.py leave-one-source-out):
  - 4 sources per domain (GQR + 3 new, leakage-screened, cap 3000)
  - background v2: wikitext+dolly+yahoo(non-ID topics), ID-topic filtered
  - per-source 85/15 train/val split; ID-val + aux-val used for ALL selection
  - seeds: data/model seeds 22/43/44 -> mean +- sd

Overfitting audit per model x seed (nothing below touches test data):
  1. train acc vs stratified 5-fold CV acc (classic gap)
  2. alpha sweep: tau_a = (1-a)-quantile of p_bg on ID-val; score(a) =
     hmean(ID-val routing acc under rejection, aux-val rejection).  Best a is
     selected HERE, on validation only.
  3. (reference) leave-one-source-out numbers live in results/cv_summary.md

Untouched test panels, evaluated once at the end (argmax + the selected tau):
  - GQR-Bench ID test / OOD test (protocol GQR score)
  - VAULT panel - fresh sources never used in any training, tuning or earlier
    panel: law = umarbutler/open-australian-legal-qa; finance =
    bitext/Bitext-retail-banking + virattt/financial-qa-10K; healthcare =
    openlifescienceai/medmcqa (validation split).  Leakage-screened (exact +
    8-word shingle) against every training-side text and the GQR test sets.
  - new-OOD panel (trec/agnews/rotten/codealpaca/gsm8k/yahoo/tweets), screened

usage: .venv/bin/python train_final_bp.py [--models svm,xgb] [--embeds mini]
       [--seeds 22,43,44] [--cap 3000]
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import xgboost  # noqa: F401  (before torch: libomp clash on macOS)

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
import retrain_bg4 as R  # noqa: E402
from cv_sources import load_sources  # noqa: E402
from eval_new_id import DOM  # noqa: E402
from eval_new_ood import SeenIndex, load_new_sets, screen  # noqa: E402

log = logging.getLogger("final")
ALPHA_GRID = (0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2)


def load_vault(cap, seed, index):
    """Fresh test-only sources; screened against `index` (training-side texts +
    GQR test sets)."""
    rng = np.random.default_rng(seed)

    def take(texts, n=cap, min_words=3):
        texts = [str(t).strip() for t in texts if t is not None and len(str(t).split()) >= min_words]
        if len(texts) > n:
            texts = [texts[i] for i in rng.choice(len(texts), n, replace=False)]
        return texts

    from datasets import load_dataset
    V = {}
    df = pd.read_json("hf://datasets/umarbutler/open-australian-legal-qa/qa.jsonl", lines=True)
    V["aus_legal_qa"] = ("law", take(df["question"].tolist()))
    ds = load_dataset("bitext/Bitext-retail-banking-llm-chatbot-training-dataset", split="train")
    V["bitext_banking"] = ("finance", take(ds["instruction"]))
    ds = load_dataset("virattt/financial-qa-10K", split="train")
    V["finqa_10k"] = ("finance", take(ds["question"]))
    ds = load_dataset("openlifescienceai/medmcqa", split="validation")
    V["medmcqa"] = ("healthcare", take(ds["question"]))
    report = {}
    for name, (dom, texts) in list(V.items()):
        keep, hits = screen(index, texts)
        report[name] = dict(domain=dom, n_raw=len(texts), n_kept=len(keep), dropped=hits)
        log.info("vault %-15s (%-10s) raw %4d kept %4d dropped %s", name, dom, len(texts), len(keep), hits or "-")
        V[name] = (dom, keep)
    return V, report


def fit(model_key, X, y, seed, proba=True):
    if model_key == "xgb":
        from xgboost import XGBClassifier
        clf = XGBClassifier(n_jobs=-1, tree_method="hist", device="cpu", objective="multi:softprob",
                            num_class=4, random_state=seed)
    else:
        from sklearn.svm import SVC
        clf = SVC(probability=proba, cache_size=4000, random_state=seed)
    clf.fit(X, y)
    return clf


def pbg(clf, X):
    return clf.predict_proba(X)[:, 3]


def route(clf, X, tau=None):
    """Labels under argmax (tau=None) or the p_bg>tau rule."""
    P = clf.predict_proba(X)
    if tau is None:
        return P.argmax(1)
    return np.where(P[:, 3] > tau, R.BG, P[:, :3].argmax(1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="svm,xgb")
    ap.add_argument("--embeds", default="mini")
    ap.add_argument("--seeds", default="22,43,44")
    ap.add_argument("--cap", type=int, default=3000)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--out", default=str(R.RESULTS / "final_bp.csv"))
    args = ap.parse_args()
    for _n in ("httpx", "urllib3", "sentence_transformers", "datasets", "transformers"):
        logging.getLogger(_n).setLevel(logging.WARNING)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(R.RESULTS / "final_bp.log")])
    out = Path(args.out)

    D = R.load_all(22, bg="v2")                       # bg v2 corpus is seed-22-built; data reuse is fine
    S, src_report = load_sources(D, args.cap, 22)     # screened sources (deterministic)
    aux_tr, aux_va = D["aux_train"], D["aux_val"]

    # training-side index for the vault screen
    idx = SeenIndex()
    idx.add(aux_tr + aux_va, "bg_v2")
    for k, v in S.items():
        idx.add(v, f"{k[0]}/{k[1]}")
    idx.add(D["val"]["text"], "gqr_val")
    idx.add(D["id_test"]["text"], "gqr_id_test")
    idx.add(D["ood_test"]["text"], "gqr_ood_test")
    V, vault_report = load_vault(1500, 22, idx)
    (R.RESULTS / "final_bp_screen.json").write_text(json.dumps(dict(sources=src_report, vault=vault_report), indent=2))

    new_ood = load_new_sets(1000, 22)
    new_ood.pop("banking77", None)
    bi = SeenIndex()
    bi.add(aux_tr + aux_va, "aux")
    for k in list(new_ood):
        new_ood[k], _ = screen(bi, new_ood[k])

    from sklearn.model_selection import StratifiedKFold
    rows = []
    for emb in args.embeds.split(","):
        F = lambda texts: R.embed(list(texts), emb, batch_size=256)  # noqa: E731
        Xs = {k: F(v) for k, v in S.items() if v}
        Xbg_tr, Xbg_va = F(aux_tr), F(aux_va)
        Xgqr_id, ygqr_id = F(list(D["id_test"]["text"])), D["id_test"]["label"].to_numpy()
        Xgqr_ood = F(list(D["ood_test"]["text"]))
        gqr_ood_ds = D["ood_test"]["dataset"].values
        XV = {n: (dom, F(t)) for n, (dom, t) in V.items() if t}
        Xood = {n: F(t) for n, t in new_ood.items() if t}
        for seed in [int(s) for s in args.seeds.split(",")]:
            rng = np.random.default_rng(seed)
            tr_parts, va_parts = [], []
            for k, X in Xs.items():
                perm = rng.permutation(len(X))
                n_va = int(len(X) * args.val_frac)
                va_parts.append((X[perm[:n_va]], np.full(n_va, DOM[k[0]])))
                tr_parts.append((X[perm[n_va:]], np.full(len(X) - n_va, DOM[k[0]])))
            Xtr = np.vstack([p[0] for p in tr_parts] + [Xbg_tr])
            ytr = np.concatenate([p[1] for p in tr_parts] + [np.full(len(Xbg_tr), R.BG)])
            Xva = np.vstack([p[0] for p in va_parts])
            yva = np.concatenate([p[1] for p in va_parts])
            for mk in args.models.split(","):
                t0 = time.time()
                clf = fit(mk, Xtr, ytr, seed)
                fit_s = time.time() - t0
                train_acc = float((clf.predict(Xtr) == ytr).mean())
                # 1. classic 5-fold CV gap (argmax, no probability for speed)
                skf = StratifiedKFold(5, shuffle=True, random_state=seed)
                cvs = []
                for tri, tei in skf.split(Xtr, ytr):
                    c = fit(mk, Xtr[tri], ytr[tri], seed, proba=False)
                    cvs.append(float((c.predict(Xtr[tei]) == ytr[tei]).mean()))
                cv_acc = float(np.mean(cvs))
                # 2. alpha sweep on validation only
                p_va, p_aux = pbg(clf, Xva), pbg(clf, Xbg_va)
                Pva = clf.predict_proba(Xva)
                sweep = {}
                for a in ALPHA_GRID:
                    tau = float(np.quantile(p_va, 1 - a))
                    acc_va = float((np.where(p_va > tau, R.BG, Pva[:, :3].argmax(1)) == yva).mean())
                    rej_aux = float((p_aux > tau).mean())
                    sweep[a] = dict(tau=round(tau, 5), val_acc=round(acc_va, 4), aux_rej=round(rej_aux, 4),
                                    score=round(2 * acc_va * rej_aux / (acc_va + rej_aux), 4))
                best_a = max(sweep, key=lambda a: sweep[a]["score"])
                tau = sweep[best_a]["tau"]
                # 3. untouched tests
                res = {}
                for rule, t in (("argmax", None), ("tau", tau)):
                    pid = route(clf, Xgqr_id, t)
                    id_acc = float((pid == ygqr_id).mean())
                    pood = route(clf, Xgqr_ood, t)
                    ood_acc = float(np.mean([float((pood[gqr_ood_ds == d] == R.BG).mean()) for d in np.unique(gqr_ood_ds)]))
                    gqr = 2 * id_acc * ood_acc / (id_acc + ood_acc)
                    vacc = {n: float((route(clf, X, t) == DOM[dom]).mean()) for n, (dom, X) in XV.items()}
                    vrej = {n: float((route(clf, X, t) == R.BG).mean()) for n, (dom, X) in XV.items()}
                    nood = {n: float((route(clf, X, t) == R.BG).mean()) for n, X in Xood.items()}
                    vault_mean = float(np.mean(list(vacc.values())))
                    nood_mean = float(np.mean(list(nood.values())))
                    res[rule] = dict(gqr_id=id_acc, gqr_ood=ood_acc, gqr=gqr, vault=vault_mean,
                                     new_ood=nood_mean,
                                     hmean_unseen=2 * vault_mean * nood_mean / (vault_mean + nood_mean),
                                     vault_per_set=vacc, vault_rej=vrej, new_ood_per_set=nood)
                log.info("%s/%s s%d | train %.4f cv %.4f gap %.4f | best_a %.3f tau %.4f (val %.3f auxrej %.3f) | "
                         "GQR argmax %.4f tau %.4f | VAULT argmax %.3f tau %.3f | newOOD argmax %.3f tau %.3f | hmean_unseen tau %.3f | %ds",
                         mk, emb, seed, train_acc, cv_acc, train_acc - cv_acc, best_a, tau,
                         sweep[best_a]["val_acc"], sweep[best_a]["aux_rej"],
                         res["argmax"]["gqr"], res["tau"]["gqr"], res["argmax"]["vault"], res["tau"]["vault"],
                         res["argmax"]["new_ood"], res["tau"]["new_ood"], res["tau"]["hmean_unseen"], int(fit_s))
                for rule in ("argmax", "tau"):
                    r = res[rule]
                    rows.append(dict(timestamp=time.strftime("%Y%m%d_%H%M%S"), model_key=mk, embedding=emb,
                                     seed=seed, rule=rule, alpha=best_a if rule == "tau" else None,
                                     tau=tau if rule == "tau" else None, train_acc=round(train_acc, 4),
                                     cv_acc=round(cv_acc, 4), cv_gap=round(train_acc - cv_acc, 4),
                                     gqr_id=round(r["gqr_id"], 4), gqr_ood=round(r["gqr_ood"], 4),
                                     gqr=round(r["gqr"], 4), vault_acc=round(r["vault"], 4),
                                     new_ood_acc=round(r["new_ood"], 4),
                                     hmean_unseen=round(r["hmean_unseen"], 4),
                                     vault_per_set=json.dumps({k: round(v, 3) for k, v in r["vault_per_set"].items()}),
                                     vault_rej=json.dumps({k: round(v, 3) for k, v in r["vault_rej"].items()}),
                                     new_ood_per_set=json.dumps({k: round(v, 3) for k, v in r["new_ood_per_set"].items()}),
                                     alpha_sweep=json.dumps(sweep) if rule == "tau" else None))
                    pd.DataFrame(rows[-1:]).to_csv(out, mode="a", index=False, header=not out.exists())
    log.info("done -> %s", out)


if __name__ == "__main__":
    main()
