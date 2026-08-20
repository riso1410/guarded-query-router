"""ID-shift check: route UNSEEN in-domain datasets (new sources for law /
finance / healthcare) through the saved 4-class models (argmax) and report
where they go.  The GQR ID-test set is drawn from the same three sources as the
training data, so ID acc ≈ 0.99 there does not show in-domain generalisation.

Panel (cap --cap per set, seed --seed; leakage-screened like eval_new_ood.py):
  finance    banking77 (mteb/banking77), fin_instruct (DeividasM/financial-instruction-aq22),
             reddit_finance (winddude/reddit_finance_43_250k title+selftext)
             [gbharti/finance-alpaca was dropped: its non-FiQA part is the GENERAL Stanford-Alpaca
              instruction set, not finance; BeIR/fiqa dropped: the GQR finance source is derived from it]
  healthcare icliniq (lavita/ChatDoctor-iCliniq patient questions), medquad
             (keivalya/MedQuad-MedicalQnADataset questions), med_flashcards
             (medalpaca/medical_meadow_medical_flashcards)
  law        legal_reddit (jonathanli/legal-advice-reddit posts), legal_qa_v1
             (dzunggg/legal-qa-v1), legal_qa_ib (ibunescu/qa_legal_dataset_train)

Reported per model × set: acc (routed to the correct domain), reject (background),
and the confusion to the other two domains.  Decision rule = argmax only.
"""

from __future__ import annotations

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
from eval_new_ood import SeenIndex, _hf, build, screen  # noqa: E402

log = logging.getLogger("new_id")
DOM = {"law": 0, "finance": 1, "healthcare": 2}


def load_new_id(cap, seed):
    rng = np.random.default_rng(seed)

    def take(texts, n=cap, min_words=3):
        texts = [str(t).strip() for t in texts if t is not None and len(str(t).split()) >= min_words]
        if len(texts) > n:
            idx = rng.choice(len(texts), n, replace=False)
            texts = [texts[i] for i in idx]
        return texts

    out = {}
    # ---- finance
    ds = _hf("mteb/banking77", "test")
    out["banking77"] = ("finance", take(ds["text"]))
    ds = _hf("DeividasM/financial-instruction-aq22", "train")
    out["fin_instruct"] = ("finance", take(ds["instruction"]))
    ds = _hf("winddude/reddit_finance_43_250k", "train")
    out["reddit_finance"] = ("finance", take([(r["title"] + "\n" + (r["selftext"] or "")).strip() for r in ds.select(range(40000))]))
    # ---- healthcare
    ds = _hf("lavita/ChatDoctor-iCliniq", "train")
    out["icliniq"] = ("healthcare", take(ds["input"]))
    ds = _hf("keivalya/MedQuad-MedicalQnADataset", "train")
    out["medquad"] = ("healthcare", take(ds["Question"]))
    ds = _hf("medalpaca/medical_meadow_medical_flashcards", "train")
    out["med_flashcards"] = ("healthcare", take(ds["input"]))
    # ---- law
    ds = _hf("jonathanli/legal-advice-reddit", "test")
    out["legal_reddit"] = ("law", take([(r["title"] + "\n" + r["body"]).strip() for r in ds]))
    ds = _hf("dzunggg/legal-qa-v1", "train")
    out["legal_qa_v1"] = ("law", take(ds["question"]))
    ds = _hf("ibunescu/qa_legal_dataset_train", "train")
    out["legal_qa_ib"] = ("law", take(ds["Question"] if "Question" in ds.column_names else ds["question"]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="xgb,svm,fasttext,widemlp,bert,modernbert")
    ap.add_argument("--variants", default="bg4,ctrl3")
    ap.add_argument("--seed", type=int, default=22)
    ap.add_argument("--cap", type=int, default=1000)
    ap.add_argument("--results", default=str(R.RESULTS / "results.csv"))
    ap.add_argument("--out", default=str(R.RESULTS / "new_id.csv"))
    ap.add_argument("--screen-only", action="store_true")
    args = ap.parse_args()
    for _n in ("httpx", "urllib3", "sentence_transformers", "datasets", "transformers"):
        logging.getLogger(_n).setLevel(logging.WARNING)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(R.RESULTS / "new_id.log")])

    D = R.load_all(args.seed)
    new = load_new_id(args.cap, args.seed)

    idx = SeenIndex()
    idx.add(D["aux_train"] + D["aux_val"], "aux(wikitext+dolly)")
    idx.add(D["train"]["text"], "gqr_train")
    idx.add(D["val"]["text"], "gqr_val")
    idx.add(D["id_test"]["text"], "gqr_id_test")
    idx.add(D["ood_test"]["text"], "gqr_ood_test")
    report = {}
    for name, (dom, texts) in list(new.items()):
        keep, hits = screen(idx, texts)
        report[name] = dict(domain=dom, n_raw=len(texts), n_kept=len(keep), dropped=hits)
        log.info("screen %-15s (%-10s) raw %4d  kept %4d  dropped %s", name, dom, len(texts), len(keep), hits or "-")
        new[name] = (dom, keep)
    (R.RESULTS / "new_id_screen.json").write_text(json.dumps(report, indent=2))
    if args.screen_only:
        return

    res = pd.read_csv(args.results)
    res = res.sort_values("timestamp").drop_duplicates(["model_key", "embedding", "variant", "rule", "seed"], keep="last")
    out = Path(args.out)
    rows = []
    for variant in args.variants.split(","):
        for mk in args.models.split(","):
            embs = ["baai", "mini", "tf_idf"] if mk in ("xgb", "svm") else ["own"]
            for emb in embs:
                sub = res[(res.model_key == mk) & (res.embedding == emb) & (res.variant == variant) & (res.seed == args.seed)]
                if sub.empty:
                    continue
                hp = json.loads(sub.iloc[0].hyperparameters)
                try:
                    M, _ = build(mk, emb, variant, D, args.seed, hp)
                except Exception:
                    log.exception("load failed %s/%s/%s", mk, emb, variant)
                    continue
                row = dict(timestamp=time.strftime("%Y%m%d_%H%M%S"), model=M.name, model_key=mk, embedding=emb,
                           variant=variant, rule="argmax", seed=args.seed,
                           gqr_id_acc=float(sub[sub.rule == "argmax"].iloc[0].id_acc))
                per_dom = {d: [] for d in DOM}
                for name, (dom, texts) in new.items():
                    if not texts:
                        continue
                    pred = R.decide(M.proba(texts), "argmax")
                    acc = float((pred == DOM[dom]).mean())
                    rej = float((pred == R.BG).mean())
                    conf = {d: float((pred == i).mean()) for d, i in DOM.items() if d != dom}
                    row[f"acc_{name}"] = round(acc, 4)
                    row[f"rej_{name}"] = round(rej, 4)
                    row[f"conf_{name}"] = json.dumps({k: round(v, 3) for k, v in conf.items()})
                    per_dom[dom].append(acc)
                for d, v in per_dom.items():
                    row[f"acc_{d}_mean"] = round(float(np.mean(v)), 4) if v else None
                row["new_id_acc_mean"] = round(float(np.mean([row[f"acc_{d}_mean"] for d in DOM])), 4)
                log.info("%s/%s/%s  GQR-ID %.3f | new-ID mean %.3f  law %.3f  finance %.3f  health %.3f | %s",
                         mk, emb, variant, row["gqr_id_acc"], row["new_id_acc_mean"], row["acc_law_mean"],
                         row["acc_finance_mean"], row["acc_healthcare_mean"],
                         {n: (row[f"acc_{n}"], row[f"rej_{n}"]) for n in new})
                rows.append(row)
                pd.DataFrame([row]).to_csv(out, mode="a", index=False, header=not out.exists())
                try:
                    M.model = None
                except Exception:
                    pass
    log.info("done: %d rows -> %s", len(rows), out)


if __name__ == "__main__":
    main()
