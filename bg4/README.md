# bg4 — BC-thesis classifiers retrained with a 4th "background" class

Outlier-exposure retraining of the GQR-Bench classic routers (XGBoost, SVM, fastText,
WideMLP, BERT-multilingual, ModernBERT): 4-way softmax over law / finance / healthcare /
**background**, where background = 9 600 auxiliary passages (½ wikitext-103 prose,
½ dolly-15k instructions — the `scorer_oe` recipe from the safe-router DP repo).
Evaluated with the GQR-Bench protocol via the `gqr` package. Runs on Apple silicon (MPS).

```bash
uv venv --python 3.12 && uv pip install -e .      # once
./run_all.sh                                      # all families, bg4 + ctrl3 (≈2 h on M5 Pro)
MODELS="bert modernbert" EXTRA="--epochs 1 --batch-size 32 --max-len 128" ./run_all.sh
.venv/bin/python retrain_bg4.py --help
.venv/bin/python summarize.py results/results.csv --md results/summary.md
```

`run_all.sh` runs one process per family (xgboost + torch share an OpenMP runtime and
deadlock in one process). Outputs: `results/results.csv` (one row per model × embedding ×
variant × rule), `results/preds/*.csv` (per-query predictions for bootstrap CIs),
`results/summary.md`, `models/` (trained weights), `cache/` (embeddings), `artifacts/`
(aux outlier corpus).

Variants / rules: `bg4/argmax`, `bg4/tau` (reject iff p_bg > τ, τ = 0.98-quantile on ID-val,
α = 0.02), control `ctrl3/msp` (3-class, reject iff max-softmax < α-quantile of ID-val MSP).

Settings of the 2026-08-19 run: seed 22; XGB/SVM sklearn defaults; fastText autotune 300 s;
WideMLP ≤30 epochs (early stop, patience 5); BERT/ModernBERT 1 epoch, lr 2e-5, bs 32,
max_len 128, fp16 autocast, best-val checkpoint. See `results/summary.md`.

Note: `4DR1455/finance_questions` was removed from the HF Hub — `gqr` needs a cached copy
in `~/.cache/huggingface` (copied from the FIIT GPU server).
