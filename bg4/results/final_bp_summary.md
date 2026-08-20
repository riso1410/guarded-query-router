# Final BP router — overfitting audit + validation-selected tau (2026-08-20, `train_final_bp.py`)

Recipe: multi-source training (GQR + 3 new sources per domain, cap 3000, leakage-screened) + **bg v2** (topic-cleaned background), frozen MiniLM embeddings, SVM / XGBoost heads, seeds 22/43/44. tau selected **on validation only** (alpha grid; score = hmean(ID-val routing acc, held-out-aux rejection)) — picked alpha=0.02 (SVM) / 0.03 (XGB) on ALL three seeds. Untouched tests: GQR-Bench, VAULT panel (fresh sources never used anywhere: open-australian-legal-qa, bitext-retail-banking, financial-qa-10K, medmcqa), new-OOD panel.

| model | rule | train | 5-fold CV | gap | GQR (±sd) | VAULT acc (±sd) | new-OOD rej | hmean(vault,ood) (±sd) |
|---|---|---|---|---|---|---|---|---|
| svm | argmax | 0.988 | 0.965 | 0.024 | 0.9079 ±0.0031 | 0.835 ±0.001 | 0.811 | 0.823 ±0.001 |
| svm | tau | 0.988 | 0.965 | 0.024 | 0.9302 ±0.0023 | 0.808 ±0.005 | 0.858 | 0.832 ±0.001 |
| xgb | argmax | 1.000 | 0.956 | 0.044 | 0.9167 ±0.0041 | 0.824 ±0.001 | 0.874 | 0.848 ±0.002 |
| xgb | tau | 1.000 | 0.956 | 0.044 | 0.9328 ±0.0027 | 0.799 ±0.001 | 0.907 | 0.850 ±0.002 |

Vault per set (SVM argmax, s22): {"aus_legal_qa": 0.825, "bitext_banking": 0.885, "finqa_10k": 0.851, "medmcqa": 0.782}  rejections {"aus_legal_qa": 0.077, "bitext_banking": 0.089, "finqa_10k": 0.062, "medmcqa": 0.185}

**Verdict — not overfitting:**
1. Classic gap small & stable: train−CV = 0.024 (SVM) / 0.044 (XGB), sd ≤ 0.0004 across seeds.
2. The leave-one-source-out CV estimate (0.845 held-out, cv_summary.md) **predicted the vault outcome** (0.82–0.84 on 4 completely fresh sources) — the generalisation estimate is honest, no hidden test fitting.
3. No per-set collapse on the vault (worst medmcqa 0.75–0.78; rejections 6–23 %) — unlike the single-source models (MedQuAD 0.39).
4. Seed variance ≤ 0.005 on every test metric — results are not seed luck.
5. Best config by unseen hmean: **XGB/MiniLM + tau(alpha=0.03): vault 0.799, new-OOD 0.907, hmean 0.850**; SVM argmax best vault acc 0.835. GQR-Bench drops to 0.93 vs the single-source 0.95–0.98 — that delta is the measured price of *not* overfitting to the benchmark's sources.
