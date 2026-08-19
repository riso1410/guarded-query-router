# New-OOD panel (never seen by any model) — plain 4-class models, argmax — 2026-08-19

Sets (cap 1000, seed 22, after leakage screen): trec n=496, agnews_ws n=1000, rotten_tomatoes n=1000, codealpaca n=999, gsm8k n=1000, yahoo_se n=999, tweets n=1000, banking77 n=995

Leakage screen (normalised exact + 8-word shingle vs aux corpus, GQR train/val/ID-test, GQR OOD-test): dropped trec: {'exact:aux(wikitext+dolly)': 4}; codealpaca: {'shingle:aux(wikitext+dolly)': 1}; yahoo_se: {'shingle:aux(wikitext+dolly)': 1}; banking77: {'shingle:gqr_train': 2, 'shingle:gqr_val': 3}.
Aux (wikitext+dolly, training only) vs GQR test sets: {'shingle:gqr_id_test': 2, 'exact:gqr_id_test': 1, 'exact:gqr_ood_test': 1} of 9600 (benign: 2 boilerplate 8-grams in long law questions, 'What is inflation?' (dolly↔finance ID-test, adverse), 'What is deep learning?' (dolly↔ml_questions, 1/128)).

`new-OOD acc` = mean per-dataset rejection rate (prediction == background); `GQR_new` = hmean(ID acc, new-OOD acc).

| model | emb | ID | GQR-Bench OOD | **new-OOD** | GQR-Bench | **GQR_new** | trec | agnews | rotten | codealpaca | gsm8k | yahoo | tweets | banking77 → finance / rejected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| XGBoost | baai | 0.993 | 0.863 | **0.875** | 0.924 | **0.930** | 0.960 | 0.913 | 0.996 | 0.864 | 0.756 | 0.901 | 0.734 | 0.89 / 0.07 |
| XGBoost | mini | 0.990 | 0.904 | **0.929** | 0.945 | **0.959** | 0.960 | 0.899 | 0.984 | 0.926 | 0.878 | 0.943 | 0.914 | 0.77 / 0.20 |
| XGBoost | tf_idf | 0.973 | 0.750 | **0.742** | 0.847 | **0.842** | 0.938 | 0.932 | 0.770 | 0.743 | 0.660 | 0.640 | 0.514 | 0.78 / 0.16 |
| SVM | baai | 0.995 | 0.873 | **0.897** | 0.930 | **0.944** | 0.964 | 0.965 | 0.999 | 0.955 | 0.810 | 0.905 | 0.683 | 0.89 / 0.09 |
| SVM | mini | 0.991 | 0.917 | **0.949** | 0.953 | **0.969** | 0.960 | 0.937 | 0.994 | 0.970 | 0.928 | 0.968 | 0.883 | 0.71 / 0.23 |
| SVM | tf_idf | 0.984 | 0.871 | **0.866** | 0.924 | **0.921** | 0.966 | 0.958 | 0.921 | 0.918 | 0.821 | 0.790 | 0.687 | 0.60 / 0.16 |
| fastText | own | 0.992 | 0.703 | **0.729** | 0.823 | **0.840** | 0.966 | 0.708 | 0.990 | 0.815 | 0.557 | 0.683 | 0.386 | 0.68 / 0.12 |
| WideMLP | own | 0.992 | 0.745 | **0.748** | 0.851 | **0.853** | 0.992 | 0.856 | 0.990 | 0.676 | 0.593 | 0.719 | 0.407 | 0.71 / 0.14 |
| BERT-multilingual | own | 0.996 | 0.546 | **0.831** | 0.705 | **0.906** | 0.976 | 0.699 | 0.963 | 0.959 | 0.960 | 0.692 | 0.565 | 0.76 / 0.22 |
| ModernBERT | own | 0.997 | 0.428 | **0.852** | 0.600 | **0.919** | 0.984 | 0.990 | 0.986 | 0.951 | 0.937 | 0.826 | 0.290 | 0.87 / 0.10 |

(Thresholded τ / 3-class-control rows remain in `results/new_ood.csv`.)
