# Cross-validation (leave-one-source-out + random 5-fold) — 4-class, argmax — 2026-08-19

Sources per domain (cap 3000, leakage-screened vs GQR val/test, background and each other): law = gqr | legal_reddit | legal_qa_v1 | legal_qa_ib; finance = gqr | banking77 | fin_instruct (DeividasM/financial-instruction-aq22) | reddit_finance (winddude/reddit_finance_43_250k); healthcare = gqr | icliniq | medquad | med_flashcards. Background v1 = raw wikitext+dolly (16 % ID-topic keyword hits); v2 = wikitext+dolly+yahoo(non-ID topics), ID-keyword- and classifier-filtered (0 hits).

Folds 1–3 hold out one *whole source per domain* (train on the other sources + bg) → `held-out` = mean routing acc on the held-out sources (mean ± sd over folds); fold 0 holds out the GQR sources (train on new sources only, test on the GQR ID test / GQR OOD test). `new-OOD` = mean rejection on the unseen OOD panel. Random 5-fold = stratified CV on the pooled sources+bg (train acc − CV acc = classic overfitting gap).

| model | emb | bg | **held-out source acc** (±sd) | law | finance | health | new-OOD rej. | hmean | fold0: GQR-ID / GQR-OOD / new-OOD | random-5-fold CV / train / gap |
|---|---|---|---|---|---|---|---|---|---|---|
| svm | baai | v1 | **0.807** ±0.144 | 0.932 | 0.590 | 0.898 | 0.811 | 0.804 | 0.899 / 0.823 / 0.767 | 0.966 / 0.980 / 0.014 |
| svm | baai | v2 | **0.813** ±0.126 | 0.934 | 0.585 | 0.919 | 0.840 | 0.823 | 0.938 / 0.910 / 0.824 | 0.971 / 0.983 / 0.013 |
| svm | mini | v1 | **0.838** ±0.057 | 0.938 | 0.679 | 0.897 | 0.856 | 0.846 | 0.885 / 0.849 / 0.810 | 0.960 / 0.986 / 0.025 |
| svm | mini | v2 | **0.845** ±0.039 | 0.939 | 0.687 | 0.908 | 0.870 | 0.857 | 0.919 / 0.889 / 0.849 | 0.966 / 0.988 / 0.022 |
| xgb | baai | v1 | **0.821** ±0.098 | 0.928 | 0.635 | 0.901 | 0.808 | 0.812 | 0.888 / 0.783 / 0.770 | 0.959 / 1.000 / 0.041 |
| xgb | baai | v2 | **0.819** ±0.087 | 0.928 | 0.616 | 0.914 | 0.854 | 0.835 | 0.921 / 0.894 / 0.838 | 0.964 / 1.000 / 0.036 |
| xgb | mini | v1 | **0.802** ±0.095 | 0.910 | 0.586 | 0.908 | 0.878 | 0.836 | 0.861 / 0.869 / 0.869 | 0.950 / 1.000 / 0.050 |
| xgb | mini | v2 | **0.820** ±0.058 | 0.916 | 0.628 | 0.916 | 0.905 | 0.860 | 0.902 / 0.895 / 0.901 | 0.957 / 1.000 / 0.043 |

Per fold (held-out sources: f1 = legal_reddit / banking77 / icliniq; f2 = legal_qa_v1 / fin_instruct / medquad; f3 = legal_qa_ib / reddit_finance / med_flashcards):

- svm/baai bg=v1: f0 law 0.95 fin 0.77 health 0.98 | OOD 0.77; f1 law 0.99 fin 0.77 health 0.99 | OOD 0.76; f2 law 0.85 fin 0.77 health 0.96 | OOD 0.88; f3 law 0.96 fin 0.24 health 0.74 | OOD 0.79
- svm/baai bg=v2: f0 law 0.95 fin 0.88 health 0.98 | OOD 0.82; f1 law 0.99 fin 0.71 health 0.99 | OOD 0.81; f2 law 0.86 fin 0.78 health 0.97 | OOD 0.89; f3 law 0.96 fin 0.26 health 0.79 | OOD 0.82
- svm/mini bg=v1: f0 law 0.90 fin 0.77 health 0.98 | OOD 0.81; f1 law 0.98 fin 0.65 health 0.99 | OOD 0.82; f2 law 0.87 fin 0.76 health 0.97 | OOD 0.92; f3 law 0.96 fin 0.62 health 0.73 | OOD 0.83
- svm/mini bg=v2: f0 law 0.94 fin 0.83 health 0.98 | OOD 0.85; f1 law 0.98 fin 0.60 health 0.99 | OOD 0.84; f2 law 0.88 fin 0.78 health 0.97 | OOD 0.92; f3 law 0.96 fin 0.68 health 0.76 | OOD 0.85
- xgb/baai bg=v1: f0 law 0.90 fin 0.78 health 0.98 | OOD 0.77; f1 law 0.99 fin 0.72 health 0.99 | OOD 0.77; f2 law 0.84 fin 0.77 health 0.95 | OOD 0.86; f3 law 0.95 fin 0.42 health 0.76 | OOD 0.79
- xgb/baai bg=v2: f0 law 0.90 fin 0.88 health 0.98 | OOD 0.84; f1 law 0.99 fin 0.65 health 0.99 | OOD 0.83; f2 law 0.85 fin 0.78 health 0.97 | OOD 0.90; f3 law 0.95 fin 0.42 health 0.79 | OOD 0.84
- xgb/mini bg=v1: f0 law 0.89 fin 0.72 health 0.97 | OOD 0.87; f1 law 0.98 fin 0.67 health 0.98 | OOD 0.86; f2 law 0.79 fin 0.74 health 0.95 | OOD 0.92; f3 law 0.96 fin 0.34 health 0.79 | OOD 0.85
- xgb/mini bg=v2: f0 law 0.92 fin 0.81 health 0.97 | OOD 0.90; f1 law 0.98 fin 0.61 health 0.98 | OOD 0.89; f2 law 0.81 fin 0.77 health 0.97 | OOD 0.93; f3 law 0.96 fin 0.50 health 0.80 | OOD 0.89

Per-OOD-set rejection, mean over folds:

| bg | model | emb | trec | agnews_ws | rotten_tomatoes | codealpaca | gsm8k | yahoo_se | tweets |
|---|---|---|---|---|---|---|---|---|---|
| v1 | svm | baai | 0.88 | 0.96 | 1.00 | 0.44 | 0.67 | 0.89 | 0.77 |
| v1 | svm | mini | 0.87 | 0.93 | 0.98 | 0.63 | 0.72 | 0.94 | 0.85 |
| v1 | xgb | baai | 0.87 | 0.94 | 0.99 | 0.49 | 0.69 | 0.89 | 0.71 |
| v1 | xgb | mini | 0.87 | 0.92 | 0.98 | 0.78 | 0.78 | 0.94 | 0.86 |
| v2 | svm | baai | 0.86 | 0.88 | 0.99 | 0.58 | 0.72 | 0.95 | 0.89 |
| v2 | svm | mini | 0.85 | 0.87 | 0.97 | 0.69 | 0.83 | 0.95 | 0.89 |
| v2 | xgb | baai | 0.85 | 0.90 | 0.99 | 0.66 | 0.72 | 0.95 | 0.88 |
| v2 | xgb | mini | 0.86 | 0.88 | 0.98 | 0.87 | 0.89 | 0.96 | 0.91 |
