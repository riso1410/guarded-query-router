| model | embedding | variant | rule | ID acc | OOD acc | **GQR** | τ | aux-val rej | latency ms | train s |
|---|---|---|---|---|---|---|---|---|---|---|
| XGBoost | baai | ctrl3 | argmax | 0.9974 | 0.0000 | **0.0000** |  |  | 6.58 | 2 |
| XGBoost | baai | ctrl3 | msp | 0.9768 | 0.6638 | **0.7905** | 0.987 |  | 6.58 | 2 |
| XGBoost | baai | oe4 | argmax | 0.9933 | 0.8630 | **0.9236** |  | 0.957 | 8.84 | 4 |
| XGBoost | baai | oe4 | tau | 0.9776 | 0.9593 | **0.9683** | 0.033 | 0.988 | 8.84 | 4 |
| XGBoost | mini | ctrl3 | argmax | 0.9960 | 0.0000 | **0.0000** |  |  | 3.72 | 2 |
| XGBoost | mini | ctrl3 | msp | 0.9801 | 0.5958 | **0.7411** | 0.987 |  | 3.72 | 2 |
| XGBoost | mini | oe4 | argmax | 0.9900 | 0.9042 | **0.9452** |  | 0.948 | 3.88 | 4 |
| XGBoost | mini | oe4 | tau | 0.9792 | 0.9594 | **0.9692** | 0.109 | 0.971 | 3.88 | 4 |
| XGBoost | tf_idf | ctrl3 | argmax | 0.9908 | 0.0000 | **0.0000** |  |  | 0.33 | 9 |
| XGBoost | tf_idf | ctrl3 | msp | 0.9787 | 0.1199 | **0.2136** | 0.804 |  | 0.33 | 9 |
| XGBoost | tf_idf | oe4 | argmax | 0.9732 | 0.7499 | **0.8471** |  | 0.903 | 0.33 | 18 |
| XGBoost | tf_idf | oe4 | tau | 0.9702 | 0.7628 | **0.8541** | 0.453 | 0.914 | 0.33 | 18 |
| SVM | baai | ctrl3 | argmax | 0.9987 | 0.0000 | **0.0000** |  |  | 6.19 | 22 |
| SVM | baai | ctrl3 | msp | 0.9772 | 0.7912 | **0.8744** | 0.995 |  | 6.19 | 22 |
| SVM | baai | oe4 | argmax | 0.9948 | 0.8728 | **0.9298** |  | 0.970 | 9.07 | 56 |
| SVM | baai | oe4 | tau | 0.9819 | 0.9620 | **0.9719** | 0.070 | 0.990 | 9.07 | 56 |
| SVM | mini | ctrl3 | argmax | 0.9956 | 0.0000 | **0.0000** |  |  | 3.84 | 54 |
| SVM | mini | ctrl3 | msp | 0.9763 | 0.7166 | **0.8265** | 0.952 |  | 3.84 | 54 |
| SVM | mini | oe4 | argmax | 0.9908 | 0.9174 | **0.9527** |  | 0.960 | 4.42 | 122 |
| SVM | mini | oe4 | tau | 0.9804 | 0.9710 | **0.9757** | 0.132 | 0.980 | 4.42 | 122 |
| SVM | tf_idf | ctrl3 | argmax | 0.9917 | 0.0000 | **0.0000** |  |  | 1.61 | 249 |
| SVM | tf_idf | ctrl3 | msp | 0.9747 | 0.3269 | **0.4896** | 0.906 |  | 1.61 | 249 |
| SVM | tf_idf | oe4 | argmax | 0.9842 | 0.8709 | **0.9241** |  | 0.960 | 2.81 | 526 |
| SVM | tf_idf | oe4 | tau | 0.9750 | 0.9154 | **0.9443** | 0.214 | 0.975 | 2.81 | 526 |
| fastText | own | ctrl3 | argmax | 0.9950 | 0.0000 | **0.0000** |  |  | 0.02 | 331 |
| fastText | own | ctrl3 | msp | 0.9761 | 0.4085 | **0.5760** | 0.990 |  | 0.02 | 331 |
| fastText | own | oe4 | argmax | 0.9917 | 0.7032 | **0.8229** |  | 0.964 | 0.02 | 312 |
| fastText | own | oe4 | tau | 0.9801 | 0.8806 | **0.9277** | 0.012 | 0.984 | 0.02 | 312 |
| WideMLP | own | ctrl3 | argmax | 0.9963 | 0.0000 | **0.0000** |  |  | 0.98 | 594 |
| WideMLP | own | ctrl3 | msp | 0.9794 | 0.7744 | **0.8649** | 0.908 |  | 0.98 | 594 |
| WideMLP | own | oe4 | argmax | 0.9918 | 0.7448 | **0.8507** |  | 0.970 | 1.03 | 547 |
| WideMLP | own | oe4 | tau | 0.9816 | 0.8931 | **0.9353** | 0.182 | 0.987 | 1.03 | 547 |
| BERT-multilingual | own | ctrl3 | argmax | 0.9991 | 0.0000 | **0.0000** |  |  | 8.89 | 469 |
| BERT-multilingual | own | ctrl3 | msp | 0.9812 | 0.6968 | **0.8149** | 0.999 |  | 8.89 | 469 |
| BERT-multilingual | own | oe4 | argmax | 0.9964 | 0.5457 | **0.7052** |  | 0.977 | 10.65 | 599 |
| BERT-multilingual | own | oe4 | tau | 0.9783 | 0.8424 | **0.9053** | 0.008 | 0.993 | 10.65 | 599 |
| ModernBERT | own | ctrl3 | argmax | 0.9998 | 0.0000 | **0.0000** |  |  | 13.10 | 789 |
| ModernBERT | own | ctrl3 | msp | 0.9822 | 0.7028 | **0.8194** | 1.000 |  | 13.10 | 789 |
| ModernBERT | own | oe4 | argmax | 0.9974 | 0.4285 | **0.5995** |  | 0.980 | 14.13 | 1012 |
| ModernBERT | own | oe4 | tau | 0.9806 | 0.7156 | **0.8274** | 0.011 | 0.994 | 14.13 | 1012 |

### Best rule per model: 4th background class (oe4) vs 3-class control (ctrl3, MSP reject)

| model | embedding | ctrl3/msp GQR | oe4 best GQR (rule) | Δ GQR | oe4 ID | oe4 OOD |
|---|---|---|---|---|---|---|
| XGBoost | baai | 0.7905 | **0.9683** (tau) | +0.1778 | 0.9776 | 0.9593 |
| XGBoost | mini | 0.7411 | **0.9692** (tau) | +0.2281 | 0.9792 | 0.9594 |
| XGBoost | tf_idf | 0.2136 | **0.8541** (tau) | +0.6405 | 0.9702 | 0.7628 |
| SVM | baai | 0.8744 | **0.9719** (tau) | +0.0975 | 0.9819 | 0.9620 |
| SVM | mini | 0.8265 | **0.9757** (tau) | +0.1492 | 0.9804 | 0.9710 |
| SVM | tf_idf | 0.4896 | **0.9443** (tau) | +0.4547 | 0.9750 | 0.9154 |
| fastText | own | 0.5760 | **0.9277** (tau) | +0.3517 | 0.9801 | 0.8806 |
| WideMLP | own | 0.8649 | **0.9353** (tau) | +0.0704 | 0.9816 | 0.8931 |
| BERT-multilingual | own | 0.8149 | **0.9053** (tau) | +0.0904 | 0.9783 | 0.8424 |
| ModernBERT | own | 0.8194 | **0.8274** (tau) | +0.0080 | 0.9806 | 0.7156 |

### Per-OOD-dataset accuracy (oe4, best rule)

| model | embedding | rule | jigsaw | olid | hate_xplain | hate_speech_slovak | dkhate | web_questions | ml_questions |
|---|---|---|---|---|---|---|---|---|---|
| XGBoost | baai | tau | 0.867 | 0.985 | 0.989 | 0.996 | 0.994 | 0.994 | 0.891 |
| XGBoost | mini | tau | 0.928 | 0.958 | 0.968 | 0.988 | 0.994 | 0.990 | 0.891 |
| XGBoost | tf_idf | tau | 0.569 | 0.837 | 0.659 | 0.776 | 0.954 | 0.951 | 0.594 |
| SVM | baai | tau | 0.887 | 0.991 | 0.991 | 0.991 | 0.988 | 0.997 | 0.891 |
| SVM | mini | tau | 0.952 | 0.964 | 0.986 | 0.972 | 0.994 | 0.992 | 0.938 |
| SVM | tf_idf | tau | 0.750 | 0.964 | 0.849 | 0.974 | 0.991 | 0.974 | 0.906 |
| fastText | own | tau | 0.666 | 0.912 | 0.785 | 0.957 | 0.964 | 0.974 | 0.906 |
| WideMLP | own | tau | 0.553 | 0.966 | 0.867 | 0.996 | 0.961 | 0.995 | 0.914 |
| BERT-multilingual | own | tau | 0.735 | 0.783 | 0.739 | 0.820 | 0.942 | 0.996 | 0.883 |
| ModernBERT | own | tau | 0.686 | 0.365 | 0.505 | 0.746 | 0.830 | 0.994 | 0.883 |
