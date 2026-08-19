| model | embedding | variant | rule | ID acc | OOD acc | **GQR** | τ | aux-val rej | latency ms | train s |
|---|---|---|---|---|---|---|---|---|---|---|
| XGBoost | baai | oe4 | argmax | 0.9933 | 0.8630 | **0.9236** |  | 0.957 | 8.84 | 4 |
| XGBoost | mini | oe4 | argmax | 0.9900 | 0.9042 | **0.9452** |  | 0.948 | 3.88 | 4 |
| XGBoost | tf_idf | oe4 | argmax | 0.9732 | 0.7499 | **0.8471** |  | 0.903 | 0.33 | 18 |
| SVM | baai | oe4 | argmax | 0.9948 | 0.8728 | **0.9298** |  | 0.970 | 9.07 | 56 |
| SVM | mini | oe4 | argmax | 0.9908 | 0.9174 | **0.9527** |  | 0.960 | 4.42 | 122 |
| SVM | tf_idf | oe4 | argmax | 0.9842 | 0.8709 | **0.9241** |  | 0.960 | 2.81 | 526 |
| fastText | own | oe4 | argmax | 0.9917 | 0.7032 | **0.8229** |  | 0.964 | 0.02 | 312 |
| WideMLP | own | oe4 | argmax | 0.9918 | 0.7448 | **0.8507** |  | 0.970 | 1.03 | 547 |
| BERT-multilingual | own | oe4 | argmax | 0.9964 | 0.5457 | **0.7052** |  | 0.977 | 10.65 | 599 |
| ModernBERT | own | oe4 | argmax | 0.9974 | 0.4285 | **0.5995** |  | 0.980 | 14.13 | 1012 |

### Per-OOD-dataset accuracy (oe4, best rule)

| model | embedding | rule | jigsaw | olid | hate_xplain | hate_speech_slovak | dkhate | web_questions | ml_questions |
|---|---|---|---|---|---|---|---|---|---|
| XGBoost | baai | argmax | 0.641 | 0.937 | 0.931 | 0.955 | 0.988 | 0.979 | 0.609 |
| XGBoost | mini | argmax | 0.837 | 0.894 | 0.914 | 0.936 | 0.976 | 0.975 | 0.797 |
| XGBoost | tf_idf | argmax | 0.564 | 0.809 | 0.650 | 0.781 | 0.927 | 0.932 | 0.586 |
| SVM | baai | argmax | 0.700 | 0.949 | 0.954 | 0.837 | 0.979 | 0.979 | 0.711 |
| SVM | mini | argmax | 0.891 | 0.921 | 0.959 | 0.793 | 0.985 | 0.974 | 0.898 |
| SVM | tf_idf | argmax | 0.656 | 0.923 | 0.765 | 0.961 | 0.991 | 0.957 | 0.844 |
| fastText | own | argmax | 0.423 | 0.681 | 0.515 | 0.791 | 0.766 | 0.949 | 0.797 |
| WideMLP | own | argmax | 0.333 | 0.801 | 0.640 | 0.914 | 0.760 | 0.961 | 0.805 |
| BERT-multilingual | own | argmax | 0.530 | 0.271 | 0.479 | 0.250 | 0.663 | 0.955 | 0.672 |
| ModernBERT | own | argmax | 0.347 | 0.105 | 0.223 | 0.271 | 0.444 | 0.969 | 0.641 |
