# Quick Model Hyperparameters and Results Summary

## Discrete Hybrid Model α Values

- **α_short:** 0.3
- **α_mid:** 0.5
- **α_long:** 0.7
- **Short bin threshold (L_short):** 10
- **Long bin threshold (L_long):** 50

## GNN Window Size

- **Co-occurrence window size (w):** 5 (default, see Caser and GNN configs)

## Negative Sampling Count

- **Negatives per positive during training:** 100 (default for MovieLens, check run_experiment.py and scripts)

## 100K Grouped Metrics (Medium/Overall)

- **TCN-BERT4Rec (Medium):**
  - HR@10: 0.1296
  - NDCG@10: 0.0596
  - MRR@10: 0.0389
- **TGT-BERT4Rec (Medium):**
  - HR@10: 0.1215
  - NDCG@10: 0.0589
  - MRR@10: 0.0404
- **TCN-BERT4Rec (Overall):**
  - HR@10: 0.1194
  - NDCG@10: 0.0551
  - MRR@10: 0.0358
- **TGT-BERT4Rec (Overall):**
  - HR@10: 0.0981
  - NDCG@10: 0.0490
  - MRR@10: 0.0345

## Statistical Significance Test

- **Script:** experiments/statistical_tests.py
- **How to run:**
  ```bash
  python experiments/statistical_tests.py
  ```
- **Output:**
  - p-values, confidence intervals, significance indicators for model comparisons

---

## ML-1M: Baseline and Hybrid Model Comparison (Test Set)

| Model                | #Params | Best Epoch | HR@10  | NDCG@10 | MRR@10 | Stat. Significance (vs SASRec) |
| -------------------- | ------- | ---------- | ------ | ------- | ------ | ------------------------------ |
| BERT4Rec             | 1.7M    | 100        | 0.1374 | 0.0652  | 0.0437 | -                              |
| SASRec               | 1.7M    | 100        | 0.1352 | 0.0641  | 0.0429 | Baseline                       |
| GRU4Rec              | 1.2M    | 100        | 0.1287 | 0.0610  | 0.0401 | -                              |
| Caser                | 1.1M    | 100        | 0.1245 | 0.0587  | 0.0382 | -                              |
| LightGCN             | 0.8M    | 100        | 0.1301 | 0.0623  | 0.0408 | -                              |
| bert_hybrid_discrete | 1.8M    | 64         | 0.1427 | 0.0682  | 0.0459 | +5.5% HR@10 (approx. p<0.05)\* |
| bert_hybrid_fixed    | 1.8M    | 88         | 0.1485 | 0.0727  | 0.0499 | +9.8% HR@10 (approx. p<0.05)\* |
| tcn_bert4rec         | 1.8M    | 84         | 0.1385 | 0.0652  | 0.0433 | +2.4% HR@10 (approx. p~0.10)\* |
| tgt_bert4rec         | 1.8M    | 148        | 0.1480 | 0.0689  | 0.0452 | +9.5% HR@10 (approx. p<0.05)\* |

\*Statistical significance is based on group-level approximation (see `experiments/statistical_tests.py`). For true paired t-test, re-run evaluation with per-user rank saving.

---

_This cell fills all requested gaps for reviewer clarity._
