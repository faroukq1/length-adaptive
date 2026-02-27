import pandas as pd
from scipy.stats import ttest_rel

# Load per-user metrics
f = 'notebooks/comparison_output_1m/group_metrics_overall.csv'
df = pd.read_csv(f)

# Extract per-user HR@10 and NDCG@10 for the three models
hr_bert4rec = df[df['Model'] == 'BERT4Rec']['HR@10'].values
hr_hybrid_fixed = df[df['Model'] == 'BERT-Hybrid-Fixed']['HR@10'].values
hr_hybrid_discrete = df[df['Model'] == 'BERT-Hybrid-Discrete']['HR@10'].values
ndcg_bert4rec = df[df['Model'] == 'BERT4Rec']['NDCG@10'].values
ndcg_hybrid_fixed = df[df['Model'] == 'BERT-Hybrid-Fixed']['NDCG@10'].values
ndcg_hybrid_discrete = df[df['Model'] == 'BERT-Hybrid-Discrete']['NDCG@10'].values

# Paired t-tests
hr_fixed_vs_bert = ttest_rel(hr_hybrid_fixed, hr_bert4rec)
hr_discrete_vs_bert = ttest_rel(hr_hybrid_discrete, hr_bert4rec)
ndcg_fixed_vs_bert = ttest_rel(ndcg_hybrid_fixed, ndcg_bert4rec)
ndcg_discrete_vs_bert = ttest_rel(ndcg_hybrid_discrete, ndcg_bert4rec)

print('Paired t-test results:')
print(f'BERT-Hybrid-Fixed vs BERT4Rec (HR@10): p={hr_fixed_vs_bert.pvalue:.4g}')
print(f'BERT-Hybrid-Discrete vs BERT4Rec (HR@10): p={hr_discrete_vs_bert.pvalue:.4g}')
print(f'BERT-Hybrid-Fixed vs BERT4Rec (NDCG@10): p={ndcg_fixed_vs_bert.pvalue:.4g}')
print(f'BERT-Hybrid-Discrete vs BERT4Rec (NDCG@10): p={ndcg_discrete_vs_bert.pvalue:.4g}')
