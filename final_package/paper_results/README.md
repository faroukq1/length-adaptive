# Paper Results Summary

## Dataset
- **Name:** MovieLens-1M
- **Users:** 6,040
- **Items:** 3,706
- **Interactions:** 1,000,209
- **Test Users:** 6,034
- **Short Sequences (<10):** 162 users (2.7%)
- **Medium Sequences (10-50):** 5,872 users (97.3%)

## Models Evaluated (8 total)

### Baselines (4 models)
1. BERT4Rec (bidirectional transformer)
2. SASRec (unidirectional transformer)
3. GRU4Rec (RNN-based)
4. LightGCN (pure GNN)

### Hybrids (4 models)
5. BERT4Rec + GNN Fixed (α=0.5)
6. BERT4Rec + GNN Discrete (bin-based fusion)
7. BERT4Rec + GNN Learnable (learned fusion)
8. BERT4Rec + GNN Continuous (neural fusion)

## Best Results

### Overall Performance (NDCG@10)
- **Best Overall:** bert_hybrid_fixed (0.068992)
- **Best Baseline:** bert4rec (0.065240)
- **Best Hybrid:** bert_hybrid_fixed (0.068992)

### Hybrid Improvement Over Best Baseline
- **NDCG@10:** 5.75%
- **HR@10:** 5.31%
- **MRR@10:** 5.93%

## Key Findings

1. **Best Overall Model:** bert_hybrid_fixed
   - NDCG@10: 0.068992
   - Type: Hybrid

2. **Best Baseline:** bert4rec
   - NDCG@10: 0.065240

3. **Best Hybrid:** bert_hybrid_fixed
   - NDCG@10: 0.068992
   - Improvement vs best baseline: 5.75%

4. **Parameter Efficiency:** 
   - BERT4Rec baseline: 340,544 parameters
   - Hybrids average overhead: +70.9% vs BERT4Rec

5. **Convergence Speed:**
   - Average best epoch (all models): 57.6
   - Range: 1-113 epochs

## Files Generated

### Tables (CSV + LaTeX)
- `table1_main_results` - Overall performance metrics (all 8 models)
- `table2_length_analysis` - Performance by sequence length
- `table3_computational_costs` - Model size and efficiency
- `table4_significance` - Statistical significance tests (hybrids vs all baselines)

### Figures (PNG, 300 DPI)
- `performance_by_length.png` - NDCG@10 comparison by sequence length
- `training_dynamics.png` - Convergence and validation analysis
- `computational_costs.png` - Parameter and efficiency analysis

### Raw Data
- All experimental results in results/ directory
- Per-model config, history, and results JSON files
