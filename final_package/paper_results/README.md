# Paper Results Summary

## Dataset
- **Name:** MovieLens-1M
- **Users:** 6,040
- **Items:** 3,706
- **Interactions:** 1,000,209
- **Test Users:** 6,034
- **Short Sequences (<10):** 162 users (2.7%)
- **Medium Sequences (10-50):** 5,872 users (97.3%)

## Models Evaluated
1. BERT4Rec (baseline)
2. BERT4Rec + GNN Fixed (α=0.5)
3. BERT4Rec + GNN Discrete (bin-based fusion)
4. BERT4Rec + GNN Learnable (learned fusion)
5. BERT4Rec + GNN Continuous (neural fusion)

## Best Results

### Overall Performance (NDCG@10)
bert_hybrid_fixed: 0.069412

### Cold-Start Performance (Short Sequences)
bert_hybrid_discrete: 0.119353

### Active User Performance (Medium Sequences)
bert_hybrid_fixed: 0.068344

## Key Findings

1. **Best Overall Model:** bert_hybrid_fixed
   - NDCG@10: 0.069412
   - Improvement vs baseline: 6.39%

2. **Parameter Efficiency:** 
   - Baseline: 340,544 parameters
   - Hybrids: +70.9% average overhead

3. **Convergence Speed:**
   - Average best epoch: 55.6
   - Range: 47-78 epochs

## Files Generated

### Tables (CSV + LaTeX)
- `table1_main_results` - Overall performance metrics
- `table2_length_analysis` - Performance by sequence length
- `table3_computational_costs` - Model size and efficiency
- `table4_significance` - Statistical significance tests

### Figures (PNG, 300 DPI)
- `performance_by_length.png` - NDCG@10 comparison by sequence length
- `training_dynamics.png` - Convergence and validation analysis
- `computational_costs.png` - Parameter and efficiency analysis

### Raw Data
- All experimental results in results/ directory
- Per-model config, history, and results JSON files
