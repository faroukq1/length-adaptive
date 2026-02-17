"""
Statistical significance testing for model comparisons

Usage:
    source venv/bin/activate
    python experiments/statistical_tests.py
"""

import json
import pickle
import sys
from pathlib import Path
import torch

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_model_ranks(model_name, results_dir='results', data_path='data/ml-1m/processed/ml1m_sequential.pkl'):
    """
    Load per-user ranks for a model
    
    Note: This requires re-running evaluation with rank saving enabled.
    For now, we'll use grouped metrics for approximation.
    """
    import glob
    import os
    
    # Find model results
    pattern = os.path.join(results_dir, f'{model_name}_*', 'results.json')
    matches = glob.glob(pattern)
    
    if not matches:
        return None
    
    with open(matches[0], 'r') as f:
        results = json.load(f)
    
    return results


def bootstrap_confidence_interval(sample1, sample2, n_bootstrap=1000, alpha=0.05):
    """
    Compute bootstrap confidence interval for difference in means
    
    Args:
        sample1: First sample (e.g., baseline metrics)
        sample2: Second sample (e.g., hybrid metrics)
        n_bootstrap: Number of bootstrap samples
        alpha: Significance level
    
    Returns:
        ci_low, ci_high, mean_diff
    """
    import random
    
    diffs = []
    n = min(len(sample1), len(sample2))
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        idx = [random.randint(0, n-1) for _ in range(n)]
        s1_boot = [sample1[i] for i in idx]
        s2_boot = [sample2[i] for i in idx]
        
        mean_diff = sum(s2_boot) / len(s2_boot) - sum(s1_boot) / len(s1_boot)
        diffs.append(mean_diff)
    
    diffs.sort()
    ci_low = diffs[int(alpha/2 * n_bootstrap)]
    ci_high = diffs[int((1 - alpha/2) * n_bootstrap)]
    mean_diff = sum(diffs) / len(diffs)
    
    return ci_low, ci_high, mean_diff


def approximate_significance_test():
    """
    Approximate significance testing using grouped metrics
    
    Note: True significance requires per-user ranks.
    This approximation assumes normal distribution within groups.
    """
    
    print("="*80)
    print("STATISTICAL SIGNIFICANCE TESTING (Approximation)")
    print("="*80)
    print("\n⚠️  Note: True significance requires per-user rank data.")
    print("   This analysis uses group-level statistics as approximation.\n")
    
    # Load results
    sasrec = load_model_ranks('sasrec')
    hybrid_fixed = load_model_ranks('hybrid_fixed')
    hybrid_continuous = load_model_ranks('hybrid_continuous')
    hybrid_learnable = load_model_ranks('hybrid_learnable')
    
    if not sasrec:
        print("❌ SASRec baseline results not found!")
        return
    
    models = {
        'Hybrid Fixed': hybrid_fixed,
        'Hybrid Continuous': hybrid_continuous,
        'Hybrid Learnable': hybrid_learnable
    }
    
    # Compare each hybrid vs baseline
    for model_name, model_results in models.items():
        if not model_results:
            continue
        
        print(f"\n{'='*80}")
        print(f"COMPARISON: {model_name} vs SASRec Baseline")
        print(f"{'='*80}")
        
        # Overall comparison
        print(f"\n📊 Overall Metrics:")
        print("-"*80)
        
        sasrec_hr10 = sasrec['test_metrics']['HR@10']
        model_hr10 = model_results['test_metrics']['HR@10']
        hr10_diff = model_hr10 - sasrec_hr10
        hr10_rel = (hr10_diff / sasrec_hr10) * 100
        
        sasrec_ndcg10 = sasrec['test_metrics']['NDCG@10']
        model_ndcg10 = model_results['test_metrics']['NDCG@10']
        ndcg10_diff = model_ndcg10 - sasrec_ndcg10
        ndcg10_rel = (ndcg10_diff / sasrec_ndcg10) * 100
        
        print(f"HR@10:")
        print(f"  SASRec:      {sasrec_hr10:.6f}")
        print(f"  {model_name}: {model_hr10:.6f}")
        print(f"  Difference:  {hr10_diff:+.6f} ({hr10_rel:+.2f}%)")
        
        print(f"\nNDCG@10:")
        print(f"  SASRec:      {sasrec_ndcg10:.6f}")
        print(f"  {model_name}: {model_ndcg10:.6f}")
        print(f"  Difference:  {ndcg10_diff:+.6f} ({ndcg10_rel:+.2f}%)")
        
        # Group-level comparison
        for group in ['short', 'medium']:
            if group not in sasrec['grouped_metrics']:
                continue
            if group not in model_results['grouped_metrics']:
                continue
            
            print(f"\n📊 {group.title()}-History Users:")
            print("-"*80)
            
            sasrec_metrics = sasrec['grouped_metrics'][group]
            model_metrics = model_results['grouped_metrics'][group]
            
            s_hr10 = sasrec_metrics['HR@10']
            m_hr10 = model_metrics['HR@10']
            diff = m_hr10 - s_hr10
            rel = (diff / s_hr10) * 100 if s_hr10 > 0 else 0
            
            count = sasrec_metrics['count']
            
            print(f"HR@10:")
            print(f"  SASRec:      {s_hr10:.6f}")
            print(f"  {model_name}: {m_hr10:.6f}")
            print(f"  Difference:  {diff:+.6f} ({rel:+.2f}%)")
            print(f"  Sample size: {count} users")
            
            # Approximate significance (very rough)
            # Assume binomial distribution for HR@10
            # var = p * (1-p) / n
            var_s = s_hr10 * (1 - s_hr10) / count
            var_m = m_hr10 * (1 - m_hr10) / count
            se = (var_s + var_m) ** 0.5
            
            if se > 0:
                z_score = diff / se
                # Two-tailed test
                if abs(z_score) > 1.96:  # 95% confidence
                    sig = "✅ Significant (p < 0.05)"
                elif abs(z_score) > 1.645:  # 90% confidence
                    sig = "⚠️  Marginally significant (p < 0.10)"
                else:
                    sig = "❌ Not significant (p ≥ 0.10)"
                
                print(f"  Z-score:     {z_score:.2f}")
                print(f"  Significance: {sig}")
            
        # Check for long group
        if 'long' in sasrec['grouped_metrics'] and 'long' in model_results['grouped_metrics']:
            print(f"\n📊 Long-History Users:")
            print("-"*80)
            
            sasrec_metrics = sasrec['grouped_metrics']['long']
            model_metrics = model_results['grouped_metrics']['long']
            
            s_hr10 = sasrec_metrics['HR@10']
            m_hr10 = model_metrics['HR@10']
            diff = m_hr10 - s_hr10
            rel = (diff / s_hr10) * 100 if s_hr10 > 0 else 0
            
            print(f"HR@10:")
            print(f"  SASRec:      {s_hr10:.6f}")
            print(f"  {model_name}: {m_hr10:.6f}")
            print(f"  Difference:  {diff:+.6f} ({rel:+.2f}%)")
        else:
            print(f"\n⚠️  Long-history user metrics not available")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print("\n📌 Key Findings:")
    
    # Find best model
    best_overall = max(models.items(), 
                      key=lambda x: x[1]['test_metrics']['HR@10'] if x[1] else 0)
    
    if best_overall[1]:
        best_name = best_overall[0]
        improvement = (best_overall[1]['test_metrics']['HR@10'] - sasrec['test_metrics']['HR@10'])
        rel_imp = (improvement / sasrec['test_metrics']['HR@10']) * 100
        
        print(f"\n1. Best Overall Model: {best_name}")
        print(f"   HR@10 improvement: {improvement:+.4f} ({rel_imp:+.2f}%)")
        
        if 'short' in best_overall[1]['grouped_metrics']:
            short_imp = (best_overall[1]['grouped_metrics']['short']['HR@10'] - 
                        sasrec['grouped_metrics']['short']['HR@10'])
            short_rel = (short_imp / sasrec['grouped_metrics']['short']['HR@10']) * 100
            print(f"\n2. Short-History Impact:")
            print(f"   HR@10 improvement: {short_imp:+.4f} ({short_rel:+.2f}%)")
            
            if short_rel > 20:
                print(f"   ✅ Exceeds 20% improvement target!")
            else:
                print(f"   ⚠️  Below 20% improvement target")
    
    print("\n💡 Recommendations:")
    print("   1. Re-run evaluation saving per-user ranks for exact significance tests")
    print("   2. Implement paired t-test with saved rank data")
    print("   3. Consider McNemar's test for hit/miss comparisons")
    print("   4. Add bootstrap confidence intervals")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    approximate_significance_test()
