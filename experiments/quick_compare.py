"""
Quick comparison of all experiment results

Usage:
    source venv/bin/activate
    python experiments/quick_compare.py
"""

import json
import os
from pathlib import Path
import sys

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_results(results_dir='results'):
    """Load all experiment results"""
    experiments = {}
    
    result_folders = []
    for p in sorted(Path(results_dir).rglob('results.json')):
        folder = str(p.parent)
        if 'comparison_output' not in folder:
            result_folders.append(folder)
    
    for folder in result_folders:
        model_name = os.path.relpath(folder, results_dir)
        
        results_path = os.path.join(folder, 'results.json')
        
        if not os.path.exists(results_path):
            continue
        
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        experiments[model_name] = results
    
    return experiments


def print_comparison():
    """Print quick comparison table"""
    
    experiments = load_results()
    
    if not experiments:
        print("No results found in results/ directory")
        return
    
    print("="*100)
    print("QUICK RESULTS COMPARISON")
    print("="*100)
    
    # Overall metrics
    print("\n📊 OVERALL PERFORMANCE (Test Set)")
    print("-"*100)
    print(f"{'Model':<25} | {'HR@10':>8} | {'NDCG@10':>8} | {'MRR@10':>8} | {'HR@20':>8} | {'Best Epoch':>10}")
    print("-"*100)
    
    # Sort by NDCG@10
    sorted_models = sorted(experiments.items(), 
                          key=lambda x: x[1]['test_metrics']['NDCG@10'], 
                          reverse=True)
    
    baseline_hr10 = None
    baseline_ndcg10 = None
    
    for model_name, results in sorted_models:
        metrics = results['test_metrics']
        hr10 = metrics['HR@10']
        ndcg10 = metrics['NDCG@10']
        mrr10 = metrics['MRR@10']
        hr20 = metrics['HR@20']
        best_epoch = results['best_epoch']
        
        # Track baseline
        if 'sasrec' in model_name.lower() and 'hybrid' not in model_name.lower():
            baseline_hr10 = hr10
            baseline_ndcg10 = ndcg10
            marker = " 📌 BASELINE"
        else:
            marker = ""
        
        print(f"{model_name:<25} | {hr10:>7.4f} | {ndcg10:>7.4f} | {mrr10:>7.4f} | {hr20:>7.4f} | {best_epoch:>10}{marker}")
    
    # Short-history performance
    print("\n🔥 SHORT-HISTORY USERS PERFORMANCE (L ≤ 10)")
    print("-"*100)
    print(f"{'Model':<25} | {'HR@10':>8} | {'NDCG@10':>8} | {'MRR@10':>8} | {'Count':>8} | {'Improvement'}")
    print("-"*100)
    
    baseline_short_hr10 = None
    
    for model_name, results in sorted_models:
        if 'short' not in results['grouped_metrics']:
            continue
        
        short_metrics = results['grouped_metrics']['short']
        hr10 = short_metrics['HR@10']
        ndcg10 = short_metrics['NDCG@10']
        mrr10 = short_metrics['MRR@10']
        count = short_metrics['count']
        
        if 'sasrec' in model_name.lower() and 'hybrid' not in model_name.lower():
            baseline_short_hr10 = hr10
            imp_str = "baseline"
        elif baseline_short_hr10:
            improvement = ((hr10 - baseline_short_hr10) / baseline_short_hr10) * 100
            imp_str = f"+{improvement:.1f}%"
        else:
            imp_str = "N/A"
        
        print(f"{model_name:<25} | {hr10:>7.4f} | {ndcg10:>7.4f} | {mrr10:>7.4f} | {count:>8} | {imp_str}")
    
    # Medium-history performance
    print("\n📈 MEDIUM-HISTORY USERS PERFORMANCE (10 < L ≤ 50)")
    print("-"*100)
    print(f"{'Model':<25} | {'HR@10':>8} | {'NDCG@10':>8} | {'MRR@10':>8} | {'Count':>8} | {'Improvement'}")
    print("-"*100)
    
    baseline_med_hr10 = None
    
    for model_name, results in sorted_models:
        if 'medium' not in results['grouped_metrics']:
            continue
        
        med_metrics = results['grouped_metrics']['medium']
        hr10 = med_metrics['HR@10']
        ndcg10 = med_metrics['NDCG@10']
        mrr10 = med_metrics['MRR@10']
        count = med_metrics['count']
        
        if 'sasrec' in model_name.lower() and 'hybrid' not in model_name.lower():
            baseline_med_hr10 = hr10
            imp_str = "baseline"
        elif baseline_med_hr10:
            improvement = ((hr10 - baseline_med_hr10) / baseline_med_hr10) * 100
            if improvement > 0:
                imp_str = f"+{improvement:.1f}%"
            else:
                imp_str = f"{improvement:.1f}%"
        else:
            imp_str = "N/A"
        
        print(f"{model_name:<25} | {hr10:>7.4f} | {ndcg10:>7.4f} | {mrr10:>7.4f} | {count:>8} | {imp_str}")
    
    # Check for long-history
    has_long = any('long' in results['grouped_metrics'] for results in experiments.values())
    
    if has_long:
        print("\n📊 LONG-HISTORY USERS PERFORMANCE (L > 50)")
        print("-"*100)
        print(f"{'Model':<25} | {'HR@10':>8} | {'NDCG@10':>8} | {'MRR@10':>8} | {'Count':>8}")
        print("-"*100)
        
        for model_name, results in sorted_models:
            if 'long' not in results['grouped_metrics']:
                continue
            
            long_metrics = results['grouped_metrics']['long']
            hr10 = long_metrics['HR@10']
            ndcg10 = long_metrics['NDCG@10']
            mrr10 = long_metrics['MRR@10']
            count = long_metrics['count']
            
            print(f"{model_name:<25} | {hr10:>7.4f} | {ndcg10:>7.4f} | {mrr10:>7.4f} | {count:>8}")
    else:
        print("\n⚠️  WARNING: Long-history user results are MISSING!")
        print("   → Re-run experiments with long_thresh=50 to get complete analysis")
    
    # Summary
    print("\n" + "="*100)
    print("📝 SUMMARY")
    print("="*100)
    
    if baseline_hr10 and baseline_ndcg10:
        best_model = sorted_models[0]
        best_name = best_model[0]
        best_hr10 = best_model[1]['test_metrics']['HR@10']
        best_ndcg10 = best_model[1]['test_metrics']['NDCG@10']
        
        hr_imp = ((best_hr10 - baseline_hr10) / baseline_hr10) * 100
        ndcg_imp = ((best_ndcg10 - baseline_ndcg10) / baseline_ndcg10) * 100
        
        print(f"\n✅ Best Model: {best_name}")
        print(f"   Overall HR@10:   {best_hr10:.4f} ({hr_imp:+.1f}% vs SASRec)")
        print(f"   Overall NDCG@10: {best_ndcg10:.4f} ({ndcg_imp:+.1f}% vs SASRec)")
        
        if baseline_short_hr10:
            best_short_hr10 = max((r['grouped_metrics'].get('short', {}).get('HR@10', 0), n) 
                                 for n, r in experiments.items() if 'short' in r['grouped_metrics'])[0]
            short_imp = ((best_short_hr10 - baseline_short_hr10) / baseline_short_hr10) * 100
            print(f"   Short HR@10:     {best_short_hr10:.4f} ({short_imp:+.1f}% vs SASRec)")
        
        if hr_imp > 2 and short_imp > 20:
            print("\n🎯 SUCCESS CRITERIA MET:")
            print("   ✓ Beat SASRec on overall HR@10 by ≥2%")
            print("   ✓ Beat SASRec on short-history HR@10 by ≥20%")
        else:
            print("\n⚠️  SUCCESS CRITERIA NOT FULLY MET:")
            if hr_imp < 2:
                print(f"   ✗ Overall improvement {hr_imp:.1f}% < 2% target")
            if short_imp < 20:
                print(f"   ✗ Short-user improvement {short_imp:.1f}% < 20% target")
    
    print("\n" + "="*100)


if __name__ == '__main__':
    print_comparison()
