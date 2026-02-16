"""
Analyze and compare results from all experiments

Usage:
    python experiments/analyze_results.py
    python experiments/analyze_results.py --results_dir results
"""

import argparse
import json
import os
import glob
from collections import defaultdict
import pandas as pd

def load_experiment_results(results_dir):
    """Load all experiment results from directory"""
    
    experiments = []
    
    # Find all result folders
    result_folders = glob.glob(os.path.join(results_dir, '*_*'))
    
    for folder in result_folders:
        # Extract model name from folder name
        folder_name = os.path.basename(folder)
        model_name = '_'.join(folder_name.split('_')[:-2])  # Remove timestamp
        
        # Load config
        config_path = os.path.join(folder, 'config.json')
        results_path = os.path.join(folder, 'results.json')
        
        if not os.path.exists(config_path) or not os.path.exists(results_path):
            continue
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        experiments.append({
            'folder': folder_name,
            'model': model_name,
            'config': config,
            'results': results
        })
    
    return experiments


def create_comparison_table(experiments):
    """Create comparison table of all experiments"""
    
    rows = []
    
    for exp in experiments:
        model = exp['model']
        results = exp['results']
        
        # Overall performance
        test_metrics = results['test_metrics']
        
        row = {
            'Model': model,
            'HR@5': test_metrics['hr@5'],
            'HR@10': test_metrics['hr@10'],
            'HR@20': test_metrics['hr@20'],
            'NDCG@5': test_metrics['ndcg@5'],
            'NDCG@10': test_metrics['ndcg@10'],
            'NDCG@20': test_metrics['ndcg@20'],
            'MRR@5': test_metrics['mrr@5'],
            'MRR@10': test_metrics['mrr@10'],
            'MRR@20': test_metrics['mrr@20'],
            'Best Epoch': results['best_epoch'],
            'Best Val NDCG@10': results['best_val_metric']
        }
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort by NDCG@10 (primary metric)
    df = df.sort_values('NDCG@10', ascending=False)
    
    return df


def create_grouped_comparison(experiments):
    """Compare performance across user groups"""
    
    groups = defaultdict(list)
    
    for exp in experiments:
        model = exp['model']
        grouped_metrics = exp['results']['grouped_metrics']
        
        for group_name, metrics in grouped_metrics.items():
            groups[group_name].append({
                'Model': model,
                'HR@10': metrics['hr@10'],
                'NDCG@10': metrics['ndcg@10'],
                'MRR@10': metrics['mrr@10']
            })
    
    # Create dataframes for each group
    group_dfs = {}
    for group_name, rows in groups.items():
        df = pd.DataFrame(rows)
        df = df.sort_values('NDCG@10', ascending=False)
        group_dfs[group_name] = df
    
    return group_dfs


def print_results(experiments):
    """Print comprehensive results"""
    
    print("\n" + "="*80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*80)
    
    # Overall comparison
    print("\n📊 Overall Performance (sorted by NDCG@10)")
    print("-"*80)
    
    df = create_comparison_table(experiments)
    
    # Format for display
    display_df = df.copy()
    for col in ['HR@5', 'HR@10', 'HR@20', 'NDCG@5', 'NDCG@10', 'NDCG@20', 
                'MRR@5', 'MRR@10', 'MRR@20', 'Best Val NDCG@10']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
    
    print(display_df.to_string(index=False))
    
    # Find best model
    best_model = df.iloc[0]
    print(f"\n🏆 Best Model: {best_model['Model']}")
    print(f"   NDCG@10: {best_model['NDCG@10']:.4f}")
    print(f"   HR@10: {best_model['HR@10']:.4f}")
    print(f"   MRR@10: {best_model['MRR@10']:.4f}")
    
    # Grouped comparison
    print("\n\n📈 Performance by User Group")
    print("-"*80)
    
    group_dfs = create_grouped_comparison(experiments)
    
    for group_name, df in sorted(group_dfs.items()):
        print(f"\n{group_name.upper()}:")
        
        display_df = df.copy()
        for col in ['HR@10', 'NDCG@10', 'MRR@10']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
        
        print(display_df.to_string(index=False))
    
    # Improvements analysis
    print("\n\n📌 Key Insights")
    print("-"*80)
    
    # Find SASRec baseline
    sasrec_exp = next((e for e in experiments if e['model'] == 'sasrec'), None)
    
    if sasrec_exp:
        sasrec_ndcg = sasrec_exp['results']['test_metrics']['ndcg@10']
        
        print(f"\nSASRec baseline NDCG@10: {sasrec_ndcg:.4f}")
        print("\nImprovements over SASRec:")
        
        for exp in experiments:
            if exp['model'] != 'sasrec':
                ndcg = exp['results']['test_metrics']['ndcg@10']
                improvement = ((ndcg - sasrec_ndcg) / sasrec_ndcg) * 100
                
                symbol = "✓" if improvement > 0 else "✗"
                print(f"  {symbol} {exp['model']}: {improvement:+.2f}%")
    
    # Group-specific insights
    print("\nBest model per group:")
    for group_name, df in sorted(group_dfs.items()):
        best = df.iloc[0]
        print(f"  • {group_name}: {best['Model']} (NDCG@10={best['NDCG@10']:.4f})")
    
    print("\n" + "="*80)
    print(f"\nTotal experiments analyzed: {len(experiments)}")
    print("\nDetailed results saved in each experiment folder:")
    for exp in experiments:
        print(f"  • {exp['folder']}")
    print("\n" + "="*80 + "\n")


def save_comparison_csv(experiments, output_dir='results'):
    """Save comparison tables as CSV files"""
    
    # Overall comparison
    df = create_comparison_table(experiments)
    df.to_csv(os.path.join(output_dir, 'overall_comparison.csv'), index=False)
    print(f"✓ Saved overall comparison to: {output_dir}/overall_comparison.csv")
    
    # Grouped comparison
    group_dfs = create_grouped_comparison(experiments)
    for group_name, df in group_dfs.items():
        filename = f'comparison_{group_name}.csv'
        df.to_csv(os.path.join(output_dir, filename), index=False)
        print(f"✓ Saved {group_name} comparison to: {output_dir}/{filename}")


def main(args):
    """Main analysis function"""
    
    # Load all experiments
    experiments = load_experiment_results(args.results_dir)
    
    if len(experiments) == 0:
        print(f"❌ No experiment results found in: {args.results_dir}")
        print("\nMake sure to run experiments first:")
        print("  python experiments/run_experiment.py --model hybrid_discrete --epochs 50")
        return
    
    # Print results
    print_results(experiments)
    
    # Save CSV files
    if args.save_csv:
        save_comparison_csv(experiments, args.results_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze experiment results')
    
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory containing experiment results')
    parser.add_argument('--save_csv', action='store_true',
                       help='Save comparison tables as CSV files')
    
    args = parser.parse_args()
    
    main(args)
