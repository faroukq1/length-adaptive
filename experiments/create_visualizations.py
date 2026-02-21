"""
Create visualizations for results

Requires: matplotlib, seaborn (install in venv if needed)

Usage:
    source venv/bin/activate
    pip install matplotlib seaborn  # if not installed
    python experiments/create_visualizations.py
"""

import json
import pickle
import sys
from pathlib import Path
import glob

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
except ImportError:
    print("❌ matplotlib not installed!")
    print("   Run: pip install matplotlib seaborn")
    sys.exit(1)


def load_results(results_dir='results'):
    """Load all experiment results"""
    experiments = {}
    
    result_folders = glob.glob(f'{results_dir}/*_*')
    
    for folder in result_folders:
        folder_name = Path(folder).name
        model_name = '_'.join(folder_name.split('_')[:-2])
        
        results_path = Path(folder) / 'results.json'
        if not results_path.exists():
            continue
        
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        experiments[model_name] = results
    
    return experiments


def plot_performance_by_length(experiments, output_dir='data/graphs'):
    """Plot performance comparison across length groups"""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Prepare data
    models = []
    short_hr = []
    medium_hr = []
    overall_hr = []
    
    for model_name, results in experiments.items():
        models.append(model_name.replace('_', '\n'))
        
        grouped = results['grouped_metrics']
        short_hr.append(grouped.get('short', {}).get('HR@10', 0) * 100)
        medium_hr.append(grouped.get('medium', {}).get('HR@10', 0) * 100)
        overall_hr.append(results['test_metrics']['HR@10'] * 100)
    
    # Create bar chart
    import numpy as np
    
    x = np.arange(len(models))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width, short_hr, width, label='Short (≤10)', color='#e74c3c')
    bars2 = ax.bar(x, medium_hr, width, label='Medium (11-50)', color='#3498db')
    bars3 = ax.bar(x + width, overall_hr, width, label='Overall', color='#2ecc71')
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('HR@10 (%)', fontsize=12, fontweight='bold')
    ax.set_title('Performance Comparison Across User Length Groups', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    autolabel(bars1)
    autolabel(bars2)
    autolabel(bars3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'performance_by_length.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_alpha_function(output_dir='data/graphs'):
    """Plot alpha as function of sequence length"""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    import numpy as np
    
    lengths = np.arange(1, 101)
    
    # Discrete bins
    discrete_alpha = np.where(lengths <= 10, 0.3,
                             np.where(lengths <= 50, 0.5, 0.7))
    
    # Continuous (sigmoid-based)
    def continuous_alpha(L, w=-0.05, b=2.0):
        return 1.0 / (1.0 + np.exp(-(w * np.log(L + 1) + b)))
    
    cont_alpha = continuous_alpha(lengths)
    
    # Fixed
    fixed_alpha = np.ones_like(lengths) * 0.5
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(lengths, fixed_alpha, '--', label='Fixed (α=0.5)', color='gray', linewidth=2)
    ax.plot(lengths, discrete_alpha, 'o-', label='Discrete Bins', color='#3498db', linewidth=2, markersize=3)
    ax.plot(lengths, cont_alpha, '-', label='Continuous', color='#e74c3c', linewidth=2)
    
    # Mark thresholds
    ax.axvline(10, color='black', linestyle=':', alpha=0.5, label='Short/Med threshold')
    ax.axvline(50, color='black', linestyle=':', alpha=0.3, label='Med/Long threshold')
    
    # Regions
    ax.axvspan(0, 10, alpha=0.1, color='red', label='Short region')
    ax.axvspan(10, 50, alpha=0.1, color='blue')
    ax.axvspan(50, 100, alpha=0.1, color='green')
    
    ax.set_xlabel('Sequence Length (L)', fontsize=12, fontweight='bold')
    ax.set_ylabel('α (Weight for SASRec)', fontsize=12, fontweight='bold')
    ax.set_title('Fusion Weight α as Function of Sequence Length', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Add annotations
    ax.text(5, 0.35, 'More GNN\n(cold-start)', ha='center', fontsize=9, style='italic')
    ax.text(30, 0.55, 'Balanced', ha='center', fontsize=9, style='italic')
    ax.text(75, 0.75, 'More SASRec\n(warm users)', ha='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'alpha_function.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_user_distribution(data_path='data/ml-1m/processed/ml1m_sequential.pkl',
                          output_dir='data/graphs'):
    """Plot histogram of user sequence lengths"""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    test_seqs = data['test_sequences']
    lengths = [len(seq['sequence']) for seq in test_seqs]
    
    # Create histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bins = range(0, max(lengths) + 5, 5)
    counts, bins, patches = ax.hist(lengths, bins=bins, color='#3498db', edgecolor='black', alpha=0.7)
    
    # Color regions
    for i, patch in enumerate(patches):
        if bins[i] <= 10:
            patch.set_facecolor('#e74c3c')  # Red for short
        elif bins[i] <= 50:
            patch.set_facecolor('#3498db')  # Blue for medium
        else:
            patch.set_facecolor('#2ecc71')  # Green for long
    
    # Add threshold lines
    ax.axvline(10, color='red', linestyle='--', linewidth=2, label='Short threshold (L=10)')
    ax.axvline(50, color='green', linestyle='--', linewidth=2, label='Long threshold (L=50)')
    
    ax.set_xlabel('Sequence Length', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Users', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of User Sequence Lengths (Test Set)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add stats
    import statistics
    mean_len = statistics.mean(lengths)
    median_len = statistics.median(lengths)
    
    stats_text = f'Mean: {mean_len:.1f}\nMedian: {median_len:.0f}\nTotal: {len(lengths)}'
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'user_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def main():
    """Generate all visualizations"""
    
    print("="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    # Check matplotlib
    try:
        import matplotlib
    except ImportError:
        print("\n❌ Error: matplotlib not installed")
        print("   Install with: pip install matplotlib")
        return
    
    print("\n[1/3] Loading experiment results...")
    experiments = load_results()
    
    if not experiments:
        print("   ❌ No experiment results found!")
        return
    
    print(f"   ✓ Loaded {len(experiments)} experiments")
    
    print("\n[2/3] Creating plots...")
    
    print("   Creating performance comparison...")
    plot_performance_by_length(experiments)
    
    print("   Creating alpha function plot...")
    plot_alpha_function()
    
    print("   Creating user distribution histogram...")
    try:
        plot_user_distribution()
    except Exception as e:
        print(f"   ⚠️  Could not create distribution plot: {e}")
    
    print("\n[3/3] Done!")
    print("\n" + "="*70)
    print("📊 Visualizations saved to: data/graphs/")
    print("   - performance_by_length.png")
    print("   - alpha_function.png")
    print("   - user_distribution.png")
    print("="*70)


if __name__ == '__main__':
    main()
