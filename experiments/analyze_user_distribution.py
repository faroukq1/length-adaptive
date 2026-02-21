"""
Analyze user distribution by sequence length

Usage:
    source venv/bin/activate
    python experiments/analyze_user_distribution.py
"""

import pickle
import sys
from pathlib import Path
from collections import Counter

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def analyze_distribution(data_path='data/ml-1m/processed/ml1m_sequential.pkl',
                        short_thresh=10, long_thresh=50):
    """Analyze user distribution across length bins"""
    
    print("="*70)
    print("USER DISTRIBUTION ANALYSIS")
    print("="*70)
    
    # Load data
    print(f"\nLoading data from: {data_path}")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # Analyze each split
    for split_name in ['train', 'val', 'test']:
        sequences = data[f'{split_name}_sequences']
        lengths = [len(seq['sequence']) for seq in sequences]
        
        # Count by bin
        short = sum(1 for l in lengths if l <= short_thresh)
        medium = sum(1 for l in lengths if short_thresh < l <= long_thresh)
        long_count = sum(1 for l in lengths if l > long_thresh)
        total = len(lengths)
        
        print(f"\n{split_name.upper()} SET:")
        print("-"*70)
        print(f"{'Group':<15} | {'Count':>8} | {'Percentage':>10} | {'Threshold'}")
        print("-"*70)
        print(f"{'Short':<15} | {short:>8,} | {short/total*100:>9.2f}% | L ≤ {short_thresh}")
        print(f"{'Medium':<15} | {medium:>8,} | {medium/total*100:>9.2f}% | {short_thresh} < L ≤ {long_thresh}")
        print(f"{'Long':<15} | {long_count:>8,} | {long_count/total*100:>9.2f}% | L > {long_thresh}")
        print("-"*70)
        print(f"{'Total':<15} | {total:>8,} | {100.0:>9.2f}%")
        
        # Statistics
        mean_len = sum(lengths) / len(lengths)
        sorted_lens = sorted(lengths)
        median_len = sorted_lens[len(lengths) // 2]
        min_len = min(lengths)
        max_len = max(lengths)
        
        print(f"\nStatistics:")
        print(f"  Mean length:   {mean_len:.2f}")
        print(f"  Median length: {median_len}")
        print(f"  Min length:    {min_len}")
        print(f"  Max length:    {max_len}")
        
        # Percentiles
        p25 = sorted_lens[len(lengths) // 4]
        p75 = sorted_lens[3 * len(lengths) // 4]
        p90 = sorted_lens[9 * len(lengths) // 10]
        p95 = sorted_lens[95 * len(lengths) // 100]
        
        print(f"  25th percentile: {p25}")
        print(f"  75th percentile: {p75}")
        print(f"  90th percentile: {p90}")
        print(f"  95th percentile: {p95}")
    
    # Overall summary
    print("\n" + "="*70)
    print("INSIGHTS:")
    print("="*70)
    
    test_seqs = data['test_sequences']
    test_lengths = [len(seq['sequence']) for seq in test_seqs]
    short_pct = sum(1 for l in test_lengths if l <= short_thresh) / len(test_lengths) * 100
    med_pct = sum(1 for l in test_lengths if short_thresh < l <= long_thresh) / len(test_lengths) * 100
    long_pct = sum(1 for l in test_lengths if l > long_thresh) / len(test_lengths) * 100
    
    print(f"\n1. Dataset Characteristics:")
    if short_pct > 50:
        print(f"   ⚠️  Majority ({short_pct:.1f}%) are short-history users")
        print(f"   → GNN component is CRITICAL for performance")
    elif med_pct > 50:
        print(f"   ℹ️  Majority ({med_pct:.1f}%) are medium-history users")
        print(f"   → Balanced fusion is important")
    else:
        print(f"   ✓  Majority ({long_pct:.1f}%) are long-history users")
        print(f"   → Sequential component dominates")
    
    print(f"\n2. Expected Performance Impact:")
    print(f"   - Short users ({short_pct:.1f}%): Hybrid should help significantly")
    print(f"   - Medium users ({med_pct:.1f}%): Moderate hybrid benefit")
    print(f"   - Long users ({long_pct:.1f}%): Minimal hybrid benefit")
    
    print(f"\n3. Overall Metric Impact:")
    if short_pct < 10:
        print(f"   ⚠️  Short users are only {short_pct:.1f}% of test set")
        print(f"   → Even large short-user gains won't move overall metrics much")
        print(f"   → This explains why overall HR@10 improvement is modest!")
    else:
        print(f"   ✓  Short users are {short_pct:.1f}% of test set")
        print(f"   → Short-user gains should impact overall metrics significantly")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    analyze_distribution()
