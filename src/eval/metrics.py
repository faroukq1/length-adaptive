import torch
import numpy as np

def hit_rate_at_k(ranks, k):
    """
    Hit Rate @ K: Percentage of times the true item is in top-K

    Args:
        ranks: [num_samples] - rank of true item (1 = best)
        k: int

    Returns:
        hr: float
    """
    hits = (ranks <= k).float()
    return hits.mean().item()


def ndcg_at_k(ranks, k):
    """
    Normalized Discounted Cumulative Gain @ K

    Args:
        ranks: [num_samples]
        k: int

    Returns:
        ndcg: float
    """
    # DCG for single relevant item
    dcg = torch.where(
        ranks <= k,
        1.0 / torch.log2(ranks.float() + 1),
        torch.zeros_like(ranks, dtype=torch.float)
    )

    # IDCG is always 1 (best possible rank = 1)
    idcg = 1.0

    ndcg = dcg / idcg
    return ndcg.mean().item()


def mrr_at_k(ranks, k):
    """
    Mean Reciprocal Rank @ K

    Args:
        ranks: [num_samples]
        k: int

    Returns:
        mrr: float
    """
    rr = torch.where(
        ranks <= k,
        1.0 / ranks.float(),
        torch.zeros_like(ranks, dtype=torch.float)
    )
    return rr.mean().item()


def compute_all_metrics(ranks, k_list=[5, 10, 20]):
    """
    Compute all metrics for multiple K values

    Args:
        ranks: [num_samples] - ranks of true items
        k_list: list of K values

    Returns:
        metrics: dict of metric_name -> value
    """
    metrics = {}
    for k in k_list:
        metrics[f'HR@{k}'] = hit_rate_at_k(ranks, k)
        metrics[f'NDCG@{k}'] = ndcg_at_k(ranks, k)
        metrics[f'MRR@{k}'] = mrr_at_k(ranks, k)
    return metrics


def compute_metrics_by_group(ranks, lengths, k_list=[5, 10, 20], 
                             short_thresh=10, long_thresh=50):
    """
    Compute metrics separately for short/medium/long history users

    Args:
        ranks: [num_samples] - ranks of true items
        lengths: [num_samples] - sequence lengths
        k_list: list of K values
        short_thresh: threshold for short users
        long_thresh: threshold for long users

    Returns:
        metrics_by_group: dict of group_name -> dict of metrics
    """
    # Split into groups
    short_mask = lengths <= short_thresh
    long_mask = lengths > long_thresh
    mid_mask = ~(short_mask | long_mask)

    results = {}

    # Short users
    if short_mask.sum() > 0:
        short_ranks = ranks[short_mask]
        results['short'] = compute_all_metrics(short_ranks, k_list)
        results['short']['count'] = short_mask.sum().item()

    # Medium users
    if mid_mask.sum() > 0:
        mid_ranks = ranks[mid_mask]
        results['medium'] = compute_all_metrics(mid_ranks, k_list)
        results['medium']['count'] = mid_mask.sum().item()

    # Long users
    if long_mask.sum() > 0:
        long_ranks = ranks[long_mask]
        results['long'] = compute_all_metrics(long_ranks, k_list)
        results['long']['count'] = long_mask.sum().item()

    # Overall
    results['overall'] = compute_all_metrics(ranks, k_list)
    results['overall']['count'] = len(ranks)

    return results


# Testing
if __name__ == '__main__':
    print("Testing Evaluation Metrics:")
    print("="*50)

    # Create dummy ranks
    # Perfect: rank 1 for all
    # Good: ranks 1-5
    # Medium: ranks 1-15
    # Bad: ranks 10-100

    print("\n1. Perfect Ranking (all rank 1):")
    ranks = torch.ones(100)
    metrics = compute_all_metrics(ranks, k_list=[5, 10, 20])
    for k, v in metrics.items():
        print(f"   {k}: {v:.4f}")

    print("\n2. Good Ranking (ranks 1-5):")
    ranks = torch.randint(1, 6, (100,))
    metrics = compute_all_metrics(ranks, k_list=[5, 10, 20])
    for k, v in metrics.items():
        print(f"   {k}: {v:.4f}")

    print("\n3. Medium Ranking (ranks 1-15):")
    ranks = torch.randint(1, 16, (100,))
    metrics = compute_all_metrics(ranks, k_list=[5, 10, 20])
    for k, v in metrics.items():
        print(f"   {k}: {v:.4f}")

    print("\n4. Bad Ranking (ranks 10-100):")
    ranks = torch.randint(10, 101, (100,))
    metrics = compute_all_metrics(ranks, k_list=[5, 10, 20])
    for k, v in metrics.items():
        print(f"   {k}: {v:.4f}")

    print("\n5. Metrics by User Group:")
    # Create ranks and lengths for different user groups
    ranks = torch.cat([
        torch.randint(1, 6, (30,)),    # Short users (good ranks)
        torch.randint(1, 11, (40,)),   # Medium users (medium ranks)
        torch.randint(1, 16, (30,))    # Long users (variable ranks)
    ])
    lengths = torch.cat([
        torch.randint(5, 11, (30,)),   # Short users
        torch.randint(11, 51, (40,)),  # Medium users
        torch.randint(51, 101, (30,))  # Long users
    ])

    grouped_metrics = compute_metrics_by_group(ranks, lengths, k_list=[10])

    for group_name, metrics in grouped_metrics.items():
        print(f"\n   {group_name.upper()}:")
        for k, v in metrics.items():
            print(f"     {k}: {v if isinstance(v, int) else f'{v:.4f}'}")

    print("\n✅ All metrics working!")
    print("="*50)
