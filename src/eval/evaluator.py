import torch
from tqdm import tqdm
from src.eval.metrics import compute_all_metrics, compute_metrics_by_group
from src.models.lightgcn_seq import LightGCNSeq

class Evaluator:
    """Evaluator for sequential recommendation models"""

    def __init__(self, model, device='cpu'):
        """
        Args:
            model: The recommendation model
            device: Device to run evaluation on
        """
        self.model = model
        self.device = device
        self.is_lightgcn = isinstance(model, LightGCNSeq)

    @torch.no_grad()
    def evaluate(self, eval_loader, edge_index, edge_weight=None, k_list=[5, 10, 20], 
                 compute_by_group=False, verbose=True, track_alpha=False, graph_emb=None):
        """
        Evaluate model on validation or test set

        Args:
            eval_loader: DataLoader for evaluation
            edge_index: Graph edge index
            edge_weight: Graph edge weights
            k_list: List of K values for metrics
            compute_by_group: Whether to compute metrics by user group
            verbose: Whether to show progress bar
            track_alpha: Whether to track alpha values (for hybrid models)
            graph_emb: Precomputed graph embeddings (for LightGCN)

        Returns:
            metrics: Dictionary of evaluation metrics
            alpha_stats: (Optional) Dictionary of alpha statistics if track_alpha=True
        """
        self.model.eval()

        all_ranks = []
        all_lengths = []
        all_alphas = [] if track_alpha else None

        iterator = tqdm(eval_loader, desc="Evaluating") if verbose else eval_loader

        for batch in iterator:
            # Move to device
            seq = batch['sequence'].to(self.device)
            lengths = batch['length'].to(self.device)
            targets = batch['target'].to(self.device)

            # Track alpha values if requested
            if track_alpha and hasattr(self.model, 'fusion'):
                alphas = self.model.fusion.compute_alpha(lengths)
                all_alphas.append(alphas.cpu())

            # Forward pass - check model type
            if self.is_lightgcn:
                seq_repr = self.model(seq, lengths, graph_emb=graph_emb)
            elif edge_index is not None and hasattr(self.model, 'gnn'):
                seq_repr = self.model(seq, lengths, edge_index, edge_weight)
            else:
                seq_repr = self.model(seq, lengths)

            # Get scores for all items
            if self.is_lightgcn:
                scores = self.model.predict(seq_repr, graph_emb=graph_emb)  # [batch_size, num_items]
            else:
                scores = self.model.predict(seq_repr)  # [batch_size, num_items]

            # Compute ranks
            ranks = self.compute_ranks(scores, targets)

            all_ranks.append(ranks)
            all_lengths.append(lengths)

        # Concatenate all results
        all_ranks = torch.cat(all_ranks)
        all_lengths = torch.cat(all_lengths)

        # Compute metrics
        if compute_by_group:
            metrics = compute_metrics_by_group(all_ranks, all_lengths, k_list)
        else:
            metrics = compute_all_metrics(all_ranks, k_list)

        # Compute alpha statistics if tracked
        if track_alpha and all_alphas:
            all_alphas = torch.cat(all_alphas)
            alpha_stats = self._compute_alpha_stats(all_alphas, all_lengths)
            return metrics, alpha_stats

        return metrics

    def compute_ranks(self, scores, targets):
        """
        Compute rank of target item in scored list

        Args:
            scores: [batch_size, num_items] - scores for all items
            targets: [batch_size] - target item indices (1-indexed)

        Returns:
            ranks: [batch_size] - rank of target (1 = best)
        """
        batch_size = scores.size(0)

        # Adjust targets to 0-indexed for scoring
        target_indices = targets - 1

        # Get scores of target items
        target_scores = scores[torch.arange(batch_size), target_indices]

        # Count how many items have higher or equal scores (rank starts at 1)
        # We add 1 because rank is 1-indexed
        ranks = (scores >= target_scores.unsqueeze(1)).sum(dim=1)

        return ranks

    def _compute_alpha_stats(self, all_alphas, all_lengths, short_thresh=10, long_thresh=50):
        """
        Compute statistics about alpha values across length groups
        
        Args:
            all_alphas: [num_samples, 1] - alpha values
            all_lengths: [num_samples] - sequence lengths
            short_thresh: threshold for short users
            long_thresh: threshold for long users
        
        Returns:
            alpha_stats: Dictionary of alpha statistics by group
        """
        all_alphas = all_alphas.squeeze()
        
        # Split by groups
        short_mask = all_lengths <= short_thresh
        long_mask = all_lengths > long_thresh
        mid_mask = ~(short_mask | long_mask)
        
        stats = {}
        
        # Short users
        if short_mask.sum() > 0:
            short_alphas = all_alphas[short_mask]
            stats['short'] = {
                'mean': short_alphas.mean().item(),
                'std': short_alphas.std().item(),
                'min': short_alphas.min().item(),
                'max': short_alphas.max().item(),
                'count': short_mask.sum().item()
            }
        
        # Medium users
        if mid_mask.sum() > 0:
            mid_alphas = all_alphas[mid_mask]
            stats['medium'] = {
                'mean': mid_alphas.mean().item(),
                'std': mid_alphas.std().item(),
                'min': mid_alphas.min().item(),
                'max': mid_alphas.max().item(),
                'count': mid_mask.sum().item()
            }
        
        # Long users
        if long_mask.sum() > 0:
            long_alphas = all_alphas[long_mask]
            stats['long'] = {
                'mean': long_alphas.mean().item(),
                'std': long_alphas.std().item(),
                'min': long_alphas.min().item(),
                'max': long_alphas.max().item(),
                'count': long_mask.sum().item()
            }
        
        # Overall
        stats['overall'] = {
            'mean': all_alphas.mean().item(),
            'std': all_alphas.std().item(),
            'min': all_alphas.min().item(),
            'max': all_alphas.max().item()
        }
        
        return stats

    def print_metrics(self, metrics, title="Evaluation Results"):
        """Pretty print evaluation metrics"""
        print("\n" + "="*60)
        print(f"{title}")
        print("="*60)

        if 'overall' in metrics:
            # Grouped metrics
            for group_name, group_metrics in metrics.items():
                print(f"\n{group_name.upper()}:")
                for metric_name, value in group_metrics.items():
                    if metric_name != 'count':
                        print(f"  {metric_name:12s}: {value:.4f}")
                    else:
                        print(f"  {metric_name:12s}: {value}")
        else:
            # Regular metrics
            for metric_name, value in metrics.items():
                print(f"  {metric_name:12s}: {value:.4f}")

        print("="*60 + "\n")


# Testing
if __name__ == '__main__':
    import pickle
    from src.models.hybrid import HybridSASRecGNN
    from src.data.dataloader import get_dataloaders

    print("="*60)
    print("TESTING EVALUATOR")
    print("="*60)

    # Load data
    print("\n[1/5] Loading data...")
    with open('data/ml-1m/processed/sequences.pkl', 'rb') as f:
        data = pickle.load(f)
    with open('data/graphs/cooccurrence_graph.pkl', 'rb') as f:
        graph_data = pickle.load(f)

    num_items = data['config']['num_items']
    edge_index = graph_data['edge_index']
    edge_weight = graph_data['edge_weight']

    print(f"✓ Loaded: {num_items} items")

    # Create dataloaders
    print("\n[2/5] Creating dataloaders...")
    train_loader, val_loader, test_loader, config = get_dataloaders(
        'data/ml-1m/processed/sequences.pkl',
        batch_size=256,
        num_workers=0
    )
    print(f"✓ Val batches: {len(val_loader)}")

    # Create model
    print("\n[3/5] Creating model...")
    model = HybridSASRecGNN(
        num_items=num_items,
        d_model=64,
        fusion_type='discrete'
    )
    print(f"✓ Model created")

    # Create evaluator
    print("\n[4/5] Creating evaluator...")
    evaluator = Evaluator(model, device='cpu')
    print(f"✓ Evaluator created")

    # Evaluate
    print("\n[5/5] Running evaluation...")
    metrics = evaluator.evaluate(
        val_loader,
        edge_index,
        edge_weight,
        k_list=[5, 10, 20],
        compute_by_group=False,
        verbose=True
    )

    evaluator.print_metrics(metrics, "Validation Results (Untrained Model)")

    # Test grouped evaluation
    print("\n[Bonus] Testing grouped evaluation...")
    grouped_metrics = evaluator.evaluate(
        val_loader,
        edge_index,
        edge_weight,
        k_list=[10],
        compute_by_group=True,
        verbose=False
    )

    evaluator.print_metrics(grouped_metrics, "Grouped Validation Results")

    print("✅ Evaluator working correctly!")
