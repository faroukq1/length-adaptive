import numpy as np
import pickle
from collections import defaultdict
from tqdm import tqdm
import torch
import scipy.sparse as sp

class CooccurrenceGraphBuilder:
    """Build item co-occurrence graph from user sequences"""

    def __init__(self, window_size=3, min_count=5):
        """
        Args:
            window_size: Sliding window size for co-occurrence
            min_count: Minimum edge weight to keep (prune rare edges)
        """
        self.window_size = window_size
        self.min_count = min_count

    def build_graph(self, sequences):
        """
        Build co-occurrence graph from sequences

        Args:
            sequences: Dict[user_id -> list of item_ids]

        Returns:
            edge_dict: Dict[(item_i, item_j) -> count]
        """
        print(f"Building co-occurrence graph (window={self.window_size})...")

        edge_dict = defaultdict(int)

        for user_id, seq in tqdm(sequences.items(), desc="Processing sequences"):
            # Slide window over sequence
            for i in range(len(seq) - self.window_size + 1):
                window = seq[i:i + self.window_size]

                # Create edges between all pairs in window
                for j in range(len(window)):
                    for k in range(j + 1, len(window)):
                        # Use sorted tuple to ensure undirected edge
                        item_i, item_j = sorted([window[j], window[k]])
                        edge_dict[(item_i, item_j)] += 1

        print(f"Total unique edges: {len(edge_dict):,}")

        return edge_dict

    def prune_edges(self, edge_dict):
        """Remove edges with count < min_count"""
        print(f"Pruning edges with count < {self.min_count}...")

        pruned = {
            edge: count for edge, count in edge_dict.items()
            if count >= self.min_count
        }

        print(f"Kept {len(pruned):,} edges ({100*len(pruned)/len(edge_dict):.1f}%)")

        return pruned

    def to_pyg_format(self, edge_dict, num_items):
        """
        Convert edge_dict to PyTorch Geometric format

        Returns:
            edge_index: [2, num_edges] tensor
            edge_weight: [num_edges] tensor
        """
        print("Converting to PyTorch Geometric format...")

        edges = []
        weights = []

        for (item_i, item_j), count in edge_dict.items():
            # Add both directions (undirected graph)
            edges.append([item_i, item_j])
            edges.append([item_j, item_i])
            weights.append(count)
            weights.append(count)

        edge_index = torch.LongTensor(edges).t().contiguous()
        edge_weight = torch.FloatTensor(weights)

        print(f"Edge index shape: {edge_index.shape}")
        print(f"Edge weight shape: {edge_weight.shape}")

        # Add self-loops (each item connects to itself)
        self_loops = torch.arange(num_items + 1).unsqueeze(0).repeat(2, 1)  # +1 for padding idx
        self_weights = torch.ones(num_items + 1)

        edge_index = torch.cat([edge_index, self_loops], dim=1)
        edge_weight = torch.cat([edge_weight, self_weights], dim=0)

        return edge_index, edge_weight

    def compute_statistics(self, edge_dict, num_items):
        """Compute graph statistics"""
        print("\nGraph Statistics:")
        print("="*50)

        # Degree distribution
        degrees = defaultdict(int)
        for (item_i, item_j) in edge_dict.keys():
            degrees[item_i] += 1
            degrees[item_j] += 1

        degree_values = list(degrees.values())
        print(f"Number of nodes: {len(degrees):,} / {num_items:,} items")
        print(f"Number of edges: {len(edge_dict):,}")
        print(f"Average degree: {np.mean(degree_values):.2f}")
        print(f"Degree std: {np.std(degree_values):.2f}")
        print(f"Min degree: {np.min(degree_values)}")
        print(f"Max degree: {np.max(degree_values)}")

        # Weight distribution
        weights = list(edge_dict.values())
        print(f"\nEdge weight distribution:")
        print(f"  Min: {np.min(weights)}")
        print(f"  Max: {np.max(weights)}")
        print(f"  Mean: {np.mean(weights):.2f}")
        print(f"  Median: {np.median(weights):.2f}")
        print(f"  95th percentile: {np.percentile(weights, 95):.2f}")

        # Density
        max_edges = num_items * (num_items - 1) / 2
        density = len(edge_dict) / max_edges
        print(f"\nGraph density: {density:.6f} ({100*density:.4f}%)")
        print("="*50)

    def build_and_save(self, sequences, num_items, output_path):
        """Full pipeline: build, prune, convert, save"""
        print("="*60)
        print("CO-OCCURRENCE GRAPH CONSTRUCTION")
        print("="*60)

        # Build graph
        edge_dict = self.build_graph(sequences)

        # Prune rare edges
        edge_dict = self.prune_edges(edge_dict)

        # Compute statistics
        self.compute_statistics(edge_dict, num_items)

        # Convert to PyG format
        edge_index, edge_weight = self.to_pyg_format(edge_dict, num_items)

        # Save
        print(f"\nSaving graph to {output_path}...")
        graph_data = {
            'edge_index': edge_index,
            'edge_weight': edge_weight,
            'edge_dict': dict(edge_dict),  # For reference
            'config': {
                'window_size': self.window_size,
                'min_count': self.min_count,
                'num_items': num_items,
                'num_edges': len(edge_dict)
            }
        }

        with open(output_path, 'wb') as f:
            pickle.dump(graph_data, f)

        print("✅ Graph construction complete!")
        print("="*60)

        return graph_data


# Usage script
if __name__ == '__main__':
    # Load preprocessed sequences
    with open('../../data/ml-1m/processed/sequences.pkl', 'rb') as f:
        data = pickle.load(f)

    # Build graph
    builder = CooccurrenceGraphBuilder(window_size=3, min_count=5)
    graph_data = builder.build_and_save(
        sequences=data['train_sequences'],
        num_items=data['config']['num_items'],
        output_path='../../data/graphs/cooccurrence_graph.pkl'
    )
