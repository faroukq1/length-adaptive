import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

class LightGCNConv(MessagePassing):
    """Single LightGCN convolution layer (no weights, just aggregation)"""

    def __init__(self):
        super().__init__(aggr='add')

    def forward(self, x, edge_index, edge_weight=None):
        """
        Args:
            x: [num_nodes, d_model] - node features
            edge_index: [2, num_edges] - graph edges
            edge_weight: [num_edges] - edge weights (optional)

        Returns:
            out: [num_nodes, d_model] - aggregated features
        """
        # Normalize by degree
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        if edge_weight is not None:
            norm = norm * edge_weight

        # Message passing
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # x_j: [num_edges, d_model]
        # norm: [num_edges]
        return norm.view(-1, 1) * x_j


class LightGCNSeq(nn.Module):
    """
    LightGCN for Sequential Recommendation
    
    This model uses LightGCN to learn item embeddings from the graph, then
    aggregates the sequence of items to make predictions.
    """

    def __init__(
        self,
        num_items,
        d_model=64,
        num_layers=2,
        dropout=0.2,
        max_len=50,
        **kwargs  # Accept extra args for compatibility
    ):
        """
        Args:
            num_items: Number of items (embeddings created for 0 to num_items)
            d_model: Embedding dimension
            num_layers: Number of GCN layers
            dropout: Dropout rate
            max_len: Maximum sequence length (for compatibility)
        """
        super().__init__()
        self.num_items = num_items
        self.d_model = d_model
        self.num_layers = num_layers

        # Item embeddings (including padding idx 0)
        self.item_emb = nn.Embedding(num_items + 1, d_model, padding_idx=0)

        # GCN layers
        self.convs = nn.ModuleList([
            LightGCNConv() for _ in range(num_layers)
        ])

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Attention weights for sequence aggregation
        self.attention = nn.Linear(d_model, 1)

        # Initialize
        nn.init.xavier_normal_(self.item_emb.weight[1:])

    def compute_graph_embeddings(self, edge_index, edge_weight=None):
        """
        Compute graph-enhanced item embeddings

        Args:
            edge_index: [2, num_edges]
            edge_weight: [num_edges] (optional)

        Returns:
            embeddings: [num_items+1, d_model]
        """
        x = self.item_emb.weight  # [num_items+1, d_model]

        # Collect embeddings from all layers
        all_embs = [x]

        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)
            all_embs.append(x)

        # Mean of all layers (including initial)
        final_emb = torch.mean(torch.stack(all_embs), dim=0)

        return final_emb

    def forward(self, seq, lengths, graph_emb=None):
        """
        Args:
            seq: [batch_size, seq_len] - item IDs (0 = padding)
            lengths: [batch_size] - actual sequence lengths (non-padded)
            graph_emb: [num_items+1, d_model] - pre-computed graph embeddings (optional)

        Returns:
            seq_repr: [batch_size, d_model] - representation of sequence
        """
        batch_size, seq_len = seq.shape
        device = seq.device

        # Use graph embeddings if provided, otherwise use base embeddings
        if graph_emb is not None:
            item_embs = graph_emb[seq]  # [batch_size, seq_len, d_model]
        else:
            item_embs = self.item_emb(seq)  # [batch_size, seq_len, d_model]

        item_embs = self.dropout(item_embs)

        # Create mask for padding positions
        mask = (seq != 0).float()  # [batch_size, seq_len]

        # Compute attention scores for each position
        attn_scores = self.attention(item_embs).squeeze(-1)  # [batch_size, seq_len]
        
        # Mask out padding positions
        attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [batch_size, seq_len]
        
        # Weighted sum
        seq_repr = torch.sum(attn_weights.unsqueeze(-1) * item_embs, dim=1)  # [batch_size, d_model]

        return seq_repr

    def predict(self, seq_repr, candidate_items=None, graph_emb=None):
        """
        Compute scores for candidate items

        Args:
            seq_repr: [batch_size, d_model]
            candidate_items: [batch_size, num_candidates] or None (score all items)
            graph_emb: [num_items+1, d_model] - pre-computed graph embeddings (optional)

        Returns:
            scores: [batch_size, num_candidates] or [batch_size, num_items]
        """
        # Use graph embeddings if provided, otherwise use base embeddings
        if graph_emb is not None:
            item_weights = graph_emb
        else:
            item_weights = self.item_emb.weight

        if candidate_items is None:
            # Score all items
            item_embs = item_weights[1:]  # Exclude padding, [num_items, d_model]
            scores = torch.matmul(seq_repr, item_embs.t())  # [batch_size, num_items]
        else:
            # Score specific candidates
            batch_size, num_candidates = candidate_items.shape
            item_embs = item_weights[candidate_items]  # [batch_size, num_candidates, d_model]
            scores = torch.bmm(
                item_embs,
                seq_repr.unsqueeze(2)
            ).squeeze(2)  # [batch_size, num_candidates]

        return scores


# Testing
if __name__ == '__main__':
    import pickle

    # Create dummy data
    batch_size = 4
    seq_len = 10
    num_items = 100

    seq = torch.randint(1, num_items + 1, (batch_size, seq_len))
    seq[:, :3] = 0  # Simulate padding
    lengths = torch.LongTensor([7, 8, 9, 10])

    # Create model
    model = LightGCNSeq(
        num_items=num_items,
        d_model=64,
        num_layers=2
    )

    # Test without graph (baseline)
    print("Testing without graph embeddings:")
    seq_repr = model(seq, lengths)
    print(f"Sequence representation shape: {seq_repr.shape}")

    scores = model.predict(seq_repr)
    print(f"Scores shape: {scores.shape}")

    # Test with graph
    print("\nTesting with graph embeddings:")
    try:
        with open('data/graphs/cooccurrence_graph.pkl', 'rb') as f:
            graph_data = pickle.load(f)
        
        edge_index = graph_data['edge_index']
        edge_weight = graph_data['edge_weight']
        
        # Compute graph embeddings
        graph_emb = model.compute_graph_embeddings(edge_index, edge_weight)
        print(f"Graph embeddings shape: {graph_emb.shape}")
        
        # Forward with graph embeddings
        seq_repr = model(seq, lengths, graph_emb=graph_emb)
        scores = model.predict(seq_repr, graph_emb=graph_emb)
        print(f"Scores with graph shape: {scores.shape}")
        
    except FileNotFoundError:
        print("Graph file not found, skipping graph test")

    print("\n✅ LightGCNSeq model working!")
