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


class LightGCN(nn.Module):
    """LightGCN for item graph encoding"""

    def __init__(self, num_items, d_model, num_layers=2):
        """
        Args:
            num_items: Number of items (embeddings created for 0 to num_items)
            d_model: Embedding dimension
            num_layers: Number of GCN layers
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

        # Initialize
        nn.init.xavier_normal_(self.item_emb.weight[1:])

    def forward(self, edge_index, edge_weight=None):
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


# Testing
if __name__ == '__main__':
    import pickle

    # Load graph
    with open('data/graphs/cooccurrence_graph.pkl', 'rb') as f:
        graph_data = pickle.load(f)

    num_items = graph_data['config']['num_items']
    edge_index = graph_data['edge_index']
    edge_weight = graph_data['edge_weight']

    # Create model
    model = LightGCN(num_items=num_items, d_model=64, num_layers=2)

    # Forward pass
    item_embeddings = model(edge_index, edge_weight)
    print(f"Item embeddings shape: {item_embeddings.shape}")
    print(f"Expected: [{num_items + 1}, 64]")

    # Check padding embedding is still zero (should be masked during training)
    print(f"Padding embedding norm: {item_embeddings[0].norm().item()}")

    print("✅ LightGCN model working!")
