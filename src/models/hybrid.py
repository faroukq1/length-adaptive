import torch
import torch.nn as nn
from src.models.sasrec import TransformerBlock
from src.models.lightgcn import LightGCN
from src.models.fusion import DiscreteFusion, LearnableFusion, ContinuousFusion

class HybridSASRecGNN(nn.Module):
    """
    Length-Adaptive Hybrid Model: SASRec + LightGCN with fusion

    This is the MAIN NOVELTY model!
    """

    def __init__(
        self,
        num_items,
        d_model=64,
        n_heads=2,
        n_blocks=2,
        d_ff=256,
        max_len=50,
        gnn_layers=2,
        dropout=0.2,
        fusion_type='discrete',  # 'discrete', 'learnable', 'continuous', or 'fixed'
        fixed_alpha=0.5,
        L_short=10,
        L_long=50
    ):
        super().__init__()
        self.num_items = num_items
        self.d_model = d_model
        self.max_len = max_len
        self.fusion_type = fusion_type

        # ===== Component 1: SASRec item embeddings =====
        self.sasrec_item_emb = nn.Embedding(num_items + 1, d_model, padding_idx=0)

        # ===== Component 2: GNN encoder =====
        self.gnn = LightGCN(num_items, d_model, gnn_layers)

        # ===== Component 3: Projection (if needed) =====
        # In our case, both have d_model, so no projection needed
        # But we include it for flexibility
        self.gnn_projection = nn.Linear(d_model, d_model)

        # ===== Component 4: Fusion mechanism =====
        if fusion_type == 'fixed':
            # No fusion object, just use fixed alpha
            self.fixed_alpha = fixed_alpha
            self.fusion = None
        elif fusion_type == 'discrete':
            self.fusion = DiscreteFusion(L_short, L_long)
        elif fusion_type == 'learnable':
            self.fusion = LearnableFusion(L_short, L_long)
        elif fusion_type == 'continuous':
            self.fusion = ContinuousFusion()
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        # ===== Component 5: SASRec Transformer =====
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_blocks)
        ])
        self.dropout = nn.Dropout(dropout)

        # Initialize
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.sasrec_item_emb.weight[1:])
        nn.init.xavier_normal_(self.pos_emb.weight)

    def get_fused_embeddings(self, lengths, edge_index, edge_weight=None):
        """
        Get user-specific fused item embeddings

        Args:
            lengths: [batch_size] - sequence lengths
            edge_index: [2, num_edges] - graph structure
            edge_weight: [num_edges] - graph weights

        Returns:
            fused_emb_table: [batch_size, num_items+1, d_model]
        """
        batch_size = lengths.size(0)

        # Get SASRec embeddings
        sasrec_emb = self.sasrec_item_emb.weight  # [num_items+1, d_model]

        # Get GNN embeddings
        gnn_emb = self.gnn(edge_index, edge_weight)  # [num_items+1, d_model]
        gnn_emb = self.gnn_projection(gnn_emb)

        # Fuse based on fusion type
        if self.fusion_type == 'fixed':
            # Fixed fusion: same alpha for all users
            sasrec_expanded = sasrec_emb.unsqueeze(0).expand(batch_size, -1, -1)
            gnn_expanded = gnn_emb.unsqueeze(0).expand(batch_size, -1, -1)
            fused = self.fixed_alpha * sasrec_expanded + (1 - self.fixed_alpha) * gnn_expanded
        else:
            # Adaptive fusion
            fused = self.fusion(sasrec_emb, gnn_emb, lengths)

        return fused

    def forward(self, seq, lengths, edge_index, edge_weight=None):
        """
        Forward pass

        Args:
            seq: [batch_size, seq_len] - item sequences
            lengths: [batch_size] - actual lengths
            edge_index: [2, num_edges] - graph structure
            edge_weight: [num_edges] - graph weights

        Returns:
            seq_repr: [batch_size, d_model] - sequence representations
        """
        batch_size, seq_len = seq.shape
        device = seq.device

        # Get user-specific fused embeddings
        fused_emb_table = self.get_fused_embeddings(lengths, edge_index, edge_weight)
        # Shape: [batch_size, num_items+1, d_model]

        # Look up fused embeddings for sequence items
        # Need to index into fused_emb_table for each user's sequence
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, seq_len)
        seq_emb = fused_emb_table[batch_indices, seq]  # [batch_size, seq_len, d_model]

        # Add positional embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_emb(positions)
        x = seq_emb + pos_emb
        x = self.dropout(x)

        # Create masks
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
        padding_mask = (seq != 0).unsqueeze(1).expand(-1, seq_len, -1)
        mask = causal_mask * padding_mask

        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        # Extract last position
        batch_indices = torch.arange(batch_size, device=device)
        last_indices = lengths - 1
        seq_repr = x[batch_indices, last_indices]

        return seq_repr

    def predict(self, seq_repr, candidate_items=None):
        """
        Predict scores for candidate items

        Args:
            seq_repr: [batch_size, d_model]
            candidate_items: [batch_size, num_candidates] or None

        Returns:
            scores: [batch_size, num_candidates] or [batch_size, num_items]
        """
        if candidate_items is None:
            # Score all items using SASRec embeddings
            item_embs = self.sasrec_item_emb.weight[1:]
            scores = torch.matmul(seq_repr, item_embs.t())
        else:
            batch_size, num_candidates = candidate_items.shape
            item_embs = self.sasrec_item_emb(candidate_items)
            scores = torch.bmm(item_embs, seq_repr.unsqueeze(2)).squeeze(2)

        return scores


# Testing
if __name__ == '__main__':
    import pickle

    # Load data
    with open('data/ml-1m/processed/sequences.pkl', 'rb') as f:
        data = pickle.load(f)
    with open('data/graphs/cooccurrence_graph.pkl', 'rb') as f:
        graph_data = pickle.load(f)

    num_items = data['config']['num_items']
    edge_index = graph_data['edge_index']
    edge_weight = graph_data['edge_weight']

    # Create dummy batch
    batch_size = 4
    seq_len = 10
    seq = torch.randint(1, num_items + 1, (batch_size, seq_len))
    seq[:, :3] = 0  # Padding
    lengths = torch.LongTensor([7, 8, 9, 10])

    print("Testing Hybrid Model:")
    print("="*50)

    # Test fixed fusion
    print("\n1. Fixed Fusion (alpha=0.5):")
    model = HybridSASRecGNN(
        num_items=num_items,
        fusion_type='fixed',
        fixed_alpha=0.5
    )
    seq_repr = model(seq, lengths, edge_index, edge_weight)
    print(f"   Output shape: {seq_repr.shape}")

    # Test discrete fusion
    print("\n2. Discrete Fusion:")
    model = HybridSASRecGNN(
        num_items=num_items,
        fusion_type='discrete',
        L_short=10,
        L_long=50
    )
    seq_repr = model(seq, lengths, edge_index, edge_weight)
    print(f"   Output shape: {seq_repr.shape}")
    alphas = model.fusion.compute_alpha(lengths)
    print(f"   Alphas: {alphas.squeeze()}")

    # Test learnable fusion
    print("\n3. Learnable Fusion:")
    model = HybridSASRecGNN(
        num_items=num_items,
        fusion_type='learnable',
        L_short=10,
        L_long=50
    )
    seq_repr = model(seq, lengths, edge_index, edge_weight)
    print(f"   Output shape: {seq_repr.shape}")
    alphas = model.fusion.compute_alpha(lengths)
    print(f"   Initial alphas: {alphas.squeeze()}")

    # Test continuous fusion
    print("\n4. Continuous Fusion:")
    model = HybridSASRecGNN(
        num_items=num_items,
        fusion_type='continuous'
    )
    seq_repr = model(seq, lengths, edge_index, edge_weight)
    print(f"   Output shape: {seq_repr.shape}")
    alphas = model.fusion.compute_alpha(lengths)
    print(f"   Alphas: {alphas.squeeze()}")

    # Test prediction
    print("\n5. Prediction:")
    scores = model.predict(seq_repr)
    print(f"   Scores shape: {scores.shape}")

    print("\n✅ Hybrid model working!")
    print("="*50)
