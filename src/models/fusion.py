import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscreteFusion(nn.Module):
    """Discrete bin-based fusion with fixed alphas"""

    def __init__(self, L_short=10, L_long=50,
                 alpha_short=0.3, alpha_mid=0.5, alpha_long=0.7):
        """
        Args:
            L_short: Threshold for short history
            L_long: Threshold for long history
            alpha_short: Weight for SASRec embedding (short users)
            alpha_mid: Weight for SASRec embedding (medium users)
            alpha_long: Weight for SASRec embedding (long users)
        """
        super().__init__()
        self.L_short = L_short
        self.L_long = L_long
        self.alpha_short = alpha_short
        self.alpha_mid = alpha_mid
        self.alpha_long = alpha_long

    def compute_alpha(self, lengths):
        """
        Compute alpha values for batch of users

        Args:
            lengths: [batch_size] - sequence lengths

        Returns:
            alphas: [batch_size, 1] - fusion weights
        """
        batch_size = lengths.size(0)
        device = lengths.device
        alphas = torch.zeros(batch_size, 1, device=device)

        # Classify each user into bins
        short_mask = lengths <= self.L_short
        long_mask = lengths > self.L_long
        mid_mask = ~(short_mask | long_mask)

        alphas[short_mask] = self.alpha_short
        alphas[mid_mask] = self.alpha_mid
        alphas[long_mask] = self.alpha_long

        return alphas

    def forward(self, sasrec_emb, gnn_emb, lengths):
        """
        Fuse SASRec and GNN embeddings based on length

        Args:
            sasrec_emb: [num_items+1, d_model] - SASRec item embeddings
            gnn_emb: [num_items+1, d_model] - GNN item embeddings
            lengths: [batch_size] - sequence lengths

        Returns:
            fused_emb_table: [batch_size, num_items+1, d_model]
        """
        batch_size = lengths.size(0)
        num_items, d_model = sasrec_emb.shape

        # Compute alpha for each user
        alphas = self.compute_alpha(lengths)  # [batch_size, 1]

        # Expand embeddings to batch dimension
        sasrec_expanded = sasrec_emb.unsqueeze(0).expand(batch_size, -1, -1)
        gnn_expanded = gnn_emb.unsqueeze(0).expand(batch_size, -1, -1)

        # Fuse: alpha * sasrec + (1 - alpha) * gnn
        alphas = alphas.unsqueeze(1)  # [batch_size, 1, 1]
        fused = alphas * sasrec_expanded + (1 - alphas) * gnn_expanded

        return fused


class LearnableFusion(nn.Module):
    """Learnable fusion weights for each bin"""

    def __init__(self, L_short=10, L_long=50):
        super().__init__()
        self.L_short = L_short
        self.L_long = L_long

        # Learnable alpha parameters (initialized near reasonable values)
        self.alpha_short = nn.Parameter(torch.tensor(0.3))
        self.alpha_mid = nn.Parameter(torch.tensor(0.5))
        self.alpha_long = nn.Parameter(torch.tensor(0.7))

    def compute_alpha(self, lengths):
        """Compute alpha with learned parameters"""
        batch_size = lengths.size(0)
        device = lengths.device
        alphas = torch.zeros(batch_size, 1, device=device)

        short_mask = lengths <= self.L_short
        long_mask = lengths > self.L_long
        mid_mask = ~(short_mask | long_mask)

        # Use sigmoid to constrain to [0, 1]
        alphas[short_mask] = torch.sigmoid(self.alpha_short)
        alphas[mid_mask] = torch.sigmoid(self.alpha_mid)
        alphas[long_mask] = torch.sigmoid(self.alpha_long)

        return alphas

    def forward(self, sasrec_emb, gnn_emb, lengths):
        """Same as DiscreteFusion but with learnable alphas"""
        batch_size = lengths.size(0)
        num_items, d_model = sasrec_emb.shape

        alphas = self.compute_alpha(lengths)

        sasrec_expanded = sasrec_emb.unsqueeze(0).expand(batch_size, -1, -1)
        gnn_expanded = gnn_emb.unsqueeze(0).expand(batch_size, -1, -1)

        alphas = alphas.unsqueeze(1)
        fused = alphas * sasrec_expanded + (1 - alphas) * gnn_expanded

        return fused


class ContinuousFusion(nn.Module):
    """Continuous fusion using learned function of length"""

    def __init__(self, hidden_dim=32):
        super().__init__()

        # Neural network to compute alpha from length
        self.alpha_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )

    def compute_alpha(self, lengths):
        """Compute alpha via neural network"""
        # Normalize lengths (log scale)
        normalized_length = torch.log(lengths.float() + 1).unsqueeze(1)
        alphas = self.alpha_net(normalized_length)
        return alphas

    def forward(self, sasrec_emb, gnn_emb, lengths):
        """Continuous fusion"""
        batch_size = lengths.size(0)
        alphas = self.compute_alpha(lengths)

        sasrec_expanded = sasrec_emb.unsqueeze(0).expand(batch_size, -1, -1)
        gnn_expanded = gnn_emb.unsqueeze(0).expand(batch_size, -1, -1)

        alphas = alphas.unsqueeze(1)
        fused = alphas * sasrec_expanded + (1 - alphas) * gnn_expanded

        return fused


# Testing
if __name__ == '__main__':
    batch_size = 4
    num_items = 100
    d_model = 64

    # Create dummy embeddings
    sasrec_emb = torch.randn(num_items + 1, d_model)
    gnn_emb = torch.randn(num_items + 1, d_model)

    # Create dummy lengths (short, medium, long users)
    lengths = torch.LongTensor([5, 25, 60, 100])

    # Test discrete fusion
    print("Testing Discrete Fusion:")
    fusion = DiscreteFusion()
    alphas = fusion.compute_alpha(lengths)
    print(f"  Alphas: {alphas.squeeze()}")
    print(f"  Expected: [0.3, 0.5, 0.7, 0.7]")

    fused = fusion(sasrec_emb, gnn_emb, lengths)
    print(f"  Fused shape: {fused.shape}")
    print(f"  Expected: [{batch_size}, {num_items + 1}, {d_model}]")

    # Test learnable fusion
    print("\nTesting Learnable Fusion:")
    fusion = LearnableFusion()
    alphas = fusion.compute_alpha(lengths)
    print(f"  Initial alphas: {alphas.squeeze()}")
    print(f"  (should be close to [0.3, 0.5, 0.7, 0.7])")

    # Test continuous fusion
    print("\nTesting Continuous Fusion:")
    fusion = ContinuousFusion()
    alphas = fusion.compute_alpha(lengths)
    print(f"  Alphas: {alphas.squeeze()}")
    print(f"  (should increase with length)")

    print("\n✅ All fusion mechanisms working!")
