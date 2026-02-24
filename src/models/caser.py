import torch
import torch.nn as nn
import torch.nn.functional as F


class Caser(nn.Module):
    """
    Caser: Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding.
    
    Paper: Jiaxi Tang and Ke Wang. "Personalized Top-N Sequential Recommendation via
    Convolutional Sequence Embedding." WSDM 2018.
    
    Architecture:
        - Horizontal CNN filters: capture sequential patterns of l consecutive items
        - Vertical CNN filter: captures weighted union-level interaction signals  
        - FC layer for final representation
    """

    def __init__(
        self,
        num_items,
        d_model=64,
        L=5,                  # Number of previous items to look at (sequence window)
        num_h_filters=16,     # Number of horizontal CNN filters per filter size
        num_v_filters=4,      # Number of vertical CNN filters
        dropout=0.2,
        max_len=50,
        **kwargs
    ):
        """
        Args:
            num_items: Number of items (item IDs are 1 to num_items, 0 is padding)
            d_model: Embedding dimension
            L: The window size (recent L items used for CNN)
            num_h_filters: Number of horizontal convolutional filters per size
            num_v_filters: Number of vertical convolutional filters
            dropout: Dropout rate
            max_len: Maximum sequence length (used for compatibility)
        """
        super().__init__()
        self.num_items = num_items
        self.d_model = d_model
        self.L = L
        self.num_h_filters = num_h_filters
        self.num_v_filters = num_v_filters

        # Item embeddings (0 = padding)
        self.item_emb = nn.Embedding(num_items + 1, d_model, padding_idx=0)

        # User embeddings (for personalization; user IDs are 1-indexed)
        # For sequential-only mode (no user ID), we skip this
        self.use_user_emb = False  # Can be enabled for user-aware mode

        # Horizontal CNN filters
        # Filter sizes: from 1 to L (each captures patterns of 1,2,...,L consecutive items)
        self.h_convs = nn.ModuleList()
        self.h_filter_sizes = list(range(1, L + 1))
        for fs in self.h_filter_sizes:
            # Input: (batch, 1, L, d_model) → Output: (batch, num_h_filters, L-fs+1, 1)
            conv = nn.Conv2d(
                in_channels=1,
                out_channels=num_h_filters,
                kernel_size=(fs, d_model)
            )
            self.h_convs.append(conv)

        # Vertical CNN filter
        # Input: (batch, 1, L, d_model) → Output: (batch, num_v_filters, L, 1)
        self.v_conv = nn.Conv2d(
            in_channels=1,
            out_channels=num_v_filters,
            kernel_size=(1, d_model)  # Each filter scans one row
        )
        # After vertical conv + reshape: (batch, L * num_v_filters)
        # Actually: (batch, num_v_filters, L, 1) → squeeze → (batch, num_v_filters * L)
        # But summing over L gives (batch, d_model // some reduction) — we use a FC
        # Standard Caser: vertical conv output is num_v_filters × d_model after FC

        # Output dimension of horizontal filters: sum of (L - fs + 1) items per filter, each pooled to 1
        self.h_out_dim = num_h_filters * len(self.h_filter_sizes)

        # Vertical output: pool the L positions using a FC
        # vertical conv output: (batch, num_v_filters, L, 1) → flatten → FC
        self.v_fc = nn.Linear(num_v_filters * L, d_model)

        # FC layer combining horizontal + vertical features
        self.fc = nn.Linear(self.h_out_dim + d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(d_model)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize all weights with Xavier/normal init"""
        nn.init.xavier_normal_(self.item_emb.weight[1:])
        for conv in self.h_convs:
            nn.init.xavier_normal_(conv.weight)
            nn.init.zeros_(conv.bias)
        nn.init.xavier_normal_(self.v_conv.weight)
        nn.init.zeros_(self.v_conv.bias)
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        nn.init.xavier_normal_(self.v_fc.weight)
        nn.init.zeros_(self.v_fc.bias)

    def forward(self, seq, lengths):
        """
        Args:
            seq: [batch_size, seq_len] - item IDs (0 = padding), left-padded
            lengths: [batch_size] - actual sequence lengths

        Returns:
            seq_repr: [batch_size, d_model] - user sequence representation
        """
        batch_size, seq_len = seq.shape
        device = seq.device

        # Extract the last L items from each sequence (window)
        # seq is left-padded, so real items are at positions [seq_len-L:] for full seqs
        # For shorter sequences, we just take what's available
        L = self.L

        # Take the last L positions from each sequence (left-padded), shape: (batch, L)
        if seq_len >= L:
            window = seq[:, -L:]  # Last L items
        else:
            # Pad on the left if sequence shorter than L
            pad = torch.zeros(batch_size, L - seq_len, dtype=torch.long, device=device)
            window = torch.cat([pad, seq], dim=1)

        # Embed window items: (batch, L, d_model)
        emb = self.item_emb(window)
        emb = self.dropout(emb)

        # Add a channel dimension: (batch, 1, L, d_model)
        emb = emb.unsqueeze(1)

        # --- Horizontal convolutions ---
        h_out_list = []
        for conv in self.h_convs:
            # conv output: (batch, num_h_filters, L - fs + 1, 1)
            out = F.relu(conv(emb))
            # Max pooling over the remaining spatial dimension
            out = out.squeeze(3)  # (batch, num_h_filters, L - fs + 1)
            out = F.adaptive_max_pool1d(out, 1)  # (batch, num_h_filters, 1)
            out = out.squeeze(2)  # (batch, num_h_filters)
            h_out_list.append(out)

        h_out = torch.cat(h_out_list, dim=1)  # (batch, h_out_dim)
        h_out = self.dropout(h_out)

        # --- Vertical convolution ---
        # v_conv output: (batch, num_v_filters, L, 1)
        v_out = F.relu(self.v_conv(emb))
        v_out = v_out.squeeze(3)          # (batch, num_v_filters, L)
        v_out = v_out.view(batch_size, -1)  # (batch, num_v_filters * L)
        v_out = self.dropout(v_out)
        v_out = self.v_fc(v_out)          # (batch, d_model)
        v_out = F.relu(v_out)

        # --- Combine ---
        combined = torch.cat([h_out, v_out], dim=1)  # (batch, h_out_dim + d_model)
        seq_repr = self.fc(combined)                   # (batch, d_model)
        seq_repr = self.bn(seq_repr)
        seq_repr = F.relu(seq_repr)

        return seq_repr

    def predict(self, seq_repr, candidate_items=None):
        """
        Compute item scores from sequence representation.

        Args:
            seq_repr: [batch_size, d_model]
            candidate_items: [batch_size, num_candidates] or None (score all items)

        Returns:
            scores: [batch_size, num_candidates] or [batch_size, num_items]
        """
        if candidate_items is None:
            # Score all items (exclude padding)
            item_embs = self.item_emb.weight[1:]  # [num_items, d_model]
            scores = torch.matmul(seq_repr, item_embs.t())  # [batch_size, num_items]
        else:
            batch_size, num_candidates = candidate_items.shape
            item_embs = self.item_emb(candidate_items)  # [batch_size, num_candidates, d_model]
            scores = torch.bmm(
                item_embs,
                seq_repr.unsqueeze(2)
            ).squeeze(2)  # [batch_size, num_candidates]
        return scores


# Testing
if __name__ == '__main__':
    batch_size = 4
    seq_len = 20
    num_items = 100

    seq = torch.randint(1, num_items + 1, (batch_size, seq_len))
    seq[:, :5] = 0  # Simulate padding
    lengths = torch.LongTensor([15, 16, 18, 20])

    model = Caser(
        num_items=num_items,
        d_model=64,
        L=5,
        num_h_filters=16,
        num_v_filters=4,
        dropout=0.2
    )

    seq_repr = model(seq, lengths)
    print(f"Sequence representation shape: {seq_repr.shape}")

    scores = model.predict(seq_repr)
    print(f"All-item scores shape: {scores.shape}")

    candidates = torch.randint(1, num_items + 1, (batch_size, 10))
    cand_scores = model.predict(seq_repr, candidates)
    print(f"Candidate scores shape: {cand_scores.shape}")

    print("✅ Caser model working!")
