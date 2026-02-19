import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention (bidirectional for BERT4Rec)"""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Linear projections
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len] or None (padding mask only, no causal mask)
        Returns:
            out: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape

        # Linear projections and split into heads
        Q = self.W_Q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        # Shape: [batch_size, n_heads, seq_len, d_k]

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # Shape: [batch_size, n_heads, seq_len, seq_len]

        # Apply mask (only padding mask, no causal mask for bidirectional attention)
        if mask is not None:
            mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights, V)
        # Shape: [batch_size, n_heads, seq_len, d_k]

        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        # Final linear projection
        out = self.W_O(out)

        return out


class PointWiseFeedForward(nn.Module):
    """Position-wise feed-forward network"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))  # GELU activation like BERT


class TransformerBlock(nn.Module):
    """Single Transformer block (attention + FFN + residual + layernorm)"""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PointWiseFeedForward(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_out = self.attention(x, mask)
        x = self.ln1(x + self.dropout(attn_out))

        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.ln2(x + self.dropout(ffn_out))

        return x


class BERT4Rec(nn.Module):
    """BERT4Rec: Bidirectional Encoder Representations from Transformers for Sequential Recommendation"""

    def __init__(
        self,
        num_items,
        d_model=64,
        n_heads=2,
        n_blocks=2,
        d_ff=256,
        max_len=50,
        dropout=0.2
    ):
        """
        Args:
            num_items: Number of items (item IDs are 1 to num_items, 0 is padding)
            d_model: Embedding dimension
            n_heads: Number of attention heads
            n_blocks: Number of transformer blocks
            d_ff: Feed-forward hidden dimension
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        self.num_items = num_items
        self.d_model = d_model
        self.max_len = max_len

        # Embeddings (item 0 is padding, so num_items+1 embeddings)
        self.item_emb = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)

        # Transformer blocks (bidirectional)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_blocks)
        ])

        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_model)

        # Initialize embeddings
        self._init_weights()

    def _init_weights(self):
        """Initialize embeddings with Xavier"""
        nn.init.xavier_normal_(self.item_emb.weight[1:])  # Skip padding
        nn.init.xavier_normal_(self.pos_emb.weight)

    def forward(self, seq, lengths):
        """
        Args:
            seq: [batch_size, seq_len] - item IDs (0 = padding)
            lengths: [batch_size] - actual sequence lengths (non-padded)

        Returns:
            seq_repr: [batch_size, d_model] - representation of last item in sequence
        """
        batch_size, seq_len = seq.shape
        device = seq.device

        # Create position indices (0 to seq_len-1)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        # Embed items and positions
        item_embs = self.item_emb(seq)  # [batch_size, seq_len, d_model]
        pos_embs = self.pos_emb(positions)  # [batch_size, seq_len, d_model]

        # Combine
        x = item_embs + pos_embs
        x = self.dropout(x)

        # Create padding mask (don't attend to padding tokens)
        # Bidirectional: each position can attend to all other positions (not just previous)
        padding_mask = (seq != 0).unsqueeze(1).expand(-1, seq_len, -1)
        # Shape: [batch_size, seq_len, seq_len]

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, padding_mask)

        x = self.ln(x)

        # Extract representation at last non-padding position
        batch_indices = torch.arange(batch_size, device=device)
        last_indices = lengths - 1
        seq_repr = x[batch_indices, last_indices]  # [batch_size, d_model]

        return seq_repr

    def predict(self, seq_repr, candidate_items=None):
        """
        Compute scores for candidate items

        Args:
            seq_repr: [batch_size, d_model]
            candidate_items: [batch_size, num_candidates] or None (score all items)

        Returns:
            scores: [batch_size, num_candidates] or [batch_size, num_items]
        """
        if candidate_items is None:
            # Score all items
            item_embs = self.item_emb.weight[1:]  # Exclude padding, [num_items, d_model]
            scores = torch.matmul(seq_repr, item_embs.t())  # [batch_size, num_items]
        else:
            # Score specific candidates
            batch_size, num_candidates = candidate_items.shape
            item_embs = self.item_emb(candidate_items)  # [batch_size, num_candidates, d_model]
            scores = torch.bmm(
                item_embs,
                seq_repr.unsqueeze(2)
            ).squeeze(2)  # [batch_size, num_candidates]

        return scores


# Testing
if __name__ == '__main__':
    # Create dummy data
    batch_size = 4
    seq_len = 10
    num_items = 100

    seq = torch.randint(1, num_items + 1, (batch_size, seq_len))
    seq[:, :3] = 0  # Simulate padding
    lengths = torch.LongTensor([7, 8, 9, 10])

    # Create model
    model = BERT4Rec(
        num_items=num_items,
        d_model=64,
        n_heads=2,
        n_blocks=2,
        max_len=50
    )

    # Forward pass
    seq_repr = model(seq, lengths)
    print(f"Sequence representation shape: {seq_repr.shape}")

    # Predict scores for all items
    scores = model.predict(seq_repr)
    print(f"Scores shape: {scores.shape}")

    # Predict scores for specific candidates
    candidates = torch.randint(1, num_items + 1, (batch_size, 10))
    candidate_scores = model.predict(seq_repr, candidates)
    print(f"Candidate scores shape: {candidate_scores.shape}")

    print("✅ BERT4Rec model working!")
