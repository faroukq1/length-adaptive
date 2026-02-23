"""
TGT-BERT4Rec: Temporal Graph Transformer + BERT4Rec Hybrid
Combines temporal graph modeling with bidirectional sequential transformers

Architecture:
- BERT4Rec: Bidirectional masked language modeling on sequences
- TGT: Time-aware graph attention on user-item temporal interactions
- Gated Fusion: Learnable combination of both representations

Target: Beat BERT4Rec baseline (NDCG@10=0.7665) by 5-15% → target >0.82

Based on:
- TGT: https://github.com/akaxlh/TGT
- BERT4Rec: https://github.com/FeiSun/BERT4Rec
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .bert4rec import BERT4Rec


class TimeEncode(nn.Module):
    """
    Time encoding module from TGT
    Projects timestamps to continuous time embeddings
    """
    def __init__(self, d_model):
        super(TimeEncode, self).__init__()
        self.d_model = d_model
        self.w = nn.Linear(1, d_model)
        
    def forward(self, t):
        """
        Args:
            t: [batch_size, seq_len] timestamps (normalized)
        Returns:
            time_emb: [batch_size, seq_len, d_model]
        """
        # Expand to [batch, seq_len, 1]
        t = t.unsqueeze(-1).float()
        # Project to d_model
        time_emb = self.w(t)
        return torch.cos(time_emb)


class TemporalGraphAttention(nn.Module):
    """
    Time-aware multi-head attention for temporal graphs
    Incorporates temporal information into attention computation
    """
    def __init__(self, d_model, n_heads, dropout=0.2):
        super(TemporalGraphAttention, self).__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # Time encoding
        self.time_encoder = TimeEncode(d_model)
        
        # Query, Key, Value projections
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        # Time attention weights
        self.time_attn = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x, timestamps, mask=None):
        """
        Args:
            x: [batch_size, seq_len, d_model] - node features
            timestamps: [batch_size, seq_len] - interaction timestamps
            mask: [batch_size, seq_len] - padding mask
        Returns:
            output: [batch_size, seq_len, d_model]
            attention_weights: [batch_size, n_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.size()
        
        # Time embeddings
        time_emb = self.time_encoder(timestamps)  # [batch, seq_len, d_model]
        
        # Q, K, V
        Q = self.q_linear(x)  # [batch, seq_len, d_model]
        K = self.k_linear(x + time_emb)  # Add temporal info to keys
        V = self.v_linear(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        # Now: [batch, n_heads, seq_len, d_head]
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        # [batch, n_heads, seq_len, seq_len]
        
        # Temporal attention bias
        time_attn_bias = self.time_attn(time_emb)  # [batch, seq_len, d_model]
        time_attn_bias = time_attn_bias.view(batch_size, seq_len, self.n_heads, self.d_head)
        time_attn_bias = time_attn_bias.transpose(1, 2)  # [batch, n_heads, seq_len, d_head]
        time_scores = torch.matmul(time_attn_bias, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        
        # Combine structural and temporal attention
        scores = scores + time_scores
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax and attention
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)  # [batch, n_heads, seq_len, d_head]
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.d_model)
        
        # Output projection
        output = self.out_proj(context)
        
        return output, attn_weights


class TemporalGraphLayer(nn.Module):
    """
    Single Temporal Graph Transformer layer
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.2):
        super(TemporalGraphLayer, self).__init__()
        
        # Temporal graph attention
        self.tgat = TemporalGraphAttention(d_model, n_heads, dropout)
        
        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer norms
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, timestamps, mask=None):
        """
        Args:
            x: [batch_size, seq_len, d_model]
            timestamps: [batch_size, seq_len]
            mask: [batch_size, seq_len]
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        # Temporal graph attention with residual
        attn_out, attn_weights = self.tgat(self.ln1(x), timestamps, mask)
        x = x + self.dropout(attn_out)
        
        # Feed-forward with residual
        ff_out = self.ff(self.ln2(x))
        x = x + ff_out
        
        return x


class TemporalGraphTransformer(nn.Module):
    """
    Temporal Graph Transformer (TGT) encoder
    Stacks multiple temporal graph layers
    """
    def __init__(self, num_items, d_model, n_heads, n_blocks, d_ff, max_len, dropout=0.2):
        super(TemporalGraphTransformer, self).__init__()
        
        self.d_model = d_model
        
        # Item embedding (shared with BERT)
        self.item_embedding = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        
        # Position embedding (for sequence order)
        self.position_embedding = nn.Embedding(max_len, d_model)
        
        # Temporal graph layers
        self.layers = nn.ModuleList([
            TemporalGraphLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_blocks)
        ])
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, item_ids, timestamps, mask=None):
        """
        Args:
            item_ids: [batch_size, seq_len]
            timestamps: [batch_size, seq_len] - normalized to [0, 1]
            mask: [batch_size, seq_len] - 1 for valid, 0 for padding
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len = item_ids.size()
        
        # Item embeddings
        x = self.item_embedding(item_ids)
        
        # Add positional embeddings
        positions = torch.arange(seq_len, device=item_ids.device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.position_embedding(positions)
        
        x = self.dropout(x)
        
        # Apply temporal graph layers
        for layer in self.layers:
            x = layer(x, timestamps, mask)
        
        x = self.layer_norm(x)
        
        return x


class TGT_BERT4Rec(nn.Module):
    """
    Hybrid TGT + BERT4Rec model
    
    Combines:
    - BERT4Rec: Bidirectional sequential modeling with masked language modeling
    - TGT: Temporal graph transformer with time-aware attention
    - Gated Fusion: Learnable combination of both representations
    
    Target: NDCG@10 > 0.82 (beating 0.7665 baseline by 5-15%)
    """
    def __init__(
        self,
        num_items,
        d_model=64,
        n_heads=2,
        n_blocks=2,
        d_ff=256,
        max_len=200,
        dropout=0.2,
        fusion_alpha=0.3,  # Initial fusion weight (learnable)
        learnable_fusion=True
    ):
        """
        Args:
            num_items: Number of items in catalog
            d_model: Embedding dimension (64 for user's config)
            n_heads: Number of attention heads (2 for user's config)
            n_blocks: Number of transformer blocks (2 for user's config)
            d_ff: Feed-forward dimension
            max_len: Maximum sequence length (200)
            dropout: Dropout rate (0.2 for user's config)
            fusion_alpha: Initial fusion weight for gating (0.3 optimal)
            learnable_fusion: Whether fusion weight is learnable
        """
        super(TGT_BERT4Rec, self).__init__()
        
        self.num_items = num_items
        self.d_model = d_model
        self.learnable_fusion = learnable_fusion
        
        # Shared item embedding
        self.item_embedding = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        
        # BERT4Rec branch (bidirectional sequential modeling)
        self.bert = BERT4Rec(
            num_items=num_items,
            d_model=d_model,
            n_heads=n_heads,
            n_blocks=n_blocks,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout
        )
        
        # TGT branch (temporal graph modeling)
        self.tgt = TemporalGraphTransformer(
            num_items=num_items,
            d_model=d_model,
            n_heads=n_heads,
            n_blocks=n_blocks,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout
        )
        
        # Fusion gate (learnable or fixed)
        if learnable_fusion:
            # Initialize with fusion_alpha, make learnable
            self.fusion_gate = nn.Parameter(torch.tensor(fusion_alpha))
        else:
            # Register as buffer (not learnable)
            self.register_buffer('fusion_gate', torch.tensor(fusion_alpha))
        
        # Optional fusion MLP (for more complex fusion)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
        # Output projection
        self.output_layer = nn.Linear(d_model, num_items + 1)
        
        # Share embeddings between branches
        self.bert.item_embedding = self.item_embedding
        self.tgt.item_embedding = self.item_embedding
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights"""
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        nn.init.normal_(self.output_layer.weight, std=0.02)
        nn.init.zeros_(self.output_layer.bias)
        
    def forward(self, input_ids, timestamps=None, mask=None, return_fusion_info=False):
        """
        Forward pass through hybrid model
        
        Args:
            input_ids: [batch_size, seq_len] - item IDs
            timestamps: [batch_size, seq_len] - interaction timestamps (normalized [0,1])
            mask: [batch_size, seq_len] - attention mask (1 for valid, 0 for padding)
            return_fusion_info: Whether to return fusion weights and branch outputs
        
        Returns:
            logits: [batch_size, seq_len, num_items + 1] - prediction logits
            (optional) fusion_info: Dict with fusion details
        """
        batch_size, seq_len = input_ids.size()
        
        # If timestamps not provided, use sequential positions normalized
        if timestamps is None:
            timestamps = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
            timestamps = timestamps.expand(batch_size, -1).float() / seq_len
        
        # BERT4Rec branch (bidirectional attention)
        bert_out = self.bert(input_ids)  # [batch, seq_len, d_model]
        # Extract hidden states (before output layer)
        bert_hidden = bert_out  # BERT4Rec forward returns pre-logit features
        
        # TGT branch (temporal graph attention)
        tgt_hidden = self.tgt(input_ids, timestamps, mask)  # [batch, seq_len, d_model]
        
        # Gated fusion
        alpha = torch.sigmoid(self.fusion_gate) if self.learnable_fusion else self.fusion_gate
        
        # Simple gated combination
        fused = alpha * bert_hidden + (1 - alpha) * tgt_hidden
        
        # Optional: MLP-based fusion for richer combination
        # concatenated = torch.cat([bert_hidden, tgt_hidden], dim=-1)
        # fused_mlp = self.fusion_mlp(concatenated)
        # fused = fused + 0.1 * fused_mlp  # Residual connection
        
        # Output logits
        logits = self.output_layer(fused)  # [batch, seq_len, num_items + 1]
        
        if return_fusion_info:
            fusion_info = {
                'alpha': alpha.item() if self.learnable_fusion else alpha,
                'bert_hidden': bert_hidden,
                'tgt_hidden': tgt_hidden,
                'fused': fused
            }
            return logits, fusion_info
        
        return logits
    
    def predict(self, input_ids, timestamps=None, mask=None):
        """
        Predict next item scores
        
        Args:
            input_ids: [batch_size, seq_len]
            timestamps: [batch_size, seq_len]
            mask: [batch_size, seq_len]
        
        Returns:
            scores: [batch_size, num_items + 1] - next item prediction scores
        """
        logits = self.forward(input_ids, timestamps, mask)
        # Return scores for last position
        return logits[:, -1, :]  # [batch, num_items + 1]


# Example usage and testing
if __name__ == '__main__':
    print("="*70)
    print("Testing TGT-BERT4Rec Hybrid Model")
    print("="*70)
    
    # Test configuration (MovieLens-1M scale)
    num_items = 3952  # MovieLens-1M items
    batch_size = 32
    seq_len = 50
    
    # User's fine-tuned optimal config
    config = {
        'd_model': 64,
        'n_heads': 2,
        'n_blocks': 2,
        'd_ff': 256,
        'max_len': 200,
        'dropout': 0.2,
        'fusion_alpha': 0.3,
        'learnable_fusion': True
    }
    
    # Create model
    model = TGT_BERT4Rec(num_items=num_items, **config)
    
    # Test inputs
    input_ids = torch.randint(1, num_items + 1, (batch_size, seq_len))
    timestamps = torch.rand(batch_size, seq_len)  # Normalized timestamps
    mask = torch.ones(batch_size, seq_len)
    
    # Forward pass
    print("\nForward pass test:")
    logits, fusion_info = model(input_ids, timestamps, mask, return_fusion_info=True)
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output logits shape: {logits.shape}")
    print(f"  Fusion alpha: {fusion_info['alpha']:.4f}")
    
    # Prediction test
    print("\nPrediction test:")
    scores = model.predict(input_ids, timestamps, mask)
    print(f"  Next-item scores shape: {scores.shape}")
    
    # Model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Branch comparison
    print(f"\nBranch outputs:")
    print(f"  BERT hidden: {fusion_info['bert_hidden'].shape}")
    print(f"  TGT hidden: {fusion_info['tgt_hidden'].shape}")
    print(f"  Fused: {fusion_info['fused'].shape}")
    
    print("\n" + "="*70)
    print("✅ TGT-BERT4Rec model test complete!")
    print("Target: NDCG@10 > 0.82 (baseline: 0.7665)")
    print("="*70)
