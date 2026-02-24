"""
TCN-BERT4Rec: Temporal Convolutional Network combined with BERT4Rec
Combines causal temporal convolutions with bidirectional transformers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .bert4rec import BERT4Rec


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network (TCN) for sequential modeling
    Uses dilated causal convolutions to capture temporal patterns
    """
    
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        """
        Args:
            num_inputs: Number of input channels (embedding dimension)
            num_channels: List of channel sizes for each TCN layer
            kernel_size: Size of convolutional kernel
            dropout: Dropout rate
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i  # Exponential dilation
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(
                TemporalBlock(
                    in_channels, 
                    out_channels, 
                    kernel_size, 
                    stride=1, 
                    dilation=dilation_size,
                    padding=(kernel_size-1) * dilation_size,
                    dropout=dropout
                )
            )
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        # TCN expects [batch, channels, seq_len]
        x = x.transpose(1, 2)  # [batch, d_model, seq_len]
        output = self.network(x)
        output = output.transpose(1, 2)  # [batch, seq_len, d_model]
        return output


class TemporalBlock(nn.Module):
    """
    Single temporal block with dilated causal convolution
    """
    
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)  # Remove future information
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        # Second convolutional layer
        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Combine layers
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        
        # Residual connection
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Args:
            x: [batch, channels, seq_len]
        Returns:
            output: [batch, channels, seq_len]
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Chomp1d(nn.Module):
    """
    Removes rightmost elements to ensure causality in TCN
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x


class TCNBERT4Rec(nn.Module):
    """
    Hybrid model combining TCN and BERT4Rec
    TCN captures causal temporal patterns, BERT4Rec captures bidirectional context
    """
    
    def __init__(
        self,
        num_items,
        d_model=64,
        n_heads=2,
        n_blocks=2,
        d_ff=256,
        max_len=200,
        tcn_channels=None,
        tcn_kernel_size=3,
        dropout=0.2,
        fusion_type='learnable'  # 'fixed', 'learnable', 'concat'
    ):
        """
        Args:
            num_items: Number of items in catalog
            d_model: Embedding dimension
            n_heads: Number of attention heads for BERT
            n_blocks: Number of transformer blocks for BERT
            d_ff: Feed-forward dimension for BERT
            max_len: Maximum sequence length
            tcn_channels: List of channel sizes for TCN layers
            tcn_kernel_size: Kernel size for TCN
            dropout: Dropout rate
            fusion_type: How to combine TCN and BERT ('fixed', 'learnable', 'concat')
        """
        super(TCNBERT4Rec, self).__init__()
        
        self.num_items = num_items
        self.d_model = d_model
        self.fusion_type = fusion_type
        
        # Shared item embedding
        self.item_embedding = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, d_model)
        
        # TCN branch (causal temporal convolutions)
        if tcn_channels is None:
            tcn_channels = [d_model, d_model, d_model]  # 3 layers by default
        
        self.tcn = TemporalConvNet(
            num_inputs=d_model,
            num_channels=tcn_channels,
            kernel_size=tcn_kernel_size,
            dropout=dropout
        )
        
        # BERT4Rec branch (bidirectional transformer)
        self.bert = BERT4Rec(
            num_items=num_items,
            d_model=d_model,
            n_heads=n_heads,
            n_blocks=n_blocks,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout
        )
        
        # Share embeddings between TCN and BERT branches
        self.bert.item_emb = self.item_embedding
        self.bert.pos_emb = self.position_embedding
        
        # Fusion layer
        if fusion_type == 'fixed':
            self.alpha = 0.5  # Fixed weight
        elif fusion_type == 'learnable':
            self.fusion_weight = nn.Parameter(torch.tensor(0.5))
        elif fusion_type == 'concat':
            self.fusion_layer = nn.Linear(d_model * 2, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Output projection
        self.output_layer = nn.Linear(d_model, num_items + 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        nn.init.normal_(self.output_layer.weight, std=0.02)
        nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, seq, lengths):
        """
        Forward pass (compatible with trainer interface)
        
        Args:
            seq: [batch_size, seq_len] - item IDs (0 = padding)
            lengths: [batch_size] - actual sequence lengths (non-padded)
        
        Returns:
            seq_repr: [batch_size, d_model] - representation of last item in sequence
        """
        batch_size, seq_len = seq.size()
        device = seq.device
        
        # Shared embeddings
        item_emb = self.item_embedding(seq)  # [batch, seq_len, d_model]
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)
        
        x = item_emb + pos_emb
        x = self.dropout(x)
        
        # TCN branch (causal)
        tcn_out = self.tcn(x)  # [batch, seq_len, d_model]
        
        # BERT branch (bidirectional) - reuse embeddings
        # We need to pass through BERT's transformer blocks only
        bert_x = item_emb + pos_emb
        bert_x = self.bert.dropout(bert_x)
        
        # Create padding mask for BERT (bidirectional attention)
        padding_mask = (seq != 0).unsqueeze(1).expand(-1, seq_len, -1)
        
        # Pass through BERT transformer blocks
        for block in self.bert.blocks:
            bert_x = block(bert_x, mask=padding_mask)
        
        bert_out = self.bert.ln(bert_x)  # [batch, seq_len, d_model]
        
        # Fusion
        if self.fusion_type == 'fixed':
            fused = self.alpha * tcn_out + (1 - self.alpha) * bert_out
        elif self.fusion_type == 'learnable':
            alpha = torch.sigmoid(self.fusion_weight)
            fused = alpha * tcn_out + (1 - alpha) * bert_out
        elif self.fusion_type == 'concat':
            concatenated = torch.cat([tcn_out, bert_out], dim=-1)
            fused = self.fusion_layer(concatenated)
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")
        
        # Final processing
        fused = self.layer_norm(fused)
        fused = self.dropout(fused)
        
        # Extract representation at last non-padding position
        batch_indices = torch.arange(batch_size, device=device)
        last_indices = lengths - 1
        seq_repr = fused[batch_indices, last_indices]  # [batch_size, d_model]
        
        return seq_repr
    
    def predict(self, seq_repr, candidate_items=None):
        """
        Compute scores for candidate items (compatible with trainer interface)
        
        Args:
            seq_repr: [batch_size, d_model]
            candidate_items: [batch_size, num_candidates] or None (score all items)
        
        Returns:
            scores: [batch_size, num_candidates] or [batch_size, num_items]
        """
        if candidate_items is None:
            # Score all items
            item_embs = self.item_embedding.weight[1:]  # Exclude padding, [num_items, d_model]
            scores = torch.matmul(seq_repr, item_embs.t())  # [batch_size, num_items]
        else:
            # Score specific candidates
            batch_size, num_candidates = candidate_items.shape
            item_embs = self.item_embedding(candidate_items)  # [batch_size, num_candidates, d_model]
            scores = torch.bmm(
                item_embs,
                seq_repr.unsqueeze(2)
            ).squeeze(2)  # [batch_size, num_candidates]
        
        return scores


# Example usage and testing
if __name__ == '__main__':
    # Test TCN-BERT4Rec model
    num_items = 1000
    batch_size = 32
    seq_len = 50
    
    model = TCNBERT4Rec(
        num_items=num_items,
        d_model=64,
        n_heads=2,
        n_blocks=2,
        tcn_channels=[64, 64, 64],
        fusion_type='learnable'
    )
    
    # Test forward pass
    input_ids = torch.randint(1, num_items + 1, (batch_size, seq_len))
    
    logits, fusion_weight = model(input_ids, return_fusion_weights=True)
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Fusion weight: {fusion_weight:.4f}")
    
    # Test prediction
    scores = model.predict(input_ids)
    print(f"Prediction scores shape: {scores.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
