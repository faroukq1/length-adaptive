import torch
import torch.nn as nn
import torch.nn.functional as F

class GRU4Rec(nn.Module):
    """GRU4Rec: Session-based Recommendations with Recurrent Neural Networks"""

    def __init__(
        self,
        num_items,
        d_model=64,
        n_layers=1,
        dropout=0.2,
        max_len=50,
        **kwargs  # Accept extra args for compatibility
    ):
        """
        Args:
            num_items: Number of items (item IDs are 1 to num_items, 0 is padding)
            d_model: Embedding and hidden dimension
            n_layers: Number of GRU layers
            dropout: Dropout rate
            max_len: Maximum sequence length (for compatibility, not used in GRU)
        """
        super().__init__()
        self.num_items = num_items
        self.d_model = d_model
        self.n_layers = n_layers

        # Item embeddings (item 0 is padding, so num_items+1 embeddings)
        self.item_emb = nn.Embedding(num_items + 1, d_model, padding_idx=0)

        # GRU layers
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout)

        # Initialize embeddings
        self._init_weights()

    def _init_weights(self):
        """Initialize embeddings with Xavier"""
        nn.init.xavier_normal_(self.item_emb.weight[1:])  # Skip padding

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

        # Embed items
        item_embs = self.item_emb(seq)  # [batch_size, seq_len, d_model]
        item_embs = self.dropout(item_embs)

        # Pack padded sequence for efficient RNN processing
        packed_input = nn.utils.rnn.pack_padded_sequence(
            item_embs, 
            lengths.cpu(), 
            batch_first=True, 
            enforce_sorted=False
        )

        # Pass through GRU
        packed_output, hidden = self.gru(packed_input)

        # Unpack sequence
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, 
            batch_first=True,
            total_length=seq_len
        )
        # output: [batch_size, seq_len, d_model]

        # Extract representation at last non-padding position
        batch_indices = torch.arange(batch_size, device=device)
        last_indices = lengths - 1
        seq_repr = output[batch_indices, last_indices]  # [batch_size, d_model]

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
    model = GRU4Rec(
        num_items=num_items,
        d_model=64,
        n_layers=2
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

    print("✅ GRU4Rec model working!")
