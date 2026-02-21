"""Quick test to verify all models can be instantiated and run"""

import torch
from src.models.sasrec import SASRec
from src.models.bert4rec import BERT4Rec
from src.models.gru4rec import GRU4Rec
from src.models.lightgcn_seq import LightGCNSeq
from src.models.hybrid import HybridSASRecGNN

print("="*60)
print("TESTING ALL MODELS")
print("="*60)

# Test parameters
num_items = 100
batch_size = 4
seq_len = 10
d_model = 64

# Create dummy data
seq = torch.randint(1, num_items + 1, (batch_size, seq_len))
seq[:, :3] = 0  # Simulate padding
lengths = torch.LongTensor([7, 8, 9, 10])

models_to_test = [
    ('SASRec', SASRec(num_items=num_items, d_model=d_model, n_heads=2, n_blocks=2)),
    ('BERT4Rec', BERT4Rec(num_items=num_items, d_model=d_model, n_heads=2, n_blocks=2)),
    ('GRU4Rec', GRU4Rec(num_items=num_items, d_model=d_model, n_layers=2)),
    ('LightGCNSeq', LightGCNSeq(num_items=num_items, d_model=d_model, num_layers=2)),
    ('Hybrid-Fixed', HybridSASRecGNN(num_items=num_items, d_model=d_model, fusion_type='fixed')),
    ('Hybrid-Discrete', HybridSASRecGNN(num_items=num_items, d_model=d_model, fusion_type='discrete')),
    ('Hybrid-Learnable', HybridSASRecGNN(num_items=num_items, d_model=d_model, fusion_type='learnable')),
    ('Hybrid-Continuous', HybridSASRecGNN(num_items=num_items, d_model=d_model, fusion_type='continuous')),
]

print("\nTesting model instantiation and forward pass...\n")

for name, model in models_to_test:
    try:
        # Test forward pass
        if 'Hybrid' in name:
            # Create fake graph
            edge_index = torch.randint(0, num_items, (2, 100))
            edge_weight = torch.rand(100)
            seq_repr = model(seq, lengths, edge_index, edge_weight)
        else:
            seq_repr = model(seq, lengths)
        
        # Test prediction
        scores = model.predict(seq_repr)
        
        # Check shapes
        assert seq_repr.shape == (batch_size, d_model), f"Wrong seq_repr shape: {seq_repr.shape}"
        assert scores.shape == (batch_size, num_items), f"Wrong scores shape: {scores.shape}"
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        
        print(f"✓ {name:20s} - {num_params:,} parameters")
        
    except Exception as e:
        print(f"✗ {name:20s} - ERROR: {e}")

print("\n" + "="*60)
print("ALL TESTS COMPLETE!")
print("="*60)
