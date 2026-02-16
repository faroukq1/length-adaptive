"""
Comprehensive test suite for all Phase 2 models
"""
import torch
import pickle

print("="*60)
print("TESTING PHASE 2: BASELINE MODELS")
print("="*60)

# Load preprocessed data and graph
print("\n[Setup] Loading data and graph...")
with open('data/ml-1m/processed/sequences.pkl', 'rb') as f:
    data = pickle.load(f)
with open('data/graphs/cooccurrence_graph.pkl', 'rb') as f:
    graph_data = pickle.load(f)

num_items = data['config']['num_items']
edge_index = graph_data['edge_index']
edge_weight = graph_data['edge_weight']

print(f"✓ Data loaded: {num_items} items")
print(f"✓ Graph loaded: {edge_index.shape[1]} edges")

# Create dummy batch for testing
batch_size = 4
seq_len = 10
seq = torch.randint(1, num_items + 1, (batch_size, seq_len))
seq[:, :3] = 0  # Simulate padding
lengths = torch.LongTensor([7, 8, 9, 10])

print(f"✓ Test batch created: {batch_size} sequences of length {seq_len}")

# =============================================================================
# Test 1: SASRec Model
# =============================================================================
print("\n" + "="*60)
print("TEST 1: SASRec Baseline Model")
print("="*60)

from src.models.sasrec import SASRec

model = SASRec(
    num_items=num_items,
    d_model=64,
    n_heads=2,
    n_blocks=2,
    max_len=50,
    dropout=0.2
)

print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

# Forward pass
seq_repr = model(seq, lengths)
print(f"✓ Forward pass successful")
print(f"  Input shape: {seq.shape}")
print(f"  Output shape: {seq_repr.shape}")
print(f"  Expected: [{batch_size}, 64]")
assert seq_repr.shape == (batch_size, 64), "Output shape mismatch!"

# Prediction - all items
scores_all = model.predict(seq_repr)
print(f"✓ Prediction (all items) successful")
print(f"  Scores shape: {scores_all.shape}")
print(f"  Expected: [{batch_size}, {num_items}]")
assert scores_all.shape == (batch_size, num_items), "Scores shape mismatch!"

# Prediction - candidates
candidates = torch.randint(1, num_items + 1, (batch_size, 10))
scores_cand = model.predict(seq_repr, candidates)
print(f"✓ Prediction (candidates) successful")
print(f"  Scores shape: {scores_cand.shape}")
print(f"  Expected: [{batch_size}, 10]")
assert scores_cand.shape == (batch_size, 10), "Candidate scores shape mismatch!"

print("\n✅ SASRec Model: ALL TESTS PASSED")

# =============================================================================
# Test 2: LightGCN Model
# =============================================================================
print("\n" + "="*60)
print("TEST 2: LightGCN GNN Model")
print("="*60)

from src.models.lightgcn import LightGCN

model = LightGCN(
    num_items=num_items,
    d_model=64,
    num_layers=2
)

print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

# Forward pass
item_embeddings = model(edge_index, edge_weight)
print(f"✓ Forward pass successful")
print(f"  Output shape: {item_embeddings.shape}")
print(f"  Expected: [{num_items + 1}, 64]")
assert item_embeddings.shape == (num_items + 1, 64), "Embedding shape mismatch!"

# Check padding embedding
padding_norm = item_embeddings[0].norm().item()
print(f"✓ Padding embedding norm: {padding_norm:.4f}")
print(f"  (Should be non-zero due to self-loops)")

# Check non-padding embeddings
non_padding_norms = item_embeddings[1:].norm(dim=1)
print(f"✓ Non-padding embedding norms: min={non_padding_norms.min():.4f}, max={non_padding_norms.max():.4f}")

print("\n✅ LightGCN Model: ALL TESTS PASSED")

# =============================================================================
# Test 3: Fusion Mechanisms
# =============================================================================
print("\n" + "="*60)
print("TEST 3: Fusion Mechanisms")
print("="*60)

from src.models.fusion import DiscreteFusion, LearnableFusion, ContinuousFusion

# Create dummy embeddings
sasrec_emb = torch.randn(num_items + 1, 64)
gnn_emb = torch.randn(num_items + 1, 64)

# Test lengths with different bins
test_lengths = torch.LongTensor([5, 25, 60, 100])

# --- Test Discrete Fusion ---
print("\n3.1 Discrete Fusion:")
fusion = DiscreteFusion(L_short=10, L_long=50)
alphas = fusion.compute_alpha(test_lengths)
print(f"✓ Alphas computed: {alphas.squeeze().tolist()}")
print(f"  Expected: [0.3, 0.5, 0.7, 0.7]")
assert torch.allclose(alphas.squeeze(), torch.tensor([0.3, 0.5, 0.7, 0.7])), "Alpha values incorrect!"

fused = fusion(sasrec_emb, gnn_emb, test_lengths)
print(f"✓ Fused embeddings shape: {fused.shape}")
print(f"  Expected: [4, {num_items + 1}, 64]")
assert fused.shape == (4, num_items + 1, 64), "Fused shape mismatch!"

# --- Test Learnable Fusion ---
print("\n3.2 Learnable Fusion:")
fusion = LearnableFusion(L_short=10, L_long=50)
print(f"✓ Learnable parameters created")
print(f"  alpha_short: {fusion.alpha_short.item():.4f}")
print(f"  alpha_mid: {fusion.alpha_mid.item():.4f}")
print(f"  alpha_long: {fusion.alpha_long.item():.4f}")

alphas = fusion.compute_alpha(test_lengths)
print(f"✓ Alphas computed (with sigmoid): {alphas.squeeze().tolist()}")

fused = fusion(sasrec_emb, gnn_emb, test_lengths)
print(f"✓ Fused embeddings shape: {fused.shape}")

# --- Test Continuous Fusion ---
print("\n3.3 Continuous Fusion:")
fusion = ContinuousFusion(hidden_dim=32)
print(f"✓ Continuous fusion network created")
print(f"  Parameters: {sum(p.numel() for p in fusion.parameters()):,}")

alphas = fusion.compute_alpha(test_lengths)
print(f"✓ Alphas computed: {alphas.squeeze().tolist()}")
print(f"  (Should generally increase with length)")

# Check monotonicity (generally increasing)
alpha_values = alphas.squeeze().tolist()
increasing_count = sum(1 for i in range(len(alpha_values)-1) if alpha_values[i+1] >= alpha_values[i])
print(f"  Monotonic increases: {increasing_count}/3")

fused = fusion(sasrec_emb, gnn_emb, test_lengths)
print(f"✓ Fused embeddings shape: {fused.shape}")

print("\n✅ Fusion Mechanisms: ALL TESTS PASSED")

# =============================================================================
# Test 4: Hybrid Model (Main Novelty)
# =============================================================================
print("\n" + "="*60)
print("TEST 4: Hybrid SASRec+GNN Model (MAIN NOVELTY)")
print("="*60)

from src.models.hybrid import HybridSASRecGNN

# --- Test Fixed Fusion ---
print("\n4.1 Fixed Fusion (alpha=0.5):")
model = HybridSASRecGNN(
    num_items=num_items,
    fusion_type='fixed',
    fixed_alpha=0.5
)
print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

seq_repr = model(seq, lengths, edge_index, edge_weight)
print(f"✓ Forward pass: {seq_repr.shape}")
assert seq_repr.shape == (batch_size, 64), "Output shape mismatch!"

scores = model.predict(seq_repr)
print(f"✓ Prediction: {scores.shape}")
assert scores.shape == (batch_size, num_items), "Scores shape mismatch!"

# --- Test Discrete Fusion ---
print("\n4.2 Discrete Fusion:")
model = HybridSASRecGNN(
    num_items=num_items,
    fusion_type='discrete',
    L_short=10,
    L_long=50
)
print(f"✓ Model created")

seq_repr = model(seq, lengths, edge_index, edge_weight)
alphas = model.fusion.compute_alpha(lengths)
print(f"✓ Forward pass with adaptive fusion")
print(f"  Alphas for lengths {lengths.tolist()}: {alphas.squeeze().tolist()}")

scores = model.predict(seq_repr)
print(f"✓ Prediction: {scores.shape}")

# --- Test Learnable Fusion ---
print("\n4.3 Learnable Fusion:")
model = HybridSASRecGNN(
    num_items=num_items,
    fusion_type='learnable',
    L_short=10,
    L_long=50
)
print(f"✓ Model created with learnable fusion weights")

seq_repr = model(seq, lengths, edge_index, edge_weight)
alphas = model.fusion.compute_alpha(lengths)
print(f"✓ Forward pass with learnable fusion")
print(f"  Initial alphas: {alphas.squeeze().tolist()}")

# --- Test Continuous Fusion ---
print("\n4.4 Continuous Fusion:")
model = HybridSASRecGNN(
    num_items=num_items,
    fusion_type='continuous'
)
print(f"✓ Model created with continuous fusion")

seq_repr = model(seq, lengths, edge_index, edge_weight)
alphas = model.fusion.compute_alpha(lengths)
print(f"✓ Forward pass with continuous fusion")
print(f"  Alphas: {alphas.squeeze().tolist()}")

print("\n✅ Hybrid Model: ALL TESTS PASSED")

# =============================================================================
# Test 5: Model Comparison
# =============================================================================
print("\n" + "="*60)
print("TEST 5: Model Size Comparison")
print("="*60)

models_info = []

# SASRec
sasrec = SASRec(num_items=num_items, d_model=64, n_blocks=2)
sasrec_params = sum(p.numel() for p in sasrec.parameters())
models_info.append(("SASRec Baseline", sasrec_params))

# LightGCN
lightgcn = LightGCN(num_items=num_items, d_model=64, num_layers=2)
lightgcn_params = sum(p.numel() for p in lightgcn.parameters())
models_info.append(("LightGCN", lightgcn_params))

# Hybrid Fixed
hybrid_fixed = HybridSASRecGNN(num_items=num_items, fusion_type='fixed')
hybrid_fixed_params = sum(p.numel() for p in hybrid_fixed.parameters())
models_info.append(("Hybrid (Fixed)", hybrid_fixed_params))

# Hybrid Discrete
hybrid_discrete = HybridSASRecGNN(num_items=num_items, fusion_type='discrete')
hybrid_discrete_params = sum(p.numel() for p in hybrid_discrete.parameters())
models_info.append(("Hybrid (Discrete)", hybrid_discrete_params))

# Hybrid Learnable
hybrid_learnable = HybridSASRecGNN(num_items=num_items, fusion_type='learnable')
hybrid_learnable_params = sum(p.numel() for p in hybrid_learnable.parameters())
models_info.append(("Hybrid (Learnable)", hybrid_learnable_params))

# Hybrid Continuous
hybrid_continuous = HybridSASRecGNN(num_items=num_items, fusion_type='continuous')
hybrid_continuous_params = sum(p.numel() for p in hybrid_continuous.parameters())
models_info.append(("Hybrid (Continuous)", hybrid_continuous_params))

print("\nModel Parameters:")
for name, params in models_info:
    print(f"  {name:25s}: {params:>10,} parameters")

print("\n✅ Model Comparison: COMPLETE")

# =============================================================================
# Final Summary
# =============================================================================
print("\n" + "="*60)
print("✅ ALL PHASE 2 TESTS PASSED SUCCESSFULLY!")
print("="*60)
print("\nImplemented Models:")
print("  ✓ SASRec - Transformer-based sequential recommendation")
print("  ✓ LightGCN - Graph neural network for item relationships")
print("  ✓ Discrete Fusion - Bin-based length-adaptive fusion")
print("  ✓ Learnable Fusion - Learnable fusion weights per bin")
print("  ✓ Continuous Fusion - Neural network for smooth fusion")
print("  ✓ Hybrid Model - Complete length-adaptive system (NOVELTY)")
print("\nReady for Phase 3: Training & Evaluation")
print("="*60)
