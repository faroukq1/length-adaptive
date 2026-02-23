"""
Test TGT-BERT4Rec Model
Quick test to verify the model is working correctly
"""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.tgt_bert4rec import TGT_BERT4Rec


def test_tgt_bert4rec():
    """Test TGT-BERT4Rec model"""
    print("="*70)
    print("Testing TGT-BERT4Rec Model")
    print("="*70)
    
    # MovieLens-1M configuration
    num_items = 3952
    batch_size = 16
    seq_len = 50
    
    # User's fine-tuned configuration
    config = {
        'num_items': num_items,
        'd_model': 64,
        'n_heads': 2,
        'n_blocks': 2,
        'd_ff': 256,
        'max_len': 200,
        'dropout': 0.2,
        'fusion_alpha': 0.3,
        'learnable_fusion': True
    }
    
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create model
    print("\n[1/5] Creating model...")
    model = TGT_BERT4Rec(**config)
    print(f"✅ Model created")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[2/5] Model statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    # Test inputs
    print(f"\n[3/5] Creating test inputs...")
    input_ids = torch.randint(1, num_items + 1, (batch_size, seq_len))
    timestamps = torch.rand(batch_size, seq_len)  # Normalized timestamps [0, 1]
    mask = torch.ones(batch_size, seq_len)
    print(f"  Input IDs: {input_ids.shape}")
    print(f"  Timestamps: {timestamps.shape}")
    print(f"  Mask: {mask.shape}")
    
    # Forward pass
    print(f"\n[4/5] Testing forward pass...")
    model.eval()
    with torch.no_grad():
        logits, fusion_info = model(input_ids, timestamps, mask, return_fusion_info=True)
    print(f"  ✅ Forward pass successful")
    print(f"  Output logits: {logits.shape}")
    print(f"  Fusion alpha: {fusion_info['alpha']:.4f}")
    
    # Prediction test
    print(f"\n[5/5] Testing prediction...")
    with torch.no_grad():
        scores = model.predict(input_ids, timestamps, mask)
    print(f"  ✅ Prediction successful")
    print(f"  Next-item scores: {scores.shape}")
    
    # Verify dimensions
    assert logits.shape == (batch_size, seq_len, num_items + 1), "Logits shape mismatch"
    assert scores.shape == (batch_size, num_items + 1), "Scores shape mismatch"
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
    print("\nModel Ready for Training:")
    print("  - Target: NDCG@10 > 0.82 (baseline: 0.7665)")
    print("  - Expected improvement: 5-15% over baseline")
    print("  - Fusion: Learnable gating (α≈0.3)")
    print("="*70)
    
    return model


if __name__ == '__main__':
    model = test_tgt_bert4rec()
    print("\n🚀 Ready to run experiments!")
    print("   python experiments/run_best_models.py --models tgt_bert4rec")
