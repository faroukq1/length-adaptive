"""
Quick GPU availability check
"""

import torch

print("="*60)
print("GPU/CUDA CHECK")
print("="*60)

# Check CUDA availability
cuda_available = torch.cuda.is_available()
print(f"\n✓ CUDA Available: {cuda_available}")

if cuda_available:
    print(f"✓ CUDA Version: {torch.version.cuda}")
    print(f"✓ GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU Count: {torch.cuda.device_count()}")
    print(f"✓ Current Device: {torch.cuda.current_device()}")
    
    # Memory info
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✓ GPU Memory: {total_memory:.1f} GB")
    
    # Test tensor on GPU
    test_tensor = torch.randn(100, 100).cuda()
    print(f"✓ Test tensor device: {test_tensor.device}")
    
    device = torch.device('cuda')
else:
    print("⚠ GPU not available - will use CPU")
    device = torch.device('cpu')

print(f"\n✓ Training will use: {device}")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

if cuda_available:
    print("✅ GPU is available and will be used for training")
    print("   Expected speed: ~20-30 it/s (~8-10 min per model)")
else:
    print("⚠️  Only CPU available - training will be slower")
    print("   Expected speed: ~2-3 it/s (~40-50 min per model)")
    print("\n   To enable GPU on Kaggle:")
    print("   Settings → Accelerator → GPU T4")

print("="*60)
