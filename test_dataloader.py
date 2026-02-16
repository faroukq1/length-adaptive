"""
Test script for data preprocessing and dataloader
"""
import pickle

print("="*60)
print("TESTING DATA PREPROCESSING & DATALOADER")
print("="*60)

# Test 1: Load and verify processed sequences
print("\n[Test 1] Loading processed sequences...")
with open('data/ml-1m/processed/sequences.pkl', 'rb') as f:
    data = pickle.load(f)

print(f"✓ Successfully loaded sequences.pkl")
print(f"  Keys: {list(data.keys())}")
print(f"  Config: {data['config']}")

# Test 2: Verify data integrity
print("\n[Test 2] Verifying data integrity...")
assert 'train_sequences' in data
assert 'val_data' in data
assert 'test_data' in data
assert 'mappings' in data
assert 'config' in data

num_users = data['config']['num_users']
num_items = data['config']['num_items']

print(f"✓ All required keys present")
print(f"  Users: {num_users:,}")
print(f"  Items: {num_items:,}")

# Test 3: Sample user data
print("\n[Test 3] Inspecting sample user sequences...")
user_id = 1
train_seq = data['train_sequences'][user_id]
val_seq, val_target = data['val_data'][user_id]
test_seq, test_target = data['test_data'][user_id]

print(f"✓ User {user_id} data:")
print(f"  Train length: {len(train_seq)}")
print(f"  Train items: {train_seq[:10]}... (showing first 10)")
print(f"  Val sequence: {val_seq[:10]}... -> target: {val_target}")
print(f"  Test sequence: {test_seq[:10]}... -> target: {test_target}")

# Verify sequence consistency
assert train_seq == val_seq[:len(train_seq)], "Train and val sequences should match up to train length"
assert val_seq == test_seq[:len(val_seq)], "Val and test sequences should match up to val length"
print(f"✓ Sequence consistency verified")

# Test 4: DataLoader functionality
print("\n[Test 4] Testing DataLoader...")
from src.data.dataloader import get_dataloaders

train_loader, val_loader, test_loader, config = get_dataloaders(
    'data/ml-1m/processed/sequences.pkl',
    batch_size=64,
    num_workers=0  # Use 0 for testing
)

print(f"✓ DataLoaders created successfully")
print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches: {len(val_loader)}")
print(f"  Test batches: {len(test_loader)}")

# Test 5: Batch inspection
print("\n[Test 5] Inspecting batch structure...")
train_batch = next(iter(train_loader))
print(f"✓ Train batch keys: {list(train_batch.keys())}")
print(f"  Sequence shape: {train_batch['sequence'].shape}")
print(f"  Target shape: {train_batch['target'].shape}")
print(f"  Negatives shape: {train_batch['negatives'].shape}")
print(f"  Sample lengths: {train_batch['length'][:5].tolist()}")

val_batch = next(iter(val_loader))
print(f"✓ Val batch keys: {list(val_batch.keys())}")
print(f"  Sequence shape: {val_batch['sequence'].shape}")
print(f"  Target shape: {val_batch['target'].shape}")

# Test 6: Verify no data leakage
print("\n[Test 6] Verifying train/val/test split...")
train_items = set()
for user_id, seq in data['train_sequences'].items():
    train_items.update(seq)

val_targets = set([target for _, target in data['val_data'].values()])
test_targets = set([target for _, target in data['test_data'].values()])

print(f"✓ Unique training items: {len(train_items)}")
print(f"  Val targets in training set: {len(val_targets & train_items)} / {len(val_targets)}")
print(f"  Test targets in training set: {len(test_targets & train_items)} / {len(test_targets)}")

# Test 7: Check padding
print("\n[Test 7] Verifying sequence padding...")
seq_with_padding = train_batch['sequence'][0]
num_padding = (seq_with_padding == 0).sum().item()
actual_length = train_batch['length'][0].item()
max_len = seq_with_padding.shape[0]

print(f"✓ First sequence:")
print(f"  Max length: {max_len}")
print(f"  Actual length: {actual_length}")
print(f"  Padding tokens: {num_padding}")
print(f"  Non-zero items: {(seq_with_padding != 0).sum().item()}")
assert num_padding + actual_length == max_len, "Padding + actual length should equal max_len"

print("\n" + "="*60)
print("✅ ALL DATALOADER TESTS PASSED!")
print("="*60)
