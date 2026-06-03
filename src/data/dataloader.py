import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class SequenceDataset(Dataset):
    """Dataset for sequential recommendation training"""

    def __init__(
        self,
        sequences,
        num_items,
        max_len=50,
        num_neg_samples=1,
        hard_neg_ratio=0.0,
    ):
        """
        Args:
            sequences: Dict[user_id -> list of item_ids]
            num_items: Total number of items (for negative sampling)
            max_len: Maximum sequence length (truncate if longer)
            num_neg_samples: Number of negative items to sample per positive
            hard_neg_ratio: Probability of drawing negatives from popularity-biased
                distribution instead of uniform sampling
        """
        self.sequences = sequences
        self.num_items = num_items
        self.max_len = max_len
        self.num_neg_samples = num_neg_samples
        self.hard_neg_ratio = float(max(0.0, min(1.0, hard_neg_ratio)))

        if self.num_neg_samples < 1:
            raise ValueError(f"num_neg_samples must be >= 1, got {self.num_neg_samples}")

        # Item-frequency distribution for optional hard-negative sampling.
        self.item_probs = None
        item_freq = np.zeros(self.num_items + 1, dtype=np.float64)
        for seq in sequences.values():
            for item in seq:
                if 1 <= item <= self.num_items:
                    item_freq[item] += 1.0

        total = item_freq[1:].sum()
        if total > 0:
            self.item_probs = item_freq[1:] / total

        # Create training instances: for each position in sequence, predict next item
        self.instances = []
        for user_id, seq in sequences.items():
            # Create instances for positions 1 to len-1 (predicting next item)
            for i in range(1, len(seq)):
                prefix = seq[:i]  # Items up to position i
                target = seq[i]   # Next item to predict
                user_items = set(seq)  # All items user has interacted with (for negative sampling)

                self.instances.append({
                    'user_id': user_id,
                    'prefix': prefix,
                    'target': target,
                    'user_items': user_items,
                    'length': len(prefix)
                })

    def _sample_negative(self, user_items):
        """Sample a negative item not present in the user's interaction history."""
        max_tries = 100
        for _ in range(max_tries):
            use_hard = (
                self.item_probs is not None and
                self.hard_neg_ratio > 0.0 and
                np.random.random() < self.hard_neg_ratio
            )

            if use_hard:
                neg_item = int(np.random.choice(self.num_items, p=self.item_probs)) + 1
            else:
                neg_item = int(np.random.randint(1, self.num_items + 1))

            if neg_item not in user_items:
                return neg_item

        # Safety fallback when rejection sampling struggles for dense users.
        candidates = np.array(list(set(range(1, self.num_items + 1)) - user_items))
        if len(candidates) == 0:
            return int(np.random.randint(1, self.num_items + 1))
        return int(np.random.choice(candidates))

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        instance = self.instances[idx]

        # Truncate or pad prefix sequence
        prefix = instance['prefix'][-self.max_len:]  # Take last max_len items
        length = len(prefix)

        # Pad to max_len
        padded_seq = [0] * self.max_len
        padded_seq[-length:] = prefix

        # Sample negative items (not in user's history)
        neg_items = []
        while len(neg_items) < self.num_neg_samples:
            neg_items.append(self._sample_negative(instance['user_items']))

        return {
            'user_id': instance['user_id'],
            'sequence': torch.LongTensor(padded_seq),
            'length': length,
            'target': instance['target'],
            'negatives': torch.LongTensor(neg_items)
        }


class EvalDataset(Dataset):
    """Dataset for evaluation (val/test)"""

    def __init__(self, eval_data, num_items, max_len=50, num_neg_eval=100):
        """
        Args:
            eval_data: Dict[user_id -> (prefix_seq, target_item)]
            num_items: Total number of items
            max_len: Maximum sequence length
            num_neg_eval: Number of negative items for ranking evaluation
        """
        self.eval_data = list(eval_data.items())
        self.num_items = num_items
        self.max_len = max_len
        self.num_neg_eval = num_neg_eval

    def __len__(self):
        return len(self.eval_data)

    def __getitem__(self, idx):
        user_id, (prefix, target) = self.eval_data[idx]

        # Truncate/pad prefix
        prefix = prefix[-self.max_len:]
        length = len(prefix)
        padded_seq = [0] * self.max_len
        padded_seq[-length:] = prefix

        # For evaluation, we'll rank target among all items (done in evaluator)
        return {
            'user_id': user_id,
            'sequence': torch.LongTensor(padded_seq),
            'length': length,
            'target': target
        }


def get_dataloaders(
    data_path,
    batch_size=256,
    max_len=50,
    num_workers=4,
    num_neg_samples=1,
    hard_neg_ratio=0.0,
):
    """Create train/val/test dataloaders"""
    import pickle

    # Load processed data
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    num_items = data['config']['num_items']

    # Create datasets
    train_dataset = SequenceDataset(
        data['train_sequences'],
        num_items,
        max_len=max_len,
        num_neg_samples=num_neg_samples,
        hard_neg_ratio=hard_neg_ratio,
    )

    val_dataset = EvalDataset(
        data['val_data'],
        num_items,
        max_len=max_len
    )

    test_dataset = EvalDataset(
        data['test_data'],
        num_items,
        max_len=max_len
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, data['config']


# Testing
if __name__ == '__main__':
    train_loader, val_loader, test_loader, config = get_dataloaders(
        'data/1m/ml-1m/processed/sequences.pkl',
        batch_size=64
    )

    print(f"Config: {config}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Test one batch
    batch = next(iter(train_loader))
    print(f"\nBatch keys: {batch.keys()}")
    print(f"Sequence shape: {batch['sequence'].shape}")
    print(f"Lengths: {batch['length']}")
    print(f"Targets: {batch['target']}")
