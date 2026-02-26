import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
from tqdm import tqdm

class ML1MPreprocessor:
    def __init__(self, raw_data_path, min_rating=4, min_seq_len=5):
        """
        Args:
            raw_data_path: Path to ratings.dat
            min_rating: Threshold for implicit feedback (rating >= threshold → positive)
            min_seq_len: Minimum sequence length to keep user
        """
        self.raw_data_path = raw_data_path
        self.min_rating = min_rating
        self.min_seq_len = min_seq_len

    def load_ratings(self):
        """Load and parse ratings.dat"""
        print("Loading ratings...")
        ratings = pd.read_csv(
            self.raw_data_path,
            sep='::',
            engine='python',
            names=['user_id', 'item_id', 'rating', 'timestamp'],
            dtype={'user_id': int, 'item_id': int, 'rating': int, 'timestamp': int}
        )
        print(f"Loaded {len(ratings):,} ratings")
        return ratings

    def filter_by_rating(self, ratings):
        """Keep only ratings >= min_rating (implicit positive feedback)"""
        print(f"Filtering ratings >= {self.min_rating}...")
        filtered = ratings[ratings['rating'] >= self.min_rating].copy()
        print(f"Kept {len(filtered):,} positive interactions ({100*len(filtered)/len(ratings):.1f}%)")
        return filtered

    def build_sequences(self, ratings):
        """Group by user and sort by timestamp to create sequences"""
        print("Building chronological sequences...")

        # Sort by user and timestamp
        ratings = ratings.sort_values(['user_id', 'timestamp'])

        # Group by user
        user_sequences = defaultdict(list)
        for _, row in tqdm(ratings.iterrows(), total=len(ratings), desc="Processing"):
            user_sequences[row['user_id']].append(row['item_id'])

        print(f"Built sequences for {len(user_sequences):,} users")
        return user_sequences

    def filter_short_sequences(self, user_sequences):
        """Remove users with too few interactions"""
        print(f"Filtering users with < {self.min_seq_len} interactions...")

        filtered = {
            user: seq for user, seq in user_sequences.items()
            if len(seq) >= self.min_seq_len
        }

        print(f"Kept {len(filtered):,} users ({100*len(filtered)/len(user_sequences):.1f}%)")

        # Print length distribution
        lengths = [len(seq) for seq in filtered.values()]
        print(f"Sequence length stats:")
        print(f"  Min: {np.min(lengths)}")
        print(f"  Max: {np.max(lengths)}")
        print(f"  Mean: {np.mean(lengths):.1f}")
        print(f"  Median: {np.median(lengths):.1f}")

        return filtered

    def remap_ids(self, user_sequences):
        """Map original IDs to continuous indices starting from 1 (0 reserved for padding)"""
        print("Remapping user and item IDs...")

        # Collect all unique items
        all_items = set()
        for seq in user_sequences.values():
            all_items.update(seq)

        # Create mappings (1-indexed, 0 reserved for padding)
        item_to_idx = {item: idx for idx, item in enumerate(sorted(all_items), start=1)}
        user_to_idx = {user: idx for idx, user in enumerate(sorted(user_sequences.keys()), start=1)}

        # Remap sequences
        remapped = {}
        for user, seq in user_sequences.items():
            new_user_id = user_to_idx[user]
            new_seq = [item_to_idx[item] for item in seq]
            remapped[new_user_id] = new_seq

        print(f"Remapped to {len(user_to_idx):,} users and {len(item_to_idx):,} items")

        # Create reverse mappings for reference
        idx_to_user = {v: k for k, v in user_to_idx.items()}
        idx_to_item = {v: k for k, v in item_to_idx.items()}

        mappings = {
            'user_to_idx': user_to_idx,
            'item_to_idx': item_to_idx,
            'idx_to_user': idx_to_user,
            'idx_to_item': idx_to_item
        }

        return remapped, mappings

    def split_sequences(self, user_sequences):
        """
        Split each sequence into train/val/test using leave-one-out

        For user with sequence [i1, i2, i3, i4, i5]:
        - Train: [i1, i2, i3]
        - Val: [i1, i2, i3, i4] with target i4
        - Test: [i1, i2, i3, i4] with target i5
        """
        print("Splitting sequences into train/val/test...")

        train_seqs = {}
        val_data = {}
        test_data = {}

        for user, seq in user_sequences.items():
            if len(seq) < 3:  # Need at least 3 items for meaningful split
                continue

            # Split
            train_seqs[user] = seq[:-2]  # All but last 2
            val_data[user] = (seq[:-2], seq[-2])  # Sequence up to second-last, target = second-last
            test_data[user] = (seq[:-1], seq[-1])  # Sequence up to last, target = last

        print(f"Split data for {len(train_seqs):,} users")
        print(f"  Train sequences: {len(train_seqs):,}")
        print(f"  Val instances: {len(val_data):,}")
        print(f"  Test instances: {len(test_data):,}")

        return train_seqs, val_data, test_data

    def preprocess(self, output_path):
        """Run full preprocessing pipeline"""
        print("="*60)
        print("MOVIELENS-1M PREPROCESSING PIPELINE")
        print("="*60)

        # Step 1: Load ratings
        ratings = self.load_ratings()

        # Step 2: Filter by rating threshold
        ratings = self.filter_by_rating(ratings)

        # Step 3: Build chronological sequences
        user_sequences = self.build_sequences(ratings)

        # Step 4: Filter short sequences
        user_sequences = self.filter_short_sequences(user_sequences)

        # Step 5: Remap IDs
        user_sequences, mappings = self.remap_ids(user_sequences)

        # Step 6: Split into train/val/test
        train_seqs, val_data, test_data = self.split_sequences(user_sequences)

        # Save processed data
        print(f"\nSaving processed data to {output_path}...")
        data = {
            'train_sequences': train_seqs,
            'val_data': val_data,
            'test_data': test_data,
            'mappings': mappings,
            'config': {
                'min_rating': self.min_rating,
                'min_seq_len': self.min_seq_len,
                'num_users': len(mappings['user_to_idx']),
                'num_items': len(mappings['item_to_idx']),
            }
        }

        with open(output_path, 'wb') as f:
            pickle.dump(data, f)

        print("✅ Preprocessing complete!")
        print(f"   Users: {data['config']['num_users']:,}")
        print(f"   Items: {data['config']['num_items']:,}")
        print("="*60)

        return data


# Usage script
if __name__ == '__main__':
    preprocessor = ML1MPreprocessor(
        raw_data_path='data/ml-1m/raw/ml-1m/ratings.dat',
        min_rating=4,  # Only ratings >= 4 are positive
        min_seq_len=5  # Keep users with >= 5 interactions
    )

    data = preprocessor.preprocess('data/ml-1m/processed/sequences.pkl')
