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


# ==============================================================================
# Amazon Electronics Preprocessor
# ==============================================================================

class AmazonElectronicsPreprocessor:
    """
    Preprocessor for the Amazon Electronics dataset.

    Expected raw file: reviews_Electronics_5.json.gz  (5-core JSON gzip)
    Available at:
        https://jmcauley.ucsd.edu/data/amazon/links.html
        → "5-core" → "Electronics"

    Each line of the JSON file is a dict with at minimum:
        {
          "reviewerID": "A...",
          "asin":       "B...",
          "overall":    5.0,
          "unixReviewTime": 1234567890
        }
    """

    def __init__(
        self,
        raw_data_path: str,
        min_rating: float = 4.0,
        min_seq_len: int = 5,
    ):
        self.raw_data_path = raw_data_path
        self.min_rating = min_rating
        self.min_seq_len = min_seq_len

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _load_raw(self):
        """Load the gzip JSON file into a DataFrame."""
        import gzip, json
        rows = []
        opener = gzip.open if self.raw_data_path.endswith('.gz') else open
        with opener(self.raw_data_path, 'rt', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                rows.append({
                    'user_id':   obj['reviewerID'],
                    'item_id':   obj['asin'],
                    'rating':    float(obj.get('overall', 0)),
                    'timestamp': int(obj.get('unixReviewTime', 0)),
                })
        df = pd.DataFrame(rows)
        print(f"  Loaded {len(df):,} raw interactions")
        return df

    def _filter_ratings(self, df):
        """Keep only positive implicit feedback (rating >= min_rating)."""
        df = df[df['rating'] >= self.min_rating].copy()
        print(f"  After rating filter (>={self.min_rating}): {len(df):,} interactions")
        return df

    def _apply_kcore(self, df, k: int = 5):
        """
        Iteratively remove users/items with fewer than k interactions
        until the dataset is stable (5-core).
        """
        print(f"  Applying {k}-core filtering …")
        while True:
            n_before = len(df)
            item_counts = df['item_id'].value_counts()
            valid_items = item_counts[item_counts >= k].index
            df = df[df['item_id'].isin(valid_items)]

            user_counts = df['user_id'].value_counts()
            valid_users = user_counts[user_counts >= k].index
            df = df[df['user_id'].isin(valid_users)]

            if len(df) == n_before:
                break
        print(f"  After {k}-core: {len(df):,} interactions, "
              f"{df['user_id'].nunique():,} users, "
              f"{df['item_id'].nunique():,} items")
        return df

    def _build_sequences(self, df):
        """Sort by timestamp and build per-user item sequences."""
        df = df.sort_values(['user_id', 'timestamp'])
        sequences = {}
        for user_id, grp in df.groupby('user_id'):
            sequences[user_id] = grp['item_id'].tolist()
        print(f"  Built sequences for {len(sequences):,} users")
        return sequences

    def _filter_short_sequences(self, sequences):
        """Drop users with fewer than min_seq_len interactions."""
        filtered = {u: s for u, s in sequences.items()
                    if len(s) >= self.min_seq_len}
        print(f"  After min_seq_len filter (>={self.min_seq_len}): "
              f"{len(filtered):,} users remain")
        return filtered

    def _remap_ids(self, sequences):
        """
        Remap raw string IDs to contiguous integers starting from 1
        (0 is reserved for padding).
        """
        all_items = sorted({item for seq in sequences.values() for item in seq})
        item2id = {item: idx + 1 for idx, item in enumerate(all_items)}
        num_items = len(item2id)

        all_users = sorted(sequences.keys())
        user2id = {u: idx + 1 for idx, u in enumerate(all_users)}

        remapped = {}
        for user, seq in sequences.items():
            remapped[user2id[user]] = [item2id[i] for i in seq]

        print(f"  Remapped → {len(remapped):,} users, {num_items:,} items")
        return remapped, num_items, user2id, item2id

    def _split_sequences(self, sequences):
        """
        Leave-one-out split (same as ML-1M preprocessor):
          - test  : last item
          - val   : second-to-last item
          - train : everything else
        """
        train_seqs, val_seqs, test_seqs = {}, {}, {}
        for user_id, seq in sequences.items():
            if len(seq) < 3:
                # Need at least 3 items: 1 train + 1 val + 1 test
                continue
            train_seqs[user_id] = seq[:-2]
            val_seqs[user_id]   = seq[:-1]   # train + val target (last element is target)
            test_seqs[user_id]  = seq         # full seq (last element is target)
        return train_seqs, val_seqs, test_seqs

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def preprocess(self, output_path: str):
        """
        Full preprocessing pipeline.  Saves a pickle at *output_path*
        with the same schema as the ML-1M preprocessor so that the
        existing dataloader works unchanged.
        """
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        print("=" * 60)
        print("AMAZON ELECTRONICS PREPROCESSING")
        print("=" * 60)

        print("\n[1/7] Loading raw data …")
        df = self._load_raw()

        print("\n[2/7] Filtering by rating …")
        df = self._filter_ratings(df)

        print("\n[3/7] Applying 5-core filtering …")
        df = self._apply_kcore(df, k=5)

        print("\n[4/7] Building sequences …")
        sequences = self._build_sequences(df)

        print("\n[5/7] Filtering short sequences …")
        sequences = self._filter_short_sequences(sequences)

        print("\n[6/7] Remapping IDs …")
        sequences, num_items, user2id, item2id = self._remap_ids(sequences)

        print("\n[7/7] Splitting (leave-one-out) …")
        train_seqs, val_seqs, test_seqs = self._split_sequences(sequences)

        # Statistics
        seq_lengths = [len(s) for s in sequences.values()]
        print("\n" + "=" * 60)
        print("DATASET STATISTICS")
        print("=" * 60)
        print(f"  Users      : {len(sequences):,}")
        print(f"  Items      : {num_items:,}")
        print(f"  Avg seq len: {sum(seq_lengths)/len(seq_lengths):.1f}")
        print(f"  Min seq len: {min(seq_lengths)}")
        print(f"  Max seq len: {max(seq_lengths)}")
        print(f"  Train users: {len(train_seqs):,}")
        print(f"  Val   users: {len(val_seqs):,}")
        print(f"  Test  users: {len(test_seqs):,}")

        data = {
            'train_sequences': train_seqs,
            'val_sequences':   val_seqs,
            'test_sequences':  test_seqs,
            'user2id': user2id,
            'item2id': item2id,
            'config': {
                'num_users': len(sequences),
                'num_items': num_items,
                'min_rating':   self.min_rating,
                'min_seq_len':  self.min_seq_len,
                'dataset':      'amazon_electronics',
            }
        }

        with open(output_path, 'wb') as f:
            pickle.dump(data, f)

        print(f"\n✅ Saved to {output_path}")
        print("=" * 60)
        return data

