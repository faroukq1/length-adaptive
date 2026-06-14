import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
from tqdm import tqdm


# ==============================================================================
# MovieLens 100K Preprocessor  (auto-download + preprocess)
# ==============================================================================

class ML100KPreprocessor:
    """
    Preprocessor for the MovieLens 100K dataset.

    Key differences from ML-1M:
      - File is a ZIP archive (not individual .dat files)
      - Ratings file is `u.data` (tab-separated, no engine='python' needed)
      - Columns: user_id  item_id  rating  timestamp  (space/tab separated)
      - IDs are already integers, no special parsing needed
      - The ZIP contains many files; we only need `ml-100k/u.data`

    Download URL:
        https://files.grouplens.org/datasets/movielens/ml-100k.zip
    """

    DOWNLOAD_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    ZIP_NAME     = "ml-100k.zip"
    RATINGS_FILE = "ml-100k/u.data"   # path inside the ZIP

    def __init__(
        self,
        raw_data_dir: str = "data/ml-100k/raw",
        min_rating:   int = 4,
        min_seq_len:  int = 5,
    ):
        """
        Args:
            raw_data_dir : Directory where the ZIP is downloaded & extracted.
            min_rating   : Implicit feedback threshold (rating >= value → positive).
            min_seq_len  : Minimum interactions per user (shorter sequences dropped).
        """
        self.raw_data_dir = raw_data_dir
        self.min_rating   = min_rating
        self.min_seq_len  = min_seq_len

    # ------------------------------------------------------------------
    # Step 0 – download & extract
    # ------------------------------------------------------------------

    def download(self):
        """Download ml-100k.zip if not already present, then extract."""
        os.makedirs(self.raw_data_dir, exist_ok=True)
        zip_path = os.path.join(self.raw_data_dir, self.ZIP_NAME)

        if not os.path.exists(zip_path):
            print(f"Downloading MovieLens 100K from:\n  {self.DOWNLOAD_URL}")
            urllib.request.urlretrieve(self.DOWNLOAD_URL, zip_path)
            print(f"  Saved to {zip_path}")
        else:
            print(f"  ZIP already exists: {zip_path}  (skipping download)")

        # Extract only if the ratings file is not already present
        ratings_path = os.path.join(self.raw_data_dir, self.RATINGS_FILE)
        if not os.path.exists(ratings_path):
            print(f"Extracting {self.ZIP_NAME} …")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(self.raw_data_dir)
            print("  Extraction complete.")
        else:
            print(f"  Already extracted: {ratings_path}")

        return ratings_path

    # ------------------------------------------------------------------
    # Step 1 – load
    # ------------------------------------------------------------------

    def load_ratings(self, ratings_path: str) -> pd.DataFrame:
        """
        Load u.data.

        Format: user_id<TAB>item_id<TAB>rating<TAB>timestamp
        (no header, tab-separated)
        """
        print("Loading ratings …")
        df = pd.read_csv(
            ratings_path,
            sep="\t",
            names=["user_id", "item_id", "rating", "timestamp"],
            dtype={"user_id": int, "item_id": int, "rating": int, "timestamp": int},
        )
        print(f"  Loaded {len(df):,} ratings from {ratings_path}")
        return df

    # ------------------------------------------------------------------
    # Step 2 – filter by rating
    # ------------------------------------------------------------------

    def filter_by_rating(self, df: pd.DataFrame) -> pd.DataFrame:
        print(f"Filtering ratings >= {self.min_rating} …")
        filtered = df[df["rating"] >= self.min_rating].copy()
        pct = 100 * len(filtered) / len(df)
        print(f"  Kept {len(filtered):,} positive interactions ({pct:.1f}%)")
        return filtered

    # ------------------------------------------------------------------
    # Step 3 – build chronological sequences
    # ------------------------------------------------------------------

    def build_sequences(self, df: pd.DataFrame) -> dict:
        print("Building chronological sequences …")
        df = df.sort_values(["user_id", "timestamp"])

        user_sequences: dict = defaultdict(list)
        for _, row in tqdm(df.iterrows(), total=len(df), desc="  Processing rows"):
            user_sequences[row["user_id"]].append(row["item_id"])

        print(f"  Built sequences for {len(user_sequences):,} users")
        return dict(user_sequences)

    # ------------------------------------------------------------------
    # Step 4 – filter short sequences
    # ------------------------------------------------------------------

    def filter_short_sequences(self, user_sequences: dict) -> dict:
        print(f"Filtering users with < {self.min_seq_len} interactions …")
        filtered = {u: s for u, s in user_sequences.items() if len(s) >= self.min_seq_len}
        pct = 100 * len(filtered) / max(len(user_sequences), 1)
        print(f"  Kept {len(filtered):,} users ({pct:.1f}%)")

        lengths = [len(s) for s in filtered.values()]
        print(f"  Sequence length — min: {np.min(lengths)}, "
              f"max: {np.max(lengths)}, "
              f"mean: {np.mean(lengths):.1f}, "
              f"median: {np.median(lengths):.1f}")
        return filtered

    # ------------------------------------------------------------------
    # Step 5 – remap IDs
    # ------------------------------------------------------------------

    def remap_ids(self, user_sequences: dict):
        """
        Map original IDs → contiguous integers starting at 1.
        Index 0 is reserved for padding.
        """
        print("Remapping user and item IDs …")
        all_items = sorted({item for seq in user_sequences.values() for item in seq})
        item_to_idx = {item: idx for idx, item in enumerate(all_items, start=1)}
        user_to_idx = {user: idx for idx, user in enumerate(sorted(user_sequences), start=1)}

        remapped = {}
        for user, seq in user_sequences.items():
            remapped[user_to_idx[user]] = [item_to_idx[i] for i in seq]

        print(f"  Remapped → {len(user_to_idx):,} users, {len(item_to_idx):,} items")

        mappings = {
            "user_to_idx": user_to_idx,
            "item_to_idx": item_to_idx,
            "idx_to_user": {v: k for k, v in user_to_idx.items()},
            "idx_to_item": {v: k for k, v in item_to_idx.items()},
        }
        return remapped, mappings

    # ------------------------------------------------------------------
    # Step 6 – leave-one-out split
    # ------------------------------------------------------------------

    def split_sequences(self, user_sequences: dict):
        """
        Leave-one-out split (identical logic to ML-1M preprocessor):
          train : seq[:-2]
          val   : (seq[:-2], seq[-2])   ← input / target
          test  : (seq[:-1], seq[-1])   ← input / target
        """
        print("Splitting into train / val / test (leave-one-out) …")
        train_seqs, val_data, test_data = {}, {}, {}

        for user, seq in user_sequences.items():
            if len(seq) < 3:
                continue
            train_seqs[user] = seq[:-2]
            val_data[user]   = (seq[:-2], seq[-2])
            test_data[user]  = (seq[:-1], seq[-1])

        print(f"  Train: {len(train_seqs):,} users | "
              f"Val: {len(val_data):,} | "
              f"Test: {len(test_data):,}")
        return train_seqs, val_data, test_data

    # ------------------------------------------------------------------
    # Public entry-point
    # ------------------------------------------------------------------

    def preprocess(self, output_path: str):
        """Run the full pipeline and save a pickle compatible with the ML-1M schema."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        print("=" * 60)
        print("MOVIELENS-100K PREPROCESSING PIPELINE")
        print("=" * 60)

        # 0. Download & extract
        ratings_path = self.download()

        # 1. Load
        ratings = self.load_ratings(ratings_path)

        # 2. Filter by rating
        ratings = self.filter_by_rating(ratings)

        # 3. Build sequences
        user_sequences = self.build_sequences(ratings)

        # 4. Filter short sequences
        user_sequences = self.filter_short_sequences(user_sequences)

        # 5. Remap IDs
        user_sequences, mappings = self.remap_ids(user_sequences)

        # 6. Split
        train_seqs, val_data, test_data = self.split_sequences(user_sequences)

        # 7. Save  (same schema as ML-1M preprocessor)
        data = {
            "train_sequences": train_seqs,
            "val_data":        val_data,
            "test_data":       test_data,
            "mappings":        mappings,
            "config": {
                "dataset":      "ml-100k",
                "min_rating":   self.min_rating,
                "min_seq_len":  self.min_seq_len,
                "num_users":    len(mappings["user_to_idx"]),
                "num_items":    len(mappings["item_to_idx"]),
            },
        }

        print(f"\nSaving to {output_path} …")
        with open(output_path, "wb") as f:
            pickle.dump(data, f)

        print("✅  Preprocessing complete!")
        print(f"   Users : {data['config']['num_users']:,}")
        print(f"   Items : {data['config']['num_items']:,}")
        print("=" * 60)

        return data


# ==============================================================================
# Entry-point
# ==============================================================================

if __name__ == "__main__":
    preprocessor = ML100KPreprocessor(
        raw_data_dir="data/ml-100k/raw",
        min_rating=4,
        min_seq_len=5,
    )
    data = preprocessor.preprocess("data/ml-100k/processed/sequences.pkl")