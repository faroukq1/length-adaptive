import os
from preprocess import ML1MPreprocessor
from graph_builder import CooccurrenceGraphBuilder
import pickle

# Download MovieLens 100k and preprocess
import urllib.request
import zipfile

ROOT_DIR = os.path.join('..', '..')
ML100K_DIR = os.path.join(ROOT_DIR, 'ml-100k')
RAW_DATA_PATH = os.path.join(ML100K_DIR, 'u.data')
GRAPHS_DIR = os.path.join(ROOT_DIR, 'data', 'graphs')
SEQUENCES_PATH = os.path.join(GRAPHS_DIR, 'sequences.pkl')
GRAPH_PATH = os.path.join(GRAPHS_DIR, 'item_graph.pkl')

# Download and extract MovieLens 100k if not present
if not os.path.exists(ML100K_DIR):
    print('Downloading MovieLens 100k...')
    url = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
    zip_path = os.path.join(ROOT_DIR, 'ml-100k.zip')
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(ROOT_DIR)
    os.remove(zip_path)
    print('Downloaded and extracted MovieLens 100k.')

# Ensure output directory exists
os.makedirs(GRAPHS_DIR, exist_ok=True)


# Custom loading for MovieLens 100k
import pandas as pd
print('Loading ratings...')
ratings = pd.read_csv(
    RAW_DATA_PATH,
    sep='\t',
    names=['user_id', 'item_id', 'rating', 'timestamp'],
    dtype={'user_id': int, 'item_id': int, 'rating': int, 'timestamp': int}
)
print(f"Loaded {len(ratings):,} ratings")

# Use ML1MPreprocessor for filtering and sequence building
preprocessor = ML1MPreprocessor(RAW_DATA_PATH, min_rating=4, min_seq_len=5)
filtered = preprocessor.filter_by_rating(ratings)
sequences = preprocessor.build_sequences(filtered)

# Save user sequences
with open(SEQUENCES_PATH, 'wb') as f:
    pickle.dump(sequences, f)
print(f"Saved user sequences to {SEQUENCES_PATH}")

# Build item co-occurrence graph
graph_builder = CooccurrenceGraphBuilder(window_size=3, min_count=5)
edge_dict = graph_builder.build_graph(sequences)

# Save graph
with open(GRAPH_PATH, 'wb') as f:
    pickle.dump(edge_dict, f)
print(f"Saved item graph to {GRAPH_PATH}")
