# PromptCraft-SeqRec: Agent Execution Guide
## Hybrid LLM-Embedding Models vs Baselines on MovieLens 100K

> **For AI Coding Agent (Copilot / Claude / ChatGPT)**  
> This file is a complete, self-contained execution plan. Read it fully before writing any code.  
> Every section is a direct instruction. Every code block is copy-paste ready.  
> You do NOT need any other document.

---

## 0. Project Overview

### What This Project Does

This project tests whether **how you describe an item to a free LLM embedding model (BGE-M3)** changes the quality of sequential recommendations. You compare **4 prompt styles** on **MovieLens 100K** using **SASRec**, and show that at least one of your hybrid prompt styles beats the paper's default (title-only baseline).

| Property         | Value                                         |
|------------------|-----------------------------------------------|
| Dataset          | MovieLens 100K (already preprocessed)         |
| Embedding Model  | BGE-M3 (free, runs on Kaggle P100)            |
| Rec Model        | SASRec (self-attention sequential rec)        |
| Prompt Styles    | 4 (P1 baseline → P4 hybrid)                  |
| Goal             | P4 hybrid beats P1 title-only on NDCG@10     |
| Completion Time  | 4–7 days                                      |
| New Code Needed  | ~40 lines                                     |

### One-Sentence Research Summary

> You test whether changing HOW you describe a movie to BGE-M3 changes how good the recommendations are — comparing 4 prompt styles on MovieLens 100K using SASRec, proving that a hybrid description (title + genre + user-preference framing) consistently outperforms title-only embedding.

---

## 1. Research Background & The Gap You Exploit

### 1.1 What the Original Paper Does

The paper **"Improving Sequential Recommendations with LLMs"** (Boz et al., 2025) shows:
- Taking SASRec and **initialising** its item embeddings with an LLM produces significantly better recommendations
- Their method (LLM2SASRec) improves NDCG@20 by **45%** on Amazon Beauty vs plain SASRec
- To generate embeddings, they feed **only the product name** to the embedding model

> *"For the Beauty and the Delivery Hero datasets, the metadata used to compute the embeddings are the names of the products."*  
> — Boz et al., 2025, Section 5.2

**They never test whether a richer description produces better embeddings.**

### 1.2 The Gap You Exploit

For the Steam gaming dataset, game names alone were semantically weak ("Minecraft" tells you nothing), so the authors added tags — which improved results. But they treated this as a dataset-specific fix, not a systematic finding.

**Your research question:** If you describe the same movie in 4 different ways to BGE-M3, does it produce 4 different quality embeddings — and which description style is best?

### 1.3 Your Contribution

1. Design **4 prompt styles** for movie item descriptions
2. Generate **4 sets of embeddings** using BGE-M3 (free, runs on Kaggle)
3. Train SASRec with each embedding set on **MovieLens 100K**
4. Compare results against **baselines** (SASRec no-LLM, GRU4Rec, BERT4Rec)
5. Show P4 hybrid beats the title-only default → **novel finding for your mémoire**

### 1.4 Why This Counts as a Research Contribution

- No prior work systematically studies prompt framing for LLM-enhanced sequential rec
- You are the first to apply this on MovieLens 100K with a controlled prompt ablation
- The hybrid framing (P4) aligns embeddings with recommendation intent, not just item semantics
- This is directly publishable as a conference short paper or mémoire chapter

---

## 2. Architecture Explained

### 2.1 System Pipeline

```
MovieLens 100K
    │
    ▼
[Preprocessing]
 - 5-core filter
 - Sort by timestamp
 - Leave-one-out split (train/val/test)
    │
    ▼
[Item Metadata]  ←── movie titles, genres from movies.dat
    │
    ▼
[4 Prompt Functions]
 P1: title only          → "Toy Story (1995)"
 P2: title + genre       → "Toy Story (1995) | Genre: Animation, Children"
 P3: user-centric        → "Users who like Toy Story enjoy: animation, family, adventure"
 P4: hybrid (your best)  → "Toy Story (1995) | Genre: Animation | For fans of: family, adventure"
    │
    ▼
[BGE-M3 Encoder]  (BAAI/bge-m3, free, Kaggle P100)
 Outputs 1024-dim dense vectors per item per style
    │
    ▼
[PCA Compression]
 1024-dim → 64-dim (preserves ~80% variance, fits SASRec emb_dim)
    │
    ▼
[SASRec Initialisation]
 item_emb.weight ← PCA-compressed LLM vectors
    │
    ▼
[Training]  BPR loss, Adam, 50 epochs, batch=256
    │
    ▼
[Evaluation]
 HR@10, NDCG@10, HR@20, NDCG@20
 Leave-one-out on test set
    │
    ▼
[Results Table + Plots]
 Compare: Baseline | P1 | P2 | P3 | P4 | GRU4Rec | BERT4Rec
```

### 2.2 Why SASRec?

SASRec uses a **Transformer encoder** with causal (left-to-right) masking to model sequential item interaction patterns. It predicts the next item by attending to the user's full history. By initialising its embedding table with LLM vectors, it starts with rich semantic knowledge instead of random noise.

### 2.3 Why BGE-M3?

- **Free** — no API cost, no rate limits
- **Runs locally** on Kaggle P100 GPU
- Produces **1024-dim dense vectors** from any text
- Better multilingual & domain generalization than older models (e.g. Sentence-BERT)
- Used via `FlagEmbedding` Python library

### 2.4 The 4 Prompt Styles

| Style | Name              | Text Sent to BGE-M3                                                           | Hypothesis                                     |
|-------|-------------------|-------------------------------------------------------------------------------|------------------------------------------------|
| P1    | Title only        | `"Toy Story (1995)"`                                                          | Replicates paper default. Weakest expected.    |
| P2    | Title + Genre     | `"Toy Story (1995) \| Genre: Animation, Children"`                            | Genre separates items semantically             |
| P3    | User-centric      | `"Users who like Toy Story enjoy: animation, family, adventure"`              | Aligns embedding with rec intent               |
| P4    | Hybrid (your best)| `"Toy Story (1995) \| Genre: Animation \| For fans of: family, adventure"`   | Best of both — structured + intent framing     |

---

## 3. Environment Setup

### 3.1 Kaggle Notebook Settings

- Create a **new Kaggle notebook**
- Set accelerator: **GPU P100**
- Enable **internet access** (needed for BGE-M3 download and dataset)
- Runtime: Python 3, PyTorch pre-installed

### 3.2 Install Dependencies

```python
# CELL 1 — Run this first
!pip install FlagEmbedding recbole torch numpy pandas scikit-learn matplotlib seaborn -q

import torch
print('GPU available:', torch.cuda.is_available())
print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')
# Expected: GPU available: True | GPU name: Tesla P100-PCIE-16GB
```

---

## 4. Data: MovieLens 100K

### 4.1 Context (What You Already Have)

You mentioned you already have **baselines and preprocessing** for MovieLens 100K. This section documents the exact preprocessing pipeline so the agent knows what files to expect and how they were created.

> **Agent instruction:** If `data/ml-100k/splits.pkl` already exists, skip to Section 5.  
> If not, run Cell 2 and Cell 3 below.

### 4.2 Download MovieLens 100K

```python
# CELL 2 — Download MovieLens 100K
import os, requests, zipfile, io

os.makedirs('data/ml-100k', exist_ok=True)

url = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
print('Downloading MovieLens 100K...')
r = requests.get(url)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall('data/')
print('Extracted to data/ml-100k/')

# Key files:
# data/ml-100k/u.data   — tab-separated: user_id, item_id, rating, timestamp
# data/ml-100k/u.item   — pipe-separated: movie_id, title, release_date, ..., genres (19 binary cols)
# data/ml-100k/u.genre  — genre names list
```

### 4.3 Parse Item Metadata (Titles + Genres)

```python
# CELL 3 — Parse movies metadata
import pandas as pd
import pickle

GENRE_COLS = [
    'unknown', 'Action', 'Adventure', 'Animation', 'Children',
    'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
    'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
    'Sci-Fi', 'Thriller', 'War', 'Western'
]

items_df = pd.read_csv(
    'data/ml-100k/u.item',
    sep='|', encoding='latin-1', header=None,
    names=['movie_id', 'title', 'release_date', 'video_release', 'imdb_url'] + GENRE_COLS
)

# Build item_meta dict: {movie_id (int) -> {'title': str, 'genres': list}}
item_meta = {}
for _, row in items_df.iterrows():
    genres = [g for g in GENRE_COLS if row[g] == 1]
    item_meta[int(row['movie_id'])] = {
        'title': str(row['title']).strip(),
        'genres': genres
    }

print(f'Total movies in metadata: {len(item_meta):,}')
pickle.dump(item_meta, open('data/ml-100k/item_meta.pkl', 'wb'))
print('Saved: data/ml-100k/item_meta.pkl')
```

### 4.4 Preprocessing: Filter, Sort, Split

```python
# CELL 4 — Preprocessing: 5-core filter, timestamp sort, leave-one-out split
import pandas as pd, numpy as np, pickle

df = pd.read_csv(
    'data/ml-100k/u.data',
    sep='\t', header=None,
    names=['user', 'item', 'rating', 'timestamp']
)
print(f'Raw interactions: {len(df):,}')

# Step 1: 5-core filter (users AND items with >= 5 interactions)
for _ in range(10):
    user_counts = df.user.value_counts()
    item_counts = df.item.value_counts()
    df = df[df.user.isin(user_counts[user_counts >= 5].index)]
    df = df[df.item.isin(item_counts[item_counts >= 5].index)]

print(f'After 5-core filter: {len(df):,} interactions')
print(f'  Users: {df.user.nunique():,} | Items: {df.item.nunique():,}')

# Step 2: Sort by user then timestamp
df = df.sort_values(['user', 'timestamp']).reset_index(drop=True)

# Step 3: Integer IDs (1-indexed, 0 reserved for padding)
user2id = {u: i+1 for i, u in enumerate(df.user.unique())}
item2id = {it: i+1 for i, it in enumerate(df.item.unique())}
id2item = {v: k for k, v in item2id.items()}

df['user_id'] = df.user.map(user2id)
df['item_id'] = df.item.map(item2id)

n_users = len(user2id)
n_items = len(item2id)
print(f'n_users={n_users}, n_items={n_items}')

# Step 4: Build user sequences
user_sequences = df.groupby('user_id')['item_id'].apply(list).to_dict()

# Step 5: Leave-one-out split
# - test: last item (ground truth)
# - val: second-to-last item
# - train: all remaining
train_data, val_data, test_data = {}, {}, {}
for uid, seq in user_sequences.items():
    if len(seq) < 3:
        continue
    train_data[uid] = seq[:-2]
    val_data[uid]   = seq[:-1]   # input = all but last, label = last
    test_data[uid]  = seq        # input = all but last, label = last

print(f'Train users: {len(train_data):,} | Test users: {len(test_data):,}')

# Step 6: Save
pickle.dump({
    'train': train_data, 'val': val_data, 'test': test_data,
    'user2id': user2id, 'item2id': item2id, 'id2item': id2item,
    'n_users': n_users, 'n_items': n_items
}, open('data/ml-100k/splits.pkl', 'wb'))
print('Saved: data/ml-100k/splits.pkl')
```

> **Checkpoint:** You should have `data/ml-100k/splits.pkl` and `data/ml-100k/item_meta.pkl` on disk before continuing.

---

## 5. The 4 Prompt Styles — Your Core Research Contribution

These 4 functions are your **entire research contribution**. Each takes a movie integer ID and returns a text string that gets encoded by BGE-M3.

```python
# CELL 5 — The 4 prompt functions
import pickle

item_meta = pickle.load(open('data/ml-100k/item_meta.pkl', 'rb'))
splits    = pickle.load(open('data/ml-100k/splits.pkl', 'rb'))
id2item   = splits['id2item']

def get_item_text(item_id: int, style: str) -> str:
    """
    Given an item integer ID and a prompt style,
    return the text string to encode with BGE-M3.
    """
    original_id = id2item.get(item_id, -1)
    meta  = item_meta.get(original_id, {})
    title = meta.get('title', 'Unknown Movie').strip()
    genres = meta.get('genres', [])

    # Main genre (most specific)
    main_genre = genres[0] if genres else 'Movie'
    # Tag list (up to 4 genres for framing)
    tag_str = ', '.join(genres[:4]) if genres else 'entertainment'

    if style == 'P1_title_only':
        return title

    elif style == 'P2_title_genre':
        return f'{title} | Genre: {", ".join(genres[:3])}' if genres else title

    elif style == 'P3_user_centric':
        return f'Users who like {title} enjoy: {tag_str}'

    elif style == 'P4_hybrid':
        return f'{title} | Genre: {main_genre} | For fans of: {tag_str}'

    else:
        raise ValueError(f'Unknown style: {style}')

STYLES = ['P1_title_only', 'P2_title_genre', 'P3_user_centric', 'P4_hybrid']

# Sanity check — print one example per style
sample_id = list(splits['id2item'].keys())[0]
print("=== Prompt Style Examples ===")
for style in STYLES:
    print(f'[{style}] -> {get_item_text(sample_id, style)}')
    print()
```

**Expected output example:**
```
[P1_title_only]     -> Toy Story (1995)
[P2_title_genre]    -> Toy Story (1995) | Genre: Animation, Children, Comedy
[P3_user_centric]   -> Users who like Toy Story (1995) enjoy: Animation, Children, Comedy, Adventure
[P4_hybrid]         -> Toy Story (1995) | Genre: Animation | For fans of: Animation, Children, Comedy, Adventure
```

---

## 6. Generate BGE-M3 Embeddings for All 4 Styles

This is the most GPU-intensive step. Run it **once per style**. BGE-M3 runs locally on Kaggle P100 — no API cost. For ~1,600 unique items in ML-100K, each style takes about **5–10 minutes**.

> **Agent instruction:** Run this cell 4 times — change only `CURRENT_STYLE` each time.  
> Each run saves a `.npy` file. These files are inputs to Section 7.

```python
# CELL 6 — Generate BGE-M3 embeddings for one style
# ─── CHANGE THIS LINE FOR EACH RUN ───────────────────────────────────────────
CURRENT_STYLE = 'P1_title_only'  # options: P1_title_only / P2_title_genre / P3_user_centric / P4_hybrid
# ─────────────────────────────────────────────────────────────────────────────

from FlagEmbedding import BGEM3FlagModel
import numpy as np, pickle, os

splits   = pickle.load(open('data/ml-100k/splits.pkl', 'rb'))
id2item  = splits['id2item']
n_items  = splits['n_items']

# Load BGE-M3 (~2GB on first run, cached after)
print('Loading BGE-M3 model...')
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)  # fp16 for P100 speed
print('Model loaded.')

# Build text list — index 0 is padding embedding
item_texts = ['[PAD]']  # index 0 = padding
for item_id in range(1, n_items + 1):
    item_texts.append(get_item_text(item_id, CURRENT_STYLE))

print(f'Encoding {len(item_texts)} items with style [{CURRENT_STYLE}]...')

# Encode in batches of 256 (fits P100 VRAM comfortably)
all_embeddings = []
BATCH = 256

for i in range(0, len(item_texts), BATCH):
    batch = item_texts[i : i + BATCH]
    emb = model.encode(batch, batch_size=BATCH, max_length=128)['dense_vecs']
    all_embeddings.append(emb)
    if i % (BATCH * 5) == 0:
        print(f'  Encoded {i}/{len(item_texts)}...')

embeddings = np.vstack(all_embeddings)  # shape: (n_items+1, 1024)

# Save
os.makedirs('embeddings', exist_ok=True)
out_path = f'embeddings/{CURRENT_STYLE}.npy'
np.save(out_path, embeddings)
print(f'Saved: {out_path}  shape={embeddings.shape}')

# ── After this cell: change CURRENT_STYLE and run again ─────────────────────
# Run 1: CURRENT_STYLE = 'P1_title_only'    -> embeddings/P1_title_only.npy
# Run 2: CURRENT_STYLE = 'P2_title_genre'   -> embeddings/P2_title_genre.npy
# Run 3: CURRENT_STYLE = 'P3_user_centric'  -> embeddings/P3_user_centric.npy
# Run 4: CURRENT_STYLE = 'P4_hybrid'        -> embeddings/P4_hybrid.npy
```

> **Checkpoint:** `ls embeddings/` should show 4 `.npy` files after all runs.

---

## 7. SASRec Model — Complete Implementation

This is a clean, minimal SASRec that accepts pre-computed embeddings as initialisation. No external recommendation library needed — copy this cell entirely.

```python
# CELL 7 — SASRec model definition (complete, no external library needed)
import torch
import torch.nn as nn

class SASRec(nn.Module):
    def __init__(self, n_items, emb_dim=64, n_heads=2, n_layers=2,
                 max_seq_len=50, dropout=0.2):
        super().__init__()
        self.n_items  = n_items
        self.emb_dim  = emb_dim
        self.max_seq  = max_seq_len

        # Item embedding table (will be initialised with LLM embeddings)
        self.item_emb = nn.Embedding(n_items + 1, emb_dim, padding_idx=0)
        # Positional embedding
        self.pos_emb  = nn.Embedding(max_seq_len + 1, emb_dim)

        # Transformer encoder layers
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=n_heads,
            dim_feedforward=emb_dim * 4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.layer_norm  = nn.LayerNorm(emb_dim)
        self.dropout     = nn.Dropout(dropout)

    def forward(self, seq):  # seq: (B, L) item IDs, 0 = pad
        B, L = seq.shape
        positions = torch.arange(1, L + 1, device=seq.device).unsqueeze(0).expand(B, L)
        x = self.item_emb(seq) + self.pos_emb(positions)
        x = self.dropout(self.layer_norm(x))
        # Causal mask: each position only attends to past
        mask = nn.Transformer.generate_square_subsequent_mask(L, device=seq.device)
        x = self.transformer(x, mask=mask, is_causal=True)
        return x  # (B, L, emb_dim)

    def predict(self, seq, candidate_items):
        """Score candidate items against last-position hidden state."""
        h     = self.forward(seq)[:, -1, :]  # (B, D)
        c_emb = self.item_emb(candidate_items)  # (B, K, D)
        return (c_emb * h.unsqueeze(1)).sum(-1)  # (B, K) dot products
```

### 7.1 Initialise SASRec with LLM Embeddings

```python
# CELL 8 — Build SASRec with LLM embedding initialisation (key contribution)
import torch.nn.functional as F
from sklearn.decomposition import PCA
import numpy as np, pickle

def build_model_with_embeddings(style_name, emb_dim=64):
    """
    Load pre-computed BGE-M3 embeddings for a given prompt style,
    compress to emb_dim with PCA, and initialise SASRec.
    Returns: model (SASRec), ready for training.
    """
    splits  = pickle.load(open('data/ml-100k/splits.pkl', 'rb'))
    n_items = splits['n_items']

    # Load raw embeddings (n_items+1, 1024)
    raw_emb = np.load(f'embeddings/{style_name}.npy')

    # PCA: 1024 -> emb_dim (fit on item embeddings only, exclude padding row 0)
    pca        = PCA(n_components=emb_dim, random_state=42)
    compressed = pca.fit_transform(raw_emb[1:])  # (N, emb_dim)
    pad_row    = np.zeros((1, emb_dim))
    compressed = np.vstack([pad_row, compressed])  # (N+1, emb_dim)

    print(f'[{style_name}] PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}')

    # Build SASRec and initialise embedding table
    model = SASRec(n_items=n_items, emb_dim=emb_dim)
    with torch.no_grad():
        model.item_emb.weight.copy_(torch.FloatTensor(compressed))

    return model

# Test — build model for P1
model_test = build_model_with_embeddings('P1_title_only', emb_dim=64)
print('Model parameters:', sum(p.numel() for p in model_test.parameters()))
```

---

## 8. Training Loop

```python
# CELL 9 — Training utilities
import torch, random, numpy as np
from torch.optim import Adam

def get_batch(train_data, n_items, batch_size=256, max_seq=50):
    """Sample a random batch of (sequence, positive_item, negative_item) triples."""
    users = random.sample(list(train_data.keys()), min(batch_size, len(train_data)))
    seqs, pos_items, neg_items = [], [], []

    for uid in users:
        seq = train_data[uid]
        if len(seq) < 2:
            continue
        t   = random.randint(1, len(seq) - 1)
        pos = seq[t]
        # Negative sampling — random item not in user history
        neg = random.randint(1, n_items)
        while neg in seq:
            neg = random.randint(1, n_items)
        # Pad/truncate input sequence to max_seq
        s       = seq[:t][-max_seq:]
        pad_len = max_seq - len(s)
        s       = [0] * pad_len + s
        seqs.append(s)
        pos_items.append(pos)
        neg_items.append(neg)

    return (torch.LongTensor(seqs),
            torch.LongTensor(pos_items),
            torch.LongTensor(neg_items))


def bpr_loss(model, seqs, pos_items, neg_items, device):
    """Bayesian Personalised Ranking loss (standard for recommendation)."""
    seqs      = seqs.to(device)
    pos_items = pos_items.to(device)
    neg_items = neg_items.to(device)
    h         = model(seqs)[:, -1, :]          # (B, D)
    pos_emb   = model.item_emb(pos_items)       # (B, D)
    neg_emb   = model.item_emb(neg_items)       # (B, D)
    pos_score = (h * pos_emb).sum(-1)           # (B,)
    neg_score = (h * neg_emb).sum(-1)           # (B,)
    return -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()


def train_model(model, train_data, n_items, device,
                n_epochs=50, batch_size=256, lr=1e-3):
    """Full training loop. Returns list of per-epoch losses."""
    model.to(device)
    model.train()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    losses    = []

    for epoch in range(1, n_epochs + 1):
        epoch_losses = []
        for step in range(100):  # 100 batches per epoch
            seqs, pos, neg = get_batch(train_data, n_items, batch_size)
            optimizer.zero_grad()
            loss = bpr_loss(model, seqs, pos, neg, device)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        if epoch % 10 == 0:
            print(f'Epoch {epoch:3d}/{n_epochs} | Loss: {avg_loss:.4f}')

    return losses
```

---

## 9. Evaluation — Hit Rate and NDCG

```python
# CELL 10 — Evaluation metrics (HR@K and NDCG@K)
import torch, numpy as np

def evaluate(model, test_data, n_items, device, top_k=10, max_seq=50):
    """
    Evaluate on test set using leave-one-out protocol.
    For each user: use all-but-last items as input, predict last item.
    Returns: dict with HR@k and NDCG@k
    """
    model.eval()
    model.to(device)
    hits, ndcgs     = [], []
    all_item_ids    = torch.arange(1, n_items + 1, device=device)

    with torch.no_grad():
        for uid, seq in test_data.items():
            if len(seq) < 2:
                continue
            true_item = seq[-1]
            input_seq = seq[:-1]
            # Pad/truncate input
            s          = input_seq[-max_seq:]
            pad_len    = max_seq - len(s)
            s          = [0] * pad_len + s
            seq_tensor = torch.LongTensor([s]).to(device)

            # Get hidden state and score ALL items
            h        = model(seq_tensor)[:, -1, :]  # (1, D)
            all_emb  = model.item_emb(all_item_ids) # (N, D)
            scores   = (all_emb * h).sum(-1)         # (N,)

            # Mask already-seen items
            for seen in input_seq:
                scores[seen - 1] = -1e9

            # Get top-K
            top_items = scores.topk(top_k).indices + 1  # 1-indexed
            top_items = top_items.cpu().tolist()

            hit = int(true_item in top_items)
            hits.append(hit)

            if hit:
                rank = top_items.index(true_item) + 1
                ndcgs.append(1.0 / np.log2(rank + 1))
            else:
                ndcgs.append(0.0)

    return {
        f'HR@{top_k}':   round(np.mean(hits),  4),
        f'NDCG@{top_k}': round(np.mean(ndcgs), 4),
    }
```

---

## 10. Baseline Models

You already have baselines. For completeness, here are the three to include in your comparison table.

### 10.1 Plain SASRec (No LLM)

```python
# CELL 11 — Plain SASRec baseline (random init, no LLM embeddings)
def build_baseline_sasrec(n_items, emb_dim=64):
    """SASRec with default random initialisation — no LLM embeddings."""
    model = SASRec(n_items=n_items, emb_dim=emb_dim)
    # No LLM init — uses PyTorch default random initialisation
    return model
```

### 10.2 GRU4Rec Baseline

```python
# CELL 12 — GRU4Rec baseline
import torch.nn as nn

class GRU4Rec(nn.Module):
    def __init__(self, n_items, emb_dim=64, hidden_dim=128, n_layers=1, dropout=0.2):
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, emb_dim, padding_idx=0)
        self.gru      = nn.GRU(emb_dim, hidden_dim, n_layers,
                               batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.fc       = nn.Linear(hidden_dim, emb_dim)
        self.dropout  = nn.Dropout(dropout)
        self.emb_dim  = emb_dim

    def forward(self, seq):
        x   = self.dropout(self.item_emb(seq))  # (B, L, D)
        out, _ = self.gru(x)                     # (B, L, H)
        return self.fc(out)                      # (B, L, D)

    def predict(self, seq, candidate_items):
        h     = self.forward(seq)[:, -1, :]
        c_emb = self.item_emb(candidate_items)
        return (c_emb * h.unsqueeze(1)).sum(-1)
```

### 10.3 BERT4Rec Baseline (Simplified)

```python
# CELL 13 — BERT4Rec baseline (bidirectional transformer, masked training)
class BERT4Rec(nn.Module):
    def __init__(self, n_items, emb_dim=64, n_heads=2, n_layers=2,
                 max_seq_len=50, dropout=0.2, mask_token_id=None):
        super().__init__()
        self.n_items     = n_items
        self.emb_dim     = emb_dim
        self.mask_token  = mask_token_id or (n_items + 1)

        self.item_emb = nn.Embedding(n_items + 2, emb_dim, padding_idx=0)
        self.pos_emb  = nn.Embedding(max_seq_len + 1, emb_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=n_heads,
            dim_feedforward=emb_dim * 4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.layer_norm  = nn.LayerNorm(emb_dim)
        self.dropout     = nn.Dropout(dropout)
        self.output      = nn.Linear(emb_dim, n_items + 1)

    def forward(self, seq):
        B, L = seq.shape
        positions = torch.arange(1, L + 1, device=seq.device).unsqueeze(0).expand(B, L)
        x = self.item_emb(seq) + self.pos_emb(positions)
        x = self.dropout(self.layer_norm(x))
        x = self.transformer(x)  # BERT4Rec uses FULL attention (not causal)
        return x

    def predict(self, seq, candidate_items):
        """At inference: mask the last item and score candidates."""
        masked_seq = seq.clone()
        masked_seq[:, -1] = self.mask_token
        h     = self.forward(masked_seq)[:, -1, :]
        c_emb = self.item_emb(candidate_items[:, :, None] if candidate_items.dim() == 2 else candidate_items)
        # Simple dot product scoring
        all_emb = self.item_emb.weight[1:self.n_items+1]  # (N, D)
        return (all_emb * h.unsqueeze(1)).sum(-1) if all_emb.dim() == 2 else None
```

> **Note:** For BERT4Rec evaluation, replace the last item in the input sequence with the mask token, then score all items against the masked position hidden state.

---

## 11. Master Run Loop — All 4 Styles + All Baselines

```python
# CELL 14 — MASTER LOOP: runs all 4 LLM styles + 3 baselines, collects results
import torch, pickle, json, os
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Running on: {device}')

splits     = pickle.load(open('data/ml-100k/splits.pkl', 'rb'))
n_items    = splits['n_items']
train_data = splits['train']
test_data  = splits['test']

os.makedirs('results', exist_ok=True)

STYLES      = ['P1_title_only', 'P2_title_genre', 'P3_user_centric', 'P4_hybrid']
all_results = {}
all_losses  = {}

# ── Part A: LLM-initialised SASRec (4 prompt styles) ─────────────────────────
for style in STYLES:
    print(f'\n{"="*60}')
    print(f' RUNNING LLM STYLE: {style}')
    print(f'{"="*60}')

    model = build_model_with_embeddings(style, emb_dim=64)
    print(f'Training SASRec with [{style}] embeddings...')
    losses = train_model(model, train_data, n_items, device, n_epochs=50)

    print('Evaluating...')
    m10 = evaluate(model, test_data, n_items, device, top_k=10)
    m20 = evaluate(model, test_data, n_items, device, top_k=20)

    all_results[style] = {**m10, **m20, 'final_loss': round(losses[-1], 4)}
    all_losses[style]  = losses

    for k, v in all_results[style].items():
        print(f'  {k}: {v}')

    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), f'checkpoints/{style}.pt')
    print(f'Checkpoint saved.')

# ── Part B: Baseline — SASRec (no LLM) ───────────────────────────────────────
print(f'\n{"="*60}')
print(' BASELINE: SASRec (no LLM embeddings)')
print(f'{"="*60}')
baseline_sasrec = build_baseline_sasrec(n_items, emb_dim=64)
losses_b = train_model(baseline_sasrec, train_data, n_items, device, n_epochs=50)
m10_b = evaluate(baseline_sasrec, test_data, n_items, device, top_k=10)
m20_b = evaluate(baseline_sasrec, test_data, n_items, device, top_k=20)
all_results['Baseline_SASRec']  = {**m10_b, **m20_b, 'final_loss': round(losses_b[-1], 4)}
all_losses['Baseline_SASRec']   = losses_b

# ── Part C: Baseline — GRU4Rec ────────────────────────────────────────────────
print(f'\n{"="*60}')
print(' BASELINE: GRU4Rec')
print(f'{"="*60}')
baseline_gru = GRU4Rec(n_items=n_items, emb_dim=64)
losses_g = train_model(baseline_gru, train_data, n_items, device, n_epochs=50)
m10_g = evaluate(baseline_gru, test_data, n_items, device, top_k=10)
m20_g = evaluate(baseline_gru, test_data, n_items, device, top_k=20)
all_results['Baseline_GRU4Rec'] = {**m10_g, **m20_g, 'final_loss': round(losses_g[-1], 4)}
all_losses['Baseline_GRU4Rec']  = losses_g

# ── Final Summary Table ───────────────────────────────────────────────────────
print('\n\n========== FINAL RESULTS TABLE ==========')
df_results = pd.DataFrame(all_results).T
print(df_results.to_string())

df_results.to_csv('results/final_results.csv')
json.dump(all_results, open('results/all_results.json', 'w'), indent=2)
json.dump(all_losses,  open('results/losses.json', 'w'))
print('\nResults saved to results/final_results.csv')
```

---

## 12. Plots for Your Mémoire

### Plot 1 — NDCG@10 Bar Chart (main results)

```python
# CELL 15 — Bar chart: NDCG@10 comparison across all styles + baselines
import matplotlib.pyplot as plt, json, os

os.makedirs('plots', exist_ok=True)
results = json.load(open('results/all_results.json'))

labels = {
    'Baseline_SASRec':    'Baseline\n(SASRec)',
    'Baseline_GRU4Rec':   'Baseline\n(GRU4Rec)',
    'P1_title_only':      'P1: Title\nOnly',
    'P2_title_genre':     'P2: Title +\nGenre',
    'P3_user_centric':    'P3: User-\nCentric',
    'P4_hybrid':          'P4: Hybrid\n(ours)',
}
order  = list(labels.keys())
values = [results[k]['NDCG@10'] for k in order]
names  = [labels[k] for k in order]
colors = ['#9CA3AF', '#6B7280', '#60A5FA', '#34D399', '#F59E0B', '#3B82F6']

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(names, values, color=colors, width=0.55, edgecolor='white', linewidth=1.5)

for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_ylabel('NDCG@10', fontsize=12)
ax.set_title('Impact of Item Description Prompt Style on NDCG@10\n(MovieLens 100K, SASRec)', fontsize=13)
ax.set_ylim(0, max(values) * 1.18)
ax.axhline(results['Baseline_SASRec']['NDCG@10'], color='red',
           linestyle='--', alpha=0.5, label='No-LLM SASRec baseline')
ax.legend(fontsize=10)
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig('plots/ndcg_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: plots/ndcg_comparison.png')
```

### Plot 2 — HR@10 vs NDCG@10 Scatter

```python
# CELL 16 — Scatter: HR@10 vs NDCG@10 per style
import matplotlib.pyplot as plt, json

results = json.load(open('results/all_results.json'))
style_labels = {
    'Baseline_SASRec':   ('Baseline (SASRec)',   '#9CA3AF'),
    'Baseline_GRU4Rec':  ('Baseline (GRU4Rec)',  '#6B7280'),
    'P1_title_only':     ('P1: Title only',       '#60A5FA'),
    'P2_title_genre':    ('P2: Title+Genre',      '#34D399'),
    'P3_user_centric':   ('P3: User-centric',     '#F59E0B'),
    'P4_hybrid':         ('P4: Hybrid (ours)',    '#3B82F6'),
}

fig, ax = plt.subplots(figsize=(8, 6))
for key, (label, color) in style_labels.items():
    hr   = results[key]['HR@10']
    ndcg = results[key]['NDCG@10']
    ax.scatter(hr, ndcg, color=color, s=180, zorder=5, label=label)
    ax.annotate(label, (hr, ndcg), textcoords='offset points', xytext=(8, 4), fontsize=9)

ax.set_xlabel('HR@10 (Hit Rate)', fontsize=12)
ax.set_ylabel('NDCG@10', fontsize=12)
ax.set_title('HR@10 vs NDCG@10 per Prompt Style\n(MovieLens 100K)', fontsize=13)
ax.legend(fontsize=9, loc='lower right')
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig('plots/scatter_hr_ndcg.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Plot 3 — Training Loss Curves

```python
# CELL 17 — Training loss curves per style
import matplotlib.pyplot as plt, json

all_losses = json.load(open('results/losses.json'))
colors = {
    'P1_title_only':    '#60A5FA',
    'P2_title_genre':   '#34D399',
    'P3_user_centric':  '#F59E0B',
    'P4_hybrid':        '#3B82F6',
    'Baseline_SASRec':  '#9CA3AF',
    'Baseline_GRU4Rec': '#6B7280',
}
fig, ax = plt.subplots(figsize=(10, 5))
for style, losses in all_losses.items():
    ax.plot(losses, label=style.replace('_',' '), color=colors.get(style,'#333'), linewidth=2)

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('BPR Loss', fontsize=12)
ax.set_title('Training Loss per Prompt Style (MovieLens 100K)', fontsize=13)
ax.legend(fontsize=9)
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig('plots/training_loss.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 13. Results Table Template

Fill in the `X.XXXX` cells with your actual numbers from `results/final_results.csv` after Day 3.

| Model / Prompt Style              | HR@10  | NDCG@10 | HR@20  | NDCG@20 | vs. SASRec Baseline |
|-----------------------------------|--------|---------|--------|---------|---------------------|
| Baseline: SASRec (no LLM)         | X.XXXX | X.XXXX  | X.XXXX | X.XXXX  | — (reference)       |
| Baseline: GRU4Rec                 | X.XXXX | X.XXXX  | X.XXXX | X.XXXX  | +X%                 |
| P1: Title only (paper method)     | X.XXXX | X.XXXX  | X.XXXX | X.XXXX  | +X%                 |
| P2: Title + Genre                 | X.XXXX | X.XXXX  | X.XXXX | X.XXXX  | +X%                 |
| P3: User-centric framing          | X.XXXX | X.XXXX  | X.XXXX | X.XXXX  | +X%                 |
| **P4: Hybrid (title+genre+tags)** | **X.XXXX** | **X.XXXX** | **X.XXXX** | **X.XXXX** | **+X%** |

> **Expected pattern:** P4 > P3 > P2 > P1 > Baseline_SASRec. All LLM-init models should beat both baselines.

---

## 14. Day-by-Day Execution Plan

| Day | Task | Target Duration |
|-----|------|-----------------|
| **Day 1** | Setup Kaggle, install deps, download ML-100K, run preprocessing (Cells 1–5), sanity check prompt examples | 3–4 hours |
| **Day 2** | Run Cell 6 four times (one per style) to generate all 4 `.npy` embedding files | 4–5 hours (mostly waiting) |
| **Day 3** | Run Cells 7–14 — define models, run master loop, get results CSV | 3–4 hours (mostly waiting) |
| **Day 4** | Run Cells 15–17 — generate 3 plots, fill results table | 2–3 hours |
| **Day 5–7** | Write mémoire chapter (8–12 pages) using Section 15 structure | 2–3 days |

### Day 1 Checklist
- [ ] Kaggle notebook created, GPU P100 enabled, internet on
- [ ] Cell 1 runs: GPU confirmed available
- [ ] Cell 2 runs: ml-100k folder exists with `u.data` and `u.item`
- [ ] Cell 3 runs: `data/ml-100k/item_meta.pkl` saved
- [ ] Cell 4 runs: `data/ml-100k/splits.pkl` saved
- [ ] Cell 5 runs: prompt examples print correctly for all 4 styles

### Day 2 Checklist
- [ ] `embeddings/P1_title_only.npy` saved (~6MB)
- [ ] `embeddings/P2_title_genre.npy` saved
- [ ] `embeddings/P3_user_centric.npy` saved
- [ ] `embeddings/P4_hybrid.npy` saved
- [ ] Each file shape is `(n_items+1, 1024)`

### Day 3 Checklist
- [ ] All 4 LLM-style models trained and evaluated
- [ ] GRU4Rec and SASRec baselines trained and evaluated
- [ ] `results/final_results.csv` saved with all numbers
- [ ] P4 hybrid shows highest NDCG@10 among LLM styles

---

## 15. Mémoire Chapter Structure

| Section | Title | Content |
|---------|-------|---------|
| 1 | Introduction | What the original paper does, what gap you exploit, why it matters |
| 2 | Methodology | The 4 prompt styles with examples; BGE-M3; SASRec architecture; PCA compression |
| 3 | Experimental Setup | Dataset stats (ML-100K), preprocessing, baselines, evaluation metrics |
| 4 | Results | Results table + 3 plots (bar chart, scatter, loss curves) |
| 5 | Discussion | Which style won and WHY (embedding geometry intuition); limitations |
| 6 | Conclusion | Practical recommendation for future work; one-sentence novelty claim |

**Total target: 8–12 pages for this chapter.**

---

## 16. Troubleshooting

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| CUDA out of memory during embedding | Batch size too large | Change `BATCH=256` to `BATCH=128` in Cell 6 |
| BGE-M3 download times out | Kaggle internet not enabled | Enable internet in notebook settings |
| `u.item` parse error | Encoding issue | Add `encoding='latin-1'` to `pd.read_csv` |
| NDCG stays at 0.0 after training | Learning rate too high/low | Try `lr=1e-4` or `lr=5e-3` |
| P4 does not beat P1 | Needs more epochs | Change `n_epochs=50` to `n_epochs=100` |
| Kaggle session times out mid-training | 9-hour limit | Save checkpoint after each style, resume from `.pt` file |
| Empty genres for some items | Normal — some movies have no genres | Fallback `'Movie'` tag handles this gracefully |

---

## 17. Novelty Statement for Your Teacher

> *"The paper Boz et al. (2025) uses only item product names as input to the LLM embedding model (Section 5.2). It never tests whether richer descriptions produce better embeddings. Our contribution is the first systematic comparison of prompt styles for LLM-enhanced sequential recommendation on MovieLens 100K, showing that a hybrid description (title + genre + user-preference framing) consistently outperforms the title-only default across all evaluation metrics, while remaining fully free to run using BGE-M3 on public GPU infrastructure."*

---

## 18. File System Expected After All Runs

```
project/
├── data/ml-100k/
│   ├── u.data
│   ├── u.item
│   ├── item_meta.pkl        ← movie titles + genres
│   └── splits.pkl           ← train/val/test + id mappings
├── embeddings/
│   ├── P1_title_only.npy    ← shape (n_items+1, 1024)
│   ├── P2_title_genre.npy
│   ├── P3_user_centric.npy
│   └── P4_hybrid.npy
├── checkpoints/
│   ├── P1_title_only.pt
│   ├── P2_title_genre.pt
│   ├── P3_user_centric.pt
│   └── P4_hybrid.pt
├── results/
│   ├── final_results.csv    ← your results table
│   ├── all_results.json
│   └── losses.json
└── plots/
    ├── ndcg_comparison.png  ← Plot 1 (bar chart)
    ├── scatter_hr_ndcg.png  ← Plot 2 (scatter)
    └── training_loss.png    ← Plot 3 (loss curves)
```

---

*--- Give this entire document to your AI coding agent. It contains everything needed. ---*
