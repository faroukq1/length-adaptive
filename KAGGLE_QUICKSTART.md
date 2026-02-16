# Running on Kaggle - Quick Start

## Method 1: Simplest - Upload as Kaggle Dataset

1. **Compress your project folder:**

   ```bash
   cd /home/farouk/files
   tar -czf length-adaptive.tar.gz length-adaptive/
   ```

2. **Create a Kaggle Dataset:**
   - Go to https://www.kaggle.com/datasets
   - Click "New Dataset"
   - Upload `length-adaptive.tar.gz`
   - Name it "length-adaptive-recommender"

3. **Create a new Kaggle Notebook:**
   - Add your dataset to the notebook
   - Run these commands:

```python
# Extract and setup
!tar -xzf /kaggle/input/length-adaptive-recommender/length-adaptive.tar.gz
%cd length-adaptive

# Install PyTorch (CPU)
!pip install -q torch --index-url https://download.pytorch.org/whl/cpu
!pip install -q torch-geometric

# Download MovieLens-1M (if not already present)
!wget -q https://files.grouplens.org/datasets/movielens/ml-1m.zip -O data/ml-1m/raw/ml-1m.zip
!unzip -q data/ml-1m/raw/ml-1m.zip -d data/ml-1m/raw/
!rm data/ml-1m/raw/ml-1m.zip

# Run preprocessing
!python src/data/preprocess.py

# Build graph
!python src/data/graph_builder.py

# Test all models
!python test_models.py
```

## Method 2: GitHub Clone

1. **Push your code to GitHub:**

   ```bash
   cd /home/farouk/files/length-adaptive
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git push -u origin main
   ```

2. **In Kaggle notebook, run:**

   ```python
   !git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
   %cd YOUR_REPO

   # Install dependencies
   !pip install -q torch --index-url https://download.pytorch.org/whl/cpu
   !pip install -q torch-geometric pandas scikit-learn tqdm

   # Download data
   !bash setup_kaggle.sh
   ```

## Method 3: Copy-Paste Files Directly

Create a new Kaggle notebook and run each cell below:

### Cell 1: Install Dependencies

```python
!pip install -q torch --index-url https://download.pytorch.org/whl/cpu
!pip install -q torch-geometric pandas scikit-learn tqdm
```

### Cell 2: Setup Directories

```python
import os
os.makedirs('data/ml-1m/raw', exist_ok=True)
os.makedirs('data/ml-1m/processed', exist_ok=True)
os.makedirs('data/graphs', exist_ok=True)
os.makedirs('src/data', exist_ok=True)
os.makedirs('src/models', exist_ok=True)
```

### Cell 3: Download Dataset

```python
!wget https://files.grouplens.org/datasets/movielens/ml-1m.zip -O data/ml-1m/raw/ml-1m.zip
!unzip -q data/ml-1m/raw/ml-1m.zip -d data/ml-1m/raw/
!rm data/ml-1m/raw/ml-1m.zip
```

### Cell 4-7: Copy Model Files

Then copy the contents of these files into separate cells:

- `src/data/preprocess.py` → Save as Cell 4 with `%%writefile src/data/preprocess.py`
- `src/data/graph_builder.py` → Save as Cell 5 with `%%writefile src/data/graph_builder.py`
- `src/models/sasrec.py` → Save as Cell 6 (continue for all model files)
- `test_models.py` → Save as final cell

Example:

```python
%%writefile src/data/preprocess.py
import pandas as pd
import numpy as np
# ... (paste entire file content)
```

### Cell 8: Run Pipeline

```python
# Preprocess
!python src/data/preprocess.py

# Build graph
!python src/data/graph_builder.py

# Test models
!python test_models.py
```

## What You Should See

After running, you'll see:

```
============================================================
TESTING PHASE 2: BASELINE MODELS
============================================================

[Setup] Loading data and graph...
✓ Data loaded: 3533 items
✓ Graph loaded: 151874 edges
...

✅ ALL PHASE 2 TESTS PASSED SUCCESSFULLY!
```

## Files You Need on Kaggle

Minimum files required:

```
src/data/
  - preprocess.py
  - graph_builder.py
src/models/
  - sasrec.py
  - lightgcn.py
  - fusion.py
  - hybrid.py
test_models.py
```

## Quick Command Reference

```bash
# On your local machine - create archive
cd /home/farouk/files
tar -czf length-adaptive.tar.gz length-adaptive/

# On Kaggle - extract and run
!tar -xzf /kaggle/input/YOUR-DATASET/length-adaptive.tar.gz
%cd length-adaptive
!bash setup_kaggle.sh
```

That's it! Your models will be ready to train on Kaggle.
