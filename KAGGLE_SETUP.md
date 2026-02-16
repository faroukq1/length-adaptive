# Kaggle Quick Start Guide

## Option 1: Using the Jupyter Notebook

1. Upload `kaggle_notebook.ipynb` to Kaggle
2. Run all cells sequentially
3. The notebook will:
   - Install dependencies
   - Download MovieLens-1M
   - Create all model files
   - Run preprocessing
   - Test all models

## Option 2: Using Shell Script with Git Clone

If your code is on GitHub, run these commands in a Kaggle notebook cell:

```bash
# Clone your repository
!git clone https://github.com/YOUR_USERNAME/length-adaptive.git
%cd length-adaptive

# Run the setup script
!chmod +x setup_kaggle.sh
!./setup_kaggle.sh
```

## Option 3: Upload Files Directly to Kaggle

1. Create a new Kaggle notebook
2. Upload all files from your local `length-adaptive` directory as a dataset
3. Add the dataset to your notebook
4. Run:

```python
# Copy files from dataset to working directory
!cp -r /kaggle/input/your-dataset-name/* /kaggle/working/
%cd /kaggle/working

# Run preprocessing
!python src/data/preprocess.py

# Build graph
!python src/data/graph_builder.py

# Test models
!python test_models.py
```

## Option 4: Run Everything in One Cell

If you don't want to upload files, you can run this single cell in Kaggle:

```python
# Download the complete project
!git clone https://github.com/YOUR_USERNAME/length-adaptive.git
%cd length-adaptive

# Install dependencies
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
!pip install -q torch-geometric pandas scikit-learn tqdm

# Download dataset
!wget https://files.grouplens.org/datasets/movielens/ml-1m.zip -O data/ml-1m/raw/ml-1m.zip
!unzip -q data/ml-1m/raw/ml-1m.zip -d data/ml-1m/raw/
!rm data/ml-1m/raw/ml-1m.zip

# Run preprocessing
!python src/data/preprocess.py

# Build graph
!python src/data/graph_builder.py

# Test all models
!python test_models.py
```

## What Gets Executed

After setup, you'll have:

- ✅ MovieLens-1M dataset preprocessed (6,034 users, 3,533 items)
- ✅ Co-occurrence graph built (74,170 edges)
- ✅ All models tested:
  - SASRec baseline
  - LightGCN GNN
  - Hybrid models (Fixed, Discrete, Learnable, Continuous fusion)

## Next Steps

After setup, you can:

1. Train the models (implement training loop)
2. Evaluate on test set
3. Compare performance across fusion strategies
4. Analyze results by user history length

## Minimal Example for Kaggle

Create a new Kaggle notebook and paste this:

```python
%%bash
# Setup
mkdir -p data/ml-1m/raw data/ml-1m/processed data/graphs src/data src/models

# Download dataset
wget -q https://files.grouplens.org/datasets/movielens/ml-1m.zip -O data/ml-1m/raw/ml-1m.zip
unzip -q data/ml-1m/raw/ml-1m.zip -d data/ml-1m/raw/
rm data/ml-1m/raw/ml-1m.zip
```

Then upload your Python files (`src/` directory) or copy-paste them into cells.
