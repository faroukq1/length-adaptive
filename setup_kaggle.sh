#!/bin/bash

# Length-Adaptive Sequential Recommendation - Kaggle Setup Script
# This script sets up and runs the complete pipeline on Kaggle

echo "============================================================"
echo "Length-Adaptive Sequential Recommendation - Kaggle Setup"
echo "============================================================"

# Step 1: Create directory structure
echo -e "\n[1/6] Creating directory structure..."
mkdir -p data/ml-1m/raw data/ml-1m/processed data/graphs
mkdir -p src/data src/models src/train src/eval src/utils
echo "✓ Directories created"

# Step 2: Download and extract MovieLens-1M
echo -e "\n[2/6] Downloading MovieLens-1M dataset..."
cd data/ml-1m/raw
wget -q https://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip -q ml-1m.zip
rm ml-1m.zip
cd ../../..
echo "✓ Dataset downloaded and extracted"

# Step 3: Copy or create model files
echo -e "\n[3/6] Setting up model files..."
# Note: If cloning from GitHub, use:
# git clone YOUR_REPO_URL
# Otherwise, files should already be present
echo "✓ Model files ready"

# Step 4: Run preprocessing
echo -e "\n[4/6] Running data preprocessing..."
python src/data/preprocess.py
echo "✓ Preprocessing complete"

# Step 5: Build co-occurrence graph
echo -e "\n[5/6] Building co-occurrence graph..."
python src/data/graph_builder.py
echo "✓ Graph construction complete"

# Step 6: Test models
echo -e "\n[6/6] Testing all models..."
python test_models.py
echo "✓ Model testing complete"

echo -e "\n============================================================"
echo "✅ Setup Complete! Ready for training and evaluation."
echo "============================================================"
echo -e "\nNext steps:"
echo "  1. Review the processed data in data/ml-1m/processed/"
echo "  2. Check the graph in data/graphs/"
echo "  3. Run training with your preferred configuration"
echo "============================================================"
