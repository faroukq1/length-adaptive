#!/bin/bash

# Run all baseline experiments with PAPER-LEVEL SETTINGS
# This trains each baseline model for 200 epochs with early stopping (patience=20)
# Expected time: ~5-6 hours (GPU) or ~25-30 hours (CPU)

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the project root (parent of scripts directory)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to project root
cd "$PROJECT_ROOT"

echo "========================================"
echo "PAPER-LEVEL EXPERIMENTS - BASELINES"
echo "========================================"
echo ""
echo "Working directory: $PROJECT_ROOT"
echo ""
echo "Training with 200 epochs, early stopping patience=20"
echo "Expected to converge at epoch 30-50"
echo ""
echo "Models:"
echo "  Baselines: SASRec, BERT4Rec, GRU4Rec, LightGCN, Caser"
echo ""
echo "⚠️  Time estimate:"
echo "   GPU: ~5-6 hours total"
echo "   CPU: ~25-30 hours total"
echo ""
echo "Starting experiments..."
echo ""

# Create results directory
mkdir -p results

# Paper-level parameters
EPOCHS=200
PATIENCE=20
BATCH_SIZE=256
LR=0.001
MAX_LEN=50
D_MODEL=64
N_HEADS=2
N_BLOCKS=2
GNN_LAYERS=2

# ============================================
# BASELINE MODELS
# ============================================

# 1. SASRec baseline
echo ""
echo "[1/5] Training SASRec (Transformer baseline)..."
python -m experiments.run_experiment \
    --model sasrec \
    --epochs $EPOCHS \
    --patience $PATIENCE \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --d_model $D_MODEL \
    --n_heads $N_HEADS \
    --n_blocks $N_BLOCKS \
    --max_len $MAX_LEN

echo ""
echo "✓ SASRec complete"
echo ""

# 2. BERT4Rec baseline
echo "[2/5] Training BERT4Rec (Bidirectional Transformer)..."
python -m experiments.run_experiment \
    --model bert4rec \
    --epochs $EPOCHS \
    --patience $PATIENCE \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --d_model $D_MODEL \
    --n_heads $N_HEADS \
    --n_blocks $N_BLOCKS \
    --max_len $MAX_LEN

echo ""
echo "✓ BERT4Rec complete"
echo ""

# 3. GRU4Rec baseline
echo "[3/5] Training GRU4Rec (RNN baseline)..."
python -m experiments.run_experiment \
    --model gru4rec \
    --epochs $EPOCHS \
    --patience $PATIENCE \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --d_model $D_MODEL \
    --n_blocks $N_BLOCKS \
    --max_len $MAX_LEN

echo ""
echo "✓ GRU4Rec complete"
echo ""

# 4. LightGCN baseline
echo "[4/5] Training LightGCN (GNN baseline)..."
python -m experiments.run_experiment \
    --model lightgcn \
    --epochs $EPOCHS \
    --patience $PATIENCE \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --d_model $D_MODEL \
    --gnn_layers $GNN_LAYERS \
    --max_len $MAX_LEN

echo ""
echo "✓ LightGCN complete"
echo ""

# 5. Caser baseline
echo "[5/5] Training Caser (CNN baseline)..."
python -m experiments.run_experiment \
    --model caser \
    --epochs $EPOCHS \
    --patience $PATIENCE \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --d_model $D_MODEL \
    --max_len $MAX_LEN

echo ""
echo "✓ Caser complete"
echo ""

# Print summary
echo "========================================"
echo "ALL BASELINE EXPERIMENTS COMPLETE!"
echo "========================================"
echo ""
echo "Results saved in: results/"
echo ""
echo "Models trained:"
echo "  ✓ SASRec (Transformer baseline)"
echo "  ✓ BERT4Rec (Bidirectional Transformer)"
echo "  ✓ GRU4Rec (RNN baseline)"
echo "  ✓ LightGCN (GNN baseline)"
echo "  ✓ Caser (CNN baseline)"
echo ""
echo "To analyze results, run:"
echo "  python -m experiments.analyze_results"
echo ""
