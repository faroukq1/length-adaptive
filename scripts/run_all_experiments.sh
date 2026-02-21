#!/bin/bash

# Run all experiments for model comparison
# This will train each model for 50 epochs and save results

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the project root (parent of scripts directory)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to project root
cd "$PROJECT_ROOT"

echo "========================================"
echo "RUNNING ALL EXPERIMENTS"
echo "========================================"
echo ""
echo "Working directory: $PROJECT_ROOT"
echo ""

# Create results directory
mkdir -p results

# Common parameters
EPOCHS=50
BATCH_SIZE=256
LR=0.001
MAX_LEN=50
D_MODEL=64
N_HEADS=2
N_BLOCKS=2

# 1. SASRec baseline
echo "[1/5] Training SASRec..."
python3 -m experiments.run_experiment \
    --model sasrec \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --d_model $D_MODEL \
    --n_heads $N_HEADS \
    --n_blocks $N_BLOCKS \
    --max_len $MAX_LEN

echo ""
echo "✓ SASRec complete"
echo ""

# 2. Hybrid with Fixed fusion
echo "[2/5] Training Hybrid (Fixed α=0.5)..."
python3 -m experiments.run_experiment \
    --model hybrid_fixed \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --d_model $D_MODEL \
    --n_heads $N_HEADS \
    --n_blocks $N_BLOCKS \
    --max_len $MAX_LEN \
    --fixed_alpha 0.5

echo ""
echo "✓ Hybrid (Fixed) complete"
echo ""

# 3. Hybrid with Discrete fusion
echo "[3/5] Training Hybrid (Discrete Bins)..."
python3 -m experiments.run_experiment \
    --model hybrid_discrete \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --d_model $D_MODEL \
    --n_heads $N_HEADS \
    --n_blocks $N_BLOCKS \
    --max_len $MAX_LEN \
    --L_short 10 \
    --L_long 50

echo ""
echo "✓ Hybrid (Discrete) complete"
echo ""

# 4. Hybrid with Learnable fusion
echo "[4/5] Training Hybrid (Learnable)..."
python3 -m experiments.run_experiment \
    --model hybrid_learnable \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --d_model $D_MODEL \
    --n_heads $N_HEADS \
    --n_blocks $N_BLOCKS \
    --max_len $MAX_LEN

echo ""
echo "✓ Hybrid (Learnable) complete"
echo ""

# 5. Hybrid with Continuous fusion
echo "[5/5] Training Hybrid (Continuous)..."
python3 -m experiments.run_experiment \
    --model hybrid_continuous \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --d_model $D_MODEL \
    --n_heads $N_HEADS \
    --n_blocks $N_BLOCKS \
    --max_len $MAX_LEN

echo ""
echo "✓ Hybrid (Continuous) complete"
echo ""

# Print summary
echo "========================================"
echo "ALL EXPERIMENTS COMPLETE!"
echo "========================================"
echo ""
echo "Results saved in: results/"
echo ""
echo "To analyze results, run:"
echo "  python3 -m experiments.analyze_results"
echo ""
