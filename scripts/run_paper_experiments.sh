#!/bin/bash

# Run all experiments with PAPER-LEVEL SETTINGS
# This trains each model for 200 epochs with early stopping (patience=20)
# Expected time: ~3-4 hours (GPU) or ~20-30 hours (CPU)

echo "========================================"
echo "PAPER-LEVEL EXPERIMENTS"
echo "========================================"
echo ""
echo "Training with 200 epochs, early stopping patience=20"
echo "Expected to converge at epoch 30-50"
echo ""
echo "⚠️  Time estimate:"
echo "   GPU: ~3-4 hours total"
echo "   CPU: ~20-30 hours total"
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

# 1. SASRec baseline
echo ""
echo "[1/5] Training SASRec (paper settings)..."
python experiments/run_experiment.py \
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

# 2. Hybrid with Fixed fusion
echo "[2/5] Training Hybrid (Fixed α=0.5)..."
python experiments/run_experiment.py \
    --model hybrid_fixed \
    --epochs $EPOCHS \
    --patience $PATIENCE \
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
python experiments/run_experiment.py \
    --model hybrid_discrete \
    --epochs $EPOCHS \
    --patience $PATIENCE \
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
python experiments/run_experiment.py \
    --model hybrid_learnable \
    --epochs $EPOCHS \
    --patience $PATIENCE \
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
python experiments/run_experiment.py \
    --model hybrid_continuous \
    --epochs $EPOCHS \
    --patience $PATIENCE \
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
echo "  python experiments/analyze_results.py"
echo ""
