#!/bin/bash

# Run all experiments with PAPER-LEVEL SETTINGS
# This trains each baseline and hybrid model for 200 epochs with early stopping (patience=20)
# Expected time: ~8-10 hours (GPU) or ~40-60 hours (CPU)

echo "========================================"
echo "PAPER-LEVEL EXPERIMENTS - ALL MODELS"
echo "========================================"
echo ""
echo "Training with 200 epochs, early stopping patience=20"
echo "Expected to converge at epoch 30-50"
echo ""
echo "Models:"
echo "  Baselines: SASRec, BERT4Rec, GRU4Rec, LightGCN"
echo "  Hybrid: Fixed, Discrete, Learnable, Continuous"
echo ""
echo "⚠️  Time estimate:"
echo "   GPU: ~8-10 hours total"
echo "   CPU: ~40-60 hours total"
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
echo "[1/8] Training SASRec (Transformer baseline)..."
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

# 2. BERT4Rec baseline
echo "[2/8] Training BERT4Rec (Bidirectional Transformer)..."
python experiments/run_experiment.py \
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
echo "[3/8] Training GRU4Rec (RNN baseline)..."
python experiments/run_experiment.py \
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
echo "[4/8] Training LightGCN (GNN baseline)..."
python experiments/run_experiment.py \
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

# ============================================
# HYBRID MODELS
# ============================================

# 5. Hybrid with Fixed fusion
echo "[5/8] Training Hybrid (Fixed α=0.5)..."
python experiments/run_experiment.py \
    --model hybrid_fixed \
    --epochs $EPOCHS \
    --patience $PATIENCE \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --d_model $D_MODEL \
    --n_heads $N_HEADS \
    --n_blocks $N_BLOCKS \
    --gnn_layers $GNN_LAYERS \
    --max_len $MAX_LEN \
    --fixed_alpha 0.5

echo ""
echo "✓ Hybrid (Fixed) complete"
echo ""

# 6. Hybrid with Discrete fusion
echo "[6/8] Training Hybrid (Discrete Bins)..."
python experiments/run_experiment.py \
    --model hybrid_discrete \
    --epochs $EPOCHS \
    --patience $PATIENCE \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --d_model $D_MODEL \
    --n_heads $N_HEADS \
    --n_blocks $N_BLOCKS \
    --gnn_layers $GNN_LAYERS \
    --max_len $MAX_LEN \
    --L_short 10 \
    --L_long 50

echo ""
echo "✓ Hybrid (Discrete) complete"
echo ""

# 7. Hybrid with Learnable fusion
echo "[7/8] Training Hybrid (Learnable)..."
python experiments/run_experiment.py \
    --model hybrid_learnable \
    --epochs $EPOCHS \
    --patience $PATIENCE \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --d_model $D_MODEL \
    --n_heads $N_HEADS \
    --n_blocks $N_BLOCKS \
    --gnn_layers $GNN_LAYERS \
    --max_len $MAX_LEN

echo ""
echo "✓ Hybrid (Learnable) complete"
echo ""

# 8. Hybrid with Continuous fusion
echo "[8/8] Training Hybrid (Continuous)..."
python experiments/run_experiment.py \
    --model hybrid_continuous \
    --epochs $EPOCHS \
    --patience $PATIENCE \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --d_model $D_MODEL \
    --n_heads $N_HEADS \
    --n_blocks $N_BLOCKS \
    --gnn_layers $GNN_LAYERS \
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
echo "Models trained:"
echo "  ✓ SASRec (Transformer baseline)"
echo "  ✓ BERT4Rec (Bidirectional Transformer)"
echo "  ✓ GRU4Rec (RNN baseline)"
echo "  ✓ LightGCN (GNN baseline)"
echo "  ✓ Hybrid Fixed (α=0.5)"
echo "  ✓ Hybrid Discrete (bins)"
echo "  ✓ Hybrid Learnable (MLP)"
echo "  ✓ Hybrid Continuous (sigmoid)"
echo ""
echo "To analyze results, run:"
echo "  python experiments/analyze_results.py"
echo ""
