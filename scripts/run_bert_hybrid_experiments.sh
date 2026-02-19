#!/bin/bash

# Run BERT4Rec + GNN Hybrid Experiments
# This script compares BERT4Rec baseline with all 4 BERT+GNN hybrid variations
# Expected time: ~6-8 hours (GPU) or ~30-40 hours (CPU)

echo "========================================"
echo "BERT4Rec + GNN HYBRID EXPERIMENTS"
echo "========================================"
echo ""
echo "This compares BERT4Rec (bidirectional) with BERT+GNN hybrids"
echo "to properly evaluate if GNN improves bidirectional transformers"
echo ""
echo "Training with 200 epochs, early stopping patience=20"
echo "Expected to converge at epoch 30-70"
echo ""
echo "Models:"
echo "  1. BERT4Rec (baseline)"
echo "  2. BERT4Rec + GNN (Fixed fusion)"
echo "  3. BERT4Rec + GNN (Discrete fusion)"
echo "  4. BERT4Rec + GNN (Learnable fusion)"
echo "  5. BERT4Rec + GNN (Continuous fusion)"
echo ""
echo "⚠️  Time estimate:"
echo "   GPU: ~6-8 hours total"
echo "   CPU: ~30-40 hours total"
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
D_FF=256

# ============================================
# BERT4Rec BASELINE (for comparison)
# ============================================

echo ""
echo "[1/5] Training BERT4Rec (baseline)..."
echo "--------------------------------------"
python experiments/run_experiment.py \
    --model bert4rec \
    --epochs $EPOCHS \
    --patience $PATIENCE \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --d_model $D_MODEL \
    --n_heads $N_HEADS \
    --n_blocks $N_BLOCKS \
    --d_ff $D_FF \
    --max_len $MAX_LEN

echo ""
echo "✓ BERT4Rec complete"
echo ""

# ============================================
# BERT4Rec + GNN HYBRID MODELS
# ============================================

# 2. Fixed Fusion
echo "[2/5] Training BERT4Rec + GNN (Fixed α=0.5)..."
echo "--------------------------------------"
python experiments/run_experiment.py \
    --model bert_hybrid_fixed \
    --epochs $EPOCHS \
    --patience $PATIENCE \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --d_model $D_MODEL \
    --n_heads $N_HEADS \
    --n_blocks $N_BLOCKS \
    --d_ff $D_FF \
    --gnn_layers $GNN_LAYERS \
    --max_len $MAX_LEN \
    --fixed_alpha 0.5

echo ""
echo "✓ BERT Hybrid Fixed complete"
echo ""

# 3. Discrete Fusion
echo "[3/5] Training BERT4Rec + GNN (Discrete bins)..."
echo "--------------------------------------"
python experiments/run_experiment.py \
    --model bert_hybrid_discrete \
    --epochs $EPOCHS \
    --patience $PATIENCE \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --d_model $D_MODEL \
    --n_heads $N_HEADS \
    --n_blocks $N_BLOCKS \
    --d_ff $D_FF \
    --gnn_layers $GNN_LAYERS \
    --max_len $MAX_LEN \
    --L_short 10 \
    --L_long 50

echo ""
echo "✓ BERT Hybrid Discrete complete"
echo ""

# 4. Learnable Fusion
echo "[4/5] Training BERT4Rec + GNN (Learnable bins)..."
echo "--------------------------------------"
python experiments/run_experiment.py \
    --model bert_hybrid_learnable \
    --epochs $EPOCHS \
    --patience $PATIENCE \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --d_model $D_MODEL \
    --n_heads $N_HEADS \
    --n_blocks $N_BLOCKS \
    --d_ff $D_FF \
    --gnn_layers $GNN_LAYERS \
    --max_len $MAX_LEN \
    --L_short 10 \
    --L_long 50

echo ""
echo "✓ BERT Hybrid Learnable complete"
echo ""

# 5. Continuous Fusion
echo "[5/5] Training BERT4Rec + GNN (Continuous function)..."
echo "--------------------------------------"
python experiments/run_experiment.py \
    --model bert_hybrid_continuous \
    --epochs $EPOCHS \
    --patience $PATIENCE \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --d_model $D_MODEL \
    --n_heads $N_HEADS \
    --n_blocks $N_BLOCKS \
    --d_ff $D_FF \
    --gnn_layers $GNN_LAYERS \
    --max_len $MAX_LEN

echo ""
echo "✓ BERT Hybrid Continuous complete"
echo ""

# ============================================
# COMPLETION
# ============================================

echo ""
echo "========================================"
echo "ALL EXPERIMENTS COMPLETE!"
echo "========================================"
echo ""
echo "Results saved in: results/"
echo ""
echo "To analyze results, run:"
echo "  python experiments/analyze_results.py"
echo ""
echo "To create visualizations, run:"
echo "  bash experiments/run_all_analysis.sh"
echo ""
