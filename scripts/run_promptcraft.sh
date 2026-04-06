#!/bin/bash

# Run PromptCraft experiments on MovieLens-100K with project-consistent settings.
# It trains:
#   1) SASRec baseline (random init)
#   2) SASRec + PromptCraft P1/P2/P3/P4 LLM initialization

set -e

# Resolve project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

echo "========================================"
echo "PROMPTCRAFT EXPERIMENTS - MOVIELENS-100K"
echo "========================================"
echo ""
echo "Working directory: $PROJECT_ROOT"
echo ""

# Configurable defaults via environment variables
EPOCHS=${EPOCHS:-200}
PATIENCE=${PATIENCE:-20}
BATCH_SIZE=${BATCH_SIZE:-256}
LR=${LR:-0.001}
MAX_LEN=${MAX_LEN:-50}
D_MODEL=${D_MODEL:-64}
N_HEADS=${N_HEADS:-2}
N_BLOCKS=${N_BLOCKS:-2}
D_FF=${D_FF:-256}
DROPOUT=${DROPOUT:-0.2}
EMB_BATCH_SIZE=${EMB_BATCH_SIZE:-256}
EVAL_EVERY=${EVAL_EVERY:-5}

echo "Training config:"
echo "  epochs=$EPOCHS, patience=$PATIENCE, batch_size=$BATCH_SIZE"
echo "  lr=$LR, d_model=$D_MODEL, n_heads=$N_HEADS, n_blocks=$N_BLOCKS"
echo "  max_len=$MAX_LEN, dropout=$DROPOUT"
echo "  embedding_batch_size=$EMB_BATCH_SIZE"
echo "  eval_every=$EVAL_EVERY (prints once every 5 epochs)"
echo ""
echo "Styles: baseline + P1_title_only + P2_title_genre + P3_user_centric + P4_hybrid"
echo ""

python3 -m experiments.run_promptcraft \
    --styles P1_title_only P2_title_genre P3_user_centric P4_hybrid \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --max_len "$MAX_LEN" \
    --d_model "$D_MODEL" \
    --n_heads "$N_HEADS" \
    --n_blocks "$N_BLOCKS" \
    --d_ff "$D_FF" \
    --dropout "$DROPOUT" \
    --embedding_batch_size "$EMB_BATCH_SIZE" \
    --eval_every "$EVAL_EVERY" \
    --quiet

echo ""
echo "========================================"
echo "PROMPTCRAFT RUN COMPLETE"
echo "========================================"
echo "Results are in:"
echo "  results/promptcraft_*"
echo "  results/promptcraft/summary_*.csv"
echo ""
