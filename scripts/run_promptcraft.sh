#!/bin/bash

# Run PromptCraft experiments on MovieLens-100K with stronger defaults.
# Default run trains BERT4Rec + PromptCraft P4_hybrid and includes baseline.
# Behavior can be changed via environment variables below.

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
MODEL=${MODEL:-bert4rec}
LOSS=${LOSS:-bpr}
STYLES=${STYLES:-P4_hybrid}
RUN_BASELINE=${RUN_BASELINE:-1}

EPOCHS=${EPOCHS:-300}
PATIENCE=${PATIENCE:-35}
BATCH_SIZE=${BATCH_SIZE:-256}
LR=${LR:-0.0005}
MAX_LEN=${MAX_LEN:-200}
D_MODEL=${D_MODEL:-128}
N_HEADS=${N_HEADS:-4}
N_BLOCKS=${N_BLOCKS:-3}
D_FF=${D_FF:-512}
DROPOUT=${DROPOUT:-0.15}
EMB_BATCH_SIZE=${EMB_BATCH_SIZE:-256}
EVAL_EVERY=${EVAL_EVERY:-5}
NUM_NEG_SAMPLES=${NUM_NEG_SAMPLES:-5}
HARD_NEG_RATIO=${HARD_NEG_RATIO:-0.4}
INIT_MIX_ALPHA=${INIT_MIX_ALPHA:-0.8}
INIT_SCALE=${INIT_SCALE:-1.0}
GRAD_CLIP_NORM=${GRAD_CLIP_NORM:-1.0}

echo "Training config:"
echo "  model=$MODEL, loss=$LOSS"
echo "  styles=$STYLES"
echo "  epochs=$EPOCHS, patience=$PATIENCE, batch_size=$BATCH_SIZE"
echo "  lr=$LR, d_model=$D_MODEL, n_heads=$N_HEADS, n_blocks=$N_BLOCKS"
echo "  max_len=$MAX_LEN, dropout=$DROPOUT, grad_clip_norm=$GRAD_CLIP_NORM"
echo "  num_neg_samples=$NUM_NEG_SAMPLES, hard_neg_ratio=$HARD_NEG_RATIO"
echo "  init_mix_alpha=$INIT_MIX_ALPHA, init_scale=$INIT_SCALE"
echo "  embedding_batch_size=$EMB_BATCH_SIZE"
echo "  eval_every=$EVAL_EVERY (prints once every 5 epochs)"
echo ""
if [ "$RUN_BASELINE" = "1" ]; then
    echo "Baseline: enabled"
    BASELINE_FLAGS=()
else
    echo "Baseline: skipped"
    BASELINE_FLAGS=(--skip_baseline)
fi

read -r -a STYLE_ARGS <<< "$STYLES"

echo "Running styles: ${STYLE_ARGS[*]}"
echo ""

python3 -m experiments.run_promptcraft \
        --model "$MODEL" \
        --loss "$LOSS" \
        --styles "${STYLE_ARGS[@]}" \
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
    --num_neg_samples "$NUM_NEG_SAMPLES" \
    --hard_neg_ratio "$HARD_NEG_RATIO" \
    --init_mix_alpha "$INIT_MIX_ALPHA" \
    --init_scale "$INIT_SCALE" \
    --embedding_batch_size "$EMB_BATCH_SIZE" \
    --eval_every "$EVAL_EVERY" \
    --grad_clip_norm "$GRAD_CLIP_NORM" \
    "${BASELINE_FLAGS[@]}" \
    --quiet

echo ""
echo "========================================"
echo "PROMPTCRAFT RUN COMPLETE"
echo "========================================"
echo "Results are in:"
echo "  results/promptcraft_*"
echo "  results/promptcraft/summary_*.csv"
echo ""
