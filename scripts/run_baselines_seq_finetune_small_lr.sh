#!/bin/bash

# --------------------------------------------------------------
# Example Running Command:
# bash scripts/run_baselines_seq_finetune_small_lr.sh [GPU_ID] [SEEDS] [DATASET] [EXTRA_NOTE]
# --------------------------------------------------------------

source "$(dirname "$0")/common_baselines.sh"

date
ulimit -n 65536
export MASTER_PORT=$(($RANDOM+32769))
export WORLD_SIZE=1

GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID

SEQ_FINETUNE_SMALL_LR=${SEQ_FINETUNE_SMALL_LR:-"0.005"}

echo "Using GPU: $GPU_ID"
echo "Running experiments on dataset: $DATASET with seeds: $SEEDS"
echo "========================================="
echo "Starting SeqFinetuneSmallLR Baseline Experiment"
echo "Dataset: $DATASET"
echo "Seeds: $SEEDS"
echo "Si-Blurry Setting: m=$N%, n=$M%"
echo "Tasks: $N_TASKS"
echo "Head LR: $SEQ_FINETUNE_SMALL_LR"
echo "Backbone LR: head LR / 100"
echo "========================================="

extra=("${@:5}") ; extract_backbone_and_filter_args "${extra[@]}"
BACKBONE_TO_USE="${PARSED_BACKBONE:-${BACKBONE:-vit_base_patch16_224}}"
run_experiment "seq_finetune_small_lr" "$BACKBONE_TO_USE" "adam_head_default_backbone_100x_small" "$SEQ_FINETUNE_SMALL_LR" \
    "${FILTERED_ARGS[@]}"

echo "========================================="
echo "SeqFinetuneSmallLR experiment completed!"
echo "Results saved in ${LOG_PATH} directory"
echo "========================================="
