#!/bin/bash

# --------------------------------------------------------------
# Example Running Command:
# bash scripts/run_baselines_seq_finetune.sh [GPU_ID] [SEEDS] [DATASET] [EXTRA_NOTE]
# --------------------------------------------------------------

source "$(dirname "$0")/common_baselines.sh"

date
ulimit -n 65536
export MASTER_PORT=$(($RANDOM+32769))
export WORLD_SIZE=1

GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID

SEQ_FINETUNE_LR=${SEQ_FINETUNE_LR:-"0.005"}

echo "Using GPU: $GPU_ID"
echo "Running experiments on dataset: $DATASET with seeds: $SEEDS"
echo "========================================="
echo "Starting SeqFinetune Baseline Experiment"
echo "Dataset: $DATASET"
echo "Seeds: $SEEDS"
echo "Si-Blurry Setting: m=$N%, n=$M%"
echo "Tasks: $N_TASKS"
echo "LR: $SEQ_FINETUNE_LR"
echo "========================================="

extra=("${@:5}") ; extract_backbone_and_filter_args "${extra[@]}"
BACKBONE_TO_USE="${PARSED_BACKBONE:-${BACKBONE:-vit_base_patch16_224}}"
run_experiment "seq_finetune" "$BACKBONE_TO_USE" "adam" "$SEQ_FINETUNE_LR" \
    "${FILTERED_ARGS[@]}"

echo "========================================="
echo "SeqFinetune experiment completed!"
echo "Results saved in ${LOG_PATH} directory"
echo "========================================="
