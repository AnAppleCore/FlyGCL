#!/bin/bash

# --------------------------------------------------------------
# Example Running Command:
# bash scripts/run_baselines_linear_probe.sh [GPU_ID] [SEEDS] [DATASET] [EXTRA_NOTE]
# --------------------------------------------------------------

source "$(dirname "$0")/common_baselines.sh"

date
ulimit -n 65536
export MASTER_PORT=$(($RANDOM+32769))
export WORLD_SIZE=1

GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID

LINEAR_PROBE_LR=${LINEAR_PROBE_LR:-"0.005"}

echo "Using GPU: $GPU_ID"
echo "Running experiments on dataset: $DATASET with seeds: $SEEDS"
echo "========================================="
echo "Starting LinearProbe Baseline Experiment"
echo "Dataset: $DATASET"
echo "Seeds: $SEEDS"
echo "Si-Blurry Setting: m=$N%, n=$M%"
echo "Tasks: $N_TASKS"
echo "LR: $LINEAR_PROBE_LR"
echo "========================================="

extra=("${@:5}") ; extract_backbone_and_filter_args "${extra[@]}"
BACKBONE_TO_USE="${PARSED_BACKBONE:-${BACKBONE:-vit_base_patch16_224}}"
run_experiment "linear_probe" "$BACKBONE_TO_USE" "adam" "$LINEAR_PROBE_LR" \
    "${FILTERED_ARGS[@]}"

echo "========================================="
echo "LinearProbe experiment completed!"
echo "Results saved in ${LOG_PATH} directory"
echo "========================================="
