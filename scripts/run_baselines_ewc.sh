#!/bin/bash

# --------------------------------------------------------------
# Example Running Command:
# bash scripts/run_baselines_ewc.sh [GPU_ID] [SEEDS] [DATASET] [EXTRA_NOTE]
# --------------------------------------------------------------

source "$(dirname "$0")/common_baselines.sh"

date
ulimit -n 65536
export MASTER_PORT=$(($RANDOM+32769))
export WORLD_SIZE=1

GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID

EWC_LR=${EWC_LR:-"5e-5"}
EWC_LAMBDA=${EWC_LAMBDA:-"1000.0"}
EWC_GAMMA=${EWC_GAMMA:-"1.0"}
echo "Using GPU: $GPU_ID"
echo "Running experiments on dataset: $DATASET with seeds: $SEEDS"
echo "========================================="
echo "Starting EWC Baseline Experiment"
echo "Dataset: $DATASET"
echo "Seeds: $SEEDS"
echo "Si-Blurry Setting: m=$N%, n=$M%"
echo "Tasks: $N_TASKS"
echo "LR: $EWC_LR"
echo "EWC lambda: $EWC_LAMBDA"
echo "EWC gamma: $EWC_GAMMA"
echo "========================================="

extra=("${@:5}") ; extract_backbone_and_filter_args "${extra[@]}"
BACKBONE_TO_USE="${PARSED_BACKBONE:-${BACKBONE:-vit_base_patch16_224}}"
run_experiment "ewc" "$BACKBONE_TO_USE" "adam" "$EWC_LR" \
    --ewc_lambda "$EWC_LAMBDA" \
    --ewc_gamma "$EWC_GAMMA" \
    "${FILTERED_ARGS[@]}"

echo "========================================="
echo "EWC experiment completed!"
echo "Results saved in ${LOG_PATH} directory"
echo "========================================="
