#!/bin/bash

# --------------------------------------------------------------
# Example Running Command:
# bash scripts/run_baselines_lwf.sh [GPU_ID] [SEEDS] [DATASET] [EXTRA_NOTE]
# --------------------------------------------------------------

source "$(dirname "$0")/common_baselines.sh"

date
ulimit -n 65536
export MASTER_PORT=$(($RANDOM+32769))
export WORLD_SIZE=1

GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID

LWF_LR=${LWF_LR:-"5e-5"}
LWF_LAMBDA=${LWF_LAMBDA:-"1.0"}
LWF_TEMPERATURE=${LWF_TEMPERATURE:-"2.0"}

echo "Using GPU: $GPU_ID"
echo "Running experiments on dataset: $DATASET with seeds: $SEEDS"
echo "========================================="
echo "Starting LwF Baseline Experiment"
echo "Dataset: $DATASET"
echo "Seeds: $SEEDS"
echo "Si-Blurry Setting: m=$N%, n=$M%"
echo "Tasks: $N_TASKS"
echo "LR: $LWF_LR"
echo "LwF lambda: $LWF_LAMBDA"
echo "LwF temperature: $LWF_TEMPERATURE"
echo "========================================="

extra=("${@:5}") ; extract_backbone_and_filter_args "${extra[@]}"
BACKBONE_TO_USE="${PARSED_BACKBONE:-${BACKBONE:-vit_base_patch16_224}}"
run_experiment "lwf" "$BACKBONE_TO_USE" "adam" "$LWF_LR" \
    --lwf_lambda "$LWF_LAMBDA" \
    --lwf_temperature "$LWF_TEMPERATURE" \
    "${FILTERED_ARGS[@]}"

echo "========================================="
echo "LwF experiment completed!"
echo "Results saved in ${LOG_PATH} directory"
echo "========================================="
