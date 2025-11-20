#!/bin/bash

# --------------------------------------------------------------
# Example Running Command:
# bash scripts/run_baselines_mvp.sh [GPU_ID] [SEEDS] [DATASET] [EXTRA_NOTE]
# --------------------------------------------------------------

# Source the common functions and variables
source "$(dirname "$0")/common_baselines.sh"

# Setup environment
date
ulimit -n 65536
export MASTER_PORT=$(($RANDOM+32769))
export WORLD_SIZE=1

# GPU Selection (can be manually specified)
GPU_ID=${1:-0}  # Default to GPU 0 if not specified
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "Using GPU: $GPU_ID"
echo "Running experiments on dataset: $DATASET with seeds: $SEEDS"

# Parse extra args to figure out effective number of tasks (override via --n_tasks if provided)
extra=("${@:5}")
EFFECTIVE_N_TASKS="$N_TASKS"
for ((i=0;i<${#extra[@]};i++)); do
    if [[ "${extra[$i]}" == "--n_tasks" ]]; then
        if (( i+1 < ${#extra[@]} )); then
            EFFECTIVE_N_TASKS="${extra[$((i+1))]}"
        fi
    fi
done

echo "========================================="
echo "Starting MVP Baseline Experiment"
echo "Dataset: $DATASET"
echo "Seeds: $SEEDS"
echo "Si-Blurry Setting: m=$N%, n=$M%"
echo "Tasks: $EFFECTIVE_N_TASKS"
echo "========================================="

# Run only MVP experiment
extract_backbone_and_filter_args "${extra[@]}"
BACKBONE_TO_USE="${PARSED_BACKBONE:-${BACKBONE:-vit_base_patch16_224}}"
run_experiment "mvp" "$BACKBONE_TO_USE" "adam" 0.005 "${FILTERED_ARGS[@]}"

echo "========================================="
echo "MVP experiment completed!"
echo "Results saved in ${LOG_PATH} directory"
echo "========================================="