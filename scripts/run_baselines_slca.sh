#!/bin/bash

# --------------------------------------------------------------
# Example Running Command:
# bash scripts/run_baselines_slca.sh [GPU_ID] [SEEDS] [DATASET] [EXTRA_NOTE]
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

echo "========================================="
echo "Starting SLCA Baseline Experiment"
echo "Dataset: $DATASET"
echo "Seeds: $SEEDS"
echo "Si-Blurry Setting: m=$N%, n=$M%"
echo "Tasks: $N_TASKS"
echo "========================================="

# Run only SLCA experiment
run_experiment "slca" "vit_base_patch16_224" "sgd_sl" 0.00005 ""

echo "========================================="
echo "SLCA experiment completed!"
echo "Results saved in ${LOG_PATH} directory"
echo "========================================="