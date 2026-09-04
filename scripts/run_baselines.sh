#!/bin/bash

# --------------------------------------------------------------
# Example Running Command:
# bash scripts/run_baselines.sh [GPU_ID] [SEEDS] [DATASET] [EXTRA_NOTE]
# --------------------------------------------------------------

# Standardized Baseline Experiments for Online Si-Blurry Setting
date
ulimit -n 65536
export MASTER_PORT=$(($RANDOM+32769))
export WORLD_SIZE=1

# GPU Selection (can be manually specified)
GPU_ID=${1:-0}  # Default to GPU 0 if not specified
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "Using GPU: $GPU_ID"

# Common experiment settings
N=50  # Disjoint Class Ratio (n) = 50%
M=10  # Blurry Sample Ratio (m) = 10%
N_TASKS=5
LOG_PATH="./results"
SEEDS=${2:-"1 2 3 4 5"}  # Default to multiple seeds, can be specified as second parameter

# Common training settings
TOPK=1
N_WORKER=8
BATCHSIZE=64
NUM_EPOCHS=1  # 1 epoch (online learning)
ONLINE_ITER=3
EVAL_PERIOD=1000
SCHED_NAME="default"
TRANSFORMS="autoaug"

# Dataset configuration
DATASET=${3:-"cifar100"}  # Default to cifar100, can be cifar100, imagenet-r, cub200

echo "Running experiments on dataset: $DATASET with seeds: $SEEDS"

# Extra note for the experiment
EXTRA_NOTE=${4:-"baseline_standard"}

# Dataset-specific paths
case $DATASET in
    "cifar100")
        DATA_DIR="/data/datasets"
        ;;
    "imagenet-r")
        DATA_DIR="/data/datasets/imagenet-r"
        ;;
    "cub200")
        DATA_DIR="/data/datasets/CUB_200_2011"
        ;;
    *)
        echo "Unsupported dataset: $DATASET"
        exit 1
        ;;
esac

# Extract --backbone/-b from extra args; outputs:
# - PARSED_BACKBONE: parsed value or empty if not provided
# - FILTERED_ARGS: extra args with backbone flags removed
extract_backbone_and_filter_args() {
    local args=("$@"); PARSED_BACKBONE=""; FILTERED_ARGS=()
    for ((i=0;i<${#args[@]};i++)); do
        if [[ "${args[$i]}" == "--backbone" || "${args[$i]}" == "-b" ]]; then
            if (( i+1 < ${#args[@]} )); then
                PARSED_BACKBONE="${args[$((i+1))]}"
                ((i++))
                continue
            fi
        fi
        FILTERED_ARGS+=("${args[$i]}")
    done
}

# Function to run experiment
run_experiment() {
    local METHOD=$1
    local BACKBONE=${2:-"vit_base_patch16_224"}
    local OPT_NAME=${3:-"adam"}
    local LR=${4:-"0.005"}
    local EXTRA_ARGS=${5:-""}
    
    local NOTE="${METHOD}_${BACKBONE}_${DATASET}_${EXTRA_NOTE}"

    mkdir -p "${LOG_PATH}/logs/${DATASET}/${NOTE}"
    
    echo "Running $METHOD experiment..."
    
    /home/hongwei/miniconda3/envs/DGIL/bin/python -W ignore main.py \
        --seeds $SEEDS \
        --note $NOTE \
        --log_path $LOG_PATH \
        --method $METHOD \
        --backbone $BACKBONE \
        --dataset $DATASET \
        --data_dir $DATA_DIR \
        --n_tasks $N_TASKS --m $M --n $N \
        --opt_name $OPT_NAME \
        --sched_name $SCHED_NAME \
        --n_worker $N_WORKER \
        --batchsize $BATCHSIZE \
        --lr $LR \
        --num_epochs $NUM_EPOCHS \
        --online_iter $ONLINE_ITER \
        --transforms $TRANSFORMS \
        --topk $TOPK \
        --eval_period $EVAL_PERIOD \
        --rnd_NM \
        --use_amp \
        $EXTRA_ARGS \
        | tee "${LOG_PATH}/logs/${DATASET}/${NOTE}/seed_${SEEDS}_log.txt" 2>&1
}

echo "========================================="
echo "Starting Baseline Experiments"
echo "Dataset: $DATASET"
echo "Seeds: $SEEDS"
echo "Si-Blurry Setting: m=$N%, n=$M%"
echo "Tasks: $N_TASKS"
echo "========================================="

# Parse extra args once to get backbone override (if any)
extra=("${@:5}") ; extract_backbone_and_filter_args "${extra[@]}"
BACKBONE_TO_USE="${PARSED_BACKBONE:-${BACKBONE:-vit_base_patch16_224}}"

# TODO: Add SLCA
# Seq FT (SL) - Sequential Fine-tuning with low backbone learning rate
run_experiment "slca" "$BACKBONE_TO_USE" "sgd_sl" 0.00005 "${FILTERED_ARGS[@]}"

# CODA-P (uses prefix tuning)
run_experiment "codaprompt" "$BACKBONE_TO_USE" "adam" 0.005 "${FILTERED_ARGS[@]}"

# L2P (uses prompt tuning)
run_experiment "l2p" "$BACKBONE_TO_USE" "adam" 0.005 "${FILTERED_ARGS[@]}"

# DualPrompt (uses prefix tuning)
run_experiment "dualprompt" "$BACKBONE_TO_USE" "adam" 0.005 "${FILTERED_ARGS[@]}"

# MVP (with contrastive loss + logit masking)
run_experiment "mvp" "$BACKBONE_TO_USE" "adam" 0.005 "${FILTERED_ARGS[@]}"

# RanPAC (uses random projection)
run_experiment "ranpac" "$BACKBONE_TO_USE" "adam" 0.005 "${FILTERED_ARGS[@]}"

echo "========================================="
echo "All experiments completed!"
echo "Results saved in ${LOG_PATH} directory"
echo "========================================="