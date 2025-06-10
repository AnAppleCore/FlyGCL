#!/bin/bash

# LSH-based FlyPrompt Experiment Script
# Tests the locality-sensitive hashing prompt selection mechanism

date
ulimit -n 65536
export MASTER_PORT=$(($RANDOM+32769))
export WORLD_SIZE=1

# GPU Selection (can be manually specified)
GPU_ID=${1:-0}  # Default to GPU 0 if not specified
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "Using GPU: $GPU_ID"

# Common experiment settings
N_TASKS=5
N=50  # Disjoint Class Ratio (m) = 50%
M=10  # Blurry Sample Ratio (n) = 10%
GPU_TRANSFORM="--gpu_transform"
USE_AMP="--use_amp"
SEEDS=${2:-"1 2 3"}  # Default to 3 seeds for quick testing

# Common training settings
BATCHSIZE=64
NUM_EPOCHS=1  # 1 epoch (online learning)
EVAL_PERIOD=1000
N_WORKER=8

# Dataset configuration
DATASET=${3:-"cifar100"}  # Default to cifar100

echo "Running LSH-FlyPrompt experiments on dataset: $DATASET with seeds: $SEEDS"

# Dataset-specific paths
case $DATASET in
    "cifar100")
        DATA_DIR="/data/datasets/CIFAR"
        ;;
    "imagenet-r")
        DATA_DIR="/data/datasets/imagenet-r"
        ;;
    "cub200")
        DATA_DIR="/data/datasets/CUB200_2011"
        ;;
    *)
        echo "Unsupported dataset: $DATASET"
        exit 1
        ;;
esac

POSTFIX=${4:-"_lsh"}

# Function to run LSH experiment
run_lsh_experiment() {
    local METHOD=$1
    local MODEL_NAME=$2
    local OPT_NAME=$3
    local LR=$4
    local MEMORY_SIZE=$5
    local ONLINE_ITER=$6
    local EXPANSION_DIM=$7
    local KEEP_RATIO=$8
    local WINNER_TYPE=$9
    local RETURN_BINARY=${10}
    local HASH_TYPE=${11}
    local EXTRA_ARGS=${12}
    
    local NOTE="${METHOD}${POSTFIX}_exp${EXPANSION_DIM}_keep${KEEP_RATIO}_${WINNER_TYPE}_${HASH_TYPE}"
    if [ "$RETURN_BINARY" = "True" ]; then
        NOTE="${NOTE}_binary"
    else
        NOTE="${NOTE}_float"
    fi

    mkdir -p "./results/logs/${DATASET}/${NOTE}"
    
    echo "Running $METHOD experiment with LSH parameters:"
    echo "  Expansion dim: $EXPANSION_DIM"
    echo "  Keep ratio: $KEEP_RATIO"
    echo "  Winner type: $WINNER_TYPE"
    echo "  Return binary: $RETURN_BINARY"
    echo "  Hash type: $HASH_TYPE"
    
    /home/yanhongwei/miniconda3/envs/DGIL/bin/python -W ignore main.py \
        --mode $METHOD \
        --dataset $DATASET \
        --n_tasks $N_TASKS --m $M --n $N \
        --seeds $SEEDS \
        --model_name $MODEL_NAME \
        --opt_name $OPT_NAME \
        --lr $LR \
        --batchsize $BATCHSIZE \
        --num_epochs $NUM_EPOCHS \
        --memory_size $MEMORY_SIZE \
        --online_iter $ONLINE_ITER \
        --data_dir $DATA_DIR \
        --note $NOTE \
        --eval_period $EVAL_PERIOD \
        --n_worker $N_WORKER \
        --selection_size 5 \
        --transforms autoaug \
        --rnd_NM \
        --expansion_dim $EXPANSION_DIM \
        --keep_ratio $KEEP_RATIO \
        --winner_type $WINNER_TYPE \
        --return_binary $RETURN_BINARY \
        --hash_type $HASH_TYPE \
        $GPU_TRANSFORM \
        $USE_AMP \
        $EXTRA_ARGS \
        > "./results/logs/${DATASET}/${NOTE}/seed_${SEEDS}_log.txt" 2>&1
}

echo "========================================="
echo "Starting LSH-based FlyPrompt Experiments"
echo "Dataset: $DATASET"
echo "Seeds: $SEEDS"
echo "Si-Blurry Setting: m=$N%, n=$M%"
echo "Tasks: $N_TASKS"
echo "Selection Size: 5 (FlyPromptLSH default)"
echo "========================================="

echo "Running LSH-based FlyPrompt variants..."

# Baseline LSH configuration (your specified parameters)
echo "1. Baseline LSH FlyPrompt (4096D, 5% sparsity, top-k, binary, chunked)"
run_lsh_experiment "FlyPromptLSH" "FlyPromptLSH" "adam" 0.005 0 3 4096 0.05 "topk" "True" "chunked" ""

# # Alternative configurations for comparison
# echo "2. LSH FlyPrompt with float representation"
# run_lsh_experiment "FlyPromptLSH" "FlyPromptLSH" "adam" 0.005 0 3 4096 0.05 "topk" "False" "chunked" ""

# echo "3. LSH FlyPrompt with threshold winner-take-all"
# run_lsh_experiment "FlyPromptLSH" "FlyPromptLSH" "adam" 0.005 0 3 4096 0.05 "threshold" "True" "chunked" ""

# echo "4. LSH FlyPrompt with overlapping hash"
# run_lsh_experiment "FlyPromptLSH" "FlyPromptLSH" "adam" 0.005 0 3 4096 0.05 "topk" "True" "overlapping" ""

# echo "5. LSH FlyPrompt with higher expansion (8192D)"
# run_lsh_experiment "FlyPromptLSH" "FlyPromptLSH" "adam" 0.005 0 3 8192 0.05 "topk" "True" "chunked" ""

# echo "6. LSH FlyPrompt with lower sparsity (10%)"
# run_lsh_experiment "FlyPromptLSH" "FlyPromptLSH" "adam" 0.005 0 3 4096 0.10 "topk" "True" "chunked" ""

# echo "========================================="
# echo "All LSH-based experiments completed!"
# echo "Results saved in results/ directory"
# echo "========================================="

# # Quick test run (optional)
# echo "Running quick test to verify implementation..."
# /home/yanhongwei/miniconda3/envs/DGIL/bin/python test_lsh_implementation.py 