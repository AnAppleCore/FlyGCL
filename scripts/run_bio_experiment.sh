#!/bin/bash

# Bio-Plausible Drosophila-Inspired FlyPrompt Experiment Script
# Tests the biological hierarchical tri-partite prompt selection mechanism

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

echo "Running Bio-Plausible FlyPrompt experiments on dataset: $DATASET with seeds: $SEEDS"

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

POSTFIX=${4:-"_bioplausible"}

# Function to run bio-plausible experiment
run_bio_experiment() {
    local METHOD=$1
    local MODEL_NAME=$2
    local OPT_NAME=$3
    local LR=$4
    local MEMORY_SIZE=$5
    local ONLINE_ITER=$6
    local EXPANSION_DIM=$7
    local CONNECTIVITY=$8
    local EMA_ALPHA=$9
    local EMA_BETA=${10}
    local KEEP_RATIO=${11}
    local EXTRA_ARGS=${12}
    
    local NOTE="${METHOD}${POSTFIX}_exp${EXPANSION_DIM}_conn${CONNECTIVITY}_ema_alpha${EMA_ALPHA}_ema_beta${EMA_BETA}"

    mkdir -p "./results/logs/${DATASET}/${NOTE}"
    
    echo "Running $METHOD Bio-Plausible experiment:"
    echo "  Bio-plausible: True"
    echo "  Projection type: sparse_binary"
    echo "  Expansion dim: $EXPANSION_DIM ($(echo "scale=1; $EXPANSION_DIM/768" | bc)x)"
    echo "  Connectivity: $CONNECTIVITY"
    echo "  EMA Alpha (part B EMA): $EMA_ALPHA"
    echo "  EMA Beta (part C EMA): $EMA_BETA"
    echo "  Keep ratio: $KEEP_RATIO"
    
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
        --bio_plausible \
        --projection_type sparse_binary \
        --expansion_dim $EXPANSION_DIM \
        --connectivity $CONNECTIVITY \
        --ema_alpha $EMA_ALPHA \
        --ema_beta $EMA_BETA \
        --keep_ratio $KEEP_RATIO \
        --winner_type topk \
        --return_binary False\
        --hash_type chunked \
        $GPU_TRANSFORM \
        $USE_AMP \
        $EXTRA_ARGS \
        > "./results/logs/${DATASET}/${NOTE}/seed_${SEEDS}_log.txt" 2>&1
}

echo "========================================="
echo "Starting Bio-Plausible Drosophila-Inspired FlyPrompt Experiments"
echo "Dataset: $DATASET"
echo "Seeds: $SEEDS"
echo "Si-Blurry Setting: m=$N%, n=$M%"
echo "Tasks: $N_TASKS"
echo "Hierarchical Architecture: γ (fast), α'/β' (medium), α/β (slow)"
echo "========================================="

echo "1. Bio-Plausible FlyPrompt (20x expansion, biological connectivity)"
run_bio_experiment "FlyPromptLSH" "FlyPromptLSH" "adam" 0.005 0 3 15360 6 0.05 0.1 0.05 ""

# echo "2. Bio-Plausible FlyPrompt (15x expansion, reduced complexity)"
# run_bio_experiment "FlyPromptLSH" "FlyPromptLSH" "adam" 0.005 0 3 11520 6 0.05 0.1 0.05 ""

# echo "3. Bio-Plausible FlyPrompt (different temporal dynamics)"
# run_bio_experiment "FlyPromptLSH" "FlyPromptLSH" "adam" 0.005 0 3 15360 6 0.1 0.05 0.05 ""

# echo "4. Bio-Plausible FlyPrompt (higher connectivity)"
# run_bio_experiment "FlyPromptLSH" "FlyPromptLSH" "adam" 0.005 0 3 15360 10 0.05 0.1 0.05 ""

echo "========================================="
echo "All Bio-Plausible experiments completed!"
echo "Results saved in results/ directory"
echo "=========================================" 