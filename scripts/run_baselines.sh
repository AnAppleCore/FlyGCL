#!/bin/bash

# Standardized Baseline Experiments for Si-Blurry Setting
# Based on the paper's Table 2 configuration

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
SEEDS=${2:-1}  # Default seed 1, can be specified as second parameter

# Common training settings
BATCHSIZE=64
NUM_EPOCHS=1  # 1 epoch (online learning)
EVAL_PERIOD=1000
N_WORKER=8

# Dataset configuration
DATASET=${3:-"cifar100"}  # Default to cifar100, can be cifar100, imagenet-r, cub200

echo "Running experiments on dataset: $DATASET with seeds: $SEEDS"

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

# Function to run experiment
run_experiment() {
    local METHOD=$1
    local MODEL_NAME=$2
    local OPT_NAME=$3
    local LR=$4
    local MEMORY_SIZE=$5
    local ONLINE_ITER=$6
    local EXTRA_ARGS=$7
    
    local NOTE="${METHOD}_${DATASET}_baseline_standard"
    
    echo "Running $METHOD experiment..."
    
    python -W ignore main.py \
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
        --transforms autoaug \
        --rnd_NM \
        $GPU_TRANSFORM \
        $USE_AMP \
        $EXTRA_ARGS
}

echo "========================================="
echo "Starting Baseline Experiments"
echo "Dataset: $DATASET"
echo "Seeds: $SEEDS"
echo "Si-Blurry Setting: m=$N%, n=$M%"
echo "Tasks: $N_TASKS"
echo "========================================="

# Simple Methods
echo "Running Simple Methods..."

# Seq FT (Sequential Fine-tuning)
run_experiment "Finetuning" "vit_finetune" "sgd" 0.005 0 3 "--init_model --init_opt"

# Seq FT (SL) - Sequential Fine-tuning with low backbone learning rate
# Note: This would require separate implementation for different learning rates

# Linear Probe - Backbone frozen, only train output layer
run_experiment "Finetuning" "vit_finetune_last" "adam" 0.005 0 3 "--fix_bcb"

# PTMs-based Continual Learning Methods
echo "Running PTMs-based CL Methods..."

# CODA-P (using prefix tuning)
run_experiment "CodaPrompt" "CodaPrompt" "adam" 0.005 0 3 ""

# L2P (Already uses prompt tuning, not prefix - this is correct as per analysis)
run_experiment "L2P" "L2P" "adam" 0.005 0 3 ""

# DualPrompt (uses prefix tuning)
run_experiment "DualPrompt" "DualPrompt" "adam" 0.005 0 3 ""

# PTMs-based Generalized Continual Learning Methods  
echo "Running PTMs-based GCL Methods..."

# MVP (with contrastive loss + logit masking)
run_experiment "mvp" "mvp" "adam" 0.005 0 3 "--use_mask --use_contrastiv --use_afs --use_mcr"

# Note: MISA implementation not found in current codebase

echo "========================================="
echo "All baseline experiments completed!"
echo "Results saved in results/ directory"
echo "=========================================" 