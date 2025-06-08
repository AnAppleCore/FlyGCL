#!/bin/bash

# L2P CIFAR100 Reproduction Script
# Single GPU, Local Execution

echo "Starting L2P reproduction on CIFAR100..."
date

# CIL CONFIG
NOTE="L2P_CIFAR100_Reproduction" # Short description of the experiment
MODE="L2P"
DATASET="cifar100"
N_TASKS=5
N=50  # Disjoint percentage (matching MISA config)
M=10  # Blurry percentage
GPU_TRANSFORM="--gpu_transform"
USE_AMP="--use_amp"
SEEDS="1 2 3" # Multiple seeds for statistical significance

# CIFAR100 specific settings (matching MISA benchmark config)
MEM_SIZE=0  # No episodic memory (matching MISA)
ONLINE_ITER=3
MODEL_NAME="L2P"
EVAL_PERIOD=1000
BATCHSIZE=64
LR=3e-2
OPT_NAME="adam"
SCHED_NAME="default"
MEMORY_EPOCH=256
DATA_DIR="/data/datasets/CIFAR"  # Specific CIFAR path (matching MISA)

# Set CUDA device (use only first GPU)
export CUDA_VISIBLE_DEVICES=0

echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Model: $MODEL_NAME"
echo "  Tasks: $N_TASKS"
echo "  N (Disjoint): $N"
echo "  M (Blurry): $M"
echo "  Memory Size: $MEM_SIZE"
echo "  Batch Size: $BATCHSIZE"
echo "  Learning Rate: $LR"
echo "  Seeds: $SEEDS"
echo "  Data Directory: $DATA_DIR"
echo ""

# Run L2P for each seed
for RND_SEED in $SEEDS
do
    echo "Running L2P with seed: $RND_SEED"
    python main.py --mode $MODE \
        --dataset $DATASET \
        --n_tasks $N_TASKS --m $M --n $N \
        --rnd_seed $RND_SEED \
        --model_name $MODEL_NAME --opt_name $OPT_NAME --sched_name $SCHED_NAME \
        --lr $LR --batchsize $BATCHSIZE \
        --memory_size $MEM_SIZE $GPU_TRANSFORM $USE_AMP --online_iter $ONLINE_ITER \
        --data_dir $DATA_DIR \
        --note $NOTE --eval_period $EVAL_PERIOD --memory_epoch $MEMORY_EPOCH \
        --n_worker 8 --rnd_NM --transforms autoaug \
        --seeds $RND_SEED
    
    echo "Completed seed $RND_SEED"
    echo "----------------------------------------"
done

echo "L2P CIFAR100 reproduction completed!"
date 