#!/bin/bash

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

# Extra note for the experiment
EXTRA_NOTE=${4:-"baseline_standard"}

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
    "tinyimagenet")
        DATA_DIR="/data/datasets"
        ;;
    "mnist")
        DATA_DIR="/data/datasets"
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
    shift 4
    local EXTRA_ARGS=("$@")
    
    local NOTE="${METHOD}_${BACKBONE}_${DATASET}_${EXTRA_NOTE}"

    mkdir -p "${LOG_PATH}/logs/${DATASET}/${NOTE}"
    
    echo "Running $METHOD experiment..."
    
    /home/yanhongwei/miniconda3/envs/DGIL/bin/python -W ignore main.py \
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
        "${EXTRA_ARGS[@]}" \
        | tee "${LOG_PATH}/logs/${DATASET}/${NOTE}/seed_${SEEDS}_log.txt" 2>&1
}