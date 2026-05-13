#!/bin/bash

# Sequential GCL Experiment Runner with Monitoring
# Runs groups of experiments sequentially, waiting for each group to complete

# Default dataset (can be overridden by command line argument)
DEFAULT_DATASET="cifar100" # Available: cifar100, imagenet-r, cub200

# Function to check if any screen sessions from a given pattern are still running
check_sessions_running() {
    local pattern=$1
    local running_sessions=$(screen -ls | grep -c "$pattern")
    return $running_sessions
}

# Function to wait for all sessions in a group to complete
wait_for_group_completion() {
    local group_name=$1

    echo "========================================="
    echo "Waiting for $group_name group to complete..."
    echo "Monitoring sessions matching pattern: $group_name"
    echo "========================================="

    while true; do
        check_sessions_running "$group_name"
        local running_count=$?

        if [ $running_count -eq 0 ]; then
            echo "✓ All $group_name sessions completed!"
            break
        else
            echo "$(date): $running_count $group_name sessions still running..."
            sleep 120  # Check every 5 minutes
        fi
    done

    echo "Group $group_name finished at $(date)"
    echo ""
}

# Function to start a group of experiments with custom GPU and seed lists
start_group() {
    local group_name=$1
    local script_path=$2
    local extra_note=$3

    # Backward-compatible parsing: if the 4th arg looks like a GPU id (number) or is empty,
    # use global DATASET and treat the rest as GPU list; otherwise take the 4th as dataset.
    local fourth_arg=${4:-}
    local dataset
    if [[ -z "$fourth_arg" || "$fourth_arg" =~ ^[0-9]+$ ]]; then
        dataset="$DATASET"
        shift 3
    else
        dataset="$fourth_arg"
        shift 4
    fi

    # Split remaining args into GPU list and optional extra args (after a "--" sentinel)
    local extra_args=()
    local gpu_list=()
    local parsing_extras=0
    for arg in "$@"; do
        if [ "$arg" = "--" ]; then
            parsing_extras=1
            continue
        fi
        if [ $parsing_extras -eq 1 ]; then
            extra_args+=("$arg")
        else
            gpu_list+=("$arg")
        fi
    done

    echo "========================================="
    echo "Starting $group_name group at $(date)"
    echo "Script: $script_path"
    echo "Dataset: $dataset"
    echo "GPUs: ${gpu_list[@]}"
    if [ ${#extra_args[@]} -gt 0 ]; then
        echo "Extra args: ${extra_args[*]}"
    fi
    echo "========================================="

    # Start sessions for each GPU in the list
    local session_counter=1
    for gpu in "${gpu_list[@]}"; do
        local session_name="${group_name}${session_counter}"
        echo "Starting session: $session_name (GPU $gpu, Session $session_counter)"
        screen -dmS "$session_name" bash "$script_path" $gpu $session_counter $dataset $extra_note "${extra_args[@]}"
        sleep 2  # Brief delay between session starts
        ((session_counter++))
    done

    echo "All $group_name sessions started!"
    echo ""
}

# Alternative function for more control - accepts both GPU and seed arrays
start_group_custom() {
    local group_name=$1
    local script_path=$2
    local extra_note=$3
    local gpu_list_str=$4
    local seed_list_str=$5
    local dataset=${6:-$DATASET}  # Use provided dataset or global DATASET variable

    # Shift to expose any additional arguments as extra args to forward.
    # Accept an optional "--" separator for consistency with start_group.
    shift 6
    local extra_args=()
    if [ "${1:-}" = "--" ]; then
        shift
    fi
    extra_args=("$@")

    # Convert string representations to arrays
    IFS=' ' read -ra gpu_list <<< "$gpu_list_str"
    IFS=' ' read -ra seed_list <<< "$seed_list_str"

    echo "========================================="
    echo "Starting $group_name group at $(date)"
    echo "Script: $script_path"
    echo "Dataset: $dataset"
    echo "GPUs: ${gpu_list[@]}"
    echo "Seeds: ${seed_list[@]}"
    if [ ${#extra_args[@]} -gt 0 ]; then
        echo "Extra args: ${extra_args[*]}"
    fi
    echo "========================================="

    # Check if GPU and seed lists have same length
    if [ ${#gpu_list[@]} -ne ${#seed_list[@]} ]; then
        echo "Error: GPU list and seed list must have the same length!"
        return 1
    fi

    # Start sessions for each GPU-seed pair
    for i in "${!gpu_list[@]}"; do
        local gpu=${gpu_list[$i]}
        local seed=${seed_list[$i]}
        local session_name="${group_name}$((i+1))"
        echo "Starting session: $session_name (GPU $gpu, Seed $seed)"
        screen -dmS "$session_name" bash "$script_path" $gpu $seed $dataset $extra_note "${extra_args[@]}"
        sleep 2  # Brief delay between session starts
    done

    echo "All $group_name sessions started!"
    echo ""
}

# Standardized start-and-wait function:
# standard_start_and_wait <methods> <backbones> <datasets> [-- <extra args...>]
# - The three primary parameters are space-separated list strings (single or multiple values).
# - You can override the default GPU list via env var GPU_LIST, e.g., export GPU_LIST="0 1 2 3 4"
# - Remaining extra args after "--" are forwarded to the baseline script (and we also inject --backbone <b>).
standard_start_and_wait() {
    local methods_str=$1
    local backbones_str=$2
    local datasets_str=$3
    shift 3

    # Parse into arrays
    IFS=' ' read -ra methods <<< "$methods_str"
    IFS=' ' read -ra backbones <<< "$backbones_str"
    IFS=' ' read -ra datasets <<< "$datasets_str"

    # Parse extra args (forwarded after "--" to the baseline script)
    local extra_args=("$@")

    # GPU list: default 0..4; can be overridden by GPU_LIST env var (space-separated)
    local gpus=()
    if [ -n "${GPU_LIST:-}" ]; then
        IFS=' ' read -ra gpus <<< "$GPU_LIST"
    else
        gpus=(0 1 2 3 4)
    fi

    for m in "${methods[@]}"; do
        for b in "${backbones[@]}"; do
            for d in "${datasets[@]}"; do
                local group_name="${m}_${b}_${d}_"
                echo "[standard_start_and_wait] Launching group: $group_name"
                # Use existing start_group to launch, then wait for the group to finish
                start_group "$group_name" "./scripts/run_baselines_${m}.sh" "standard" "$d" \
                    "${gpus[@]}" -- --backbone "$b" "${extra_args[@]}"
                wait_for_group_completion "$group_name"
            done
        done
    done
}

# Parse command line arguments
DATASET=${1:-$DEFAULT_DATASET}

# Main execution flow
echo "========================================="
echo "Sequential GCL Experiment Runner Started"
echo "Dataset: $DATASET"
echo "$(date)"
echo "========================================="

# Session Groups — Usage Guide
# -----------------------------------------------------------------------------
# Syntax
#   1) start_group <group_name> <script> <extra_note> [dataset|first_gpu] <gpu...> [-- <extra args...>]
#      - If the 4th arg is empty or a number, the global $DATASET is used;
#        otherwise the 4th arg is treated as the dataset name.
#      - Use "--" to separate the GPU list from additional main.py args to forward.
#   2) start_group_custom <group_name> <script> <extra_note> "<gpu list>" "<seed list>" [dataset] [-- <extra args...>]
#      - GPU list and seed list must have the same length.
#      - Any args after dataset are forwarded to the baseline script.
# -----------------------------------------------------------------------------
# Examples (commented; copy and adjust):
#
# Example A: start_group using global DATASET, GPUs 0..4, with extra FlyPrompt args
# start_group "flyprompt_ex1" "./scripts/run_baselines_flyprompt.sh" "tuned_len12_bs32" \
#   0 1 2 3 4 \
#   -- --len_prompt 12 --batchsize 32 --online_iter 5
#
# Example B: start_group with explicit dataset and extra args
# start_group "flyprompt_cifar" "./scripts/run_baselines_flyprompt.sh" "cifar_tuned" \
#   "cifar100" 0 1 2 \
#   -- --len_prompt 10 --pos_prompt 0 2 4 --ema_ratio 0.9 0.99
#
# Example C: start_group_custom with paired GPUs and seeds, explicit dataset and extra args
# start_group_custom "flyprompt_pair" "./scripts/run_baselines_flyprompt.sh" "pair_run" \
#   "0 1 2" "42 43 44" "imagenet-r" \
#   -- --batchsize 48 --eval_period 500
#
# Note: You can still tweak defaults in scripts/common_baselines.sh, or override via extra args.
# -----------------------------------------------------------------------------

for BACKBONE_TO_RUN in vit_base_patch16_224 vit_base_patch16_224_mepo_21k_1k vit_base_patch16_224_21k_ibot vit_base_patch16_224_ibot vit_base_patch16_224_dino vit_base_patch16_224_mocov3; do
for DATASET_TO_RUN in  cifar100 imagenet-r cub200; do
for METHOD_TO_RUN in flyprompt l2p dualprompt codaprompt mvp ranpac; do
for N_TASKS_TO_RUN in 5; do

start_group "${METHOD_TO_RUN}_${BACKBONE_TO_RUN}_${DATASET_TO_RUN}_${N_TASKS_TO_RUN}_standard" "./scripts/run_baselines_${METHOD_TO_RUN}.sh" "tasks${N_TASKS_TO_RUN}_standard" $DATASET_TO_RUN 0 1 2 3 4 \
-- --backbone $BACKBONE_TO_RUN --n_tasks $N_TASKS_TO_RUN

wait_for_group_completion "${METHOD_TO_RUN}_${BACKBONE_TO_RUN}_${DATASET_TO_RUN}_${N_TASKS_TO_RUN}_standard"

done
done
done
done

# -----------------------------------------------------------------------------

# vit_base_patch16_224 vit_base_patch16_224_in1k vit_base_patch16_224_21k_ibot
# vit_base_patch16_224_ibot vit_base_patch16_224_dino vit_base_patch16_224_mocov3
# vit_base_patch16_224_mepo_21k vit_base_patch16_224_mepo_21k_1k vit_base_patch16_224_mepo_ibot_21k

# cifar100 imagenet-r cub200

# l2p dualprompt codaprompt mvp ranpac

# standard_start_and_wait "flyprompt" "vit_base_patch16_224 vit_base_patch16_224_mepo_21k_1k vit_base_patch16_224_21k_ibot vit_base_patch16_224_ibot vit_base_patch16_224_dino vit_base_patch16_224_mocov3" "cifar100 imagenet-r cub200"


echo "========================================="
echo "All GCL experiment groups completed!"
echo "Finished at $(date)"
echo "========================================="