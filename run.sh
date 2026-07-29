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

# for BACKBONE_TO_RUN in vit_base_patch16_224 vit_base_patch16_224_mepo_21k_1k vit_base_patch16_224_21k_ibot vit_base_patch16_224_ibot vit_base_patch16_224_dino vit_base_patch16_224_mocov3; do
# for DATASET_TO_RUN in  cifar100 imagenet-r cub200; do
# for METHOD_TO_RUN in flyprompt l2p dualprompt codaprompt mvp ranpac; do
# for N_TASKS_TO_RUN in 5; do

# start_group "${METHOD_TO_RUN}_${BACKBONE_TO_RUN}_${DATASET_TO_RUN}_${N_TASKS_TO_RUN}_standard" "./scripts/run_baselines_${METHOD_TO_RUN}.sh" "tasks${N_TASKS_TO_RUN}_standard" $DATASET_TO_RUN 0 1 2 3 4 \
# -- --backbone $BACKBONE_TO_RUN --n_tasks $N_TASKS_TO_RUN

# wait_for_group_completion "${METHOD_TO_RUN}_${BACKBONE_TO_RUN}_${DATASET_TO_RUN}_${N_TASKS_TO_RUN}_standard"

# done
# done
# done
# done

# -----------------------------------------------------------------------------
# FlyPrompt analytic DAN gain full rerun.
# Uses stage-dependent z-scored analytic evidence as a multiplicative class gain.
# Runs all five seeds in one parallel batch on GPUs 0/1/2/3/5.
# Override with, for example: ANALYTIC_DAN_GPU_LIST="0 1 2 3 5" bash run.sh
# -----------------------------------------------------------------------------
# ANALYTIC_DAN_GPU_LIST=${ANALYTIC_DAN_GPU_LIST:-"0 1 2 3 5"}
# ANALYTIC_DAN_SEEDS="1 2 3 4 5"
# ANALYTIC_DAN_BATCH_ID=1
#
# for BACKBONE_TO_RUN in vit_base_patch16_224; do
# for DATASET_TO_RUN in cifar100 imagenet-r cub200; do
# for METHOD_TO_RUN in flyprompt; do
# for N_TASKS_TO_RUN in 5; do
#
# GROUP_NAME="${METHOD_TO_RUN}_${BACKBONE_TO_RUN}_${DATASET_TO_RUN}_${N_TASKS_TO_RUN}_analytic_dan_gain_batch${ANALYTIC_DAN_BATCH_ID}"
# start_group_custom "$GROUP_NAME" "./scripts/run_baselines_${METHOD_TO_RUN}.sh" "tasks${N_TASKS_TO_RUN}_analytic_dan_gain" \
#     "$ANALYTIC_DAN_GPU_LIST" "$ANALYTIC_DAN_SEEDS" "$DATASET_TO_RUN" \
#     -- --backbone "$BACKBONE_TO_RUN" --n_tasks "$N_TASKS_TO_RUN" \
#        --use_analytic_head --use_analytic_gain \
#        --analytic_gain_max_lambda 0.5 --analytic_gain_schedule quadratic
#
# wait_for_group_completion "$GROUP_NAME"
# ((ANALYTIC_DAN_BATCH_ID++))
#
# done
# done
# done
# done

# -----------------------------------------------------------------------------
# FlyPrompt MISA-pretrain + analytic DAN gain sweep (vit_base_patch16_224).
# Already completed for vit_base_patch16_224; commented out to avoid reruns.
# -----------------------------------------------------------------------------
# MISA_GPU_LIST="0 1 2 3 3"
# MISA_SEEDS="1 2 3 4 5"
# MISA_CKPT_ROOT="./checkpoints/FlyPrompt_MISA_Pretrain_Prompt"
# for MISA_DATASET in cifar100 imagenet-r cub200; do
# for MISA_EPOCH in 022 025; do
#     for MISA_DIR in add sub; do
#         MISA_CKPT="${MISA_CKPT_ROOT}/flyprompt_misa_${MISA_DIR}_ddp_bs256_ep32_seed1/epoch_${MISA_EPOCH}/flyprompt_misa_prompt_${MISA_DIR}_ddp_bs256_ep32_seed1.pt"
#         if [ ! -f "$MISA_CKPT" ]; then echo "missing checkpoint: $MISA_CKPT"; exit 1; fi
#         start_group_custom "misa_dan_${MISA_DATASET}_ep${MISA_EPOCH}_${MISA_DIR}_" \
#             "./scripts/run_baselines_flyprompt.sh" \
#             "tasks5_analytic_dan_gain_misa_${MISA_DIR}_ep${MISA_EPOCH}" \
#             "$MISA_GPU_LIST" "$MISA_SEEDS" "$MISA_DATASET" \
#             -- --backbone vit_base_patch16_224 --n_tasks 5 \
#                --use_analytic_head --use_analytic_gain \
#                --analytic_gain_max_lambda 0.5 --analytic_gain_schedule quadratic \
#                --load_pt --flyprompt_pt_path "$MISA_CKPT"
#         wait_for_group_completion "misa_dan_${MISA_DATASET}_ep${MISA_EPOCH}_${MISA_DIR}_"
#     done
# done
# done

# -----------------------------------------------------------------------------
# Multi-backbone sweep: flyprompt / flyadapter / flylora x 6 configs x 3 datasets
# Runs on 5 new backbones (vit_base_patch16_224 already complete).
# Uses backbone- and method-specific MISA checkpoints from checkpoints/MISA_BBSub.
# Each checkpoint was pretrained for its corresponding backbone and PEFT method.
#
# Config matrix:
#   1 standard
#   2 analytic DAN gain
#   3 misa + standard   (sub ep022)
#   4 misa + standard   (sub ep025)
#   5 misa + DAN gain   (sub ep022)
#   6 misa + DAN gain   (sub ep025)
#
# Override knobs via env, e.g.:
#   MB_BACKBONES="vit_base_patch16_224_dino" bash run.sh
#   MB_DRY_RUN=1 bash run.sh
# -----------------------------------------------------------------------------
MB_GPU_LIST=${MB_GPU_LIST:-"1 2 4 5 6"}
MB_SEEDS=${MB_SEEDS:-"1 2 3 4 5"}
MB_BACKBONES=${MB_BACKBONES:-"vit_base_patch16_224_mepo_21k_1k vit_base_patch16_224_21k_ibot vit_base_patch16_224_ibot vit_base_patch16_224_dino vit_base_patch16_224_mocov3"}
MB_METHODS=${MB_METHODS:-"flyprompt flyadapter flylora"}
MB_DATASETS=${MB_DATASETS:-"cifar100 imagenet-r cub200"}
MB_MISA_EPOCHS=${MB_MISA_EPOCHS:-"022 025"}
MB_INTER_GROUP_SLEEP=${MB_INTER_GROUP_SLEEP:-2}
MB_DRY_RUN=${MB_DRY_RUN:-0}
MB_RUN_STANDARD=${MB_RUN_STANDARD:-1}
MB_RUN_DAN=${MB_RUN_DAN:-1}
MB_RUN_MISA_STANDARD=${MB_RUN_MISA_STANDARD:-1}
MB_RUN_MISA_DAN=${MB_RUN_MISA_DAN:-1}

# ridge per backbone (matches the existing vit_base_patch16_224 flyprompt runs)
mb_ridge_for() {
    case "$1" in
        vit_base_patch16_224)             echo 10000 ;;
        vit_base_patch16_224_mepo_21k_1k) echo 1000000 ;;
        vit_base_patch16_224_21k_ibot)    echo 10000000 ;;
        vit_base_patch16_224_ibot)        echo 10000000 ;;
        vit_base_patch16_224_dino)        echo 10000000 ;;
        vit_base_patch16_224_mocov3)      echo 1000000 ;;
        *) echo "10000" ;;
    esac
}

MB_MISA_CKPT_ROOT=${MB_MISA_CKPT_ROOT:-"./checkpoints/MISA_BBSub"}
MB_MISA_NOTE_SUFFIX=${MB_MISA_NOTE_SUFFIX:-"bbspec"}

# Backbone-specific MISA checkpoint for a method+backbone+epoch.
mb_misa_ckpt_for() {
    local method=$1 backbone=$2 epoch=$3
    local mt bt file
    mt="$(mb_method_tag "$method")"
    bt="$(mb_backbone_tag "$backbone")"
    case "$method" in
        flyprompt)  file="fp_sub_s1.pt" ;;
        flyadapter) file="fa_sub_s1.pt" ;;
        flylora)    file="fl_sub_s1.pt" ;;
        *) file="" ;;
    esac
    echo "${MB_MISA_CKPT_ROOT}/${mt}_${bt}/epoch_${epoch}/${file}"
}

# structural args for the PEFT expert (none for flyprompt)
mb_struct_args_for() {
    case "$1" in
        flyadapter) echo "--fly_adapter_down_dim 10 --fly_adapter_layers 5" ;;
        flylora)    echo "--fly_lora_rank 5 --fly_lora_alpha 1.0 --fly_lora_layers 5" ;;
        *) echo "" ;;
    esac
}

# baseline script path for a method
mb_script_for() {
    case "$1" in
        flyprompt)  echo "./scripts/run_baselines_flyprompt.sh" ;;
        flyadapter) echo "./scripts/run_baselines_flyadapter.sh" ;;
        flylora)    echo "./scripts/run_baselines_flylora.sh" ;;
        *) echo "" ;;
    esac
}

# Short tags are used only for screen session names. Result notes stay unchanged.
mb_method_tag() {
    case "$1" in
        flyprompt)  echo "fp" ;;
        flyadapter) echo "fa" ;;
        flylora)    echo "fl" ;;
        *) echo "$1" ;;
    esac
}

mb_backbone_tag() {
    case "$1" in
        vit_base_patch16_224)             echo "base" ;;
        vit_base_patch16_224_mepo_21k_1k) echo "mepo" ;;
        vit_base_patch16_224_21k_ibot)    echo "i21" ;;
        vit_base_patch16_224_ibot)        echo "ibot" ;;
        vit_base_patch16_224_dino)        echo "dino" ;;
        vit_base_patch16_224_mocov3)      echo "moco" ;;
        *) echo "$1" ;;
    esac
}

mb_dataset_tag() {
    case "$1" in
        cifar100)   echo "cf" ;;
        imagenet-r) echo "imr" ;;
        cub200)     echo "cub" ;;
        *) echo "$1" ;;
    esac
}

mb_group_name() {
    local method=$1 backbone=$2 dataset=$3 config_tag=$4
    echo "mb_$(mb_method_tag "$method")_$(mb_backbone_tag "$backbone")_$(mb_dataset_tag "$dataset")_${config_tag}_"
}

mb_result_complete() {
    local method=$1 backbone=$2 dataset=$3 note=$4
    local dir="./results/logs/${dataset}/${method}_${backbone}_${dataset}_${note}"
    local seed
    [ -d "$dir" ] || return 1
    for seed in $MB_SEEDS; do
        [ -s "$dir/seed_${seed}_log.txt" ] || return 1
        [ -s "$dir/seed_${seed}.npy" ] || return 1
        [ -s "$dir/seed_${seed}_eval.npy" ] || return 1
        [ -s "$dir/seed_${seed}_eval_time.npy" ] || return 1
    done
    return 0
}

mb_run_group() {
    local group=$1 script=$2 note=$3 dataset=$4
    shift 4
    local extra_args=("$@")

    if [ "$MB_SKIP_COMPLETED" = "1" ] && mb_result_complete "$MB_METHOD" "$MB_BB" "$dataset" "$note"; then
        echo "[SKIP] complete: ${MB_METHOD}_${MB_BB}_${dataset}_${note}"
        return 0
    fi

    if [ "$MB_DRY_RUN" = "1" ]; then
        echo "[DRY_RUN] start_group_custom $group | note=$note | $MB_GPU_LIST | $MB_SEEDS | ${extra_args[*]} | screen_len_with_seed=$(( ${#group} + 1 ))"
    else
        start_group_custom "$group" "$script" \
            "$note" "$MB_GPU_LIST" "$MB_SEEDS" "$dataset" \
            -- "${extra_args[@]}"
        wait_for_group_completion "$group"
        sleep "$MB_INTER_GROUP_SLEEP"
    fi
}

MB_SKIP_COMPLETED=${MB_SKIP_COMPLETED:-1}

echo "========================================="
echo "Multi-backbone sweep started at $(date)"
echo "Backbones: $MB_BACKBONES"
echo "Methods:   $MB_METHODS"
echo "Datasets:  $MB_DATASETS"
echo "GPUs:      $MB_GPU_LIST"
echo "Seeds:     $MB_SEEDS"
echo "Sections:  standard=$MB_RUN_STANDARD dan=$MB_RUN_DAN misa_std=$MB_RUN_MISA_STANDARD misa_dan=$MB_RUN_MISA_DAN"
echo "DRY_RUN:   $MB_DRY_RUN"
echo "Skip completed: $MB_SKIP_COMPLETED"
echo "========================================="

for MB_BB in $MB_BACKBONES; do
    MB_RIDGE="$(mb_ridge_for "$MB_BB")"
    for MB_METHOD in $MB_METHODS; do
        MB_SCRIPT="$(mb_script_for "$MB_METHOD")"
        read -ra MB_STRUCT <<< "$(mb_struct_args_for "$MB_METHOD")"
        for MB_DATASET in $MB_DATASETS; do

            # --- 1) standard ---
            if [ "$MB_RUN_STANDARD" = "1" ]; then
                MB_GROUP="$(mb_group_name "$MB_METHOD" "$MB_BB" "$MB_DATASET" "t5std")"
                mb_run_group "$MB_GROUP" "$MB_SCRIPT" "tasks5_standard" "$MB_DATASET" \
                    --backbone "$MB_BB" --n_tasks 5 --rp_ridge "$MB_RIDGE" "${MB_STRUCT[@]}"
            fi

            # --- 2) analytic DAN gain ---
            if [ "$MB_RUN_DAN" = "1" ]; then
                MB_GROUP="$(mb_group_name "$MB_METHOD" "$MB_BB" "$MB_DATASET" "t5dan")"
                mb_run_group "$MB_GROUP" "$MB_SCRIPT" "tasks5_analytic_dan_gain" "$MB_DATASET" \
                    --backbone "$MB_BB" --n_tasks 5 --rp_ridge "$MB_RIDGE" "${MB_STRUCT[@]}" \
                    --use_analytic_head --use_analytic_gain \
                    --analytic_gain_max_lambda 0.5 --analytic_gain_schedule quadratic
            fi

            # --- 3/4) backbone-specific MISA + standard (sub ep022 / ep025) ---
            if [ "$MB_RUN_MISA_STANDARD" = "1" ]; then
                for MB_EPOCH in $MB_MISA_EPOCHS; do
                    MB_CKPT="$(mb_misa_ckpt_for "$MB_METHOD" "$MB_BB" "$MB_EPOCH")"
                    if [ ! -f "$MB_CKPT" ]; then echo "missing checkpoint: $MB_CKPT"; exit 1; fi
                    MB_GROUP="$(mb_group_name "$MB_METHOD" "$MB_BB" "$MB_DATASET" "t5${MB_MISA_NOTE_SUFFIX}sub${MB_EPOCH#0}")"
                    mb_run_group "$MB_GROUP" "$MB_SCRIPT" "tasks5_misa_${MB_MISA_NOTE_SUFFIX}_sub_ep${MB_EPOCH}" "$MB_DATASET" \
                        --backbone "$MB_BB" --n_tasks 5 --rp_ridge "$MB_RIDGE" "${MB_STRUCT[@]}" \
                        --load_pt --flyprompt_pt_path "$MB_CKPT"
                done
            fi

            # --- 5/6) backbone-specific MISA + DAN gain (sub ep022 / ep025) ---
            if [ "$MB_RUN_MISA_DAN" = "1" ]; then
                for MB_EPOCH in $MB_MISA_EPOCHS; do
                    MB_CKPT="$(mb_misa_ckpt_for "$MB_METHOD" "$MB_BB" "$MB_EPOCH")"
                    if [ ! -f "$MB_CKPT" ]; then echo "missing checkpoint: $MB_CKPT"; exit 1; fi
                    MB_GROUP="$(mb_group_name "$MB_METHOD" "$MB_BB" "$MB_DATASET" "t5dan${MB_MISA_NOTE_SUFFIX}sub${MB_EPOCH#0}")"
                    mb_run_group "$MB_GROUP" "$MB_SCRIPT" "tasks5_analytic_dan_gain_misa_${MB_MISA_NOTE_SUFFIX}_sub_ep${MB_EPOCH}" "$MB_DATASET" \
                        --backbone "$MB_BB" --n_tasks 5 --rp_ridge "$MB_RIDGE" "${MB_STRUCT[@]}" \
                        --use_analytic_head --use_analytic_gain \
                        --analytic_gain_max_lambda 0.5 --analytic_gain_schedule quadratic \
                        --load_pt --flyprompt_pt_path "$MB_CKPT"
                done
            fi

        done
    done
done

echo "========================================="
echo "All GCL experiment groups completed!"
echo "Finished at $(date)"
echo "========================================="
