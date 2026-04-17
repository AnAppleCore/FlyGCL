#!/bin/bash
# FlyPrompt + Whitened Subspace Router experiments
# Waits until GPU 2 is idle for 10 consecutive minutes, then runs sequentially.

set -e

GPU_ID=2
export CUDA_VISIBLE_DEVICES=$GPU_ID
export MASTER_PORT=$(($RANDOM+32769))
export WORLD_SIZE=1

IDLE_REQUIRED=600   # seconds of consecutive GPU idleness required
POLL_INTERVAL=30    # check every 30 seconds

cd /home/hongwei/FlyGCL

echo "============================================"
echo "FlyPrompt + WS Router Experiments"
echo "GPU: $GPU_ID"
echo "Will start after GPU $GPU_ID is idle for ${IDLE_REQUIRED}s"
echo "Started at: $(date)"
echo "============================================"

gpu_has_processes() {
    local count
    count=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $GPU_ID 2>/dev/null | sed '/^$/d' | wc -l)
    [ "$count" -gt 0 ]
}

idle_seconds=0
if gpu_has_processes; then
    echo "[$(date)] GPU $GPU_ID is busy. Polling every ${POLL_INTERVAL}s..."
    while true; do
        sleep $POLL_INTERVAL
        if gpu_has_processes; then
            idle_seconds=0
        else
            idle_seconds=$((idle_seconds + POLL_INTERVAL))
            echo "[$(date)] GPU $GPU_ID idle for ${idle_seconds}s / ${IDLE_REQUIRED}s"
            if [ "$idle_seconds" -ge "$IDLE_REQUIRED" ]; then
                echo "[$(date)] GPU $GPU_ID confirmed idle. Starting experiments."
                break
            fi
        fi
    done
else
    echo "[$(date)] GPU $GPU_ID is already idle. Waiting ${IDLE_REQUIRED}s to confirm..."
    while [ "$idle_seconds" -lt "$IDLE_REQUIRED" ]; do
        sleep $POLL_INTERVAL
        if gpu_has_processes; then
            idle_seconds=0
            echo "[$(date)] GPU $GPU_ID became busy again. Resetting idle timer."
        else
            idle_seconds=$((idle_seconds + POLL_INTERVAL))
            echo "[$(date)] GPU $GPU_ID idle for ${idle_seconds}s / ${IDLE_REQUIRED}s"
        fi
    done
    echo "[$(date)] GPU $GPU_ID confirmed idle. Starting experiments."
fi

PYTHON=/home/hongwei/miniconda3/envs/DGIL/bin/python

COMMON_ARGS="--seeds 1 --method flyprompt --n_tasks 5 --router_type ws --ws_k 32 \
--batchsize 64 --n_worker 8 --opt_name adam --lr 0.005 --num_epochs 1 --online_iter 3 \
--transforms autoaug --sched_name default --topk 1 --eval_period 1000 \
--rnd_NM --use_amp --n 50 --m 10"

echo ""
echo "============================================"
echo "[1/3] CIFAR-100"
echo "============================================"
NOTE="flyprompt_vit_base_patch16_224_cifar100_flyprompt_ws"
mkdir -p results/logs/cifar100/$NOTE
$PYTHON -W ignore main.py \
    --note $NOTE --log_path ./results \
    --backbone vit_base_patch16_224 \
    --dataset cifar100 --data_dir /data/datasets \
    $COMMON_ARGS \
    2>&1 | tee results/logs/cifar100/${NOTE}/seed_1_log.txt
echo "[$(date)] CIFAR-100 done."

echo ""
echo "============================================"
echo "[2/3] ImageNet-R"
echo "============================================"
NOTE="flyprompt_vit_base_patch16_224_imagenet-r_flyprompt_ws"
mkdir -p "results/logs/imagenet-r/$NOTE"
$PYTHON -W ignore main.py \
    --note $NOTE --log_path ./results \
    --backbone vit_base_patch16_224 \
    --dataset imagenet-r --data_dir /data/datasets/imagenet-r \
    $COMMON_ARGS \
    2>&1 | tee "results/logs/imagenet-r/${NOTE}/seed_1_log.txt"
echo "[$(date)] ImageNet-R done."

echo ""
echo "============================================"
echo "[3/3] CUB-200"
echo "============================================"
NOTE="flyprompt_vit_base_patch16_224_cub200_flyprompt_ws"
mkdir -p results/logs/cub200/$NOTE
$PYTHON -W ignore main.py \
    --note $NOTE --log_path ./results \
    --backbone vit_base_patch16_224 \
    --dataset cub200 --data_dir /data/datasets/CUB_200_2011 \
    $COMMON_ARGS \
    2>&1 | tee results/logs/cub200/${NOTE}/seed_1_log.txt
echo "[$(date)] CUB-200 done."

echo ""
echo "============================================"
echo "All 3 experiments completed at $(date)"
echo "============================================"
