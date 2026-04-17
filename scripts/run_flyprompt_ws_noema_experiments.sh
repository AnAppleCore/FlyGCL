#!/bin/bash
# FlyPrompt + WS Router + NO EMA ensemble experiments
# Runs sequentially on GPU 2.

set -e

GPU_ID=2
export CUDA_VISIBLE_DEVICES=$GPU_ID
export MASTER_PORT=$(($RANDOM+32769))
export WORLD_SIZE=1

cd /home/hongwei/FlyGCL

PYTHON=/home/hongwei/miniconda3/envs/DGIL/bin/python

COMMON_ARGS="--seeds 1 --method flyprompt --n_tasks 5 --router_type ws --ws_k 32 \
--no_ema_ensemble \
--batchsize 64 --n_worker 8 --opt_name adam --lr 0.005 --num_epochs 1 --online_iter 3 \
--transforms autoaug --sched_name default --topk 1 --eval_period 1000 \
--rnd_NM --use_amp --n 50 --m 10"

echo "============================================"
echo "FlyPrompt + WS Router + NO EMA Experiments"
echo "GPU: $GPU_ID"
echo "Started at: $(date)"
echo "============================================"

echo ""
echo "============================================"
echo "[1/3] CIFAR-100"
echo "============================================"
NOTE="flyprompt_vit_base_patch16_224_cifar100_flyprompt_ws_noema"
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
NOTE="flyprompt_vit_base_patch16_224_imagenet-r_flyprompt_ws_noema"
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
NOTE="flyprompt_vit_base_patch16_224_cub200_flyprompt_ws_noema"
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
