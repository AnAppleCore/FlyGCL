#!/usr/bin/env bash
# Official buffer-free SinglePrompt on the shared FlyGCL CIFAR-100 stream.
# Requires real CIFAR-100 data and ViT-B/16 ImageNet-21K pretrained weights.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
DATA_DIR="${DATA_DIR:-./data/CIFAR}"
python main.py \
  --method singleprompt --backbone vit_base_patch16_224 \
  --dataset cifar100 --data_dir "$DATA_DIR" \
  --n_tasks 5 --n 50 --m 10 --rnd_NM \
  --batchsize 64 --online_iter 3 --num_epochs 1 \
  --lr 0.005 --opt_name adam --sched_name default \
  --eval_period 1000 --transforms autoaug \
  --pos_prompt 0 1 2 3 4 --len_prompt 20 \
  --singleprompt_logit_type cos_sim --singleprompt_eval_mode seen \
  --seeds 1 --note singleprompt_cifar100
