#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
cd "$REPO_ROOT"
exec "${PYTHON_BIN:-python}" main.py \
  --method slda --backbone vit_base_patch16_224 \
  --dataset cifar100 --data_dir "${DATA_DIR:-./data/CIFAR}" \
  --n_tasks 5 --n 50 --m 10 --rnd_NM \
  --batchsize 64 --online_iter 1 --num_epochs 1 \
  --eval_period 1000 --transforms autoaug \
  --seeds 1 --note slda_cifar100 "$@"
