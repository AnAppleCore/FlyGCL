#!/usr/bin/env bash
# MISA ICLR 2025 downstream on FlyGCL shared CIFAR-100 Si-Blurry stream.
# Real training; NOT invoked by automated CPU smoke tests.
set -euo pipefail

# Work regardless of the caller's working directory.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

DATA_DIR="${DATA_DIR:-./data/CIFAR}"
MISA_PROMPT_DIR="${MISA_PROMPT_DIR:-./dongkyu_work/misa/checkpoints}"

for prompt in g_prompt.pt e_prompt.pt; do
  if [[ ! -s "$MISA_PROMPT_DIR/$prompt" ]]; then
    echo "MISA: required official prompt missing or empty: $MISA_PROMPT_DIR/$prompt" >&2
    echo "See dongkyu_work/misa/checkpoints/README.md" >&2
    exit 2
  fi
done

# The pretrained ViT-B/16 is loaded/downloaded by models/vit.py through
# the common FlyGCL backbone pathway. Python 3.10 + timm 0.6.12 recommended.
python main.py \
  --method misa --backbone vit_base_patch16_224 \
  --dataset cifar100 --data_dir "$DATA_DIR" \
  --n_tasks 5 --n 50 --m 10 --rnd_NM \
  --batchsize 64 --online_iter 3 --num_epochs 1 \
  --lr 0.005 --opt_name adam --sched_name default \
  --eval_period 1000 --transforms autoaug \
  --misa_prompt_dir "$MISA_PROMPT_DIR" \
  --seeds 1 --note misa_cifar100
