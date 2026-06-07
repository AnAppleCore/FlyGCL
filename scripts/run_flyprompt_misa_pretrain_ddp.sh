#!/bin/bash
# --------------------------------------------------------------
# FlyPrompt MISA-style prompt DDP pretraining launcher (placeholder paths).
#
# Before running, please replace the following placeholders to match
# your machine layout:
#   - PROJECT_DIR : absolute path to this FlyGCL repository
#   - PYTHON      : python interpreter inside your conda env (DGIL by default)
#   - IMAGENET_ROOT : path to ImageNet-1k (must contain a 'train/' folder
#                     organised as ImageFolder)
# --------------------------------------------------------------
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/path/to/FlyGCL}
PYTHON=${PYTHON:-python}
IMAGENET_ROOT=${IMAGENET_ROOT:-/path/to/ImageNet}
LOG_DIR=${PROJECT_DIR}/results/logs/flyprompt_misa_pretrain_ddp
CKPT_DIR=${PROJECT_DIR}/checkpoints/FlyPrompt_MISA_Pretrain_Prompt

SEED=1
EPOCHS=32
LR=1e-4
RHO=0.1
NUM_WORKERS=8
BACKBONE=vit_base_patch16_224
LEN_PROMPT=20
POS_PROMPT="0 1 2 3 4"
NUM_INIT_CLASSES=1000
NUM_ID_CLASSES=900
OOD_ACTIVE_CLASSES=10

NPROC_PER_NODE=2
GLOBAL_BATCH_SIZE=256
PER_RANK_BATCH_SIZE=128
GLOBAL_OOD_BATCH_SIZE=64
PER_RANK_OOD_BATCH_SIZE=32

mkdir -p "${LOG_DIR}" "${CKPT_DIR}"

run_one() {
  local name=$1
  local gpus=$2
  local port=$3
  local extra_flag=$4
  local screen_name="flyprompt_misa_${name}"
  local exp_name="flyprompt_misa_${name}_ddp_bs${GLOBAL_BATCH_SIZE}_ep${EPOCHS}_seed${SEED}"
  local log_file="${LOG_DIR}/${exp_name}.log"
  local ckpt_dir="${CKPT_DIR}/${exp_name}"
  local ckpt_file="${ckpt_dir}/flyprompt_misa_prompt_${name}_ddp_bs${GLOBAL_BATCH_SIZE}_ep${EPOCHS}_seed${SEED}.pt"

  mkdir -p "${ckpt_dir}"

  if screen -list | grep -q "[.]${screen_name}[[:space:]]"; then
    echo "screen ${screen_name} already exists"
    exit 1
  fi

  screen -dmS "${screen_name}" bash -lc "
    set -euo pipefail
    cd '${PROJECT_DIR}'
    CUDA_VISIBLE_DEVICES='${gpus}' '${PYTHON}' -m torch.distributed.run \
      --nproc_per_node ${NPROC_PER_NODE} \
      --master_port ${port} \
      pretrain_flyprompt_misa.py \
      --imagenet_root '${IMAGENET_ROOT}' \
      --backbone '${BACKBONE}' \
      --num_init_classes ${NUM_INIT_CLASSES} \
      --num_id_classes ${NUM_ID_CLASSES} \
      --ood_active_classes ${OOD_ACTIVE_CLASSES} \
      --len_prompt ${LEN_PROMPT} \
      --pos_prompt ${POS_PROMPT} \
      --image_size 224 \
      --batch_size ${PER_RANK_BATCH_SIZE} \
      --ood_batch_size ${PER_RANK_OOD_BATCH_SIZE} \
      --epochs ${EPOCHS} \
      --lr ${LR} \
      --rho ${RHO} \
      --weight_decay 0.0 \
      --num_workers ${NUM_WORKERS} \
      --seed ${SEED} \
      --device cuda \
      --log_interval 50 \
      --save_every_epoch \
      ${extra_flag} \
      --save_path '${ckpt_file}' \
      > '${log_file}' 2>&1
  "

  echo "started ${screen_name} on GPUs ${gpus}"
  echo "log: ${log_file}"
  echo "ckpt: ${ckpt_file}"
}

run_one sub 1,2 29661 ""
run_one add 3,4 29662 "--fam_perturb_add"
