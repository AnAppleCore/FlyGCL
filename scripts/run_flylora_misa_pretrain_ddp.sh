#!/bin/bash
# --------------------------------------------------------------
# FlyLoRA MISA-style LoRA DDP pretraining launcher on ImageNet-1k.
#
# Default task layout follows the FlyPrompt MISA pretraining launcher:
#   - sub: negative/sub OOD-gradient perturbation on GPUs 1,2
#   - add: positive/add OOD-gradient perturbation on GPUs 3,4
#
# Override paths or hyperparameters via environment variables, e.g.:
#   IMAGENET_ROOT=/data/datasets/ImageNet EPOCHS=32 bash scripts/run_flylora_misa_pretrain_ddp.sh
# Inspect commands without launching screen jobs:
#   DRY_RUN=1 bash scripts/run_flylora_misa_pretrain_ddp.sh
# --------------------------------------------------------------
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/hongwei/FlyGCL}
PYTHON=${PYTHON:-/home/hongwei/miniconda3/envs/DGIL/bin/python}
IMAGENET_ROOT=${IMAGENET_ROOT:-/data/datasets/ImageNet}
DRY_RUN=${DRY_RUN:-0}

LOG_DIR=${LOG_DIR:-${PROJECT_DIR}/results/logs/flylora_misa_pretrain_ddp}
CKPT_DIR=${CKPT_DIR:-${PROJECT_DIR}/checkpoints/FlyLoRA_MISA_Pretrain_LoRA}

SEED=${SEED:-1}
EPOCHS=${EPOCHS:-32}
LR=${LR:-1e-4}
RHO=${RHO:-0.1}
NUM_WORKERS=${NUM_WORKERS:-8}
BACKBONE=${BACKBONE:-vit_base_patch16_224}
NUM_INIT_CLASSES=${NUM_INIT_CLASSES:-1000}
NUM_ID_CLASSES=${NUM_ID_CLASSES:-900}
OOD_ACTIVE_CLASSES=${OOD_ACTIVE_CLASSES:-10}
FLY_LORA_RANK=${FLY_LORA_RANK:-5}
FLY_LORA_ALPHA=${FLY_LORA_ALPHA:-1.0}
FLY_LORA_LAYERS=${FLY_LORA_LAYERS:-5}

NPROC_PER_NODE=${NPROC_PER_NODE:-2}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-256}
PER_RANK_BATCH_SIZE=${PER_RANK_BATCH_SIZE:-128}
GLOBAL_OOD_BATCH_SIZE=${GLOBAL_OOD_BATCH_SIZE:-64}
PER_RANK_OOD_BATCH_SIZE=${PER_RANK_OOD_BATCH_SIZE:-32}
SUB_GPUS=${SUB_GPUS:-1,2}
ADD_GPUS=${ADD_GPUS:-3,4}
SUB_PORT=${SUB_PORT:-29781}
ADD_PORT=${ADD_PORT:-29782}
RUN_SUB=${RUN_SUB:-1}
RUN_ADD=${RUN_ADD:-1}

mkdir -p "${LOG_DIR}" "${CKPT_DIR}"

if [ ! -d "${PROJECT_DIR}" ]; then
  echo "PROJECT_DIR does not exist: ${PROJECT_DIR}" >&2
  exit 1
fi

if [ ! -x "${PYTHON}" ]; then
  echo "PYTHON is not executable: ${PYTHON}" >&2
  exit 1
fi

if [ ! -d "${IMAGENET_ROOT}" ]; then
  echo "IMAGENET_ROOT does not exist: ${IMAGENET_ROOT}" >&2
  exit 1
fi

if [ ! -d "${IMAGENET_ROOT}/train" ] && [ ! -d "${IMAGENET_ROOT}/n01440764" ]; then
  echo "IMAGENET_ROOT must be an ImageNet root containing train/ or an ImageFolder train directory: ${IMAGENET_ROOT}" >&2
  exit 1
fi

run_one() {
  local name=$1
  local gpus=$2
  local port=$3
  local extra_flag=$4
  local screen_name="flylora_misa_${name}"
  local exp_name="flylora_misa_${name}_ddp_bs${GLOBAL_BATCH_SIZE}_ep${EPOCHS}_seed${SEED}_rank${FLY_LORA_RANK}_layers${FLY_LORA_LAYERS}"
  local log_file="${LOG_DIR}/${exp_name}.log"
  local ckpt_dir="${CKPT_DIR}/${exp_name}"
  local ckpt_file="${ckpt_dir}/flylora_misa_lora_${name}_ddp_bs${GLOBAL_BATCH_SIZE}_ep${EPOCHS}_seed${SEED}.pt"
  local train_cmd

  mkdir -p "${ckpt_dir}"

  train_cmd="set -euo pipefail; cd '${PROJECT_DIR}'; CUDA_VISIBLE_DEVICES='${gpus}' '${PYTHON}' -m torch.distributed.run --nproc_per_node ${NPROC_PER_NODE} --master_port ${port} pretrain_flyprompt_misa.py --expert_type lora --imagenet_root '${IMAGENET_ROOT}' --backbone '${BACKBONE}' --num_init_classes ${NUM_INIT_CLASSES} --num_id_classes ${NUM_ID_CLASSES} --ood_active_classes ${OOD_ACTIVE_CLASSES} --fly_lora_rank ${FLY_LORA_RANK} --fly_lora_alpha ${FLY_LORA_ALPHA} --fly_lora_layers ${FLY_LORA_LAYERS} --image_size 224 --batch_size ${PER_RANK_BATCH_SIZE} --ood_batch_size ${PER_RANK_OOD_BATCH_SIZE} --epochs ${EPOCHS} --lr ${LR} --rho ${RHO} --weight_decay 0.0 --num_workers ${NUM_WORKERS} --seed ${SEED} --device cuda --log_interval 50 --save_every_epoch ${extra_flag} --save_path '${ckpt_file}' > '${log_file}' 2>&1"

  if [ "${DRY_RUN}" = "1" ]; then
    echo "[dry-run] screen: ${screen_name}"
    echo "[dry-run] GPUs: ${gpus}"
    echo "[dry-run] log: ${log_file}"
    echo "[dry-run] ckpt: ${ckpt_file}"
    echo "[dry-run] command: ${train_cmd}"
    return
  fi

  if screen -list | grep -q "[.]${screen_name}[[:space:]]"; then
    echo "screen ${screen_name} already exists" >&2
    exit 1
  fi

  screen -dmS "${screen_name}" bash -lc "${train_cmd}"

  echo "started ${screen_name} on GPUs ${gpus}"
  echo "log: ${log_file}"
  echo "ckpt: ${ckpt_file}"
}

if [ "${RUN_SUB}" = "1" ]; then
  run_one sub "${SUB_GPUS}" "${SUB_PORT}" ""
fi

if [ "${RUN_ADD}" = "1" ]; then
  run_one add "${ADD_GPUS}" "${ADD_PORT}" "--fam_perturb_add"
fi
