#!/bin/bash
# --------------------------------------------------------------
# Backbone-specific MISA sub pretraining launcher.
#
# Runs 5 backbones x 3 expert methods with four single-GPU jobs in parallel.
# Only negative/sub perturbation is trained, matching the downstream MISA
# configurations in run.sh.
#
# Defaults use GPUs 0 1 2 4. Inspect commands without launching:
#   DRY_RUN=1 bash scripts/run_backbone_specific_misa_sub_pretrain_4gpu.sh
# --------------------------------------------------------------
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/hongwei/FlyGCL}
PYTHON=${PYTHON:-/home/hongwei/miniconda3/envs/DGIL/bin/python}
IMAGENET_ROOT=${IMAGENET_ROOT:-/data/datasets/ImageNet}
DRY_RUN=${DRY_RUN:-0}

GPU_LIST=${GPU_LIST:-"0 1 2 4"}
BACKBONES=${BACKBONES:-"vit_base_patch16_224_mepo_21k_1k vit_base_patch16_224_21k_ibot vit_base_patch16_224_ibot vit_base_patch16_224_dino vit_base_patch16_224_mocov3"}
METHODS=${METHODS:-"flyprompt flyadapter flylora"}

BASE_LOG_DIR=${BASE_LOG_DIR:-${PROJECT_DIR}/results/logs/misa_bbspec_pretrain}
BASE_CKPT_DIR=${BASE_CKPT_DIR:-${PROJECT_DIR}/checkpoints/MISA_BBSub}
SCREEN_PREFIX=${SCREEN_PREFIX:-misa_bbspec_sub}
SKIP_COMPLETED=${SKIP_COMPLETED:-1}

SEED=${SEED:-1}
EPOCHS=${EPOCHS:-32}
LR=${LR:-1e-4}
RHO=${RHO:-0.1}
NUM_WORKERS=${NUM_WORKERS:-8}
NUM_INIT_CLASSES=${NUM_INIT_CLASSES:-1000}
NUM_ID_CLASSES=${NUM_ID_CLASSES:-900}
OOD_ACTIVE_CLASSES=${OOD_ACTIVE_CLASSES:-10}
IMAGE_SIZE=${IMAGE_SIZE:-224}
BATCH_SIZE=${BATCH_SIZE:-128}
OOD_BATCH_SIZE=${OOD_BATCH_SIZE:-32}
LOG_INTERVAL=${LOG_INTERVAL:-50}

LEN_PROMPT=${LEN_PROMPT:-20}
POS_PROMPT=${POS_PROMPT:-"0 1 2 3 4"}
FLY_ADAPTER_DOWN_DIM=${FLY_ADAPTER_DOWN_DIM:-10}
FLY_ADAPTER_LAYERS=${FLY_ADAPTER_LAYERS:-5}
FLY_LORA_RANK=${FLY_LORA_RANK:-5}
FLY_LORA_ALPHA=${FLY_LORA_ALPHA:-1.0}
FLY_LORA_LAYERS=${FLY_LORA_LAYERS:-5}

MASTER_PORT_BASE=${MASTER_PORT_BASE:-29910}
POLL_SECONDS=${POLL_SECONDS:-300}

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

mkdir -p "${BASE_LOG_DIR}" "${BASE_CKPT_DIR}"

method_tag() {
  case "$1" in
    flyprompt)  echo "fp" ;;
    flyadapter) echo "fa" ;;
    flylora)    echo "fl" ;;
    *) echo "$1" ;;
  esac
}

backbone_tag() {
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

expert_type_for() {
  case "$1" in
    flyprompt)  echo "prompt" ;;
    flyadapter) echo "adapter" ;;
    flylora)    echo "lora" ;;
    *) echo "" ;;
  esac
}

ckpt_filename_for() {
  local method=$1
  case "$method" in
    flyprompt)  echo "fp_sub_s${SEED}.pt" ;;
    flyadapter) echo "fa_sub_s${SEED}.pt" ;;
    flylora)    echo "fl_sub_s${SEED}.pt" ;;
    *) echo "" ;;
  esac
}

method_extra_args_for() {
  case "$1" in
    flyprompt)
      echo "--len_prompt ${LEN_PROMPT} --pos_prompt ${POS_PROMPT}"
      ;;
    flyadapter)
      echo "--fly_adapter_down_dim ${FLY_ADAPTER_DOWN_DIM} --fly_adapter_layers ${FLY_ADAPTER_LAYERS}"
      ;;
    flylora)
      echo "--fly_lora_rank ${FLY_LORA_RANK} --fly_lora_alpha ${FLY_LORA_ALPHA} --fly_lora_layers ${FLY_LORA_LAYERS}"
      ;;
    *)
      echo ""
      ;;
  esac
}

is_screen_running() {
  local screen_name=$1
  screen -list | grep -q "[.]${screen_name}[[:space:]]"
}

count_running_jobs() {
  local count=0
  local screen_name
  for screen_name in "${RUNNING_SCREENS[@]}"; do
    if is_screen_running "${screen_name}"; then
      count=$((count + 1))
    fi
  done
  echo "$count"
}

cleanup_finished_jobs() {
  local kept=()
  local screen_name
  for screen_name in "${RUNNING_SCREENS[@]}"; do
    if is_screen_running "${screen_name}"; then
      kept+=("${screen_name}")
    fi
  done
  RUNNING_SCREENS=("${kept[@]}")
}

wait_for_slot() {
  while [ "$(count_running_jobs)" -ge "${#GPUS[@]}" ]; do
    echo "$(date): $(count_running_jobs) MISA pretrain jobs still running; waiting for a free GPU..."
    sleep "${POLL_SECONDS}"
    cleanup_finished_jobs
  done
}

next_free_gpu() {
  local gpu
  for gpu in "${GPUS[@]}"; do
    local busy=0
    local screen_name
    for screen_name in "${RUNNING_SCREENS[@]}"; do
      if [[ "${SCREEN_TO_GPU[$screen_name]}" = "$gpu" ]] && is_screen_running "${screen_name}"; then
        busy=1
        break
      fi
    done
    if [ "$busy" -eq 0 ]; then
      echo "$gpu"
      return 0
    fi
  done
  return 1
}

start_job() {
  local method=$1
  local backbone=$2
  local gpu=$3
  local job_index=$4
  local expert_type
  local ckpt_name
  local ckpt_dir
  local ckpt_file
  local log_dir
  local log_file
  local screen_name
  local port
  local extra_args
  local train_cmd

  expert_type=$(expert_type_for "$method")
  ckpt_name=$(ckpt_filename_for "$method")
  if [ -z "$expert_type" ] || [ -z "$ckpt_name" ]; then
    echo "Unsupported method: $method" >&2
    exit 1
  fi

  local mt
  local bt
  local job_tag
  mt=$(method_tag "$method")
  bt=$(backbone_tag "$backbone")
  job_tag="${mt}_${bt}"

  ckpt_dir="${BASE_CKPT_DIR}/${job_tag}"
  ckpt_file="${ckpt_dir}/${ckpt_name}"
  log_dir="${BASE_LOG_DIR}"
  log_file="${log_dir}/${job_tag}.log"
  screen_name="${SCREEN_PREFIX}_${job_tag}"
  port=$((MASTER_PORT_BASE + job_index))
  extra_args=$(method_extra_args_for "$method")

  mkdir -p "${ckpt_dir}" "${log_dir}"

  if [ "${SKIP_COMPLETED}" = "1" ] && [ -s "${ckpt_dir}/epoch_022/${ckpt_name}" ] && [ -s "${ckpt_dir}/epoch_025/${ckpt_name}" ]; then
    echo "[SKIP] existing epoch_022 and epoch_025 checkpoints for ${method} ${backbone}"
    return 0
  fi

  if is_screen_running "${screen_name}"; then
    echo "screen ${screen_name} already exists" >&2
    exit 1
  fi

  train_cmd="set -euo pipefail; cd '${PROJECT_DIR}'; CUDA_VISIBLE_DEVICES='${gpu}' '${PYTHON}' -m torch.distributed.run --nproc_per_node 1 --master_port ${port} pretrain_flyprompt_misa.py --expert_type '${expert_type}' --imagenet_root '${IMAGENET_ROOT}' --backbone '${backbone}' --num_init_classes ${NUM_INIT_CLASSES} --num_id_classes ${NUM_ID_CLASSES} --ood_active_classes ${OOD_ACTIVE_CLASSES} ${extra_args} --image_size ${IMAGE_SIZE} --batch_size ${BATCH_SIZE} --ood_batch_size ${OOD_BATCH_SIZE} --epochs ${EPOCHS} --lr ${LR} --rho ${RHO} --weight_decay 0.0 --num_workers ${NUM_WORKERS} --seed ${SEED} --device cuda --log_interval ${LOG_INTERVAL} --save_every_epoch --save_path '${ckpt_file}' > '${log_file}' 2>&1"

  if [ "${DRY_RUN}" = "1" ]; then
    echo "[DRY_RUN] screen=${screen_name} gpu=${gpu} port=${port}"
    echo "[DRY_RUN] log=${log_file}"
    echo "[DRY_RUN] ckpt=${ckpt_file}"
    echo "[DRY_RUN] command=${train_cmd}"
    return 0
  fi

  screen -dmS "${screen_name}" bash -lc "${train_cmd}"
  RUNNING_SCREENS+=("${screen_name}")
  SCREEN_TO_GPU["${screen_name}"]="${gpu}"

  echo "started ${screen_name} | method=${method} | backbone=${backbone} | GPU=${gpu}"
  echo "log: ${log_file}"
  echo "ckpt: ${ckpt_file}"
}

read -ra GPUS <<< "${GPU_LIST}"
if [ "${#GPUS[@]}" -eq 0 ]; then
  echo "GPU_LIST is empty" >&2
  exit 1
fi

declare -a RUNNING_SCREENS=()
declare -A SCREEN_TO_GPU=()

JOB_INDEX=0

echo "========================================="
echo "Backbone-specific MISA sub pretraining"
echo "Backbones: ${BACKBONES}"
echo "Methods:   ${METHODS}"
echo "GPUs:      ${GPU_LIST}"
echo "CKPT root: ${BASE_CKPT_DIR}"
echo "LOG root:  ${BASE_LOG_DIR}"
echo "DRY_RUN:   ${DRY_RUN}"
echo "========================================="

for backbone in ${BACKBONES}; do
  for method in ${METHODS}; do
    wait_for_slot
    cleanup_finished_jobs
    gpu=$(next_free_gpu)
    if [ -z "${gpu:-}" ]; then
      echo "No free GPU found after wait_for_slot" >&2
      exit 1
    fi
    start_job "${method}" "${backbone}" "${gpu}" "${JOB_INDEX}"
    JOB_INDEX=$((JOB_INDEX + 1))
    sleep 2
  done
done

if [ "${DRY_RUN}" = "1" ]; then
  echo "DRY_RUN complete. No jobs launched."
  exit 0
fi

while [ "$(count_running_jobs)" -gt 0 ]; do
  echo "$(date): $(count_running_jobs) MISA pretrain jobs still running..."
  sleep "${POLL_SECONDS}"
  cleanup_finished_jobs
done

echo "========================================="
echo "All backbone-specific MISA sub pretraining jobs completed."
echo "Finished at $(date)"
echo "========================================="
