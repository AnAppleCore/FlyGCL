#!/bin/bash
set -euo pipefail

PROJECT_DIR=/home/yanhongwei/FlyGCL
BASELINE_SCRIPT=${PROJECT_DIR}/scripts/run_baselines_flyprompt.sh
CKPT_ROOT=${PROJECT_DIR}/checkpoints/FlyPrompt_MISA_Pretrain_Prompt
MANAGER_LOG=${PROJECT_DIR}/results/logs/flyprompt_misa_epoch_sweep_cub200.log

BACKBONE=vit_base_patch16_224
DATASET=cub200
GPU_LIST=(0 1 2 3 4)
SEED_LIST=(1 2 3 4 5)
EPOCHS=$(seq 1 32)
SKIP_EPOCH=3

mkdir -p "$(dirname "${MANAGER_LOG}")"

log_msg() {
  echo "$(date '+%F %T') | $*" | tee -a "${MANAGER_LOG}"
}

session_exists() {
  local session_name=$1
  screen -list | grep -q "[.]${session_name}[[:space:]]"
}

wait_for_pattern_done() {
  local pattern=$1
  while true; do
    local count
    count=$(screen -list | grep -c "${pattern}" || true)
    if [ "${count}" -eq 0 ]; then
      log_msg "completed group pattern=${pattern}"
      break
    fi
    log_msg "waiting pattern=${pattern} running=${count}"
    sleep 120
  done
}

run_direction_epoch() {
  local direction=$1
  local epoch=$2
  local epoch_tag
  printf -v epoch_tag "%03d" "${epoch}"

  local ckpt_path="${CKPT_ROOT}/flyprompt_misa_${direction}_ddp_bs256_ep32_seed1/epoch_${epoch_tag}/flyprompt_misa_prompt_${direction}_ddp_bs256_ep32_seed1.pt"
  if [ ! -f "${ckpt_path}" ]; then
    log_msg "missing checkpoint direction=${direction} epoch=${epoch_tag} path=${ckpt_path}"
    exit 1
  fi

  local group="flyprompt_${BACKBONE}_${DATASET}_misa_${direction}_ep${epoch_tag}_"
  log_msg "starting direction=${direction} epoch=${epoch_tag} checkpoint=${ckpt_path}"

  for i in "${!SEED_LIST[@]}"; do
    local seed=${SEED_LIST[$i]}
    local gpu=${GPU_LIST[$i]}
    local session_name="${group}${seed}"

    if session_exists "${session_name}"; then
      log_msg "screen ${session_name} already exists"
      exit 1
    fi

    log_msg "start session=${session_name} gpu=${gpu} seed=${seed}"
    screen -dmS "${session_name}" bash -lc "
      set -euo pipefail
      cd '${PROJECT_DIR}'
      bash '${BASELINE_SCRIPT}' '${gpu}' '${seed}' '${DATASET}' 'misa_${direction}_ep${epoch_tag}' \
        --backbone '${BACKBONE}' \
        --load_pt \
        --flyprompt_pt_path '${ckpt_path}'
    "
    sleep 2
  done
}

log_msg "FlyPrompt MISA cub200 epoch sweep started"
log_msg "epochs=1..32 skip=${SKIP_EPOCH} directions=sub,add seeds=${SEED_LIST[*]} gpus=${GPU_LIST[*]}"

for epoch in ${EPOCHS}; do
  if [ "${epoch}" -eq "${SKIP_EPOCH}" ]; then
    log_msg "skip epoch=$(printf '%03d' "${epoch}") because epoch 3 downstream tests already exist"
    continue
  fi

  printf -v epoch_tag "%03d" "${epoch}"
  log_msg "epoch ${epoch_tag} begin"

  run_direction_epoch sub "${epoch}"
  run_direction_epoch add "${epoch}"

  wait_for_pattern_done "flyprompt_${BACKBONE}_${DATASET}_misa_sub_ep${epoch_tag}_"
  wait_for_pattern_done "flyprompt_${BACKBONE}_${DATASET}_misa_add_ep${epoch_tag}_"

  log_msg "epoch ${epoch_tag} done"
done

log_msg "FlyPrompt MISA cub200 epoch sweep finished"
