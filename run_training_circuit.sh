#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
  set -a
  source "${PROJECT_ROOT}/.env"
  set +a
fi
cd "${PROJECT_ROOT}"

: "${CIRCUIT_DATA_ROOT:?Set CIRCUIT_DATA_ROOT in .env or the environment}"

export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.cache/huggingface}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}"
TASK_NAME="${TASK_NAME:-zebra-cot}"
EPOCHS="${EPOCHS:-15}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
LATENT_SIZE="${LATENT_SIZE:-4}"
CE_WEIGHT="${CE_WEIGHT:-1.0}"
SIM_WEIGHT="${SIM_WEIGHT:-0.6}"
WARM_UP_STEPS="${WARM_UP_STEPS:-100}"
SAVE_STEPS="${SAVE_STEPS:-200}"
DATA_PATH="${DATA_PATH:-${CIRCUIT_DATA_ROOT}/combine/merged_5000_rl.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs}"
SAVE_MODEL_PATH="${SAVE_MODEL_PATH:-${OUTPUT_ROOT}/circuit_baseline}"
LOG_FILE="${LOG_FILE:-${OUTPUT_ROOT}/logs/circuit_baseline.log}"
MASTER_PORT="${MASTER_PORT:-29501}"

if [[ ! -f "${DATA_PATH}" ]]; then
  echo "Training JSONL not found: ${DATA_PATH}" >&2
  echo "Set DATA_PATH and CIRCUIT_DATA_ROOT in .env." >&2
  exit 1
fi

mkdir -p "${SAVE_MODEL_PATH}" "$(dirname "${LOG_FILE}")" "${HF_HOME}"

NUM_PROCESSES="${NUM_PROCESSES:-$(python - <<'PY'
try:
    import torch
    print(torch.cuda.device_count() or 1)
except Exception:
    print(1)
PY
)}"

echo "Launching Circuit-MLLM training with ${NUM_PROCESSES} process(es)"
accelerate launch \
  --num_processes "${NUM_PROCESSES}" \
  --main_process_port "${MASTER_PORT}" \
  src/main.py \
  --model "${MODEL_NAME}" \
  --epochs "${EPOCHS}" \
  --task "${TASK_NAME}" \
  --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
  --stage stage1 \
  --warm_up_steps "${WARM_UP_STEPS}" \
  --data_path "${DATA_PATH}" \
  --log_file "${LOG_FILE}" \
  --latent_size "${LATENT_SIZE}" \
  --ce_weight "${CE_WEIGHT}" \
  --sim_weight "${SIM_WEIGHT}" \
  --save_model_path "${SAVE_MODEL_PATH}" \
  --cache_dir "${HF_HOME}" \
  --save_steps "${SAVE_STEPS}"

echo "Training finished. Model saved to ${SAVE_MODEL_PATH}"
