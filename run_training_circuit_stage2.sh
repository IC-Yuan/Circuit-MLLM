#!/bin/bash
set -euo pipefail

export HF_HOME="/data/jydeng/hugging_face_project/huggingface"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

export HF_HUB_OFFLINE=1

MODEL_NAME="/data/jydeng/LLM/circuit_llm_qwen/Qwen2.5-VL-7B-Instruct"
TASK_NAME="zebra-cot"
EPOCHS=2
GRAD_ACCUM_STEPS=8
LATENT_SIZE=8
CE_WEIGHT=1
WARM_UP_STEPS=50
SAVE_STEPS=2
DATA_PATH="/data/jydeng/circuit_clip/ams/QA_dataset/latent_visual_org_image_sequence/combine/merged_5000_rl.jsonl"
LOAD_MODEL_PATH="/data/jydeng/latent_visual/circuit_mllm/circuit/output/02_11/checkpoint-1000"
SAVE_MODEL_PATH="/data/jydeng/latent_visual/circuit_mllm/circuit/output/02_12_stage2"
LOG_FILE="/data/jydeng/latent_visual/circuit_mllm/logs/circuit/train_02_12_stage2.log"


mkdir -p "$(dirname "$SAVE_MODEL_PATH")" "$(dirname "$LOG_FILE")"


NUM_PROCESSES=${NUM_PROCESSES:-$(python - <<'PY'
try:
    import torch
    print(torch.cuda.device_count() or 1)
except Exception:
    print(1)
PY
)}
echo "Using ${NUM_PROCESSES} processes"

CUDA_VISIBLE_DEVICES="0,1,4,5,6,7" accelerate launch \
  src/main.py \
  --model "${MODEL_NAME}" \
  --epochs "${EPOCHS}" \
  --task "${TASK_NAME}" \
  --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
  --stage stage2 \
  --warm_up_steps "${WARM_UP_STEPS}" \
  --data_path "${DATA_PATH}" \
  --log_file "${LOG_FILE}" \
  --latent_size "${LATENT_SIZE}" \
  --ce_weight "${CE_WEIGHT}" \
  --load_model_path "${LOAD_MODEL_PATH}" \
  --save_model_path "${SAVE_MODEL_PATH}" \
  --cache_dir "${HF_HOME}" \
  --save_steps "${SAVE_STEPS}" \

#--use_lora \
echo "training finished, save model to ${SAVE_MODEL_PATH}"
