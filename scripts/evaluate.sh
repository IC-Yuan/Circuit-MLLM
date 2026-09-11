#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
  set -a
  source "${PROJECT_ROOT}/.env"
  set +a
fi

TASK="${1:?Usage: scripts/evaluate.sh TASK MODEL_DIR INPUT_DIR IMAGE_ROOT [GPU_ID]}"
MODEL_DIR="${2:?Missing MODEL_DIR}"
INPUT_DIR="${3:?Missing INPUT_DIR}"
IMAGE_ROOT="${4:?Missing IMAGE_ROOT}"
GPU_ID="${5:-${CUDA_VISIBLE_DEVICES:-0}}"

case "${TASK}" in
  connection_identification) SCRIPT="Connection_Identification_eval.py" ;;
  connection_identification_ours) SCRIPT="Connection_Identification_eval_ours.py" ;;
  connection_judge) SCRIPT="Connection_judge_eval.py" ;;
  connection_judge_ours) SCRIPT="Connection_judge_eval_ours.py" ;;
  element_classification) SCRIPT="Element_Classification_eval.py" ;;
  element_classification_ours) SCRIPT="Element_Classification_eval_ours.py" ;;
  total_counting) SCRIPT="Total_Counting_eval.py" ;;
  total_counting_ours) SCRIPT="Total_Counting_eval_ours.py" ;;
  type_wise_counting) SCRIPT="Type_wise_Counting_eval.py" ;;
  type_wise_counting_ours) SCRIPT="Type_wise_Counting_eval_ours.py" ;;
  *) echo "Unknown task: ${TASK}" >&2; exit 2 ;;
esac

if [[ ! -d "${MODEL_DIR}" || ! -d "${INPUT_DIR}" || ! -d "${IMAGE_ROOT}" ]]; then
  echo "MODEL_DIR, INPUT_DIR and IMAGE_ROOT must all exist." >&2
  exit 1
fi

OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/evaluation/${TASK}}"
CACHE_DIR="${CACHE_DIR:-${HF_HOME:-${PROJECT_ROOT}/.cache/huggingface}}"
mkdir -p "${OUTPUT_DIR}" "${CACHE_DIR}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export CIRCUIT_EVAL_IMAGE_ROOT="${IMAGE_ROOT}"
export BASE_MODEL_ID="${BASE_MODEL_ID:-Qwen/Qwen2.5-VL-7B-Instruct}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

shopt -s nullglob
files=("${INPUT_DIR}"/*.jsonl)
if [[ ${#files[@]} -eq 0 ]]; then
  echo "No JSONL files found in ${INPUT_DIR}" >&2
  exit 1
fi

for input_file in "${files[@]}"; do
  name="$(basename "${input_file}" .jsonl)"
  python "${PROJECT_ROOT}/benchresults/${SCRIPT}" \
    --model_dir "${MODEL_DIR}" \
    --test_data_path "${input_file}" \
    --task_name "zebra-cot" \
    --output_json_path "${OUTPUT_DIR}/${name}_results.jsonl" \
    --cache_dir "${CACHE_DIR}" \
    --max_new_tokens "${MAX_NEW_TOKENS:-4096}" \
    --temperature "${TEMPERATURE:-0.0}" \
    --top_p "${TOP_P:-1.0}"
done

echo "Evaluation results written to ${OUTPUT_DIR}"
