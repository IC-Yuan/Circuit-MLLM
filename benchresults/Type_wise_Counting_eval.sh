#!/bin/bash
set -e  # 任何命令失败即退出

# ------------------------------
# 默认配置（可通过命令行覆盖）
# ------------------------------
DEFAULT_MODEL_DIR="/data/jydeng/latent_visual/circuit_mllm/circuit/output/02_16_sequence_06/checkpoint-1200"
DEFAULT_INPUT_DIR="/data/jydeng/circuit_clip/amsbench/AMSBench/Type_wise_Counting_Task"
DEFAULT_OUTPUT_DIR="/data/jydeng/latent_visual/circuit_mllm/benchresults/report/02_16_sequence_06_step_1200"
DEFAULT_EVAL_SCRIPT="/data/jydeng/latent_visual/circuit_mllm/benchresults/Type_wise_Counting_eval.py"
DEFAULT_CACHE_DIR="/data/jydeng/hugging_face_project"
DEFAULT_GPU_ID="2"
DEFAULT_TASK_NAME="zebra-cot"          # 非 vsp 任务走多标签字母评估分支
DEFAULT_MAX_NEW_TOKENS=4096
DEFAULT_TEMPERATURE=0.0
DEFAULT_TOP_P=1.0

# ------------------------------
# 显示帮助信息
# ------------------------------
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -m MODEL_DIR      Path to model directory (default: $DEFAULT_MODEL_DIR)"
    echo "  -i INPUT_DIR      Path to input JSONL directory (default: $DEFAULT_INPUT_DIR)"
    echo "  -o OUTPUT_DIR     Path to output directory (default: $DEFAULT_OUTPUT_DIR)"
    echo "  -g GPU_ID         GPU device ID (default: $DEFAULT_GPU_ID)"
    echo "  -t TASK_NAME      Task name (default: $DEFAULT_TASK_NAME)"
    echo "  -s EVAL_SCRIPT    Path to evaluation script (default: $DEFAULT_EVAL_SCRIPT)"
    echo "  -c CACHE_DIR      HuggingFace cache dir (default: $DEFAULT_CACHE_DIR)"
    echo "  -h                Show this help"
    exit 1
}

# ------------------------------
# 解析命令行参数
# ------------------------------
MODEL_DIR="$DEFAULT_MODEL_DIR"
INPUT_DIR="$DEFAULT_INPUT_DIR"
OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
GPU_ID="$DEFAULT_GPU_ID"
TASK_NAME="$DEFAULT_TASK_NAME"
EVAL_SCRIPT="$DEFAULT_EVAL_SCRIPT"
CACHE_DIR="$DEFAULT_CACHE_DIR"
MAX_NEW_TOKENS="$DEFAULT_MAX_NEW_TOKENS"
TEMPERATURE="$DEFAULT_TEMPERATURE"
TOP_P="$DEFAULT_TOP_P"

while getopts "m:i:o:g:t:s:c:h" opt; do
    case "$opt" in
        m) MODEL_DIR="$OPTARG" ;;
        i) INPUT_DIR="$OPTARG" ;;
        o) OUTPUT_DIR="$OPTARG" ;;
        g) GPU_ID="$OPTARG" ;;
        t) TASK_NAME="$OPTARG" ;;
        s) EVAL_SCRIPT="$OPTARG" ;;
        c) CACHE_DIR="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

# ------------------------------
# 根据任务类型调整生成参数
# ------------------------------
if [[ "$TASK_NAME" == "vsp-spatial-planning-cot" ]]; then
    MAX_NEW_TOKENS=256      # 空间规划任务不需要太长输出
else
    MAX_NEW_TOKENS=4096     # 连接识别任务需要较长 CoT
fi

# ------------------------------
# 环境与路径检查
# ------------------------------
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES="$GPU_ID"

if [ ! -d "$MODEL_DIR" ]; then
    echo "❌ Error: Model directory not found: $MODEL_DIR"
    exit 1
fi
if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ Error: Input directory not found: $INPUT_DIR"
    exit 1
fi
if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "❌ Error: Evaluation script not found: $EVAL_SCRIPT"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# ------------------------------
# 日志记录（同时输出到终端和文件）
# ------------------------------
LOG_FILE="$OUTPUT_DIR/evaluation.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "========================================================"
echo "🚀 Starting evaluation at $(date)"
echo "   Model dir      : $MODEL_DIR"
echo "   Input dir      : $INPUT_DIR"
echo "   Output dir     : $OUTPUT_DIR"
echo "   Task name      : $TASK_NAME"
echo "   GPU ID         : $GPU_ID"
echo "   Max new tokens : $MAX_NEW_TOKENS"
echo "========================================================"

# ------------------------------
# 遍历输入目录中的所有 .jsonl 文件
# ------------------------------
for input_file in "$INPUT_DIR"/*.jsonl; do
    [ -e "$input_file" ] || continue

    filename=$(basename "$input_file")
    output_filename="${filename%.jsonl}_results.jsonl"
    output_file="$OUTPUT_DIR/$output_filename"

    echo "--------------------------------------------------------"
    echo "📄 Evaluating: $filename"
    echo "   → Output: $output_file"
    echo "--------------------------------------------------------"

    python "$EVAL_SCRIPT" \
        --model_dir "$MODEL_DIR" \
        --test_data_path "$input_file" \
        --task_name "$TASK_NAME" \
        --output_json_path "$output_file" \
        --cache_dir "$CACHE_DIR" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --temperature "$TEMPERATURE" \
        --top_p "$TOP_P"

    echo "✅ Finished: $filename"
done

# ------------------------------
# 汇总所有结果文件（提取 summary 行）
# ------------------------------
SUMMARY_FILE="$OUTPUT_DIR/all_results_summary.json"
echo "[" > "$SUMMARY_FILE"
first=1
for result_file in "$OUTPUT_DIR"/*_results.jsonl; do
    if [ -f "$result_file" ]; then
        # 提取最后一行（summary）
        summary=$(tail -n 1 "$result_file")
        if [ "$first" -eq 1 ]; then
            first=0
        else
            echo "," >> "$SUMMARY_FILE"
        fi
        echo "  $summary" >> "$SUMMARY_FILE"
    fi
done
echo "]" >> "$SUMMARY_FILE"

echo "========================================================"
echo "🎉 All evaluations completed!"
echo "   Results saved in: $OUTPUT_DIR"
echo "   Summary file    : $SUMMARY_FILE"
echo "   Log file        : $LOG_FILE"
echo "========================================================"