#!/bin/bash

export HF_HUB_OFFLINE=1

#MODEL_DIR="/data/jydeng/latent_visual/ILVR/weight"
#MODEL_DIR="/data/jydeng/latent_visual/ILVR/circuit/output/checkpoint-1560"
MODEL_DIR="/data/jydeng/latent_visual/circuit_mllm/circuit/output/02_11/checkpoint-1000"


INPUT_DIR="/data/jydeng/circuit_clip/amsbench/AMSBench/Connection_Identification_Task"
#OUTPUT_DIR="/data/jydeng/latent_visual/ILVR/test-result"
OUTPUT_DIR="/data/jydeng/latent_visual/circuit_mllm/testresult"


EVAL_SCRIPT_PATH="/data/jydeng/latent_visual/circuit_mllm/test.py"
CACHE_DIR="/data/jydeng/hugging_face_project"
TASK_NAME="zebra-cot"
MAX_NEW_TOKENS=4096  
TEMPERATURE=0.0    
TOP_P=1.0


GPU_ID="4"


if [ ! -d "$INPUT_DIR" ]; then
    echo "error: file '$INPUT_DIR' miss¡£"
    exit 1
fi


mkdir -p "$OUTPUT_DIR"
echo "save to: $OUTPUT_DIR"



for input_file in "$INPUT_DIR"/*.jsonl; do

    [ -e "$input_file" ] || continue


    filename=$(basename "$input_file")
    

    output_filename="${filename/_with_prompt.jsonl/_results.jsonl}"
    output_file="$OUTPUT_DIR/$output_filename"

    echo "--------------------------------------------------------"
    echo "eval: $filename"
    echo "output:   $output_file"
    echo "--------------------------------------------------------"

    CUDA_VISIBLE_DEVICES=$GPU_ID python "$EVAL_SCRIPT_PATH" \
        --model_dir "$MODEL_DIR" \
        --test_data_path "$input_file" \
        --task_name "$TASK_NAME" \
        --output_json_path "$output_file" \
        --cache_dir "$CACHE_DIR" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --temperature "$TEMPERATURE" \
        --top_p "$TOP_P"

    echo " $filename"
done

echo "========================================================"
echo "finished"
echo "========================================================"