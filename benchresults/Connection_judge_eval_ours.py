import os
import re
import json
import argparse
import logging
import time
from typing import List, Dict, Any, Union

import torch
from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)

try:
    from mathruler.grader import extract_boxed_content
except Exception:
    extract_boxed_content = None

# ========== 确认以下两个路径正确 ==========
BASE_MODEL_ID = "/data/jydeng/LLM/circuit_llm_qwen/Qwen2.5-VL-7B-Instruct"
BASE_DATASET_DIR = '/data/jydeng/circuit_clip/amsbench/AMSBench_circuit_mllm/Connection_Identification_Task_ours/img'
# ===========================================


def get_eval_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to the trained model directory.")
    parser.add_argument("--test_data_path", type=str, required=True,
                        help="Path to the test data JSONL file.")
    parser.add_argument("--task_name", type=str, required=True,
                        help="Task name (used for bookkeeping and branching logic).")
    parser.add_argument("--output_json_path", type=str, default="evaluation_results.jsonl",
                        help="Path to save per-sample evaluation results.")
    parser.add_argument("--cache_dir", type=str, default="./cache",
                        help="Hugging Face cache directory.")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    return parser.parse_args()


def load_processor(model_dir: str, cache_dir: str):
    """优先从微调目录加载 processor，失败则回退到基础模型。"""
    try:
        return AutoProcessor.from_pretrained(model_dir, cache_dir=cache_dir)
    except Exception:
        logging.warning("Failed to load processor from model_dir; falling back to the base model.")
        return AutoProcessor.from_pretrained(BASE_MODEL_ID, cache_dir=cache_dir)


def load_model(model_dir: str, cache_dir: str):
    """优先 flash_attention_2，失败回退 eager。"""
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_dir,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
            attn_implementation="flash_attention_2",
        )
    except Exception as e:
        logging.warning(f"flash_attention_2 failed ({e}); falling back to eager attention.")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_dir,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
            attn_implementation="eager",
        )
    model.eval()
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass
    return model


def _resolve_image_paths(image_input: Union[str, List[str]]) -> List[str]:
    """解析图像路径：相对路径拼接 BASE_DATASET_DIR，绝对路径直接返回。"""
    if image_input is None:
        return []
    if isinstance(image_input, str):
        p = image_input
        if not os.path.isabs(p):
            p = os.path.join(BASE_DATASET_DIR, p)
        return [p]
    if isinstance(image_input, list):
        outs = []
        for p in image_input:
            if not os.path.isabs(p):
                p = os.path.join(BASE_DATASET_DIR, p)
            outs.append(p)
        return outs
    return []


def open_images(paths: List[str]) -> List[Image.Image]:
    """加载图像，显式检查文件存在性。"""
    imgs = []
    for p in paths:
        if os.path.isdir(p):
            raise IsADirectoryError(f"Expected an image file, but received a directory path: {p}")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Image not found: {p}")
        imgs.append(Image.open(p).convert("RGB"))
    return imgs


# ------------------ 多标签字母答案归一化 ------------------
def normalize_option_a_b(ans: str) -> str:
    """
    专门针对 A/B 二选一任务的归一化。
    只提取出现的第一个明确的 A 或 B。
    """
    if not ans:
        return ""
    
    # 转大写并去除首尾空白
    s = str(ans).upper().strip()
    
    # 1. 简单情况：直接就是 A 或 B (可能带标点)
    # 移除标点符号
    clean_s = re.sub(r'[^\w\s]', '', s)
    if clean_s.strip() == 'A': return 'A'
    if clean_s.strip() == 'B': return 'B'

    # 2. 复杂情况：从文本中提取 "Option A" 或 "Answer: B" 等
    # 优先匹配全字匹配
    match = re.search(r'\b([AB])\b', s)
    if match:
        return match.group(1)
    
    # 3. 兜底：如果没找到单词边界的 A/B，找第一个出现的字符 A 或 B
    # 注意：这可能会误伤（例如单词 "About" 会提取出 A），但结合 extract_final_answer 通常够用
    match_fallback = re.search(r'[AB]', s)
    if match_fallback:
        return match_fallback.group(0)
        
    return ""

# ------------------ 修正 extract_final_answer，移除长度截断 ------------------
_yes_set = {"yes", "true", "a"}
_no_set = {"no", "false", "b"}

def extract_final_answer(text: str) -> str:
    """
    提取最终答案，优先匹配 'final answer is: XXX' 等模式。
    【重要】不再截断前50字符，确保完整捕获多字母答案。
    """
    s = text.strip()
    patterns = [
        r"final\s*answer\s*(?:is)?\s*[:：]\s*(.+)",
        r"answer\s*(?:is)?\s*[:：]\s*(.+)",
    ]
    for pat in patterns:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m:
            cand = m.group(1).strip()
            cand = re.split(r"[\n\r]", cand)[0]   # 只取第一行
            return cand
    tokens = re.findall(r"[A-Za-z]+", s.lower())
    only = [t for t in tokens if t in (_yes_set | _no_set)]
    if len(only) == 1:
        return only[0].capitalize() if only[0] in {"yes", "no"} else only[0]
    return s


def _strip_special_tokens(s: str) -> str:
    """移除常见的结束标记，保留 latent token。"""
    end_markers = [
        "<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end|>",
        "</s>", "<s>", "[/INST]", "[INST]",
    ]
    for m in end_markers:
        s = s.replace(m, "")
    return s.strip()


def extract_assistant_content(text: str) -> str:
    """提取 assistant 回复内容（保留 latent token）。"""
    s = text or ""
    # ChatML 格式
    m = re.search(r"<\|im_start\|>\s*assistant\s*(.*?)(?:<\|im_end\|>|$)", s, flags=re.S | re.I)
    if m:
        return _strip_special_tokens(m.group(1))
    # 其他格式
    m = re.search(r"<\|assistant\|>\s*(.*?)(?:<\|im_end\|>|<\|endoftext\|>|<\|eot_id\|>|<\|end\|>|$)",
                  s, flags=re.S | re.I)
    if m:
        return _strip_special_tokens(m.group(1))
    # 兜底
    m = re.search(r"(?:^|\n|\r)assistant\s*[:：]?\s*(.*)$", s, flags=re.S | re.I)
    if m:
        return _strip_special_tokens(m.group(1))
    return _strip_special_tokens(s)


def normalize_for_match(ans: str) -> str:
    """用于 yes/no 任务的归一化（保留供其他任务使用）。"""
    a = ans.strip()
    low = a.lower()
    if low in _yes_set:
        return "Yes"
    if low in _no_set:
        return "No"
    return a


def run_one_example(
    model,
    processor,
    sample: Dict[str, Any],
    gen_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """单样本推理，返回原始输出、提取的最终答案、推理时间。"""
    text_input = sample.get("text_input", "")
    img_list = _resolve_image_paths(sample.get("image_input", []))
    images = open_images(img_list) if img_list else None

    # 构造对话：图像占位符 + 文本
    content = []
    if images:
        for _ in images:
            content.append({"type": "image"})
    content.append({"type": "text", "text": text_input})

    conversations = [{"role": "user", "content": content}]
    prompt = processor.apply_chat_template(
        conversations,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[prompt],
        images=images,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

    # 生成
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    with torch.inference_mode():
        out_ids = model.generate(**inputs, **gen_kwargs, tokenizer=processor.tokenizer)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_time = time.perf_counter() - start_time

    out_text = processor.batch_decode(out_ids, skip_special_tokens=False)[0].strip()
    extracted = extract_final_answer(out_text)

    return {
        "raw_output": out_text,
        "extracted_final_answer": extracted,
        "inference_time": inference_time,
    }


def main():
    args = get_eval_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    logging.info(f"Loading model from: {args.model_dir}")
    model = load_model(args.model_dir, args.cache_dir)
    processor = load_processor(args.model_dir, args.cache_dir)

    # 加载测试数据
    data = []
    with open(args.test_data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    if not data:
        logging.error("Test dataset is empty. Exiting.")
        return

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=(args.temperature > 0),
    )

    total = len(data)
    success = 0                     # 完全匹配且gold非空的样本数
    valid_samples = 0              # gold非空的样本数（用于计算平均指标）
    sum_precision = 0.0
    sum_recall = 0.0
    sum_f1 = 0.0
    sum_exact_match = 0.0
    sum_jaccard = 0.0
    results = []
    total_inference_time = 0.0

    logging.info(f"Starting evaluation: num_samples={total}")
    for i, sample in enumerate(data):
        try:
            pred = run_one_example(model, processor, sample, gen_kwargs)
            sample_infer_time = pred.get("inference_time")
            if sample_infer_time is not None:
                total_inference_time += sample_infer_time

            out_text = pred["raw_output"]
            assistant_only = extract_assistant_content(out_text)

            # ---------- 任务分支 ----------
            # ========== A/B 二选一 判别任务 ==========
            gold = sample.get("original_final_answer", "")
            
            # 使用新的归一化函数
            pred_norm = normalize_option_a_b(pred["extracted_final_answer"])
            gold_norm = normalize_option_a_b(gold)

            ok = False
            
            # 只有当 gold 有效且两者相等时才算对
            if gold_norm in ['A', 'B']:
                valid_samples += 1
                if pred_norm == gold_norm:
                    ok = True
                    success += 1

            results.append({
                "index": i,
                "task_name": args.task_name,
                "image_input": sample.get("image_input", []),
                "prediction": assistant_only,       # 原始回复
                "extracted": pred["extracted_final_answer"], 
                "gold_final_answer": gold,
                "normalized_prediction": pred_norm,
                "normalized_gold": gold_norm,
                "match": bool(ok),                 # 是否正确
                "inference_time_sec": sample_infer_time,
            })
            # ===================================================

            if (i + 1) % 10 == 0 or (i + 1) == total:
                # 这里的 accuracy 是当前的累计正确率
                logging.info(f"[{i+1}/{total}] Current Accuracy = {success/(i+1):.4f}")
        
        except Exception as e:
            logging.error(f"[{i}] Evaluation failed: {e}")
            results.append({
                "index": i,
                "task_name": args.task_name,
                "error": str(e),
                "inference_time_sec": None,
            })

    # ---------- 汇总统计 (修改后) ----------
    # 整体准确率 (基于总样本数)
    acc = success / total if total > 0 else 0.0
    # 有效样本准确率 (去除 gold 不为 A/B 的坏数据)
    valid_acc = success / valid_samples if valid_samples > 0 else 0.0
    
    avg_inference_time = total_inference_time / total if total > 0 else 0.0

    # 写入输出文件
    with open(args.output_json_path, "w", encoding="utf-8") as fout:
        for r in results:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
        # 汇总行
        fout.write(json.dumps({
            "summary": True,
            "num_samples": total,
            "valid_samples": valid_samples,
            "average_inference_time_sec": avg_inference_time,
            "accuracy": acc,
            "valid_accuracy": valid_acc, # 仅供参考
            "success": success,
        }, ensure_ascii=False) + "\n")

    # 打印最终指标
    logging.info(
        f"Evaluation finished: {success}/{total} correct; Accuracy={acc:.4f}"
    )
    print(f"\n========== Final Metrics ==========")
    print(f"Total Samples    = {total}")
    print(f"Correct Count    = {success}")
    print(f"Accuracy         = {acc:.4f}")
    if valid_samples != total:
         print(f"Valid Accuracy   = {valid_acc:.4f} (Based on {valid_samples} valid golds)")
    print(f"====================================\n")

def extract_path_from_text(generated_text: str) -> str:
    if extract_boxed_content is not None:
        s = extract_boxed_content(generated_text)
        if s:
            return s
    m = re.search(r"\\boxed\{([UDLRudlr]+)\}", generated_text)
    if m:
        return m.group(1)
    m = re.search(r"final\s*answer\s*(?:is)?\s*[:：]\s*([UDLRudlr]+)", generated_text, re.I)
    return m.group(1) if m else ""


if __name__ == "__main__":
    main()