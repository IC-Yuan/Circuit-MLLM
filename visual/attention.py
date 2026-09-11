import os
# 必须在 import torch 之前设置
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# ========== 测试配置与可视化参数 ==========
MODEL_DIR = os.environ.get("MODEL_DIR", "")
# QUESTION = "What is connected to R?\nSelect all that apply.\nA. D1\nB. D2\nOptions:\nA. D1\nB. D2"
# QUESTION = "What is connected to C1?"
IMAGE_PATH = os.environ.get("IMAGE_PATH", "")
QUESTION = "What is connected to RD?"

image_filename = os.path.basename(IMAGE_PATH)  # 提取文件名，例如 "04142.png"
image_number = os.path.splitext(image_filename)[0]  # 去掉扩展名，得到 "04142"


# 可视化参数
SHARPEN_POWER = 2.0
ATTN_ALPHA = 0.55
# ========================================


def safe_get_prefill_lastlayer_hidden(outputs):
    """
    outputs.hidden_states[0][-1] 取 prefill 阶段最后一层 hidden states
    兼容返回形状可能是 [B,S,D] 或 [S,D]
    """
    hs0_last = outputs.hidden_states[0][-1]
    if hs0_last.dim() == 3:
        return hs0_last[0]  # [S, D]
    elif hs0_last.dim() == 2:
        return hs0_last     # [S, D]
    else:
        raise RuntimeError(f"Unexpected prefill hs shape: {hs0_last.shape}")


def safe_get_step_last_hidden(outputs, step_idx):
    """
    取生成第 step_idx 个 token 时，最后一层最后一个位置的 hidden state
    outputs.hidden_states[step_idx] 是该 step 的所有层 hidden states（实现可能不同）
    这里假设 outputs.hidden_states[step_idx][-1] 是最后一层
    兼容 [B,S,D] / [S,D] / [D]
    """
    hs_last = outputs.hidden_states[step_idx][-1]
    if hs_last.dim() == 3:
        return hs_last[0, -1, :]  # [D]
    elif hs_last.dim() == 2:
        return hs_last[-1, :]     # [D]
    elif hs_last.dim() == 1:
        return hs_last            # [D]
    else:
        raise RuntimeError(f"Unexpected step hs shape: {hs_last.shape}")


def find_vision_span(inputs, processor):
    """
    从 input_ids 中定位视觉 token 段：(<|vision_start|>, <|vision_end|>)
    返回 (pos_start, pos_end, n_vis)
    """
    v_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    v_end_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")

    input_ids_list = inputs["input_ids"][0].tolist()
    if v_start_id not in input_ids_list or v_end_id not in input_ids_list:
        raise RuntimeError("Cannot find <|vision_start|> / <|vision_end|> in input_ids")

    pos_start = input_ids_list.index(v_start_id) + 1
    pos_end = input_ids_list.index(v_end_id)  # end token 的位置（不包含）
    n_vis = pos_end - pos_start
    if n_vis <= 0:
        raise RuntimeError(f"Invalid vision span: start={pos_start}, end={pos_end}, n_vis={n_vis}")

    return pos_start, pos_end, n_vis


def guess_hw_from_grid_thw(inputs, n_vis):
    """
    尝试用 image_grid_thw 推二维网格（如果可用且匹配）。
    若不匹配，返回 None。
    """
    if "image_grid_thw" not in inputs:
        return None

    grid_thw = inputs["image_grid_thw"][0].tolist()  # [t,h,w]
    if len(grid_thw) != 3:
        return None

    t, h_grid, w_grid = grid_thw

    # 你原代码里用了 //2，这是和具体 merge/patch 有关。这里保留尝试，但会校验匹配。
    # 先尝试两种：不 //2 和 //2（很多实现里确实是 //2）
    candidates = []

    # candidate 1: no divide
    if t > 0 and h_grid > 0 and w_grid > 0:
        candidates.append((t, h_grid, w_grid, t * h_grid * w_grid))

    # candidate 2: divide by 2
    if t > 0 and h_grid // 2 > 0 and w_grid // 2 > 0:
        candidates.append((t, h_grid // 2, w_grid // 2, t * (h_grid // 2) * (w_grid // 2)))

    # candidate 3: divide by 4 (少数情况下)
    if t > 0 and h_grid // 4 > 0 and w_grid // 4 > 0:
        candidates.append((t, h_grid // 4, w_grid // 4, t * (h_grid // 4) * (w_grid // 4)))

    # 找一个 token 数最匹配的
    candidates.sort(key=lambda x: abs(x[3] - n_vis))
    best = candidates[0]
    t_b, h_b, w_b, n_b = best

    if n_b != n_vis:
        return None

    # 如果 t>1，就把时间维拼到高度方向
    H = t_b * h_b
    W = w_b
    return H, W


def fallback_square_hw(n_vis):
    """
    如果无法从 grid_thw 得到网格，就找一个接近正方形的 (H,W)
    """
    H = int(math.floor(math.sqrt(n_vis)))
    while H > 1 and (n_vis % H) != 0:
        H -= 1
    W = n_vis // H
    return H, W


def main():
    print("1. 正在加载 Processor...")
    processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)

    print("2. 正在加载 Model (提取 Hidden States)...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_DIR,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model.eval()

    print(f"3. 加载测试图片: {IMAGE_PATH}")
    raw_img = Image.open(IMAGE_PATH).convert("RGB")
    width, height = raw_img.size

    print(f"4. 构造输入问题: '{QUESTION}'")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": QUESTION},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(
        text=[prompt],
        images=[raw_img],
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

    # --- 精确定位视觉 token 段 ---
    pos_start, pos_end, n_vis = find_vision_span(inputs, processor)
    print(f"[vision span] pos_start={pos_start}, pos_end={pos_end}, n_vis={n_vis}")

    print("5. 模型开始推理并抓取 Hidden States...")
    input_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.0,
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

    print("6. 开始计算 Hidden State 图像相关度...")
    generated_tokens = outputs.sequences[0][input_len:]

    # prefill 阶段最后一层 hidden states: [S, D]
    prefill_hs = safe_get_prefill_lastlayer_hidden(outputs)
    print(f"[prefill_hs] shape={tuple(prefill_hs.shape)} (expect [S,D])")

    # 截取视觉 token 特征并做 L2 归一化
    if pos_start + n_vis > prefill_hs.shape[0]:
        raise RuntimeError(
            f"Vision span out of range: pos_start({pos_start}) + n_vis({n_vis}) > seq_len({prefill_hs.shape[0]})"
        )

    H_vis = prefill_hs[pos_start: pos_start + n_vis, :]  # [N_vis, D]
    H_vis = H_vis.to(torch.float32)
    H_vis_norm = H_vis / (H_vis.norm(dim=-1, keepdim=True) + 1e-8)  # [N_vis, D]

    all_heatmaps = []
    tokens_log = []

    # 为了可视化，需要一个 (H,W) 网格来 reshape
    hw = guess_hw_from_grid_thw(inputs, n_vis)
    if hw is None:
        H_grid, W_grid = fallback_square_hw(n_vis)
        print(f"[grid] use fallback square-ish grid: H={H_grid}, W={W_grid} (n_vis={n_vis})")
    else:
        H_grid, W_grid = hw
        print(f"[grid] from image_grid_thw: H={H_grid}, W={W_grid} (n_vis={n_vis})")

    # 遍历生成 token，算每个 token 与所有视觉 token 的余弦相似度
    for step_idx, token_id in enumerate(generated_tokens):
        if token_id.item() == processor.tokenizer.eos_token_id:
            break

        t_text = processor.tokenizer.decode(token_id)
        tokens_log.append(t_text)

        # 获取当前 token 对应的 hidden state 向量 h_t
        if step_idx == 0:
            # 生成第一个词的依据通常来自 prefill 序列最后一个 token 的 hidden state
            h_t = prefill_hs[-1, :]
        else:
            h_t = safe_get_step_last_hidden(outputs, step_idx)

        h_t = h_t.to(torch.float32)
        h_t_norm = h_t / (h_t.norm(dim=-1, keepdim=True) + 1e-8)  # [D]

        # 余弦相似度
        sim_scores = torch.matmul(H_vis_norm, h_t_norm)  # [N_vis]
        sim_scores = sim_scores.cpu().numpy()

        # 鲁棒归一化到 0~1
        v_min = np.percentile(sim_scores, 1)
        v_max = np.percentile(sim_scores, 99)
        if v_max > v_min:
            sim_normalized = np.clip((sim_scores - v_min) / (v_max - v_min), 0, 1)
        else:
            sim_normalized = np.zeros_like(sim_scores)

        # 幂律锐化
        sim_normalized = np.power(sim_normalized, SHARPEN_POWER)

        # reshape 回二维网格
        # 若 H_grid*W_grid != n_vis（极少数情况下），则改用 fallback
        if H_grid * W_grid != n_vis:
            H_grid, W_grid = fallback_square_hw(n_vis)

        heatmap = sim_normalized.reshape(H_grid, W_grid)
        all_heatmaps.append(heatmap)

    # --- 绘图阶段 (网格布局) ---
    print("7. 开始绘制特征相似度热力图...")
    num_plots = len(all_heatmaps)
    if num_plots == 0:
        print("[-] 没有生成任何可视化 token（可能模型直接输出 eos）")
        return

    cols = 8
    rows = math.ceil(num_plots / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

    if isinstance(axes, np.ndarray):
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]

    for idx in range(len(axes_flat)):
        ax = axes_flat[idx]
        if idx < num_plots:
            ax.imshow(raw_img)

            h_map = all_heatmaps[idx]
            # resize 到原图大小
            h_img = Image.fromarray((h_map * 255).astype(np.uint8)).resize(
                (width, height), resample=Image.BICUBIC
            )
            ax.imshow(np.array(h_img), cmap='jet', alpha=ATTN_ALPHA, extent=(0, width, height, 0))

            t_text = tokens_log[idx].replace("\n", "\\n").strip()
            ax.set_title(f"'{t_text}'", color="black", fontsize=14, fontweight="bold")
            ax.axis("off")
        else:
            ax.axis("off")

    plt.tight_layout()

    # 将图片保存时加上原始图片的序号
    save_dir = os.environ.get("VISUAL_OUTPUT_DIR", "./outputs/visual")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"circuit_hidden_state_to_image_{image_number}_sequence.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")

    print("=" * 60)
    print(f"✅ Hidden State 到图像的相关度可视化已保存至: {save_path}")
    print("✨ 模型完整输出:")
    print("=" * 60)

    out_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=False)
    if "<|im_start|>assistant" in out_text:
        answer = out_text.split("<|im_start|>assistant")[-1].strip()
    else:
        answer = out_text
    print(answer)
    print("=" * 60)

if __name__ == "__main__":
    main()
