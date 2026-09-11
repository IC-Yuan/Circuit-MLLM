import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3' 
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ================= 配置区域 =================
device = "cuda"
model_path = os.environ.get("MODEL_DIR", "Qwen/Qwen2.5-VL-7B-Instruct")
image_path = os.environ.get("IMAGE_PATH", "")

MAX_PIXELS = 1024 * 38 * 38 
question_text = "What is connected to R?"

# 短语建议保持简洁，增加匹配成功率
TARGET_PHRASES = [
    "In the given circuit",
    "resistor R is",
    "connected in series",
    "diodes D1 and D2"
]

TARGET_LAYERS = [18, 20, 22] 
SHARPEN_POWER = 1.0           # 稍微提高，让背景更干净
ATTN_ALPHA = 0.6              
NOISE_THRESHOLD = 50        # 提高阈值（保留前10%），过滤掉背景微弱噪声
# ===========================================

def simplify_string(text):
    """将文本简化为纯字母数字，彻底消除 LaTeX 和特殊符号干扰"""
    return re.sub(r'[^a-zA-Z0-9]', '', text).lower()

print(f"🚀 正在加载模型...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, attn_implementation="eager", device_map="auto"
).eval()
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

v_start_id = processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')

messages = [
    {"role": "system", "content": "You are an IC expert."},
    {"role": "user", "content": [{"type": "image", "image": image_path, "max_pixels": MAX_PIXELS}, 
                                {"type": "text", "text": f"{question_text}\n"}]}
]

image_inputs, _ = process_vision_info(messages)
text_query = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text_query], images=image_inputs, padding=True, return_tensors="pt").to(device)

grid_thw = processor.image_processor(images=image_inputs)["image_grid_thw"]
t, h_grid, w_grid = grid_thw[0][0], grid_thw[0][1], grid_thw[0][2]
h_feat_per_tile, w_feat_per_tile = h_grid // 2, w_grid // 2
total_visual_tokens = t * h_feat_per_tile * w_feat_per_tile

generated_ids = inputs['input_ids']
max_new_tokens = 150 
all_heatmaps = []
tokens_log = []

print("🧠 模型推理中...")

for step in range(max_new_tokens):
    with torch.no_grad():
        outputs = model(
            input_ids=generated_ids, 
            pixel_values=inputs.get('pixel_values'), 
            image_grid_thw=inputs.get('image_grid_thw'), 
            output_attentions=True
        )
        
        next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)
        if next_token_id.item() == processor.tokenizer.eos_token_id: break
        
        ids_list = generated_ids[0].tolist()
        try:
            pos_start = ids_list.index(v_start_id) + 1
            # 聚合层注意
            layer_attns = [outputs.attentions[l][0, :, -1, pos_start:pos_start+total_visual_tokens] for l in TARGET_LAYERS]
            combined_attn = torch.stack(layer_attns).mean(dim=0).max(dim=0)[0]
            combined_attn = combined_attn.to(torch.float32).cpu().numpy()
            
            # --- 关键降噪 1: 抑制第一个 Sink Token ---
            combined_attn[0] = 0 
            
            tile_arrays = combined_attn.reshape(t, h_feat_per_tile, w_feat_per_tile)
            heatmap = np.concatenate(tile_arrays, axis=0)

            # --- 关键降噪 2: 边缘强行置零 ---
            heatmap[0, :] = 0; heatmap[-1, :] = 0
            heatmap[:, 0] = 0; heatmap[:, -1] = 0
            
            all_heatmaps.append(heatmap)
            tokens_log.append(processor.tokenizer.decode(next_token_id[0]))
        except Exception: pass
        
        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

# --- 改进的匹配逻辑：基于纯字符流滑动窗口 ---

def get_phrase_overlays(phrases, tokens, heatmaps):
    overlays = []
    final_labels = []
    current_search_start_idx = 0
    
    for phrase in phrases:
        target_stream = simplify_string(phrase)
        if not target_stream: continue
        
        best_range = None
        # 在剩余 token 中进行滑动窗口查找
        for i in range(current_search_start_idx, len(tokens)):
            # 这里的窗口大小 (25) 需要覆盖包含 LaTeX 噪音的短语长度
            for j in range(i + 1, min(i + 25, len(tokens) + 1)):
                # 将该范围内的所有 token 合并并简化
                current_window_text = "".join(tokens[i:j])
                current_stream = simplify_string(current_window_text)
                
                # 如果目标字符流在当前窗口字符流中出现
                if target_stream in current_stream:
                    best_range = (i, j)
                    break
            if best_range: break
            
        if best_range:
            start, end = best_range
            # 聚合热力图
            phrase_map = np.mean([heatmaps[idx] for idx in range(start, end)], axis=0)
            
            # --- 关键降噪 3: 局部阈值过滤 ---
            # 只有大于全图第 NOISE_THRESHOLD 百分位强度的区域才保留
            thresh = np.percentile(phrase_map, NOISE_THRESHOLD)
            phrase_map = np.where(phrase_map > thresh, phrase_map, 0)
            
            # 归一化
            phrase_map = np.power(phrase_map, SHARPEN_POWER)
            if phrase_map.max() > 0:
                phrase_map /= phrase_map.max()
            
            overlays.append(phrase_map)
            final_labels.append(phrase)
            current_search_start_idx = end # 从下一个位置继续搜
            print(f"✅ 匹配成功: '{phrase}' -> 对应 Token 序列: '{''.join(tokens[start:end]).strip()}'")
        else:
            print(f"⚠️ 无法匹配短语: '{phrase}' (目标字符流: {target_stream})")

    return final_labels, overlays

print("\n🎨 正在执行降噪与短语聚合...")
print(f"模型完整输出原文: {''.join(tokens_log)}\n")

found_labels, phrase_heatmaps = get_phrase_overlays(TARGET_PHRASES, tokens_log, all_heatmaps)

# --- 绘图阶段 ---
if not phrase_heatmaps:
    print("❌ 没有任何匹配成功的短语。")
else:
    raw_img = Image.open(image_path).convert("RGB")
    width, height = raw_img.size
    num_plots = len(phrase_heatmaps)
    
    fig, axes = plt.subplots(1, num_plots, figsize=(num_plots * 5, 5))
    if num_plots == 1: axes = [axes]

    for i in range(num_plots):
        ax = axes[i]
        ax.imshow(raw_img)
        
        h_map = phrase_heatmaps[i]
        # 插值平滑
        h_img = Image.fromarray((h_map * 255).astype(np.uint8)).resize((width, height), resample=Image.BICUBIC)
        # 高斯模糊
        h_img_final = h_img.filter(ImageFilter.GaussianBlur(radius=width/100))
        
        # 叠加
        ax.imshow(np.array(h_img_final), cmap='jet', alpha=ATTN_ALPHA, extent=(0, width, height, 0))
        
        ax.set_title(f"Part {i+1}:\n{found_labels[i]}", fontsize=9, fontweight='bold')
        ax.axis("off")

    plt.tight_layout()
    save_name = "./circuit_final_cleaned.png"
    plt.savefig(save_name, dpi=200, bbox_inches='tight')
    print(f"\n✅ 任务完成！结果已保存至: {save_name}")
