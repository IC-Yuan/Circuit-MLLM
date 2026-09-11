import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import os
# 必须在 import torch 和 transformers 之前设置！
# 这会将物理卡 3 映射为逻辑卡 0，后续模型加载会自动跑到这张卡上
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# ========== 直接在这里写死测试配置 ==========
MODEL_DIR = os.environ.get("MODEL_DIR", "")
IMAGE_PATH = os.environ.get("IMAGE_PATH", "")
QUESTION = "What is connected to R?"
# ===========================================

def main():
    print("1. 正在加载 Processor...")
    processor = AutoProcessor.from_pretrained(MODEL_DIR)

    print("2. 正在加载 Model (优先尝试 flash_attention_2)...")
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_DIR,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
    except Exception as e:
        print(f"   [-] flash_attention_2 失败 ({e})，回退到 eager 模式...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_DIR,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )
    model.eval()

    print(f"3. 加载测试图片: {IMAGE_PATH}")
    image = Image.open(IMAGE_PATH).convert("RGB")

    print(f"4. 构造输入问题: '{QUESTION}'")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": QUESTION}
            ]
        }
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(
        text=[prompt],
        images=[image],
        return_tensors="pt",
        padding=True,
    )
    # 将 tensor 放到模型同一设备
    inputs = {k: v.to(model.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

    print("5. 模型开始推理 (生成中)...")
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.0,  # 贪婪解码，保证结果稳定
            do_sample=False
        )

    print("6. 解码输出...\n")
    # skip_special_tokens=False 以便你能看到所有的 latent token 行为
    out_text = processor.batch_decode(output_ids, skip_special_tokens=False)[0]

    # 为了更清晰地看 assistant 的回答，简单切分一下 prompt 部分
    if "<|im_start|>assistant" in out_text:
        answer = out_text.split("<|im_start|>assistant")[-1].strip()
    else:
        answer = out_text

    print("=" * 60)
    print("✨ 模型原始输出:")
    print("=" * 60)
    print(answer)
    print("=" * 60)

if __name__ == "__main__":
    main()
