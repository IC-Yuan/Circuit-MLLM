import os
import re
import json
import logging
from tqdm import tqdm
from functools import partial

import torch
from torch.nn import functional as F
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLConfig,
    AutoProcessor,
)
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

from qwen_vl_utils import process_vision_info

from utils_deepseed import *
from task_deepseed import *
from trainer import CustomTrainerStage1, CustomTrainerStage2
import warnings

# ==============================================================
# Collate Functions
# ==============================================================
def _collect_latent_segment_lengths(input_ids, latent_start_idx, latent_end_idx, latent_token_idx):
    """Collect latent_pad counts for each complete latent segment in a batch tensor."""
    lengths = []
    for seq in input_ids:
        s_pos = (seq == latent_start_idx).nonzero().squeeze(-1).tolist()
        e_pos = (seq == latent_end_idx).nonzero().squeeze(-1).tolist()
        if not s_pos or not e_pos:
            continue
        for s in s_pos:
            end_candidates = [e for e in e_pos if e > s]
            if not end_candidates:
                continue
            e = end_candidates[0]
            seg = seq[s+1:e]
            lengths.append(int((seg == latent_token_idx).sum().item()))
    return lengths


def _validate_latent_template_or_raise(batch, latent_start_idx, latent_end_idx, latent_token_idx, expected_latent_size, stage_name):
    lengths = _collect_latent_segment_lengths(
        batch["input_ids"], latent_start_idx, latent_end_idx, latent_token_idx
    )
    if not lengths:
        raise ValueError(f"[{stage_name}] No complete latent segments found in collated batch.")
    bad = [x for x in lengths if int(x) != int(expected_latent_size)]
    if bad:
        raise ValueError(
            f"[{stage_name}] Latent template mismatch: expected={expected_latent_size}, "
            f"observed(unique)={sorted(set(lengths))}, bad_count={len(bad)}"
        )

def collate_fn_stage1(examples, processor, args):
    #读取数据，替换tab,<|vision_start|>替换为<|latent_start|>
    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]
    texts = [place_input_image(t) for t in texts]
    texts = [place_output_image(t) for t in texts]
    texts = replace_visual_spectial_tokens(texts)
    #提取需要处理的图片
    image_inputs, _ = process_vision_info(examples)
    #遍历所有的对话数据，把“Assistant（助手）”回复内容里的图片全部删掉，只保留文本；而“User（用户）”输入的内容保持不变。
    user_examples = remove_assistant_images(examples)
    user_text = [processor.apply_chat_template(example, tokenize=False) for example in user_examples]
    user_text = replace_visual_spectial_tokens(user_text)
    #提取user的图片，也就是输入图片
    user_image_inputs, _ = process_vision_info(user_examples)
    #用processor处理文字和图片，使用的是
    user_batch = processor(text=user_text, images=user_image_inputs, return_tensors="pt", padding=True)

    assistant_examples = remove_user_images(examples)
    assistant_text = [processor.apply_chat_template(example, tokenize=False) for example in assistant_examples]
    assistant_text = replace_visual_spectial_tokens(assistant_text)
    assistant_image_inputs, _ = process_vision_info(assistant_examples)
    assistant_batch = processor(text=assistant_text, images=assistant_image_inputs, return_tensors="pt", padding=True)
    
    # ================== 【新增】Mask 提取逻辑开始 ==================
    # 目标：生成一个 List[Tensor]，顺序必须与 assistant_image_inputs 里的图片顺序严格一致
    # 假设 assistant_examples 结构是 [{"role": "assistant", "content": [{"type": "image", "mask_npy": ...}, ...]}, ...]
    helper_masks = []
    for example in assistant_examples:
        for message in example:
            if message["role"] == "assistant":
                content = message.get("content", [])
                if not isinstance(content, list):
                    continue
                for item in content:
                    if item.get("type") == "image":
                        mask_npy = item.get("mask_npy", None)
                        
                        if mask_npy is not None:
                            mask_tensor = torch.from_numpy(mask_npy).float()
                            helper_masks.append(mask_tensor)
                        else:
                            helper_masks.append(None)
    
    if 'image_grid_thw' in assistant_batch:
        num_helper_imgs = assistant_batch['image_grid_thw'].shape[0]
        assert len(helper_masks) == num_helper_imgs, \
            f"Mask数量 ({len(helper_masks)}) 与 Helper图片数量 ({num_helper_imgs}) 不一致！请检查 remove_user_images 是否误删了数据。"
    # ================== 【新增】Mask 提取逻辑结束 ==================

    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
    
    if 'pixel_values' in user_batch:
        batch['pixel_values'] = user_batch['pixel_values']
        batch['image_grid_thw'] = user_batch['image_grid_thw']
    
    if 'pixel_values' in assistant_batch:
        batch['pixel_values_latent'] = assistant_batch['pixel_values']
        batch['image_grid_thw_latent'] = assistant_batch['image_grid_thw']
        batch['helper_masks'] = helper_masks

    latent_token_idx = processor.tokenizer("<|latent_pad|>", return_tensors="pt")["input_ids"][0,0].item()
    latent_start_idx = processor.tokenizer("<|latent_start|>", return_tensors="pt")["input_ids"][0,0].item()
    latent_end_idx   = processor.tokenizer("<|latent_end|>", return_tensors="pt")["input_ids"][0,0].item()
    pad_token_idx    = processor.tokenizer("<|endoftext|>", return_tensors="pt")["input_ids"][0,0].item()

    new_input_ids, new_attention_mask = process_batch(
        batch["input_ids"], batch["attention_mask"],
        start_token=latent_start_idx, end_token=latent_end_idx,
        replacement_token=latent_token_idx, replacement_length=args.latent_size,
        pad_token=pad_token_idx
    )
    batch["input_ids"] = new_input_ids
    batch["attention_mask"] = new_attention_mask
    
    answer_start_token_pattern = processor.tokenizer("<|im_start|>assistant", return_tensors="pt")["input_ids"][0]

    labels = generate_labels_with_latent_template(
        batch["input_ids"],
        answer_start_token_pattern,
        pad_token_idx,
        int(latent_start_idx),
        int(latent_end_idx),
        int(latent_token_idx),
        latent_ce_ratio=0,
    )
    batch["labels"] = labels

    image_out_mask = mask_latent_output_tokens_all_segments(
        batch["input_ids"], latent_start_idx, latent_end_idx, latent_token_idx
    )
    batch["image_out_mask"] = image_out_mask
        # Add user_image_inputs and assistant_image_inputs to batch
    batch["user_image_inputs"] = user_image_inputs
    batch["assistant_image_inputs"] = assistant_image_inputs
    return batch

def collate_fn_stage2(examples, processor, args):
    # 1. 基础文本处理 (与 Stage 1 保持一致)
    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]
    texts = [place_input_image(text) for text in texts]
    texts = [place_output_image(text) for text in texts]
    texts = replace_visual_spectial_tokens(texts)
    
    # 2. 处理所有输入图像
    image_inputs, _ = process_vision_info(examples)

    # 3. 处理 User 部分 (用于多模态理解，与 Stage 1 保持一致)
    user_examples = remove_assistant_images(examples)
    user_text = [processor.apply_chat_template(example, tokenize=False) for example in user_examples]
    user_text = replace_visual_spectial_tokens(user_text)
    user_image_inputs, _ = process_vision_info(user_examples)
    user_batch = processor(text=user_text, images=user_image_inputs, return_tensors="pt", padding=True)

    # 4. 生成主 Batch
    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
    
    # 5. 转移 User 图片信息
    if 'pixel_values' in user_batch:
        batch['pixel_values'] = user_batch['pixel_values']
        batch['image_grid_thw'] = user_batch['image_grid_thw']

    # ---------------------------------------------------------------------------------
    # 注意：Stage 2 通常不需要 assistant_batch 的像素级监督 (pixel_values_latent) 和 Masks，
    # 因此这里跳过了 Stage 1 中关于 assistant_batch 和 helper_masks 的逻辑。
    # ---------------------------------------------------------------------------------

    # 6. 提取 Special Tokens (同步 Stage 1 的 .item() 写法，防止 Tensor 维度问题)
    latent_token_idx = processor.tokenizer("<|latent_pad|>", return_tensors="pt")["input_ids"][0,0].item()
    latent_start_idx = processor.tokenizer("<|latent_start|>", return_tensors="pt")["input_ids"][0,0].item()
    latent_end_idx   = processor.tokenizer("<|latent_end|>", return_tensors="pt")["input_ids"][0,0].item()
    pad_token_idx    = processor.tokenizer("<|endoftext|>", return_tensors="pt")["input_ids"][0,0].item()

    # 7. 处理 Latent 占位符 (同步 Stage 1 的参数写法)
    new_input_ids, new_attention_mask = process_batch(
        batch["input_ids"], batch["attention_mask"], 
        start_token=latent_start_idx, end_token=latent_end_idx, 
        replacement_token=latent_token_idx, replacement_length=args.latent_size, 
        pad_token=pad_token_idx
    )

    batch["input_ids"] = new_input_ids
    batch["attention_mask"] = new_attention_mask

    # 8. 生成 Labels (同步 Stage 1 使用的新函数 generate_labels_with_latent_template)
    answer_start_token_pattern = processor.tokenizer("<|im_start|>assistant", return_tensors="pt")["input_ids"][0]

    labels = generate_labels_with_latent_template(
        batch["input_ids"], 
        answer_start_token_pattern, 
        pad_token_idx, 
        int(latent_start_idx),
        int(latent_end_idx),
        int(latent_token_idx),
        latent_ce_ratio=0, # Stage 2 如果不计算 latent 的 CE Loss，保持为 0；如果需要计算，请改为 1.0 或 args.latent_ce_ratio
    )
    batch["labels"] = labels
    _validate_latent_template_or_raise(
        batch,
        latent_start_idx,
        latent_end_idx,
        latent_token_idx,
        int(args.latent_size),
        "stage2",
    )
    
    return batch

    

# ==============================================================
# Main Training Function
# ==============================================================
def main_train():
    seed_everything(seed=42)
    args = get_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(args.log_file, mode='a', encoding='utf-8'),
            logging.StreamHandler()
        ],
    )
    logging.info('==' * 20)
    logging.info(args)
    logging.info('==' * 20)

    cache_dir = args.cache_dir
    os.environ['HF_HOME'] = cache_dir
    
    logging.info(f"Loading processor from: {args.model}")
    processor = AutoProcessor.from_pretrained(args.model, cache_dir=cache_dir, trust_remote_code=True)
    
    new_tokens = ["<|latent_pad|>", "<|latent_start|>", "<|latent_end|>"]
    processor.tokenizer.add_tokens(new_tokens, special_tokens=True)

    if args.stage in ['stage1']: 
        logging.info(f"Loading model (Stage 1) from: {args.model}")
        model_path = args.model
        config = Qwen2_5_VLConfig.from_pretrained(model_path, cache_dir=cache_dir, trust_remote_code=True)
        grad_checkpointing = True
    
    if args.stage in ['stage2']: 
        logging.info(f"Loading model (Stage 2) from: {args.model}")
        model_path = args.load_model_path
        config = Qwen2_5_VLConfig.from_pretrained(model_path, trust_remote_code=True)
        grad_checkpointing = False
    
    config.compress_strategy = args.compress_strategy
    config.latent_size = args.latent_size
    config.stage = args.stage

    latent_token_idx = processor.tokenizer("<|latent_pad|>", return_tensors="pt")["input_ids"][0,0].item()
    latent_start_idx = processor.tokenizer("<|latent_start|>", return_tensors="pt")["input_ids"][0,0].item()
    latent_end_idx   = processor.tokenizer("<|latent_end|>", return_tensors="pt")["input_ids"][0,0].item()
    config.latent_token_id = int(latent_token_idx)
    config.latent_start_id = int(latent_start_idx)
    config.latent_end_id   = int(latent_end_idx)
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        cache_dir=cache_dir if args.stage == 'stage1' else None,
        trust_remote_code=True
    )
    print("load over")
    if args.stage in ['stage1']: model.resize_token_embeddings(len(processor.tokenizer))

    for param in model.visual.parameters():
        param.requires_grad = False
    
    if torch.cuda.is_available():
        # accelerate/torchrun 会注入 LOCAL_RANK；单卡/普通运行通常没有
        if "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["LOCAL_RANK"])
            n = torch.cuda.device_count()  # 注意：这是“当前进程可见”的逻辑GPU数量（受 CUDA_VISIBLE_DEVICES 影响）
            if local_rank >= n:
                raise RuntimeError(f"LOCAL_RANK={local_rank} but only {n} CUDA devices are visible")
            torch.cuda.set_device(local_rank)
            
    logging.info(f"Moving model to CUDA device: {torch.cuda.current_device()} ...")
    

    preprocess_function = task_preporcess_config[args.task]
    train_dataset = load_jsonl_dataset(args.data_path)
    train_dataset = [preprocess_function(sample) for sample in train_dataset]

    
    if args.stage in ['stage1']:
        CustomTrainer = CustomTrainerStage1
        collate_fn = collate_fn_stage1
    else:
        CustomTrainer = CustomTrainerStage2
        collate_fn = collate_fn_stage2
    
        
    collate_fn = partial(collate_fn, processor=processor, args=args)

    peft_config = None
    if getattr(args, "use_lora", False):
        logging.info("Enabling LoRA...")
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules.split(","),
            bias="none",
            task_type="CAUSAL_LM",
        )

    training_args = SFTConfig(
        output_dir=args.save_model_path,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warm_up_steps,
        learning_rate=1e-5,
        weight_decay=0.01,
        logging_steps=20,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=4,
        optim="adamw_torch_fused" if args.stage == 'stage1' else "adamw_torch",
        bf16=True,
        push_to_hub=False,
        remove_unused_columns=False,
        gradient_checkpointing=grad_checkpointing,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        report_to=[],
        logging_dir='./logs/',
        logging_strategy='steps',
        max_seq_length=32768,
        deepspeed="configs/config_stage2.json",
        ddp_find_unused_parameters=False if args.stage == 'stage2' else None,
    )
    if args.stage in ['stage1']:
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=collate_fn,
            processing_class=processor.tokenizer,
            peft_config=peft_config,
            sim_weight=getattr(args, "sim_weight", 1.0),
            ema_tau=getattr(args, "ema_tau", 0.999),
            coverage_p=getattr(args, "coverage_p", 0.9),
            image_pool_k=getattr(args, "image_pool_k", 8),
            helper_group_L=getattr(args, "helper_group_L", 256),
            ce_weight=getattr(args, "ce_weight", 1.0),
        )
    elif args.stage in ['stage2']:
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=collate_fn,
            processing_class=processor.tokenizer,
        )
    

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None:
            logging.info(f"last checkpoint: {last_checkpoint}，continue。")
        else:
            logging.info(f"output log {training_args.output_dir} exists but no checkpoint， train from start。")
    else:
        logging.info("no checkpoint，train from start。")

    logging.info("start training (DeepSpeed ZeRO-3 Mode)...")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    final_model_path = training_args.output_dir

    if trainer.is_world_process_zero():
        logging.info(f"training finish，save model: {final_model_path}")
        processor.save_pretrained(final_model_path)

    trainer.save_model(final_model_path)

    if trainer.is_world_process_zero():
        logging.info("all saved。")
        
    logging.info("finish。")

if __name__ == "__main__":
    main_train()