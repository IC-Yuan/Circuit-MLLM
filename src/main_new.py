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
from task_deepseed import *

from utils_deepseed import (
    get_args,
    seed_everything,
    place_input_image,
    place_output_image,
    replace_visual_spectial_tokens,
    remove_assistant_images,
    remove_user_images,
    load_jsonl_dataset,
    generate_labels_with_latent_template,
)
from utils_deepseed_new import (
    get_ordered_latent_pad_token_ids,
    build_ordered_latent_pad_pattern,
    process_batch_with_latent_pattern,
    generate_labels_with_ordered_latent_template,
    mask_latent_output_tokens_ordered_segments,
)
from trainer_new import CustomTrainerStage1New, CustomTrainerStage2New
import warnings


def collate_fn_stage1_new(examples, processor, args):
    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]
    texts = [place_input_image(t) for t in texts]
    texts = [place_output_image(t) for t in texts]
    texts = replace_visual_spectial_tokens(texts)

    image_inputs, _ = process_vision_info(examples)

    user_examples = remove_assistant_images(examples)
    user_text = [processor.apply_chat_template(example, tokenize=False) for example in user_examples]
    user_text = replace_visual_spectial_tokens(user_text)
    user_image_inputs, _ = process_vision_info(user_examples)
    user_batch = processor(text=user_text, images=user_image_inputs, return_tensors="pt", padding=True)

    assistant_examples = remove_user_images(examples)
    assistant_text = [processor.apply_chat_template(example, tokenize=False) for example in assistant_examples]
    assistant_text = replace_visual_spectial_tokens(assistant_text)
    assistant_image_inputs, _ = process_vision_info(assistant_examples)
    assistant_batch = processor(text=assistant_text, images=assistant_image_inputs, return_tensors="pt", padding=True)

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

    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
    if "pixel_values" in user_batch:
        batch["pixel_values"] = user_batch["pixel_values"]
        batch["image_grid_thw"] = user_batch["image_grid_thw"]
    if "pixel_values" in assistant_batch:
        batch["pixel_values_latent"] = assistant_batch["pixel_values"]
        batch["image_grid_thw_latent"] = assistant_batch["image_grid_thw"]
        batch["helper_masks"] = helper_masks

    latent_pad_token_ids = get_ordered_latent_pad_token_ids(processor.tokenizer)
    latent_start_idx = processor.tokenizer("<|latent_start|>", return_tensors="pt")["input_ids"][0, 0].item()
    latent_end_idx = processor.tokenizer("<|latent_end|>", return_tensors="pt")["input_ids"][0, 0].item()
    pad_token_idx = processor.tokenizer("<|endoftext|>", return_tensors="pt")["input_ids"][0, 0].item()

    ordered_pattern = build_ordered_latent_pad_pattern(latent_pad_token_ids)
    new_input_ids, new_attention_mask = process_batch_with_latent_pattern(
        batch["input_ids"], batch["attention_mask"],
        start_token=latent_start_idx,
        end_token=latent_end_idx,
        replacement_pattern=ordered_pattern,
        pad_token=pad_token_idx,
    )
    batch["input_ids"] = new_input_ids
    batch["attention_mask"] = new_attention_mask

    answer_start_token_pattern = processor.tokenizer("<|im_start|>assistant", return_tensors="pt")["input_ids"][0]
    batch["labels"] = generate_labels_with_ordered_latent_template(
        batch["input_ids"], answer_start_token_pattern, pad_token_idx,
        int(latent_start_idx), int(latent_end_idx), latent_pad_token_ids, latent_ce_ratio=0,
    )
    batch["image_out_mask"] = mask_latent_output_tokens_ordered_segments(
        batch["input_ids"], latent_start_idx, latent_end_idx, latent_pad_token_ids
    )
    batch["user_image_inputs"] = user_image_inputs
    batch["assistant_image_inputs"] = assistant_image_inputs
    return batch


def collate_fn_stage2_new(examples, processor, args):
    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]
    texts = [place_input_image(t) for t in texts]
    texts = [place_output_image(t) for t in texts]
    texts = replace_visual_spectial_tokens(texts)

    image_inputs, _ = process_vision_info(examples)
    user_examples = remove_assistant_images(examples)
    user_text = [processor.apply_chat_template(example, tokenize=False) for example in user_examples]
    user_text = replace_visual_spectial_tokens(user_text)
    user_image_inputs, _ = process_vision_info(user_examples)
    user_batch = processor(text=user_text, images=user_image_inputs, return_tensors="pt", padding=True)

    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
    if "pixel_values" in user_batch:
        batch["pixel_values"] = user_batch["pixel_values"]
        batch["image_grid_thw"] = user_batch["image_grid_thw"]

    latent_pad_token_ids = get_ordered_latent_pad_token_ids(processor.tokenizer)
    latent_start_idx = processor.tokenizer("<|latent_start|>", return_tensors="pt")["input_ids"][0, 0].item()
    latent_end_idx = processor.tokenizer("<|latent_end|>", return_tensors="pt")["input_ids"][0, 0].item()
    pad_token_idx = processor.tokenizer("<|endoftext|>", return_tensors="pt")["input_ids"][0, 0].item()

    ordered_pattern = build_ordered_latent_pad_pattern(latent_pad_token_ids)
    batch["input_ids"], batch["attention_mask"] = process_batch_with_latent_pattern(
        batch["input_ids"], batch["attention_mask"],
        start_token=latent_start_idx,
        end_token=latent_end_idx,
        replacement_pattern=ordered_pattern,
        pad_token=pad_token_idx,
    )

    answer_start_token_pattern = processor.tokenizer("<|im_start|>assistant", return_tensors="pt")["input_ids"][0]
    batch["labels"] = generate_labels_with_ordered_latent_template(
        batch["input_ids"], answer_start_token_pattern, pad_token_idx,
        int(latent_start_idx), int(latent_end_idx), latent_pad_token_ids, latent_ce_ratio=0,
    )
    return batch


def main_train_new():
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

    if args.stage == "stage1":
        logging.info(f"Loading processor from: {args.model}")
        processor = AutoProcessor.from_pretrained(args.model, cache_dir=cache_dir, trust_remote_code=True)
        processor.tokenizer.add_tokens(
            ["<|latent_pad_1|>", "<|latent_pad_2|>", "<|latent_pad_3|>", "<|latent_pad_4|>", "<|latent_start|>", "<|latent_end|>"],
            special_tokens=True,
        )
        logging.info(f"Loading model (Stage 1) from: {args.model}")
        model_path = args.model
        config = Qwen2_5_VLConfig.from_pretrained(model_path, cache_dir=cache_dir, trust_remote_code=True)
        grad_checkpointing = True
    else:
        logging.info(f"Loading processor from: {args.load_model_path}")
        processor = AutoProcessor.from_pretrained(args.load_model_path, cache_dir=cache_dir, trust_remote_code=True)
        model_path = args.load_model_path
        logging.info(f"Loading model (Stage 2) from: {args.load_model_path}")
        config = Qwen2_5_VLConfig.from_pretrained(model_path, trust_remote_code=True)
        grad_checkpointing = False

    latent_pad_token_ids = get_ordered_latent_pad_token_ids(processor.tokenizer)
    config.latent_token_id = int(latent_pad_token_ids[0])
    config.latent_token_ids = [int(x) for x in latent_pad_token_ids]
    config.latent_start_id = int(processor.tokenizer("<|latent_start|>", return_tensors="pt")["input_ids"][0, 0].item())
    config.latent_end_id = int(processor.tokenizer("<|latent_end|>", return_tensors="pt")["input_ids"][0, 0].item())
    config.compress_strategy = args.compress_strategy
    config.latent_size = args.latent_size
    config.stage = args.stage

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        cache_dir=cache_dir if args.stage == "stage1" else None,
        trust_remote_code=True,
    )
    if args.stage == "stage1":
        model.resize_token_embeddings(len(processor.tokenizer))

    for p in model.visual.parameters():
        p.requires_grad = False
    
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
        CustomTrainer = CustomTrainerStage1New
        collate_fn = collate_fn_stage1_new
    else:
        CustomTrainer = CustomTrainerStage2New
        collate_fn = collate_fn_stage2_new

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
        logging_steps=1,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
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
    main_train_new()
