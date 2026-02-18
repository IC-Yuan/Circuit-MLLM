from trl import SFTTrainer
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import deepspeed
from typing import List
# HAWP imports
from hawp.hawp.fsl.config import cfg as model_config
from hawp.hawp.ssl.models import MODELS
import numpy as np
import cv2
from transformers import AutoImageProcessor, AutoModel
import logging
from DeepLSD.deeplsd.models.deeplsd_inference import DeepLSD

try:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb
except Exception:
    apply_multimodal_rotary_pos_emb = None

try:
    import deepspeed
    _HAS_DS = True
except ImportError:
    _HAS_DS = False


class _EMATeacher:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.teacher = copy.deepcopy(model)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, student: nn.Module):
        d = self.decay
        for p_t, p_s in zip(self.teacher.parameters(), student.parameters()):
            p_t.data.mul_(d).add_(p_s.data, alpha=(1.0 - d))


class CustomTrainerStage1(SFTTrainer):
    def __init__(
        self,
        *args,
        sim_weight: float = 1.0,
        ema_tau: float = 0.999,
        coverage_p: float = 0.9,
        image_pool_k: int = 8,
        ce_weight: float = 1.0,
        helper_group_L: int = 256,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.sim_weight = float(sim_weight)
        self.coverage_p = float(coverage_p)
        self.helper_group_L = int(helper_group_L)
        self.ce_weight = float(ce_weight)
        self.image_pool_k = int(image_pool_k)
        self._ema = _EMATeacher(self.model, decay=float(ema_tau))

    def _find_latent_segments(self, input_ids, latent_start_id, latent_end_id, latent_pad_id):
        ids = input_ids[0].tolist()
        segments = []
        t = 0
        T = len(ids)
        while t < T:
            if ids[t] == latent_start_id:
                s = t
                e = s + 1
                while e < T and ids[e] != latent_end_id:
                    e += 1
                pad_pos = [i for i in range(s, e+1) if ids[i] == latent_pad_id]
                segments.append(pad_pos)
                t = e + 1
            else:
                t += 1
        return segments

    def _build_firstK_mask(self, input_ids, segments, K_list):
        B, T = input_ids.shape
        mask = torch.zeros(B, T, dtype=torch.bool, device=input_ids.device)
        assert len(segments) == len(K_list)
        for pads, K in zip(segments, K_list):
            if K > 0 and len(pads) > 0:
                take = min(K, len(pads))
                for i in pads[:take]:
                    mask[0, i] = True
        return mask

    def _get_special_ids(self):
        tok = self.processing_class
        get_id = lambda s: tok(s, return_tensors="pt")["input_ids"][0,0].item()
        latent_pad_id   = get_id("<|latent_pad|>")
        latent_start_id = get_id("<|latent_start|>")
        latent_end_id   = get_id("<|latent_end|>")
        special_ids = {latent_pad_id, latent_start_id, latent_end_id}
        try:
            vision_start_id = get_id("<|vision_start|>")
            vision_end_id   = get_id("<|vision_end|>")
            special_ids.update({vision_start_id, vision_end_id})
        except Exception:
            pass
        img_token_id = getattr(self.model.config, "image_token_id", None)
        if img_token_id is None:
            img_token_id = 151655
        return special_ids, int(img_token_id), int(latent_start_id), int(latent_end_id), int(latent_pad_id)
    
    def get_hawp_model(self, device='cuda'):
        # Load HAWP model configuration
        cfg_path = '/data/jydeng/latent_visual/circuit_mllm/hawp/hawp/ssl/config/hawpv3.yaml'
        model_config.merge_from_file(cfg_path)
        
        # Create and load HAWP model
        model = MODELS['HAWP'](model_config, gray_scale=True)
        model = model.eval().to(device)
        
        weight_path = '/data/share/JYD/weights/hawp/hawpv3-imagenet-03a84.pth'
        state_dict = torch.load(weight_path, map_location='cpu')
        model.load_state_dict(state_dict)
        
        _hawp_model = model
        return _hawp_model
    
    @torch.no_grad()
    def _teacher_build_latents(self, inputs, k, p):
        # 1. 准备数据
        device = self.model.device
        ids = inputs["input_ids"].to(device)
        
        # 获取 helper_masks (List[Tensor])
        # 注意：这里我们不做 pop，因为如果在中途 return，inputs 还没被修改是安全的
        # 我们统一在最后或者 return 前做清理
        raw_masks = inputs.get("helper_masks", None)
        
        tea = self._ema.teacher.to(device).eval()
        special_ids, image_token_id, latent_start_id, latent_end_id, latent_pad_id = self._get_special_ids()

        seg_pad_indices = self._find_latent_segments(ids, latent_start_id, latent_end_id, latent_pad_id)
        
        # 定义一个清理函数，方便在多处 return 时调用
        def cleanup_inputs():
            if "helper_masks" in inputs:
                inputs.pop("helper_masks")

        if len(seg_pad_indices) == 0:
            cleanup_inputs()
            return None, torch.zeros_like(ids, dtype=torch.bool)

        # 2. 获取 Visual Features
        pv = inputs.get("pixel_values_latent", None)
        thw = inputs.get("image_grid_thw_latent", None)
        if pv is None or thw is None:
            cleanup_inputs()
            return None, torch.zeros_like(ids, dtype=torch.bool)
             
        pv = pv.to(device).to(tea.visual.dtype)
        thw = thw.to(device)
        
        # 获取所有 helper images 的视觉特征
        patch_all = tea.visual(pv, grid_thw=thw) 
        
        # 计算 grid 信息 (Qwen2-VL 特定逻辑)
        num_imgs = int(thw.shape[0])
        s_merge = int(getattr(tea.visual, "spatial_merge_size", 2))
        thw_long = thw.to(dtype=torch.long)
        
        tokens_per_img = (thw_long[:,0] * (thw_long[:,1]//s_merge) * (thw_long[:,2]//s_merge))
        ends_img = torch.cumsum(tokens_per_img, dim=0).tolist()
        starts_img = [0] + ends_img[:-1]
        slices_per_img = [(int(st), int(ed)) for st, ed in zip(starts_img, ends_img)]

        latents_list = []
        Kstars = []

        # 3. 循环处理每个 Latent Segment
        for seg_idx, pad_pos in enumerate(seg_pad_indices):
            if len(pad_pos) == 0:
                Kstars.append(0); continue
            
            assert seg_idx < num_imgs, "Latent段数超过了Helper图片数"
            st_img, ed_img = slices_per_img[seg_idx]
            ei = patch_all[st_img:ed_img, :] # [N_tokens, Dim]

            selected_tokens = None
            
            # --- Mask 处理逻辑 ---
            # 检查是否有 Mask 且不是 None (我们在 collate_fn 里可能填了 None 占位)
            if raw_masks is not None and seg_idx < len(raw_masks) and raw_masks[seg_idx] is not None:
                # 拿 Mask
                curr_mask = raw_masks[seg_idx].to(device).float()
                
                # 维度调整为 [1, 1, H, W] 以适配 pooling
                if curr_mask.dim() == 2:
                    curr_mask = curr_mask.unsqueeze(0).unsqueeze(0)
                elif curr_mask.dim() == 3:
                    curr_mask = curr_mask.unsqueeze(0)

                # 计算 Visual Grid 尺寸 (H_grid, W_grid)
                grid_h = int(thw_long[seg_idx, 1] // s_merge)
                grid_w = int(thw_long[seg_idx, 2] // s_merge)

                # 【关键修改 1】：Mask 下采样策略
                # 使用 Adaptive Max Pooling。
                # 只要原始 Mask 在某个 Grid 区域内有任意非0值 (沾到一点)，
                # Max Pool 后的结果就是非0。这避免了 Nearest 插值漏掉边缘的问题。
                resized_mask = F.adaptive_max_pool2d(curr_mask, output_size=(grid_h, grid_w))
                
                # 展平
                flat_mask = resized_mask.flatten()

                # 安全截断/填充 (防止 grid 计算误差)
                if flat_mask.shape[0] > ei.shape[0]:
                    flat_mask = flat_mask[:ei.shape[0]]
                elif flat_mask.shape[0] < ei.shape[0]:
                    padding = torch.zeros(ei.shape[0] - flat_mask.shape[0], device=device)
                    flat_mask = torch.cat([flat_mask, padding])

                # 核心筛选：只看 > 0
                mask_bool = flat_mask > 0.0
                candidates = ei[mask_bool] # [M, Dim]
                
                num_cand = candidates.shape[0]

                if num_cand == 0:
                    # Case A: Mask 全黑 (异常保护) -> 退化为取前 k 个
                    selected_tokens = ei[:k]
                elif num_cand <= k:
                    # Case B: 候选数量少于 k -> 全都要
                    selected_tokens = candidates
                else:
                    input_tensor = candidates.transpose(0, 1).unsqueeze(0)
                    pooled = F.adaptive_avg_pool1d(input_tensor, output_size=k)
                    selected_tokens = pooled.squeeze(0).transpose(0, 1)

            else:
                # 无 Mask -> 兜底方案，取前 k 个 (通常是左上角)
                selected_tokens = ei[:k]
            # --------------------
            Kstar = min(len(pad_pos), selected_tokens.shape[0])
            Kstars.append(Kstar)

            if Kstar > 0:
                latents_list.append(selected_tokens[:Kstar])
        
        # 【关键修改 3】：清理 inputs，防止 Student 模型报错
        cleanup_inputs()

        if len(latents_list) == 0:
             return None, torch.zeros_like(ids, dtype=torch.bool)
             
        latents = torch.cat(latents_list, dim=0).unsqueeze(0)
        firstK_mask = self._build_firstK_mask(ids, seg_pad_indices, Kstars)
        
        return latents, firstK_mask
    
    def _teacher_build_latents_sequence(self, inputs, k, p):
        # 1. 准备数据
        device = self.model.device
        ids = inputs["input_ids"].to(device)
        # 获取 Mask
        raw_masks = inputs.get("helper_masks", None)
        
        tea = self._ema.teacher.to(device).eval()
        special_ids, image_token_id, latent_start_id, latent_end_id, latent_pad_id = self._get_special_ids()
        seg_pad_indices = self._find_latent_segments(ids, latent_start_id, latent_end_id, latent_pad_id)
        
        # 定义清理函数
        def cleanup_inputs():
            if "helper_masks" in inputs: inputs.pop("helper_masks")

        if len(seg_pad_indices) == 0:
            cleanup_inputs(); return None, torch.zeros_like(ids, dtype=torch.bool)

        # 2. 获取 Visual Features
        pv = inputs.get("pixel_values_latent", None)
        thw = inputs.get("image_grid_thw_latent", None)
        if pv is None or thw is None:
            cleanup_inputs(); return None, torch.zeros_like(ids, dtype=torch.bool)
             
        pv = pv.to(device).to(tea.visual.dtype)
        thw = thw.to(device)
        
        # 提取特征
        patch_all = tea.visual(pv, grid_thw=thw) 
        
        # Grid 计算逻辑 (Qwen2-VL)
        num_imgs = int(thw.shape[0])
        s_merge = int(getattr(tea.visual, "spatial_merge_size", 2))
        thw_long = thw.to(dtype=torch.long)
        # 计算每张图的 Token 范围
        tokens_per_img = (thw_long[:,0] * (thw_long[:,1]//s_merge) * (thw_long[:,2]//s_merge))
        ends_img = torch.cumsum(tokens_per_img, dim=0).tolist()
        starts_img = [0] + ends_img[:-1]
        slices_per_img = [(int(st), int(ed)) for st, ed in zip(starts_img, ends_img)]

        latents_list = []
        Kstars = []

        # 3. 循环处理每个 Latent Segment
        for seg_idx, pad_pos in enumerate(seg_pad_indices):
            if len(pad_pos) == 0:
                Kstars.append(0); continue
            
            assert seg_idx < num_imgs
            st_img, ed_img = slices_per_img[seg_idx]
            ei = patch_all[st_img:ed_img, :] # [N_tokens, Dim]

            selected_tokens = None
            
            # --- Mask 处理核心逻辑 ---
            if raw_masks is not None and seg_idx < len(raw_masks) and raw_masks[seg_idx] is not None:
                curr_mask = raw_masks[seg_idx].to(device).float()
                
                # 维度调整 [H, W] -> [1, 1, H, W]
                if curr_mask.dim() == 2: curr_mask = curr_mask.unsqueeze(0).unsqueeze(0)
                elif curr_mask.dim() == 3: curr_mask = curr_mask.unsqueeze(0)

                # 计算 Feature Map 的 Grid 尺寸
                grid_h = int(thw_long[seg_idx, 1] // s_merge)
                grid_w = int(thw_long[seg_idx, 2] // s_merge)

                # 【关键点 1】：使用 Adaptive Max Pool 
                # 只要原始 Mask 区域里有值（比如 2.1），池化后该 grid 就变成 2.1
                # 只要沾到一点，就不会漏掉。
                resized_mask = F.adaptive_max_pool2d(curr_mask, output_size=(grid_h, grid_w))
                flat_mask_values = resized_mask.flatten() # [N_tokens]

                # 对齐长度 (安全处理 grid 计算误差)
                if flat_mask_values.shape[0] > ei.shape[0]:
                    flat_mask_values = flat_mask_values[:ei.shape[0]]
                elif flat_mask_values.shape[0] < ei.shape[0]:
                    padding = torch.zeros(ei.shape[0] - flat_mask_values.shape[0], device=device)
                    flat_mask_values = torch.cat([flat_mask_values, padding])

                # 【关键点 2】：筛选所有非 0 区域
                # 这里 > 0.0 会把 1(根), 2.x(线), 3.x(连接), 4(孤立) 全部选中
                valid_indices = torch.nonzero(flat_mask_values > 0.0).squeeze(1) 
                
                if valid_indices.numel() == 0:
                    # 兜底：如果全黑，退化为取前 k 个空间特征
                    selected_tokens = ei[:k]
                else:
                    valid_tokens = ei[valid_indices]          # [M, Dim]
                    valid_values = flat_mask_values[valid_indices] # [M]

                    # 【关键点 3】：判断并排序
                    # 如果 mask 最大值 > 1.0 (说明包含 2.x, 3.x, 4 等顺序信息)
                    if valid_values.max() > 1.0 + 1e-4:
                        # argsort 默认升序 (Ascending)
                        # 顺序自动变成: 1 -> 2.x -> 3.x -> 4
                        # 4 (孤立目标) 因为数值最大，自然会被排到最后
                        sort_idx = torch.argsort(valid_values, descending=False)
                        sorted_tokens = valid_tokens[sort_idx]
                    else:
                        # 如果只有 0 和 1 (或者全是 1)，保持原始的空间顺序 (Raster Scan)
                        # 避免不必要的排序打乱空间结构
                        sorted_tokens = valid_tokens

                    # 【关键点 4】：特征压缩
                    num_cand = sorted_tokens.shape[0]
                    if num_cand <= k:
                        # 数量不足 k，直接取所有有效特征
                        selected_tokens = sorted_tokens
                    else:
                        # 数量超过 k，使用 Adaptive Avg Pool 压缩
                        # 在有序序列上做 Avg Pool，能较好地保留整体拓扑结构
                        input_tensor = sorted_tokens.transpose(0, 1).unsqueeze(0) # [1, Dim, M]
                        pooled = F.adaptive_avg_pool1d(input_tensor, output_size=k)
                        selected_tokens = pooled.squeeze(0).transpose(0, 1) # [k, Dim]

            else:
                # 无 Mask 兜底
                selected_tokens = ei[:k]

            # --- 结果收集 ---
            Kstar = min(len(pad_pos), selected_tokens.shape[0])
            Kstars.append(Kstar)

            if Kstar > 0:
                latents_list.append(selected_tokens[:Kstar])
        
        # 清理 Mask 输入防止污染 Student
        cleanup_inputs()

        if len(latents_list) == 0:
             return None, torch.zeros_like(ids, dtype=torch.bool)
             
        latents = torch.cat(latents_list, dim=0).unsqueeze(0)
        # 生成 Mask，标记前 Kstar 个位置有效
        firstK_mask = self._build_firstK_mask(ids, seg_pad_indices, Kstars)
        
        return latents, firstK_mask
    
    def _teacher_build_latents_sequence_circuit_expert(self, inputs, k, p):
        # =========================================================================
        # 1. 准备数据与环境
        # =========================================================================
        device = self.model.device
        ids = inputs["input_ids"].to(device)
        # 获取 Mask (Method 1 逻辑)
        raw_masks = inputs.get("helper_masks", None)
        
        tea = self._ema.teacher.to(device).eval()
        special_ids, image_token_id, latent_start_id, latent_end_id, latent_pad_id = self._get_special_ids()
        seg_pad_indices = self._find_latent_segments(ids, latent_start_id, latent_end_id, latent_pad_id)
        
        # 定义清理函数
        def cleanup_inputs():
            if "helper_masks" in inputs: inputs.pop("helper_masks")

        if len(seg_pad_indices) == 0:
            cleanup_inputs(); return None, torch.zeros_like(ids, dtype=torch.bool)

        # 获取 Grid 信息 (注意：我们依然需要 thw 来决定空间分辨率，但不需要原始 pixel_values)
        thw = inputs.get("image_grid_thw_latent", None)
        # 确保有图片输入用于专家模型
        if thw is None or 'assistant_image_inputs' not in inputs:
            cleanup_inputs(); return None, torch.zeros_like(ids, dtype=torch.bool)
            
        thw = thw.to(device)
        num_imgs = int(thw.shape[0])
        s_merge = int(getattr(tea.visual, "spatial_merge_size", 2))
        thw_long = thw.to(dtype=torch.long)
        
        # =========================================================================
        # 2. 多专家特征提取 (Method 2 逻辑)
        # =========================================================================
        
        # --- A. HAWP (Wireframe) ---
        hawp_model = self.get_hawp_model(device)
        hawp_user_features_list = []
        with torch.no_grad():
            for img_pil in inputs['assistant_image_inputs']:
                img_np = np.array(img_pil)
                if len(img_np.shape) == 3:
                    gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                else:
                    gray_img = img_np

                # 防止图像过小导致 HAWP 报错
                h, w = gray_img.shape[:2]
                min_hw = min(h, w)
                min_required = 64 
                if min_hw < min_required:
                    scale = float(min_required) / float(min_hw)
                    new_w = int(round(w * scale))
                    new_h = int(round(h * scale))
                    gray_img = cv2.resize(gray_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

                # Convert to tensor and normalize
                image_tensor = torch.from_numpy(gray_img).float() / 255.0
                image_tensor = image_tensor[None, None].to(device)  # [1, 1, H, W]
                
                # Get features from backbone
                outputs, features = hawp_model.backbone(image_tensor)
                hawp_user_features_list.append(features)

        # --- B. DeepLSD (Lines) ---
        conf = {
            'detect_lines': True, 
            'line_detection_params': {
                'merge': False, 'filtering': True, 'grad_thresh': 3, 'grad_nfa': True,
            }
        }
        # 建议：实际部署时不要在这里 load 模型，应在 __init__ 加载
        ckpt_path = '/data/share/JYD/weights/deeplsd/deeplsd_md.tar'
        ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
        net = DeepLSD(conf)
        net.load_state_dict(ckpt['model'])
        net = net.to(device).eval()
        
        deeplsd_user_outputs = []
        with torch.no_grad():
            for img_pil in inputs['assistant_image_inputs']:
                img_np = np.array(img_pil)
                if len(img_np.shape) == 3:
                    gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                else:
                    gray_img = img_np
                
                a = {'image': torch.tensor(gray_img, dtype=torch.float, device=device)[None, None] / 255.}
                base = net.backbone(a['image'])
                deeplsd_user_outputs.append(base)

        # --- C. DINOv2 (Semantic) ---
        dinov2_processor = AutoImageProcessor.from_pretrained('/data/share/JYD/weights/dinov2-giant')
        dinov2_model = AutoModel.from_pretrained('/data/share/JYD/weights/dinov2-giant')
        dinov2_model = dinov2_model.to(device).eval()

        inputs_dino = dinov2_processor(images=inputs['assistant_image_inputs'], return_tensors="pt")
        inputs_dino = {k: v.to(device) for k, v in inputs_dino.items()}
        with torch.no_grad():
            outputs = dinov2_model(**inputs_dino)
        last_hidden_states = outputs.last_hidden_state # [B, L, D]

        # DINO Patch 分割逻辑 (处理 batch > 1 的情况)
        dinov2_batch_size = last_hidden_states.shape[0]
        dinov2_seq_len = last_hidden_states.shape[1]
        
        if dinov2_batch_size == 1 and num_imgs > 1:
            patches_per_img = (dinov2_seq_len - 1) // num_imgs
            dinov2_patches_list = []
            for i in range(num_imgs):
                start_idx = 1 + i * patches_per_img
                end_idx = 1 + (i + 1) * patches_per_img
                if i == num_imgs - 1: end_idx = dinov2_seq_len
                dinov2_patches_list.append(last_hidden_states[:, start_idx:end_idx, :])
        else:
            dinov2_patches_list = [last_hidden_states[i:i+1, 1:, :] for i in range(min(dinov2_batch_size, num_imgs))]
            while len(dinov2_patches_list) < num_imgs:
                dinov2_patches_list.append(dinov2_patches_list[-1])

        # =========================================================================
        # 3. 特征融合与投影 (Expert Fusion & Projection)
        # =========================================================================
        
        pv  = inputs.get("pixel_values_latent", None)
        thw = inputs.get("image_grid_thw_latent", None)
        if pv is None or thw is None:
            return None, torch.zeros_like(ids, dtype=torch.bool)
        pv  = pv.to(device).to(tea.visual.dtype)
        thw = thw.to(device)
        
        # 计算目标维度信息
        num_imgs  = int(thw.shape[0])
        s_merge = int(getattr(tea.visual, "spatial_merge_size", 2))
        thw_long = thw.to(dtype=torch.long)
        tokens_per_img = (thw_long[:,0] * (thw_long[:,1]//s_merge) * (thw_long[:,2]//s_merge))
        
        # 直接用当前输入计算原始视觉特征
        patch_all_original = tea.visual(pv, grid_thw=thw)
        target_dim = patch_all_original.shape[-1]  # 例如 3584
        
        # 3.2 初始化投影层
        if not hasattr(self, '_feature_projection'):
            input_dim = 256 + 64 + 1536 # HAWP + DeepLSD + DINOv2
            self._feature_projection = nn.Linear(input_dim, target_dim).to(device)
            nn.init.xavier_uniform_(self._feature_projection.weight)
            nn.init.zeros_(self._feature_projection.bias)
        
        dinov2_batch_size = last_hidden_states.shape[0]
        dinov2_seq_len = last_hidden_states.shape[1]
        dinov2_hidden_dim = last_hidden_states.shape[2]

        if dinov2_batch_size == 1 and num_imgs > 1:
            # 如果batch_size=1，可能需要手动分割
            # 假设每张图像有 (dinov2_seq_len - 1) // num_imgs 个patches（去掉CLS token）
            patches_per_img = (dinov2_seq_len - 1) // num_imgs
            dinov2_patches_list = []
            for i in range(num_imgs):
                start_idx = 1 + i * patches_per_img  # 跳过CLS token
                end_idx = 1 + (i + 1) * patches_per_img
                if i == num_imgs - 1:
                    end_idx = dinov2_seq_len  # 最后一张图像取剩余所有
                dinov2_patches_list.append(last_hidden_states[:, start_idx:end_idx, :])  # [1, patches, 1536]
        else:
            # 如果batch_size等于图像数量，直接使用
            dinov2_patches_list = [last_hidden_states[i:i+1, 1:, :] for i in range(min(dinov2_batch_size, num_imgs))]
            # 如果图像数量更多，需要重复或处理
            while len(dinov2_patches_list) < num_imgs:
                dinov2_patches_list.append(dinov2_patches_list[-1])

        # 3.3 逐图融合并生成 patch_all
        patch_list = []
        # 同时计算每个图像的 token 范围，供后续 mask 逻辑使用
        tokens_per_img = [] 

        for img_idx in range(num_imgs):
            # 获取该图像的目标空间分辨率 (这是 Method 1 Mask 对齐的关键)
            T, H_target, W_target = thw_long[img_idx].tolist()
            H_spatial = H_target // s_merge
            W_spatial = W_target // s_merge
            
            # 记录 token 数量
            tokens_per_img.append(T * H_spatial * W_spatial)

            # 获取特征
            hawp_feat = hawp_user_features_list[img_idx]
            deeplsd_feat = deeplsd_user_outputs[img_idx]
            dinov2_patches = dinov2_patches_list[img_idx]

            # 空间对齐 (Interpolate)
            hawp_feat_aligned = F.interpolate(hawp_feat, size=(H_spatial, W_spatial), mode='bilinear', align_corners=False)
            deeplsd_feat_aligned = F.interpolate(deeplsd_feat, size=(H_spatial, W_spatial), mode='bilinear', align_corners=False)
            
            # DINOv2 处理
            num_patches = dinov2_patches.shape[1]
            dinov2_hw = int(np.ceil(np.sqrt(num_patches)))
            # Reshape 1D -> 2D
            if dinov2_hw * dinov2_hw == num_patches:
                dinov2_feat_spatial = dinov2_patches.view(1, dinov2_hw, dinov2_hw, -1).permute(0, 3, 1, 2)
            else:
                dinov2_feat_2d = dinov2_patches.view(1, num_patches, -1).permute(0, 2, 1)
                dinov2_feat_spatial = F.adaptive_avg_pool1d(dinov2_feat_2d, output_size=H_spatial * W_spatial).view(1, -1, H_spatial, W_spatial)

            if dinov2_feat_spatial.shape[2] != H_spatial or dinov2_feat_spatial.shape[3] != W_spatial:
                dinov2_feat_aligned = F.interpolate(dinov2_feat_spatial, size=(H_spatial, W_spatial), mode='bilinear', align_corners=False)
            else:
                dinov2_feat_aligned = dinov2_feat_spatial

            # Concat
            combined_feat = torch.cat([hawp_feat_aligned, deeplsd_feat_aligned, dinov2_feat_aligned], dim=1) # [1, 1856, H, W]
            
            # 展平空间维度：[1, 1856, H, W] -> [1, 1856, H*W] -> [H*W, 1856]
            combined_feat_flat = combined_feat.flatten(2).permute(0, 2, 1).squeeze(0)  # [H*W, 1856]
            del combined_feat  # 释放原始特征
            
            # 投影到目标维度：[H*W, 1856] -> [H*W, target_dim]
            combined_feat_proj = self._feature_projection(combined_feat_flat)  # [H*W, target_dim]
            del combined_feat_flat  # 释放展平后的特征
            
            # 如果有多个时间步，需要重复
            if T > 1:
                combined_feat_proj = combined_feat_proj.repeat(T, 1)  # [T*H*W, target_dim]
            
            patch_list.append(combined_feat_proj)

        # 外部三路特征拼接后的 token 表示
        patch_all_ext = torch.cat(patch_list, dim=0)  # [n, target_dim]
        
        # 将原始视觉特征和外部三路特征在通道维度拼接，再线性融合回 target_dim
        if not hasattr(self, '_visual_fusion'):
            self._visual_fusion = nn.Linear(target_dim * 2, target_dim).to(device)
            nn.init.xavier_uniform_(self._visual_fusion.weight)
            nn.init.zeros_(self._visual_fusion.bias)
        
        patch_all_concat = torch.cat([patch_all_original, patch_all_ext], dim=-1)  # [n, 2*target_dim]
        patch_all = self._visual_fusion(patch_all_concat)  # [n, target_dim]
        
        # 计算切片索引 (同 Method 1)
        ends_img = torch.cumsum(torch.tensor(tokens_per_img, device=device), dim=0).tolist()
        starts_img = [0] + ends_img[:-1]
        slices_per_img = [(int(st), int(ed)) for st, ed in zip(starts_img, ends_img)]

        # 清理中间显存
        del hawp_user_features_list, deeplsd_user_outputs, dinov2_patches_list, patch_list
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        # =========================================================================
        # 4. Mask 序列化处理逻辑 (Method 1 核心逻辑)
        # =========================================================================
        latents_list = []
        Kstars = []

        for seg_idx, pad_pos in enumerate(seg_pad_indices):
            if len(pad_pos) == 0:
                Kstars.append(0); continue
            
            assert seg_idx < num_imgs
            st_img, ed_img = slices_per_img[seg_idx]
            # 这里拿到的 ei 已经是投影后的专家混合特征了
            ei = patch_all[st_img:ed_img, :] # [N_tokens, Dim]

            selected_tokens = None
            
            # --- Mask 处理 ---
            if raw_masks is not None and seg_idx < len(raw_masks) and raw_masks[seg_idx] is not None:
                curr_mask = raw_masks[seg_idx].to(device).float()
                
                if curr_mask.dim() == 2: curr_mask = curr_mask.unsqueeze(0).unsqueeze(0)
                elif curr_mask.dim() == 3: curr_mask = curr_mask.unsqueeze(0)

                # 计算 Feature Map 的 Grid 尺寸 (必须与 ei 的空间尺寸一致)
                grid_h = int(thw_long[seg_idx, 1] // s_merge)
                grid_w = int(thw_long[seg_idx, 2] // s_merge)

                # 【关键点 1】Adaptive Max Pool 确保不漏掉细微 Mask
                resized_mask = F.adaptive_max_pool2d(curr_mask, output_size=(grid_h, grid_w))
                flat_mask_values = resized_mask.flatten() 

                # 长度对齐
                if flat_mask_values.shape[0] > ei.shape[0]:
                    flat_mask_values = flat_mask_values[:ei.shape[0]]
                elif flat_mask_values.shape[0] < ei.shape[0]:
                    padding = torch.zeros(ei.shape[0] - flat_mask_values.shape[0], device=device)
                    flat_mask_values = torch.cat([flat_mask_values, padding])

                # 【关键点 2】筛选有效区域
                valid_indices = torch.nonzero(flat_mask_values > 0.0).squeeze(1)
                
                if valid_indices.numel() == 0:
                    selected_tokens = ei[:k]
                else:
                    valid_tokens = ei[valid_indices]
                    valid_values = flat_mask_values[valid_indices]

                    # 【关键点 3】根据 Mask 值排序 (拓扑顺序：1.0 -> 2.x -> 3.x)
                    if valid_values.max() > 1.0 + 1e-4:
                        sort_idx = torch.argsort(valid_values, descending=False)
                        sorted_tokens = valid_tokens[sort_idx]
                    else:
                        sorted_tokens = valid_tokens

                    # 【关键点 4】特征压缩
                    num_cand = sorted_tokens.shape[0]
                    if num_cand <= k:
                        selected_tokens = sorted_tokens
                    else:
                        # 有序压缩
                        input_tensor = sorted_tokens.transpose(0, 1).unsqueeze(0)
                        pooled = F.adaptive_avg_pool1d(input_tensor, output_size=k)
                        selected_tokens = pooled.squeeze(0).transpose(0, 1)

            else:
                # 无 Mask 兜底
                selected_tokens = ei[:k]

            # --- 结果收集 ---
            Kstar = min(len(pad_pos), selected_tokens.shape[0])
            Kstars.append(Kstar)

            if Kstar > 0:
                latents_list.append(selected_tokens[:Kstar])
        
        # 清理 Mask 输入
        cleanup_inputs()

        if len(latents_list) == 0:
             return None, torch.zeros_like(ids, dtype=torch.bool)
             
        latents = torch.cat(latents_list, dim=0).unsqueeze(0)
        # 生成 Mask
        firstK_mask = self._build_firstK_mask(ids, seg_pad_indices, Kstars)
        
        return latents, firstK_mask
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        k = getattr(self.model.config, "latent_size", 8)
        #teacher_latents, firstK_mask = self._teacher_build_latents(inputs, k=k, p=self.coverage_p)
        #teacher_latents, firstK_mask = self._teacher_build_latents_sequence(inputs, k=k, p=self.coverage_p)
        teacher_latents, firstK_mask = self._teacher_build_latents_sequence_circuit_expert(inputs, k=k, p=self.coverage_p)
        if "helper_masks" in inputs:
            inputs.pop("helper_masks")
        with torch.no_grad():
            pv_lat = inputs.get("pixel_values_latent", None)
            thw_lat = inputs.get("image_grid_thw_latent", None)
            if pv_lat is not None and thw_lat is not None:
                _ = self.model.visual(
                    pv_lat.to(self.model.device).to(self.model.visual.dtype),
                    grid_thw=thw_lat.to(self.model.device)
                )
        
        if teacher_latents is None:
            ce_loss, outputs = super().compute_loss(
                model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
            )
            return (ce_loss, outputs) if return_outputs else ce_loss

        if teacher_latents.dim() == 2:
            teacher_latents = teacher_latents.unsqueeze(0)
        S = int(firstK_mask.sum().item())
        if teacher_latents.shape[1] != S:
            ce_loss, outputs = super().compute_loss(
                model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
            )
            return (ce_loss, outputs) if return_outputs else ce_loss

        mod_inputs = dict(inputs)
        mod_inputs.pop("pixel_values_latent", None)
        mod_inputs["latent_hidden_states"] = teacher_latents.to(self.model.device).to(self.model.dtype)
        mod_inputs["image_out_mask"] = firstK_mask

        ce_loss, outputs = super().compute_loss(
            model, mod_inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        if self.sim_weight == 0.0:
            return (ce_loss, outputs) if return_outputs else ce_loss

        pred_h = outputs.hidden_states
        inp_h  = outputs.inputs_embeds
        B, T, H = pred_h.shape
        if T <= 1:
            return (ce_loss, outputs) if return_outputs else ce_loss

        mask = mod_inputs["image_out_mask"][:, -(T - 1):].to(pred_h.device).bool()
        if not mask.any():
            return (ce_loss, outputs) if return_outputs else ce_loss

        pred = pred_h[..., :-1, :][mask].contiguous().float()
        gt   = inp_h[...,  1:, :][mask].contiguous().detach().float()
        gt = gt + 0.01 * torch.randn_like(gt)

        cos = F.cosine_similarity(gt, pred, dim=-1).mean()
        sim_loss = 1.0 - cos
        loss = self.ce_weight * ce_loss + self.sim_weight * sim_loss
        # ================== 【新增】打印逻辑开始 ==================
        # 这里的判断是为了防止多卡训练(DeepSpeed)时每个GPU都打印一遍，造成刷屏
        # self.state.global_step % 1 == 0 表示每一步都打印，嫌太快可以改成 10 或 100
        if self.is_world_process_zero() and self.state.global_step % 10 == 0:
            logging.info(
                f"[Step {self.state.global_step}] "
                f"Total: {loss.item():.4f} | "
                f"CE: {ce_loss.item():.4f} | "
                f"Sim: {sim_loss.item():.4f} | "
                f"Cos: {cos.item():.4f}"
            )
        # ================== 【新增】打印逻辑结束 ==================
        return (loss, outputs) if return_outputs else loss

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if not hasattr(self, "_ema") or self._ema is None:
            return
        d = float(self._ema.decay)
        with torch.no_grad():
            if _HAS_DS and any(hasattr(p, "ds_id") for p in self.model.parameters()):
                for p_t, p_s in zip(self._ema.teacher.parameters(), self.model.parameters()):
                    with deepspeed.zero.GatheredParameters(p_s, modifier_rank=0):
                        if p_s.data.numel() == 0:
                            continue
                        p_data = p_s.data
                        if p_data.device != p_t.data.device:
                            p_data = p_data.to(p_t.data.device)
                        p_t.data.mul_(d).add_(p_data, alpha=(1.0 - d))
            else:
                self._ema.update(self.model)

class CustomTrainerStage2(SFTTrainer):
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss and additionally compute token accuracies
        """
        (ce_loss, outputs) = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        loss = ce_loss
        if self.is_world_process_zero() and self.state.global_step % 10 == 0:
            logging.info(
                f"[Step {self.state.global_step}] "
                f"Total: {loss.item():.4f} | "
                f"CE: {ce_loss.item():.4f} | "
            )
        return (loss, outputs) if return_outputs else loss