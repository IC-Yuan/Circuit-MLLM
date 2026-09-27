from trl import SFTTrainer
import copy
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        ce_weight: float = 1.0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.sim_weight = float(sim_weight)
        self.ce_weight = float(ce_weight)
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

    def _get_latent_token_ids(self):
        tok = self.processing_class
        def get_id(token):
            return tok(token, return_tensors="pt")["input_ids"][0, 0].item()

        return get_id("<|latent_start|>"), get_id("<|latent_end|>"), get_id("<|latent_pad|>")
    
    def _teacher_build_latents(self, inputs, k):
        # 1. 准备数据
        device = self.model.device
        ids = inputs["input_ids"].to(device)
        
        # 获取 helper_masks (List[Tensor])
        # 注意：这里我们不做 pop，因为如果在中途 return，inputs 还没被修改是安全的
        # 我们统一在最后或者 return 前做清理
        raw_masks = inputs.get("helper_masks", None)
        
        tea = self._ema.teacher.to(device).eval()
        latent_start_id, latent_end_id, latent_pad_id = self._get_latent_token_ids()

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
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        k = getattr(self.model.config, "latent_size", 8)
        teacher_latents, firstK_mask = self._teacher_build_latents(inputs, k=k)
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
        return (loss, outputs) if return_outputs else loss
