import os
import logging

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib.cm as cm
from PIL import Image


def visualize_expert_features(
    original_images,
    hawp_features_list,
    deeplsd_outputs,
    last_hidden_states,
    dinov2_model,
    inputs_dino,
    save_dir: str = "./expert_visualizations",
    step: int = None,
):
    """
    可视化三个专家模型的特征，将特征转换为热力图并叠加到原图上

    Args:
        original_images: List[PIL.Image] - 原始图像列表
        hawp_features_list: List[Tensor] - HAWP特征列表，每个元素形状为 [1, C, H, W]
        deeplsd_outputs: List[Tensor] - DeepLSD特征列表，每个元素形状为 [1, C, H, W]
        last_hidden_states: Tensor - DINOv2特征，形状为 [batch_size, seq_len, hidden_dim]
        dinov2_model: DINOv2模型，用于获取patch_size等信息
        inputs_dino: dict - DINOv2的输入，包含pixel_values等信息
        save_dir: str - 保存目录
        step: int - 当前训练步数（可选，用于命名文件）
    """
    os.makedirs(save_dir, exist_ok=True)

    num_images = len(original_images)

    # 处理DINOv2特征：将序列特征转换为空间特征图
    dinov2_batch_size = last_hidden_states.shape[0]
    dinov2_seq_len = last_hidden_states.shape[1]

    # 获取DINOv2的patch_size
    patch_size = None
    if hasattr(dinov2_model.config, "patch_size"):
        patch_size = dinov2_model.config.patch_size
    elif hasattr(dinov2_model.config, "vision_config") and hasattr(
        dinov2_model.config.vision_config, "patch_size"
    ):
        patch_size = dinov2_model.config.vision_config.patch_size
    elif hasattr(dinov2_model.config, "image_patch_size"):
        patch_size = dinov2_model.config.image_patch_size

    # 获取输入图像尺寸
    H_in, W_in = inputs_dino["pixel_values"].shape[-2], inputs_dino["pixel_values"].shape[-1]

    # 分割DINOv2特征到每张图像
    if dinov2_batch_size == 1 and num_images > 1:
        patches_per_img = (dinov2_seq_len - 1) // num_images
        dinov2_patches_list = []
        for i in range(num_images):
            start_idx = 1 + i * patches_per_img
            end_idx = 1 + (i + 1) * patches_per_img
            if i == num_images - 1:
                end_idx = dinov2_seq_len
            dinov2_patches_list.append(last_hidden_states[:, start_idx:end_idx, :])
    else:
        dinov2_patches_list = [
            last_hidden_states[i : i + 1, 1:, :] for i in range(min(dinov2_batch_size, num_images))
        ]
        while len(dinov2_patches_list) < num_images:
            dinov2_patches_list.append(dinov2_patches_list[-1])

    # 逐图像处理
    for img_idx in range(num_images):
        img_pil = original_images[img_idx]
        img_np = np.array(img_pil)

        # 处理图像格式：确保是RGB格式
        if len(img_np.shape) == 2:
            # 灰度图转RGB
            img_np_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif img_np.shape[2] == 4:
            # RGBA转RGB
            img_np_rgb = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        else:
            img_np_rgb = img_np.copy()

        img_h, img_w = img_np_rgb.shape[:2]

        # 保存原始图像
        prefix = f"step_{step}_" if step is not None else ""
        orig_path = os.path.join(save_dir, f"{prefix}img_{img_idx}_original.png")
        img_pil.save(orig_path)

        # 1. 处理HAWP特征
        hawp_feat = hawp_features_list[img_idx]  # [1, C, H, W]
        hawp_feat_np = hawp_feat.squeeze(0).cpu().numpy()  # [C, H, W]
        # 对通道维度求平均，得到单通道热力图
        hawp_heatmap = np.mean(np.abs(hawp_feat_np), axis=0)  # [H, W]
        # 归一化
        hawp_heatmap = (hawp_heatmap - hawp_heatmap.min()) / (
            hawp_heatmap.max() - hawp_heatmap.min() + 1e-8
        )
        # 调整到原图尺寸
        hawp_heatmap_resized = cv2.resize(hawp_heatmap, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        # 应用颜色映射
        hawp_colormap = cm.jet(hawp_heatmap_resized)[:, :, :3]  # [H, W, 3]
        hawp_colormap = (hawp_colormap * 255).astype(np.uint8)
        # PIL图像是RGB，cv2.addWeighted也使用RGB格式
        hawp_overlay = cv2.addWeighted(img_np_rgb, 0.6, hawp_colormap, 0.4, 0)
        hawp_path = os.path.join(save_dir, f"{prefix}img_{img_idx}_hawp_heatmap.png")
        Image.fromarray(hawp_overlay).save(hawp_path)

        # 2. 处理DeepLSD特征
        deeplsd_feat = deeplsd_outputs[img_idx]  # [1, C, H, W]
        deeplsd_feat_np = deeplsd_feat.squeeze(0).cpu().numpy()  # [C, H, W]
        deeplsd_heatmap = np.mean(np.abs(deeplsd_feat_np), axis=0)  # [H, W]
        deeplsd_heatmap = (deeplsd_heatmap - deeplsd_heatmap.min()) / (
            deeplsd_heatmap.max() - deeplsd_heatmap.min() + 1e-8
        )
        deeplsd_heatmap_resized = cv2.resize(
            deeplsd_heatmap, (img_w, img_h), interpolation=cv2.INTER_LINEAR
        )
        deeplsd_colormap = cm.viridis(deeplsd_heatmap_resized)[:, :, :3]
        deeplsd_colormap = (deeplsd_colormap * 255).astype(np.uint8)
        deeplsd_overlay = cv2.addWeighted(img_np_rgb, 0.6, deeplsd_colormap, 0.4, 0)
        deeplsd_path = os.path.join(save_dir, f"{prefix}img_{img_idx}_deeplsd_heatmap.png")
        Image.fromarray(deeplsd_overlay).save(deeplsd_path)

        # 3. 处理DINOv2特征
        dinov2_patches = dinov2_patches_list[img_idx]  # [1, num_patches, D]
        num_patches = dinov2_patches.shape[1]
        D = dinov2_patches.shape[2]

        # 将序列特征转换为2D空间特征图
        if patch_size is not None:
            Hp = int(H_in) // int(patch_size)
            Wp = int(W_in) // int(patch_size)
            expected = Hp * Wp

            if expected == num_patches:
                dinov2_feat_spatial = (
                    dinov2_patches.view(1, Hp, Wp, D).permute(0, 3, 1, 2).squeeze(0)
                )  # [D, Hp, Wp]
            else:
                # Fallback: 尝试推断空间尺寸
                aspect = float(W_in) / float(H_in + 1e-6)
                Hp_guess = int(round((num_patches / aspect) ** 0.5))
                Hp_guess = max(1, Hp_guess)
                Wp_guess = max(1, num_patches // Hp_guess)

                if Hp_guess * Wp_guess == num_patches:
                    dinov2_feat_spatial = (
                        dinov2_patches.view(1, Hp_guess, Wp_guess, D)
                        .permute(0, 3, 1, 2)
                        .squeeze(0)
                    )
                else:
                    # 使用1D池化
                    dinov2_feat_2d = dinov2_patches.permute(0, 2, 1)  # [1, D, N]
                    target_spatial = int(np.sqrt(num_patches))
                    dinov2_feat_spatial = F.adaptive_avg_pool1d(
                        dinov2_feat_2d, output_size=target_spatial * target_spatial
                    ).view(D, target_spatial, target_spatial)
        else:
            # 使用1D池化
            target_spatial = int(np.sqrt(num_patches))
            dinov2_feat_2d = dinov2_patches.permute(0, 2, 1)  # [1, D, N]
            dinov2_feat_spatial = F.adaptive_avg_pool1d(
                dinov2_feat_2d, output_size=target_spatial * target_spatial
            ).view(D, target_spatial, target_spatial)

        dinov2_feat_np = dinov2_feat_spatial.cpu().numpy()  # [D, Hp, Wp]
        dinov2_heatmap = np.mean(np.abs(dinov2_feat_np), axis=0)  # [Hp, Wp]
        dinov2_heatmap = (dinov2_heatmap - dinov2_heatmap.min()) / (
            dinov2_heatmap.max() - dinov2_heatmap.min() + 1e-8
        )
        dinov2_heatmap_resized = cv2.resize(
            dinov2_heatmap, (img_w, img_h), interpolation=cv2.INTER_LINEAR
        )
        dinov2_colormap = cm.plasma(dinov2_heatmap_resized)[:, :, :3]
        dinov2_colormap = (dinov2_colormap * 255).astype(np.uint8)
        dinov2_overlay = cv2.addWeighted(img_np_rgb, 0.6, dinov2_colormap, 0.4, 0)
        dinov2_path = os.path.join(save_dir, f"{prefix}img_{img_idx}_dinov2_heatmap.png")
        Image.fromarray(dinov2_overlay).save(dinov2_path)

        logging.info(f"Saved visualizations for image {img_idx} to {save_dir}")


