"""PEFT-SAM 分割模型：冻结 SAM image_encoder + prompt_encoder，只训练 mask_decoder + 可选 LoRA。
PEFT-SAM segmentation model: freeze SAM image_encoder + prompt_encoder,
train only mask_decoder + optional LoRA.

2026 SOTA: PEFT-MedSAM 通过仅微调解码器 + LoRA adapter 在医学分割上达 IoU 0.8918。
2026 SOTA: PEFT-MedSAM achieves IoU 0.8918 on medical segmentation by fine-tuning
only the mask decoder with LoRA adapters on the image encoder.

用法 / Usage:
    from .model_sam import SamSegModel, build_sam
    model = SamSegModel(num_labels=2, use_lora=True)
    # 训练 / training:  logits = model(pixel_values=img, gt_mask=mask).logits
    # 推理 / inference: logits = model(pixel_values=img, input_boxes=pred_boxes).logits
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SamModel

from .config import PRETRAINED

# 尝试导入 peft 用于 LoRA 微调 / Try importing peft for LoRA fine-tuning
try:
    from peft import LoraConfig, get_peft_model
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False


# ============================================================
# 工具函数 / Utility Functions
# ============================================================

def get_bbox_from_mask(mask, pad=5):
    """从 GT mask 计算 bbox，用于 SAM box prompt。空 mask 返回全图默认框。
    Compute bbox from GT mask for SAM box prompt. Empty mask → full-image default box.

    Args:
        mask: [B, H, W] 或 [H, W] 0/1 二值 mask / binary mask
        pad: bbox 膨胀像素数 / padding pixels around the lesion

    Returns:
        boxes: [B, 1, 4] 格式 (x1, y1, x2, y2) 像素坐标 / pixel coords
    """
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)  # [H, W] → [1, H, W]
    B, H, W = mask.shape
    boxes_list = []
    for b in range(B):
        m = mask[b]
        if m.sum() == 0:
            # 空 mask（无病灶）→ 全图默认框，SAM 应预测全背景
            # Empty mask (no lesion) → full-image default box, SAM should predict all bg
            boxes_list.append([0, 0, W - 1, H - 1])
        else:
            ys, xs = torch.where(m > 0)
            x1 = max(0, xs.min().item() - pad)
            y1 = max(0, ys.min().item() - pad)
            x2 = min(W - 1, xs.max().item() + pad)
            y2 = min(H - 1, ys.max().item() + pad)
            boxes_list.append([x1, y1, x2, y2])
    boxes = torch.tensor(boxes_list, dtype=torch.float32)  # [B, 4]
    return boxes.unsqueeze(1)  # [B, 1, 4] — SAM 期望每个 prompt 一个维度


# ============================================================
# SamSegModel — PEFT-SAM wrapper / PEFT-SAM 包装器
# ============================================================

class SamSegModel(nn.Module):
    """SAM 分割模型 wrapper：冻结 encoder，训练 decoder，接口对齐 SegFormer (.logits)。
    SAM segmentation wrapper: freeze encoder, train decoder, API matching SegFormer (.logits).

    参数 / Parameters:
        num_labels: 分类数（二分类=2）/ number of classes (binary=2)
        use_lora: 是否对 image_encoder 加 LoRA adapter / whether to apply LoRA to encoder
        lora_rank: LoRA 秩 / LoRA rank (default 4)
    """

    def __init__(self, num_labels=2, use_lora=False, lora_rank=4):
        super().__init__()
        self.num_labels = num_labels

        # 加载 SAM 模型 / Load SAM model
        sam_path = PRETRAINED.get('sam', 'facebook/sam-vit-base')
        print(f'[SamSegModel] Loading SAM from {sam_path}...', flush=True)
        self.sam = SamModel.from_pretrained(sam_path)

        # 冻结 encoder / Freeze encoder
        self._freeze_encoder()

        # 可选 LoRA 微调 image_encoder / Optional LoRA on image_encoder
        self.use_lora = use_lora and HAS_PEFT
        if use_lora and not HAS_PEFT:
            print('[SamSegModel] peft not installed, skipping LoRA. '
                  'Install with: pip install peft', flush=True)
        if self.use_lora:
            self._apply_lora(lora_rank)

        # mask_decoder 保持可训练 / mask_decoder remains trainable
        self._unfreeze_decoder()

        # 统计参数量 / Log parameter counts
        self._log_params()

    def _freeze_encoder(self):
        """冻结 vision_encoder 和 prompt_encoder。
        Freeze vision_encoder and prompt_encoder."""
        if hasattr(self.sam, 'vision_encoder'):
            self.sam.vision_encoder.requires_grad_(False)
            print('[SamSegModel] Frozen: vision_encoder', flush=True)
        if hasattr(self.sam, 'prompt_encoder'):
            self.sam.prompt_encoder.requires_grad_(False)
            print('[SamSegModel] Frozen: prompt_encoder', flush=True)

    def _unfreeze_decoder(self):
        """确保 mask_decoder 可训练 / Ensure mask_decoder is trainable."""
        if hasattr(self.sam, 'mask_decoder'):
            self.sam.mask_decoder.requires_grad_(True)
            print('[SamSegModel] Trainable: mask_decoder', flush=True)

    def _apply_lora(self, rank=4):
        """对 vision_encoder 的 attention 层加 LoRA adapter。
        Apply LoRA adapters to vision_encoder attention layers.

        SAM ViT 的 attention 使用 qkv 合并投影，target 用 ["qkv"] 覆盖 Q/K/V。
        SAM ViT uses fused qkv projection; target ["qkv"] covers Q/K/V together.
        """
        encoder = self.sam.vision_encoder

        # 尝试找到 attention 中的 qkv 层 / Try to find qkv layer in attention modules
        # SAM 使用 SamVisionLayer 内的 Attention，其中的 qkv 是 nn.Linear
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=rank * 2,
            target_modules=["qkv"],  # SAM ViT 的 attention 合并 QKV 投影 / SAM's fused QKV projection
            lora_dropout=0.1,
            bias="none",
        )
        try:
            peft_model = get_peft_model(encoder, lora_config)
            # 替换 self.sam.vision_encoder 为 LoRA 版本 / Replace with LoRA version
            self.sam.vision_encoder = peft_model
            print(f'[SamSegModel] LoRA rank={rank} applied to vision_encoder (target: qkv)', flush=True)
        except Exception as e:
            # 如果 qkv 名字不匹配，尝试其他常见名字 / If qkv doesn't match, try other common names
            print(f'[SamSegModel] LoRA with target=["qkv"] failed: {e}', flush=True)
            try:
                lora_config2 = LoraConfig(
                    r=rank,
                    lora_alpha=rank * 2,
                    target_modules=["q_proj", "v_proj"],  # HF 标准命名 / HF standard naming
                    lora_dropout=0.1,
                    bias="none",
                )
                peft_model2 = get_peft_model(encoder, lora_config2)
                self.sam.vision_encoder = peft_model2
                print(f'[SamSegModel] LoRA rank={rank} applied to vision_encoder (target: q_proj, v_proj)', flush=True)
            except Exception as e2:
                print(f'[SamSegModel] LoRA application failed: {e2}, continuing without LoRA', flush=True)
                self.use_lora = False

    def _log_params(self):
        """统计并打印可训练参数 / Count and log trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'[SamSegModel] Total params: {total/1e6:.1f}M | '
              f'Trainable: {trainable/1e6:.2f}M ({100*trainable/total:.1f}%)', flush=True)

    def forward(self, pixel_values, input_boxes=None, gt_mask=None):
        """SAM 分割前向传播 / SAM segmentation forward pass.

        Args:
            pixel_values: [B, 3, H, W] ImageNet 归一化后的图像 / ImageNet-normalized images
            input_boxes: [B, N, 4] SAM box prompts，推理时使用 / for inference
            gt_mask:     [B, H, W] 0/1 GT mask，训练时自动算 bbox / for training, auto-compute bbox

        Returns:
            带 .logits 属性的 namespace 对象，logits 形状 [B, num_labels, H, W]
            Object with .logits attribute, shape [B, num_labels, H, W]
        """
        B, _, H, W = pixel_values.shape

        # ---- 确定 input_boxes / Determine input_boxes ----
        if input_boxes is None and gt_mask is not None:
            # 训练模式：从 GT mask 提取 bbox / Training: extract bbox from GT mask
            input_boxes = get_bbox_from_mask(gt_mask).to(pixel_values.device)
        elif input_boxes is None:
            # 无任何 prompt：用全图默认框 / No prompt at all: use full-image default box
            input_boxes = torch.tensor(
                [[[0, 0, W - 1, H - 1]]], device=pixel_values.device
            ).repeat(B, 1, 1)

        # ---- SAM 前向 / SAM forward ----
        outputs = self.sam(
            pixel_values=pixel_values,
            input_boxes=input_boxes,
            multimask_output=True,  # 输出 3 个候选 mask，按 iou_scores 选最佳
                                    # Output 3 candidate masks, pick best via iou_scores
        )

        # pred_masks: [B, 1, 3, 256, 256] → squeeze(point_batch) → [B, 3, 256, 256]
        pred_masks = outputs.pred_masks.squeeze(1)  # [B, 3, Hm=256, Wm=256]
        # iou_scores: [B, 1, 3] → squeeze → [B, 3]
        iou_scores = outputs.iou_scores.squeeze(1)  # [B, 3]

        # ---- 按 IoU 分数选最佳 mask / Pick best mask by IoU score ----
        best_idx = iou_scores.argmax(dim=1)  # [B]
        best_mask = pred_masks[torch.arange(B, device=pred_masks.device), best_idx]  # [B, 256, 256]

        # ---- 上采样到输入分辨率 / Upsample to input resolution ----
        if best_mask.shape[-2:] != (H, W):
            best_mask = F.interpolate(
                best_mask.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False
            ).squeeze(1)  # [B, H, W]

        # ---- 转为 2 通道 logits 对齐 SegFormer API / Convert to 2-channel logits ----
        # best_mask 是 pre-sigmoid logit。
        # 2 通道 CE: class 0 (bg) = 0, class 1 (fg) = best_mask
        # 则 P(fg) = softmax([0, z])[1] = sigmoid(z)，数学等价于 SAM 的 sigmoid 输出。
        #
        # best_mask is pre-sigmoid logit.
        # 2-channel CE: class 0 (bg) = 0, class 1 (fg) = best_mask
        # Then P(fg) = softmax([0, z])[1] = sigmoid(z), mathematically equivalent to SAM sigmoid.
        logits = torch.stack([
            torch.zeros_like(best_mask),  # bg channel
            best_mask,                     # fg channel (pre-sigmoid logit)
        ], dim=1)  # [B, 2, H, W]

        # 返回 namespace 对象，.logits 属性对齐 SegFormer / Return namespace with .logits
        return type('SAMOut', (), {'logits': logits})()


# ============================================================
# Builder 函数 / Builder Function
# ============================================================

def build_sam(num_labels=2, use_lora=False, lora_rank=4):
    """构建 PEFT-SAM 模型 / Build PEFT-SAM model.

    Args:
        num_labels: 分类数（二分类=2）/ number of classes
        use_lora: 是否对编码器加 LoRA / whether to add LoRA to encoder
        lora_rank: LoRA 秩 / LoRA rank

    Returns:
        SamSegModel 实例 / SamSegModel instance
    """
    return SamSegModel(num_labels=num_labels, use_lora=use_lora, lora_rank=lora_rank)
