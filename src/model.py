"""分割模型：SegFormer（MiT backbone）或 DeepLabV3+（ResNet101 + ASPP）。
Segmentation models: SegFormer (MiT backbone) or DeepLabV3+ (ResNet101 + ASPP).

DeepLabV3+ 优势：ASPP 多尺度池化直接利好不规则边界（HD95），且 ResNet 卷积编码器
对 2700 张数据量更友好（不像纯 ViT 那么吃数据）。
DeepLabV3+ advantage: ASPP multi-scale pooling directly helps irregular boundaries (HD95),
and the ResNet conv. encoder is more data-efficient than pure ViT for 2700 images.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

from .config import PRETRAINED


# ----- SegFormer -----
def build_segformer(variant='b2', num_labels=2):
    """variant: 'b0'/'b1'/'b2'/'b3'（越大越准越慢）。/ bigger = more accurate but slower."""
    path = PRETRAINED[variant]
    model = SegformerForSemanticSegmentation.from_pretrained(
        path, num_labels=num_labels, ignore_mismatched_sizes=True
    )
    return model


# ----- DeepLabV3+ ResNet101 -----
class DeepLabModel(nn.Module):
    """DeepLabV3+ wrapper：forward(pixel_values=)->.logits，接口对齐 SegFormer。
    DeepLabV3+ wrapper: forward(pixel_values=)->.logits, matching SegFormer's API."""
    def __init__(self, num_labels=2):
        super().__init__()
        import torchvision
        weights = torchvision.models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT
        self.backbone = torchvision.models.segmentation.deeplabv3_resnet101(weights=weights)
        # 替换分类头为 2/5 通道 / replace classifier head for 2/5 classes
        self.backbone.classifier[-1] = nn.Conv2d(256, num_labels, 1)
        if hasattr(self.backbone, 'aux_classifier') and self.backbone.aux_classifier is not None:
            self.backbone.aux_classifier[-1] = nn.Conv2d(256, num_labels, 1)

    def forward(self, pixel_values):
        out = self.backbone(pixel_values)
        # 返回同名 attribute .logits 对齐 SegFormer API / return .logits to match SegFormer
        return type('DLOut', (), {'logits': out['out']})()


def build_deeplabv3(num_labels=2):
    """DeepLabV3+ ResNet101，COCO/VOC 预训练。适合边界精细、数据量不大的情况。
    DeepLabV3+ ResNet101, COCO/VOC pretrained. Good for fine boundaries & limited data."""
    return DeepLabModel(num_labels)


# ----- 统一 builder / unified builder -----
def build_model(model_type='segformer', variant='b2', num_labels=2):
    """model_type: 'segformer' or 'deeplab'. Returns a segmentation model."""
# ----- ConvNeXt + FPN -----
class ConvNeXtSeg(nn.Module):
    """ConvNeXt backbone + FPN decoder。CNN 天然高频感知强于 ViT，边界更锐。
    ConvNeXt backbone + FPN decoder. CNN naturally better at HF than ViT, sharper boundaries."""
    def __init__(self, version='base', num_labels=2):
        super().__init__()
        import timm
        self.backbone = timm.create_model(f'convnext_{version}', pretrained=True,
                                          features_only=True, out_indices=(0, 1, 2, 3))
        info = self.backbone.feature_info
        chs = [info.channels(i) for i in range(4)]
        self.up_proj = nn.ModuleList()
        self.up_fuse = nn.ModuleList()
        for i in range(3, 0, -1):
            self.up_proj.append(nn.Conv2d(chs[i], chs[i - 1], 1))
            self.up_fuse.append(nn.Sequential(
                nn.Conv2d(chs[i - 1] * 2, chs[i - 1], 3, padding=1),
                nn.BatchNorm2d(chs[i - 1]),
                nn.ReLU(inplace=True),
            ))
        self.head = nn.Conv2d(chs[0], num_labels, 1)

    def forward(self, pixel_values):
        feats = self.backbone(pixel_values)  # [1/4,1/8,1/16,1/32]
        x = feats[-1]
        for i in range(3):
            x = self.up_proj[i](x)
            x = F.interpolate(x, size=feats[-2 - i].shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, feats[-2 - i]], dim=1)
            x = self.up_fuse[i](x)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        return type('DLOut', (), {'logits': self.head(x)})()


def build_convnext(version='base', num_labels=2):
    return ConvNeXtSeg(version, num_labels)


# ----- Attention wrapper (通道注意力，即插即用) / Attention wrapper (channel attention, plug-in) -----
class AttnWrapper(nn.Module):
    """在模型 logits 后加通道注意力，即插即用不改变训练逻辑。
    Apply channel attention on logits, plug-and-play, no training-logic change."""
    def __init__(self, model, channels):
        super().__init__()
        from .attention import ChannelAttention2D
        self.model = model
        self.attn = ChannelAttention2D(channels)

    def forward(self, pixel_values):
        out = self.model(pixel_values=pixel_values)
        out.logits = self.attn(out.logits)  # 即插通道精炼 / per-channel refinement
        return out


# ----- 统一 builder / unified builder -----
def build_model(model_type='segformer', variant='b2', num_labels=2):
    """model_type: 'segformer' / 'deeplab' / 'convnext'. Return a segmentation model."""
    if model_type == 'convnext':
        return build_convnext(variant, num_labels)
    if model_type == 'deeplab':
        return build_deeplabv3(num_labels)
    return build_segformer(variant, num_labels)
