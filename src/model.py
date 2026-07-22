"""分割模型：SegFormer（MiT backbone）或 DeepLabV3+（ResNet101 + ASPP）。
Segmentation models: SegFormer (MiT backbone) or DeepLabV3+ (ResNet101 + ASPP).

DeepLabV3+ 优势：ASPP 多尺度池化直接利好不规则边界（HD95），且 ResNet 卷积编码器
对 2700 张数据量更友好（不像纯 ViT 那么吃数据）。
DeepLabV3+ advantage: ASPP multi-scale pooling directly helps irregular boundaries (HD95),
and the ResNet conv. encoder is more data-efficient than pure ViT for 2700 images.
"""
import torch
import torch.nn as nn
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
    if model_type == 'deeplab':
        return build_deeplabv3(num_labels)
    return build_segformer(variant, num_labels)
