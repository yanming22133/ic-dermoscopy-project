"""SegFormer 分割模型。用本地预训练权重（ADE20K 微调版），替换头为 num_labels=2。
SegFormer segmentation model. Uses local pretrained weights (ADE20K-finetuned),
replaces the head with num_labels=2."""
import torch
from transformers import SegformerForSemanticSegmentation

from .config import PRETRAINED


def build_segformer(variant='b2', num_labels=2):
    """variant: 'b0'（轻量，快速）或 'b2'（主力）。返回 SegformerForSemanticSegmentation。
    variant: 'b0' (lightweight, fast) or 'b2' (main). Returns SegformerForSemanticSegmentation."""
    path = PRETRAINED[variant]
    model = SegformerForSemanticSegmentation.from_pretrained(
        path, num_labels=num_labels, ignore_mismatched_sizes=True
    )
    return model
