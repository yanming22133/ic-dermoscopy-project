"""Task2: 逐属性独立头（GIIN CELM 精简版）+ 属性图卷积层。
Task2: per-attribute independent heads (simplified GIIN CELM) + attribute GCN layer.

CELM: 5 个独立的 1×1 Conv 小头，每个属性学自己的特征子空间——全局特征
共享 encoder，但每个属性有专属的 projection 头。比共享 softmax 更能区分属性。
5 independent 1×1 Conv heads, each attribute learns its own feature subspace.
Shared encoder features, but per-attribute projections — better discrimination than shared softmax.

GCN: 5 属性成对条件概率矩阵 → 一层 GCN → 修正 logits。推理侧，不改训练。
5-attribute pairwise conditional prob matrix → 1-layer GCN → corrected logits. Inference-only.
"""
import torch
import torch.nn as nn
import numpy as np


class PerAttrHead(nn.Module):
    """GIIN CELM 精简版：5 个独立 1×1 Conv → 逐属性 sigmoid logits。
    simplified GIIN CELM: 5 independent 1×1 Conv → per-attr sigmoid logits."""

    def __init__(self, in_channels=256, num_attrs=5):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, 64, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, 1),
            )
            for _ in range(num_attrs)
        ])

    def forward(self, x):
        """x: [B,C,H,W] decoder feature → [B,5,H,W] logits"""
        return torch.cat([h(x) for h in self.heads], dim=1)


class AttrGCN(nn.Module):
    """CKTG 精简版：5 属性条件概率矩阵 → 一层 GCN → 修正 logits。
    Simplified CKTG: 5-attr conditional prob matrix → 1-layer GCN → refined logits.

    训练时从 GT 统计成对共现概率（P(a|b)），推理时用图卷积平滑预测。
    Training: pairwise co-occurrence from GT → prob matrix. Inference: GCN smooths predictions."""

    def __init__(self, num_attrs=5):
        super().__init__()
        self.num_attrs = num_attrs
        self.weight = nn.Parameter(torch.eye(num_attrs) * 0.9 + 0.01)  # init near identity

    def forward(self, prob):
        """prob: [N,5] (per-image attribute probs from mask means) → refined [N,5]"""
        # 图平滑：prob_refined = prob * self_loop + (1-self_loop) * (A * prob) / deg
        adj = self.weight / (self.weight.sum(dim=1, keepdim=True) + 1e-6)  # normalized adj
        refined = prob @ adj.T * 0.7 + prob * 0.3  # weighted smoothing
        return refined.clamp(0, 1)


def build_attr_graph_from_gt(train_ids, load_fn, num_attrs=5):
    """从 GT 统计 5 属性共现概率矩阵 [5,5]。
    P(i|j) = count(i=1 & j=1) / count(j=1).
    Returns torch.Tensor [5,5]."""
    cooc = np.zeros((num_attrs, num_attrs), dtype=np.float64)
    cnt = np.zeros(num_attrs, dtype=np.float64)
    for iid in train_ids:
        masks = load_fn(iid)  # [H,W,5] 0/1
        present = (masks.sum(axis=(0,1)) > 0).astype(int)  # [5]
        for i in range(num_attrs):
            if present[i]:
                cnt[i] += 1
                for j in range(num_attrs):
                    if present[j]:
                        cooc[i,j] += 1
    # P(j|i)
    for i in range(num_attrs):
        if cnt[i] > 0:
            cooc[i] /= cnt[i]
    return torch.tensor(cooc, dtype=torch.float32)
