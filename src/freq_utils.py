"""频域工具：Haar DWT + 频域解耦损失（LL Dice + HH MSE）。
Frequency-domain utilities: Haar DWT + decoupled freq loss (LL Dice + HH MSE).

ViT backbone（如 MiT/SegFormer）天然偏低频（LL），高频（HH）梯度不足导致边界粗糙。
把损失按 DWT 子带拆开：LL 管形状(Dice)，HH 管边界(MSE)，独立注入高频梯度。
ViT backbones (e.g., MiT/SegFormer) are naturally biased toward low freq (LL);
insufficient high-freq (HH) gradients cause rough boundaries.
Split loss by DWT subband: LL→shape (Dice), HH→boundary (MSE), independent HF injection.
"""
import torch
import torch.nn.functional as F


def haar_dwt_2d(x):
    """Haar DWT 分解，返回 (LL, HH) 各 [B,C,H/2,W/2]。
    Haar DWT decomposition, returns (LL, HH) each [B,C,H/2,W/2].
    忽略中频 LH/HL，只保留低频形状(LL)和高频边缘(HH)。"""
    ll = (x[:, :, 0::2, 0::2] + x[:, :, 0::2, 1::2] +
          x[:, :, 1::2, 0::2] + x[:, :, 1::2, 1::2]) / 4.0
    hh = (x[:, :, 0::2, 0::2] - x[:, :, 0::2, 1::2] -
          x[:, :, 1::2, 0::2] + x[:, :, 1::2, 1::2]) / 4.0
    return ll, hh


def freq_loss(prob, target):
    """频域解耦损失：LL→Dice（形状），HH→MSE（边界对齐）。默认权重 1:1。
    Frequency decoupled loss: LL→Dice (shape), HH→MSE (boundary alignment)."""
    ll_p, hh_p = haar_dwt_2d(prob)
    ll_g, hh_g = haar_dwt_2d(target)
    # LL Dice / LL dice loss
    inter = (ll_p * ll_g).sum(dim=(2, 3))
    denom = ll_p.sum(dim=(2, 3)) + ll_g.sum(dim=(2, 3)) + 1e-6
    dice_ll = (1 - (2 * inter / denom)).mean()
    # HH MSE / HH edge-matching loss
    mse_hh = F.mse_loss(hh_p, hh_g)
    return dice_ll + mse_hh
