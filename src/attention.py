"""轻量注意力模块：SE 通道注意力，SegFormer 解码器输出后即插即用。
Lightweight attention modules: SE channel attention, plug-and-play after SegFormer decoder.

作用：对分割 logits 做通道重标定，抑制 noise 通道、增强边界敏感通道。
相当于对特征做"降维精炼"——压缩→学习重要通道→膨胀→加权。
Purpose: re-calibrate logit channels, suppress noise, boost boundary-sensitive channels.
Equivalent to "dimensionality-refinement": compress→learn importance→expand→reweight.
"""
import torch.nn as nn


class ChannelAttention2D(nn.Module):
    """SE 通道注意力，输入 [B,C,H,W] → 输出 [B,C,H,W]（逐通道加权后）。
    SE channel attention, input [B,C,H,W] → output [B,C,H,W] (per-channel re-weight)."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),               # [B,C,1,1]
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x)
