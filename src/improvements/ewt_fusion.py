"""EWT 跨频融合模块（WA-NET EWT + LSF-Mamba FRE 组合）。
EWT cross-band fusion module (combining WA-NET EWT + LSF-Mamba FRE).

在 SAM decoder 256² 特征上做 Haar DWT —— LL/LH/HL/HH 分别过 1×1 Conv ——
通道注意力融合 —— 门控残差加回主路。不改 SAM 内部结构，即插即用。
Applied on SAM decoder 256² features. Haar DWT → per-band 1×1 Conv →
channel attention fusion → gated residual back to main path. Plug-and-play.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Haar DWT decomposition / Haar DWT 分解 ---
def haar_decompose(x):
    """x: [B,C,H,W] -> LL, LH, HL, HH each [B,C,H/2,W/2]"""
    ll = (x[:,:,0::2,0::2] + x[:,:,0::2,1::2] + x[:,:,1::2,0::2] + x[:,:,1::2,1::2]) * 0.25
    lh = (x[:,:,0::2,0::2] - x[:,:,0::2,1::2] + x[:,:,1::2,0::2] - x[:,:,1::2,1::2]) * 0.25
    hl = (x[:,:,0::2,0::2] + x[:,:,0::2,1::2] - x[:,:,1::2,0::2] - x[:,:,1::2,1::2]) * 0.25
    hh = (x[:,:,0::2,0::2] - x[:,:,0::2,1::2] - x[:,:,1::2,0::2] + x[:,:,1::2,1::2]) * 0.25
    return ll, lh, hl, hh


class EWT_Fusion(nn.Module):
    """WA-NET EWT + LSF-Mamba FRE：跨频拼接 + 通道注意力 + 门控残差。
    WA-NET EWT + LSF-Mamba FRE: cross-band fusion + ch-attn + gated residual."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        # 四个子带各 1×1 Conv —— 降到 channels//4 再拼
        hc = max(1, channels // 4)
        self.ll_conv = nn.Conv2d(channels, hc, 1)
        self.lh_conv = nn.Conv2d(channels, hc, 1)
        self.hl_conv = nn.Conv2d(channels, hc, 1)
        self.hh_conv = nn.Conv2d(channels, hc, 1)
        # 融合后的通道注意力
        fused_ch = hc * 4
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(fused_ch, max(1, fused_ch // reduction), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, fused_ch // reduction), fused_ch, 1),
            nn.Sigmoid(),
        )
        # 投影回原通道 + 门控
        self.proj = nn.Conv2d(fused_ch, channels, 1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """x: [B,C,H,W] — SAM decoder feature / SAM 解码特征"""
        ll, lh, hl, hh = haar_decompose(x)  # each [B,C,H/2,W/2]
        # 上采样回原分辨率
        ll = F.interpolate(ll, size=x.shape[-2:], mode='bilinear', align_corners=False)
        lh = F.interpolate(lh, size=x.shape[-2:], mode='bilinear', align_corners=False)
        hl = F.interpolate(hl, size=x.shape[-2:], mode='bilinear', align_corners=False)
        hh = F.interpolate(hh, size=x.shape[-2:], mode='bilinear', align_corners=False)
        # 各带过 1×1 Conv
        f_ll = self.ll_conv(ll)
        f_lh = self.lh_conv(lh)
        f_hl = self.hl_conv(hl)
        f_hh = self.hh_conv(hh)
        # 拼接 + SE 注意力
        fused = torch.cat([f_ll, f_lh, f_hl, f_hh], dim=1)
        fused = self.se(fused) * fused
        fused = self.proj(fused)
        # 门控残差加回 / gated residual
        g = self.gate(x)
        return x + g * fused
