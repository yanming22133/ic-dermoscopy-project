"""独立边缘检测支路（WA-NET BRM 精简版）+ Sobel 边缘监督损失。
Edge detection branch (simplified WA-NET BRM) + Sobel edge-supervised loss.

对 logits 做 Sobel 边缘提取 → 3×3 Conv → sigmoid gate → 乘回原 mask。
对 GT mask 同样做 Sobel → MSE 作为边缘损失。
Sobel edge on logits → 3×3 Conv → sigmoid gate → multiply back to mask.
Sobel on GT mask → MSE as edge-supervised loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def sobel_edges(x):
    """Sobel 边缘提取。x: [B,C,H,W] → edges: [B,C,H,W]"""
    sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    # Expand to match channels
    sobel_x = sobel_x.repeat(x.shape[1],1,1,1)
    sobel_y = sobel_y.repeat(x.shape[1],1,1,1)
    gx = F.conv2d(F.pad(x,(1,1,1,1),mode='reflect'), sobel_x, groups=x.shape[1])
    gy = F.conv2d(F.pad(x,(1,1,1,1),mode='reflect'), sobel_y, groups=x.shape[1])
    return (gx.abs() + gy.abs()).clamp(0, 1)


class EdgeBranch(nn.Module):
    """独立边缘支路：对输入做 Sobel → Conv → gate → 乘回。
    Edge branch: Sobel → Conv → gate → multiply back."""

    def __init__(self, channels):
        super().__init__()
        self.refine = nn.Conv2d(channels, channels, 3, padding=1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """x: [B,2,H,W] logits → refined logits"""
        edges = sobel_edges(x)
        refined = self.refine(edges)
        g = self.gate(refined)
        return x * g


def edge_supervised_loss(pred_prob, gt_mask):
    """WA-NET edge-supervised loss：Sobel(pred) vs Sobel(GT) 的 MSE。
    pred_prob: [B,1,H,W] before threshold. gt_mask: [B,1,H,W] 0/1."""
    pe = sobel_edges(pred_prob)
    te = sobel_edges(gt_mask.float())
    return F.mse_loss(pe, te)
