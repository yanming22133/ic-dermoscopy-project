"""分割评估指标：Dice、IoU、95% Hausdorff（tutorial p46 要求）。
Segmentation metrics: Dice, IoU, 95% Hausdorff (tutorial p46).

pred / gt 都是二值 2D 数组（0/1）。
pred / gt are binary 2D arrays (0/1).
"""
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def dice_score(pred, gt, smooth=1e-6):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = (pred & gt).sum()
    return (2 * inter + smooth) / (pred.sum() + gt.sum() + smooth)


def iou_score(pred, gt, smooth=1e-6):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return (inter + smooth) / (union + smooth)


def hausdorff95(pred, gt):
    """95% Hausdorff 距离，用距离变换高效计算。
    95% Hausdorff distance, computed efficiently via distance transform."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if not pred.any() or not gt.any():
        return float('nan')  # 有一边为空，无法定义 / one side empty, undefined
    pred_bd = pred ^ binary_erosion(pred)
    gt_bd = gt ^ binary_erosion(gt)
    if not pred_bd.any() or not gt_bd.any():
        return 0.0
    dt_gt = distance_transform_edt(~gt_bd)   # 到 gt 边界的距离图 / distance map to gt boundary
    dt_pred = distance_transform_edt(~pred_bd)
    d_pred2gt = dt_gt[pred_bd]
    d_gt2pred = dt_pred[gt_bd]
    return float(np.percentile(np.concatenate([d_pred2gt, d_gt2pred]), 95))
