"""SAM 边界精修（Tier1）：用 SegFormer 粗 mask 的 bbox 当 box prompt，
让 SAM 输出边界更锐利的病灶 mask，降 Hausdorff、提 IoU。
SAM boundary refinement (Tier1): use the SegFormer rough mask's bbox as a box
prompt so SAM produces sharper lesion boundaries, lowering Hausdorff and raising IoU.

权重需先下到 pretrained/sam-vit-base（跑 scripts/download_weights.py）。
Weights must be fetched first to pretrained/sam-vit-base (run scripts/download_weights.py).
注意：本模块的输出张量形状依 transformers 版本而异，首次用需 GPU 验证。
Note: output tensor shapes vary by transformers version; verify on GPU on first use.
"""
import numpy as np
import torch
import torch.nn.functional as F

from .config import PRETRAINED


def load_sam(device):
    from transformers import SamModel, SamProcessor
    path = PRETRAINED['sam']
    model = SamModel.from_pretrained(path).to(device).eval()
    proc = SamProcessor.from_pretrained(path)
    return model, proc


@torch.no_grad()
def refine_mask(model, proc, image_np, rough_mask, device):
    """image_np: HxWx3 uint8 RGB；rough_mask: HxW 0/1。返回精修后的 0/1 mask。
    image_np: HxWx3 uint8 RGB; rough_mask: HxW 0/1. Returns refined 0/1 mask."""
    H, W = rough_mask.shape
    if rough_mask.sum() == 0:
        return rough_mask  # 无病灶不精修 / no lesion, skip
    ys, xs = np.where(rough_mask > 0)
    box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    inputs = proc(image_np, input_boxes=[[box]], return_tensors='pt').to(device)
    out = model(**inputs)
    masks = out.pred_masks      # 形如 (1,1,3,1,256,256) / e.g. (1,1,3,1,256,256)
    iou = out.iou_scores        # 形如 (1,1,3) / e.g. (1,1,3)
    nm = iou.shape[-1]          # mask 数量 = 3 / number of masks = 3
    # 防御性 reshape 到 [nm, 256, 256] / defensive reshape to [nm,256,256]
    masks = masks.reshape(nm, masks.shape[-2], masks.shape[-1])
    iou = iou.reshape(-1)
    best = int(iou.argmax())    # 选 IoU 最高的 mask / pick highest-IoU mask
    mask = torch.sigmoid(masks[best])  # [256,256]
    mask = (mask > 0.5).float()[None, None]
    mask = F.interpolate(mask, size=(H, W), mode='bilinear', align_corners=False)[0, 0]
    return (mask.cpu().numpy() > 0.5).astype(np.uint8)
