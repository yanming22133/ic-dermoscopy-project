"""Task2 属性检测训练：5 通道多标签分割（每个属性独立 sigmoid 二值）。
Task2 attribute detection training: 5-channel multi-label segmentation
(each attribute is an independent sigmoid binary mask).

- 损失 / Loss: 逐通道 BCE + Dice，稀疏类加权 / per-channel BCE + Dice, sparse classes weighted
- presence / presence: p_attr = mean(sigmoid(logits_attr)) over lesion ROI；status 阈值 0.60/0.40
  p_attr = mean(sigmoid(logits_attr)) over lesion ROI; status thresholds 0.60/0.40
- 评估 / Eval: 逐属性 mask Dice/IoU + presence 准确率 / per-attr mask Dice/IoU + presence accuracy

用法 / Usage:
  f:\\anacondaenvs\\pytorch\\python.exe -m src.train_task2 --variant b2 --epochs 50 --batch 8
冒烟 / Smoke:
  f:\\anacondaenvs\\pytorch\\python.exe -u -m src.train_task2 --variant b0 --epochs 1 --batch 2 --max_train 20 --max_val 20
"""
import os
import argparse
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import SEED, IMG_SIZE, OUT_DIR, ATTRS_FILE, ATTRS_JSON, STATUS_HI, STATUS_LO
from .data import get_splits, Task2Dataset, load_task1_mask
from .model import build_segformer

# 逐通道权重（稀疏类加重）/ per-channel weights (sparse classes up-weighted)
# 顺序同 ATTRS_FILE: pigment, negative, streaks, milia, globules
# presence rate ~ 46% / 10% / 8% / 20% / 22% -> 稀疏的 negative/streaks 加重
CHAN_WEIGHTS = torch.tensor([1.0, 2.0, 2.5, 1.5, 1.0])


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def multilabel_loss(logits, target, weights):
    """logits/target: [B,5,H,W]。逐通道 BCE+Dice 加权和。
    logits/target: [B,5,H,W]. Weighted sum of per-channel BCE+Dice."""
    prob = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none').mean(dim=(0, 2, 3))  # [5]
    # 逐通道 Dice / per-channel Dice
    inter = (prob * target).sum(dim=(0, 2, 3))
    denom = prob.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3)) + 1e-6
    dice = 1 - (2 * inter / denom)  # [5]
    w = weights.to(logits.device)
    return (w * (bce + dice)).mean()


@torch.no_grad()
def evaluate(model, loader, device):
    """返回逐属性 Dice/IoU 和 presence 准确率。
    Returns per-attribute Dice/IoU and presence accuracy."""
    model.eval()
    n = len(ATTRS_FILE)
    dices = [[] for _ in range(n)]; ious = [[] for _ in range(n)]
    pres_correct = [0] * n; pres_total = [0] * n
    for img, mask, ids in loader:
        img = img.to(device); mask = mask.to(device)
        out = model(pixel_values=img).logits
        out = F.interpolate(out, size=mask.shape[-2:], mode='bilinear', align_corners=False)
        prob = torch.sigmoid(out)  # [B,5,H,W]
        pred = (prob >= 0.5).cpu().numpy().astype(np.uint8)
        gt = mask.cpu().numpy().astype(np.uint8)
        for b, iid in enumerate(ids):
            # ROI 用 GT 病灶 mask（val 阶段隔离 Task2 质量）/ ROI = GT lesion mask (isolate Task2 on val)
            roi = load_task1_mask(iid).astype(bool)
            for c in range(n):
                p, g = pred[b, c], gt[b, c]
                inter = (p & g).sum(); union = (p | g).sum()
                if union > 0:
                    dices[c].append(2 * inter / (p.sum() + g.sum() + 1e-6))
                    ious[c].append(inter / (union + 1e-6))
                # presence / presence
                p_attr = float(prob[b, c].cpu().numpy()[roi].mean()) if roi.any() else 0.0
                status = 'present' if p_attr >= STATUS_HI else ('absent' if p_attr <= STATUS_LO else 'uncertain')
                gt_present = bool(g.any())
                pred_present = (status == 'present')
                pres_correct[c] += int(pred_present == gt_present)
                pres_total[c] += 1
    dice_m = [float(np.mean(d)) if d else float('nan') for d in dices]
    iou_m = [float(np.mean(i)) if i else float('nan') for i in ious]
    acc_m = [pres_correct[c] / max(pres_total[c], 1) for c in range(n)]
    return dice_m, iou_m, acc_m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', default='b2', choices=['b0', 'b2'])
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--size', type=int, default=IMG_SIZE)
    ap.add_argument('--out', default=os.path.join(OUT_DIR, 'task2_segformer'))
    ap.add_argument('--max_train', type=int, default=0)
    ap.add_argument('--max_val', type=int, default=0)
    ap.add_argument('--num_workers', type=int, default=0)
    args = ap.parse_args()

    set_seed(SEED)
    os.makedirs(args.out, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', device, '| variant:', args.variant, '| size:', args.size, flush=True)

    tr, va, te = get_splits()
    if args.max_train > 0: tr = tr[:args.max_train]
    if args.max_val > 0: va = va[:args.max_val]
    print(f'split: train {len(tr)} / val {len(va)} / test-local {len(te)}', flush=True)

    tr_dl = DataLoader(Task2Dataset(tr, True, args.size), batch_size=args.batch,
                       shuffle=True, num_workers=args.num_workers, pin_memory=True)
    va_dl = DataLoader(Task2Dataset(va, False, args.size), batch_size=args.batch,
                       shuffle=False, num_workers=args.num_workers)

    model = build_segformer(args.variant, num_labels=5).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))

    best = -1.0
    best_path = os.path.join(args.out, 'best.pth')
    for ep in range(args.epochs):
        model.train(); tot = 0.0
        for img, mask, _ in tr_dl:
            img = img.to(device); mask = mask.to(device)
            with torch.amp.autocast('cuda', enabled=(device == 'cuda')):
                out = model(pixel_values=img).logits
                out = F.interpolate(out, size=mask.shape[-2:], mode='bilinear', align_corners=False)
                loss = multilabel_loss(out, mask, CHAN_WEIGHTS)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item()
        d, i, a = evaluate(model, va_dl, device)
        mean_dice = float(np.nanmean(d))
        print(f'ep {ep+1}/{args.epochs} loss={tot/len(tr_dl):.4f} meanDice={mean_dice:.4f}', flush=True)
        print('  per-attr Dice: ' + ' '.join(f'{ATTRS_FILE[k]}={d[k]:.3f}' for k in range(len(ATTRS_FILE))), flush=True)
        print('  per-attr presenceAcc: ' + ' '.join(f'{ATTRS_FILE[k]}={a[k]:.3f}' for k in range(len(ATTRS_FILE))), flush=True)
        if mean_dice > best:
            best = mean_dice
            torch.save({'model': model.state_dict(), 'variant': args.variant, 'size': args.size,
                        'mean_dice': mean_dice, 'epoch': ep + 1}, best_path)
            json.dump({'mean_dice': mean_dice, 'dice': d, 'iou': i, 'presence_acc': a, 'epoch': ep + 1},
                      open(os.path.join(args.out, 'best_metrics.json'), 'w'), indent=2)
    print(f'best val meanDice={best:.4f} -> {best_path}', flush=True)


if __name__ == '__main__':
    main()
