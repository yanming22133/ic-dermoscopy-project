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
from .data import get_splits, Task2Dataset, load_task1_mask, compute_sample_weights, build_cache
from .model import build_segformer

# 逐通道权重（稀疏类加重）/ per-channel weights (sparse classes up-weighted)
# 顺序同 ATTRS_FILE: pigment, negative, streaks, milia, globules
# presence rate ~ 46% / 10% / 8% / 20% / 22% -> 稀疏的 negative/streaks 加重
CHAN_WEIGHTS = torch.tensor([1.0, 2.0, 2.5, 1.5, 1.0])


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def focal_loss(logits, target, gamma=2.0, alpha=0.25):
    """Tier1: Focal Loss，降易分样本权重，提稀疏类 recall。
    Tier1: Focal Loss, down-weights easy samples, boosts sparse-class recall.
    返回逐通道损失 [5]。/ Returns per-channel loss [5]."""
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    p = torch.sigmoid(logits)
    pt = p * target + (1 - p) * (1 - target)
    return (alpha * (1 - pt) ** gamma * bce).mean(dim=(0, 2, 3))  # [5]


def edge_loss_per_channel(logits, target):
    """T3: 逐属性边缘损失（WA-NET edge-supervised loss 简化版）。
    Per-attribute edge loss (simplified WA-NET edge-supervised loss).
    对 present 的属性，用 Sobel 算子提取边缘后算 MSE。"""
    import torch.nn.functional as F
    prob = torch.sigmoid(logits)  # [B,5,H,W]
    # Sobel kernel
    sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=logits.dtype, device=logits.device).view(1,1,3,3)
    sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=logits.dtype, device=logits.device).view(1,1,3,3)
    loss = 0.0
    n = 0
    for c in range(logits.shape[1]):
        if target[:,c].sum() > 0:  # 只在有 GT 的属性上算（避免稀疏类全背景）
            pe = prob[:,c:c+1]; te = target[:,c:c+1].float()
            pe_x = F.conv2d(F.pad(pe,(1,1,1,1),mode='reflect'), sobel_x)
            pe_y = F.conv2d(F.pad(pe,(1,1,1,1),mode='reflect'), sobel_y)
            te_x = F.conv2d(F.pad(te,(1,1,1,1),mode='reflect'), sobel_x)
            te_y = F.conv2d(F.pad(te,(1,1,1,1),mode='reflect'), sobel_y)
            loss += F.mse_loss(pe_x, te_x) + F.mse_loss(pe_y, te_y)
            n += 1
    return loss / max(n, 1)


def multilabel_loss(logits, target, weights, edge_weight=0.0):
    """logits/target: [B,5,H,W]。逐通道 Focal+Dice 加权和 + 可选边缘损失。
    logits/target: [B,5,H,W]. Per-channel Focal+Dice + optional edge loss."""
    prob = torch.sigmoid(logits)
    foc = focal_loss(logits, target)  # [5]  Tier1: Focal 替代 BCE
    inter = (prob * target).sum(dim=(0, 2, 3))
    denom = prob.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3)) + 1e-6
    dice = 1 - (2 * inter / denom)  # [5]
    w = weights.to(logits.device)
    loss = (w * (foc + dice)).mean()
    if edge_weight > 0:
        loss = loss + edge_weight * edge_loss_per_channel(logits, target)
    return loss


@torch.no_grad()
def evaluate(model, loader, device):
    """返回逐属性 Dice/IoU 和 presence 准确率。
    Returns per-attribute Dice/IoU and presence accuracy."""
    model.eval()
    n = len(ATTRS_FILE)
    dices = [[] for _ in range(n)]; ious = [[] for _ in range(n)]
    pres_correct = [0] * n; pres_total = [0] * n
    roi_cache = {}
    for img, mask, ids in loader:
        img = img.to(device); mask = mask.to(device)
        out = model(pixel_values=img).logits
        out = F.interpolate(out, size=mask.shape[-2:], mode='bilinear', align_corners=False)
        prob = torch.sigmoid(out)  # [B,5,H,W]
        pred = (prob >= 0.5).cpu().numpy().astype(np.uint8)
        gt = mask.cpu().numpy().astype(np.uint8)
        for b, iid in enumerate(ids):
            if iid not in roi_cache:
                roi_cache[iid] = load_task1_mask(iid).astype(bool)
            roi = roi_cache[iid]
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
    ap.add_argument('--variant', default='b2', choices=['b0','b1','b2','b3'])
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--size', type=int, default=IMG_SIZE)
    ap.add_argument('--out', default=os.path.join(OUT_DIR, 'task2_segformer'))
    ap.add_argument('--max_train', type=int, default=0)
    ap.add_argument('--max_val', type=int, default=0)
    ap.add_argument('--num_workers', type=int, default=0)
    ap.add_argument('--balanced', type=int, default=1, help='Tier1: 稀疏类平衡采样 1/0 / sparse-class balanced sampling')
    ap.add_argument('--patience', type=int, default=10, help='早停耐心；0=不早停 / early stop patience; 0=off')
    ap.add_argument('--accum_steps', type=int, default=1, help='梯度累积；实际batch×accum=等效batch / grad accum')
    ap.add_argument('--resume', action='store_true', help='从 last.pth 断点续训 / resume from last.pth')
    ap.add_argument('--cosine_lr', action='store_true', help='余弦退火学习率 / cosine annealing LR')
    ap.add_argument('--edge_loss', type=float, default=0.0, help='T3: 逐属性边缘损失权重 (WA-NET) / per-attr edge loss weight')
    args = ap.parse_args()

    set_seed(SEED)
    os.makedirs(args.out, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', device, '| variant:', args.variant, '| size:', args.size, flush=True)

    tr, va, te = get_splits()
    if args.max_train > 0: tr = tr[:args.max_train]
    if args.max_val > 0: va = va[:args.max_val]
    print(f'split: train {len(tr)} / val {len(va)} / test-local {len(te)}', flush=True)

    build_cache(tr + va)  # 一次性预处理缓存 / one-time preprocess cache

    # Tier1: 平衡采样（含稀有属性的样本权重高）/ balanced sampling (rare-attr samples up-weighted)
    from torch.utils.data import WeightedRandomSampler
    if args.balanced and args.max_train == 0:
        sw = compute_sample_weights(tr)
        sampler = WeightedRandomSampler(sw, num_samples=len(tr), replacement=True)
        tr_dl = DataLoader(Task2Dataset(tr, True, args.size), batch_size=args.batch,
                           sampler=sampler, num_workers=args.num_workers, pin_memory=True)
        print(f'balanced sampling on (rare-attr weight=3x)', flush=True)
    else:
        tr_dl = DataLoader(Task2Dataset(tr, True, args.size), batch_size=args.batch,
                           shuffle=True, num_workers=args.num_workers, pin_memory=True)
    va_dl = DataLoader(Task2Dataset(va, False, args.size), batch_size=args.batch,
                       shuffle=False, num_workers=args.num_workers)

    model = build_segformer(args.variant, num_labels=5).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs) if args.cosine_lr else None

    best = -1.0
    no_improve = 0
    start_ep = 0
    best_path = os.path.join(args.out, 'best.pth')
    last_path = os.path.join(args.out, 'last.pth')
    accum = max(1, args.accum_steps)
    if args.resume and os.path.exists(last_path):  # 断点续训 / resume
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        start_ep = ck.get('epoch', 0)
        best = ck.get('best', -1.0)
        no_improve = ck.get('no_improve', 0)
        print(f'resumed from epoch {start_ep}, best={best:.4f}', flush=True)

    from tqdm import tqdm
    for ep in range(start_ep, args.epochs):
        model.train(); tot = 0.0
        opt.zero_grad()
        pbar = tqdm(tr_dl, desc=f'ep {ep+1}/{args.epochs}', leave=False)
        for i, (img, mask, _) in enumerate(pbar):
            img = img.to(device); mask = mask.to(device)
            with torch.amp.autocast('cuda', enabled=(device == 'cuda')):
                out = model(pixel_values=img).logits
                out = F.interpolate(out, size=mask.shape[-2:], mode='bilinear', align_corners=False)
                loss = multilabel_loss(out, mask, CHAN_WEIGHTS, edge_weight=args.edge_loss) / accum  # 梯度累积 / grad accum
            scaler.scale(loss).backward()
            if (i + 1) % accum == 0 or (i + 1) == len(tr_dl):
                scaler.step(opt); scaler.update(); opt.zero_grad()
            tot += loss.item() * accum
            pbar.set_postfix({'loss': f'{loss.item()*accum:.3f}'})
        d, i, a = evaluate(model, va_dl, device)
        mean_dice = float(np.nanmean(d))
        print(f'ep {ep+1}/{args.epochs} loss={tot/len(tr_dl):.4f} meanDice={mean_dice:.4f}', flush=True)
        if sched is not None:
            sched.step()  # 余弦退火 / cosine annealing
        print('  per-attr Dice: ' + ' '.join(f'{ATTRS_FILE[k]}={d[k]:.3f}' for k in range(len(ATTRS_FILE))), flush=True)
        print('  per-attr presenceAcc: ' + ' '.join(f'{ATTRS_FILE[k]}={a[k]:.3f}' for k in range(len(ATTRS_FILE))), flush=True)
        if mean_dice > best:
            best = mean_dice
            no_improve = 0
            torch.save({'model': model.state_dict(), 'variant': args.variant, 'size': args.size,
                        'mean_dice': mean_dice, 'epoch': ep + 1}, best_path)
            json.dump({'mean_dice': mean_dice, 'dice': d, 'iou': i, 'presence_acc': a, 'epoch': ep + 1},
                      open(os.path.join(args.out, 'best_metrics.json'), 'w'), indent=2)
        else:
            no_improve += 1
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'epoch': ep + 1,
                    'best': best, 'no_improve': no_improve}, last_path)  # 崩了能续 / crash recovery
        if args.patience > 0 and no_improve >= args.patience:
            print(f'early stopping at ep {ep+1} (no improve {no_improve} epochs)', flush=True)
            break
    print(f'best val meanDice={best:.4f} -> {best_path}', flush=True)

    # 最终在 val 上带 presence 评估最佳 checkpoint / final eval on best ckpt
    print('--- final eval on best checkpoint ---', flush=True)
    ck = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ck['model'])
    d, i, a = evaluate(model, va_dl, device)
    print('  per-attr Dice: ' + ' '.join(f'{ATTRS_FILE[k]}={d[k]:.3f}' for k in range(len(ATTRS_FILE))), flush=True)
    print('  per-attr presenceAcc: ' + ' '.join(f'{ATTRS_FILE[k]}={a[k]:.3f}' for k in range(len(ATTRS_FILE))), flush=True)
    json.dump({'dice': d, 'iou': i, 'presence_acc': a}, open(os.path.join(args.out, 'final_val_metrics.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
