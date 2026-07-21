"""Task1 病灶分割训练：SegFormer + BCE/Dice 损失 + AMP + val 评估（Dice/IoU/HD95）。
Task1 lesion segmentation training: SegFormer + BCE/Dice loss + AMP + val eval (Dice/IoU/HD95).

用法（用正确环境跑）/ Usage (run with the correct env):
  f:\\anacondaenvs\\pytorch\\python.exe -m src.train_task1 --variant b2 --epochs 50 --batch 8
冒烟测试 / Smoke test:
  f:\\anacondaenvs\\pytorch\\python.exe -m src.train_task1 --variant b0 --epochs 1 --batch 2 --max_train 20
"""
import os
import argparse
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import SEED, IMG_SIZE, OUT_DIR
from .data import get_splits, Task1Dataset
from .model import build_segformer
from .metrics import dice_score, iou_score, hausdorff95


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def dice_loss(prob, target, smooth=1e-6):
    """prob: [B,1,H,W] 病灶概率；target: [B,1,H,W] 0/1。
    prob: [B,1,H,W] lesion probability; target: [B,1,H,W] 0/1."""
    target = target.float()
    inter = (prob * target).sum(dim=(2, 3))
    denom = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return 1 - ((2 * inter + smooth) / (denom + smooth)).mean()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    dices, ious, hds = [], [], []
    for img, mask, _ in loader:
        img = img.to(device)
        mask = mask.to(device)
        out = model(pixel_values=img).logits
        out = F.interpolate(out, size=mask.shape[-2:], mode='bilinear', align_corners=False)
        prob = F.softmax(out, dim=1)[:, 1]
        pred = (prob >= 0.5).cpu().numpy().astype(np.uint8)
        gt = mask.cpu().numpy().astype(np.uint8)
        for p, g in zip(pred, gt):
            dices.append(dice_score(p, g))
            ious.append(iou_score(p, g))
            hd = hausdorff95(p, g)
            if not np.isnan(hd):
                hds.append(hd)
    return float(np.mean(dices)), float(np.mean(ious)), (float(np.mean(hds)) if hds else float('nan'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', default='b2', choices=['b0', 'b2'])
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--size', type=int, default=IMG_SIZE)
    ap.add_argument('--out', default=os.path.join(OUT_DIR, 'task1_segformer'))
    ap.add_argument('--max_train', type=int, default=0, help='>0 时只取这么多训练图，用于冒烟测试 / if >0, use only this many train images (smoke test)')
    ap.add_argument('--max_val', type=int, default=0, help='>0 时只评估这么多 val 图，用于冒烟测试 / if >0, evaluate only this many val images (smoke test)')
    ap.add_argument('--num_workers', type=int, default=0)
    args = ap.parse_args()

    set_seed(SEED)
    os.makedirs(args.out, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', device, '| variant:', args.variant, '| size:', args.size, flush=True)

    tr, va, te = get_splits()
    if args.max_train > 0:
        tr = tr[:args.max_train]
    if args.max_val > 0:
        va = va[:args.max_val]
    print(f'split: train {len(tr)} / val {len(va)} / test-local {len(te)}', flush=True)

    tr_dl = DataLoader(Task1Dataset(tr, True, args.size), batch_size=args.batch,
                       shuffle=True, num_workers=args.num_workers, pin_memory=True)
    va_dl = DataLoader(Task1Dataset(va, False, args.size), batch_size=args.batch,
                       shuffle=False, num_workers=args.num_workers)

    model = build_segformer(args.variant).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))

    best = -1.0
    best_path = os.path.join(args.out, 'best.pth')
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for img, mask, _ in tr_dl:
            img = img.to(device)
            mask = mask.to(device)
            with torch.amp.autocast('cuda', enabled=(device == 'cuda')):
                out = model(pixel_values=img).logits
                out = F.interpolate(out, size=mask.shape[-2:], mode='bilinear', align_corners=False)
                ce = F.cross_entropy(out, mask)
                prob = F.softmax(out, dim=1)[:, 1:2]  # [B,1,H,W]
                loss = ce + dice_loss(prob, mask.unsqueeze(1).float())
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item()
        d, i, h = evaluate(model, va_dl, device)
        print(f'ep {ep+1}/{args.epochs}  loss={tot/len(tr_dl):.4f}  val Dice={d:.4f} IoU={i:.4f} HD95={h:.2f}')
        if d > best:
            best = d
            torch.save({'model': model.state_dict(), 'variant': args.variant, 'size': args.size,
                        'dice': d, 'iou': i, 'hd95': h, 'epoch': ep + 1}, best_path)
            json.dump({'dice': d, 'iou': i, 'hd95': h, 'epoch': ep + 1},
                      open(os.path.join(args.out, 'best_metrics.json'), 'w'), indent=2)
    print(f'best val Dice={best:.4f} -> {best_path}')


if __name__ == '__main__':
    main()
