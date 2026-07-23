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
from .data import get_splits, Task1Dataset, build_cache
from .model import build_model, AttnWrapper
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


def tversky_loss(prob, target, alpha=0.3, beta=0.7, smooth=1e-6):
    """Tier1: Tversky 边界损失。beta>alpha 偏重 recall（少漏病灶），降 Hausdorff。
    Tier1: Tversky boundary loss. beta>alpha emphasizes recall, lowers Hausdorff."""
    target = target.float()
    tp = (prob * target).sum(dim=(2, 3))
    fp = (prob * (1 - target)).sum(dim=(2, 3))
    fn = ((1 - prob) * target).sum(dim=(2, 3))
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1 - tversky.mean()


def signed_distance_np(mask01):
    """水平集距离图 φ：病灶内为负、外为正。Boundary Loss 用。
    Level-set φ: negative inside lesion, positive outside. For Boundary Loss."""
    from scipy.ndimage import distance_transform_edt
    if mask01.sum() == 0:
        return np.zeros_like(mask01, dtype=np.float32)
    dt_in = distance_transform_edt(mask01)      # 病灶内到边界 / inside-to-boundary
    dt_out = distance_transform_edt(1 - mask01)  # 病灶外到边界 / outside-to-boundary
    return (dt_out - dt_in).astype(np.float32)  # 内负外正 / neg inside, pos outside


def boundary_loss(prob, phi):
    """Boundary Loss（Kervadec）：prob [B,1,H,W]，phi [B,1,H,W]（内负外正）。
    Boundary Loss: prob [B,1,H,W], phi [B,1,H,W] (neg inside). 直接优化边界距离，降 HD95。"""
    return (prob * phi).mean()


@torch.no_grad()
def evaluate(model, loader, device, compute_hd=True, model_type='segformer'):
    """评估。compute_hd=False 时跳过慢的 Hausdorff（训练中每轮用，快）。
    Eval. compute_hd=False skips the slow Hausdorff (used during training, fast)."""
    model.eval()
    dices, ious, hds = [], [], []
    for img, mask, _ in loader:
        img = img.to(device)
        mask_gpu = mask.to(device)
        # peft_sam 需要 gt_mask 做 box prompt / peft_sam needs gt_mask as box prompt
        if model_type == 'peft_sam':
            out = model(pixel_values=img, gt_mask=mask_gpu).logits
        else:
            out = model(pixel_values=img).logits
        out = F.interpolate(out, size=mask_gpu.shape[-2:], mode='bilinear', align_corners=False)
        prob = F.softmax(out, dim=1)[:, 1]
        pred = (prob >= 0.5).cpu().numpy().astype(np.uint8)
        gt = mask_gpu.cpu().numpy().astype(np.uint8)
        for p, g in zip(pred, gt):
            dices.append(dice_score(p, g))
            ious.append(iou_score(p, g))
            if compute_hd:
                hd = hausdorff95(p, g)
                if not np.isnan(hd):
                    hds.append(hd)
    hd_mean = float(np.mean(hds)) if hds else float('nan')
    return float(np.mean(dices)), float(np.mean(ious)), hd_mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='peft_sam', choices=['segformer','deeplab','peft_sam'])
    ap.add_argument('--variant', default='vit-b', choices=['b0','b1','b2','b3','vit-b','vit-l','vit-h'])
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--size', type=int, default=IMG_SIZE)
    ap.add_argument('--out', default=os.path.join(OUT_DIR, 'task1_segformer'))
    ap.add_argument('--max_train', type=int, default=0, help='>0 时只取这么多训练图，用于冒烟测试 / if >0, use only this many train images (smoke test)')
    ap.add_argument('--max_val', type=int, default=0, help='>0 时只评估这么多 val 图，用于冒烟测试 / if >0, evaluate only this many val images (smoke test)')
    ap.add_argument('--num_workers', type=int, default=0)
    ap.add_argument('--patience', type=int, default=10, help='早停耐心；0=不早停 / early stop patience; 0=off')
    ap.add_argument('--accum_steps', type=int, default=1, help='梯度累积步数；实际batch×accum=等效batch / grad accum')
    ap.add_argument('--resume', action='store_true', help='从 last.pth 断点续训 / resume from last.pth')
    ap.add_argument('--boundary_loss', action='store_true', help='加 Boundary Loss 降 HD95（慢，需 phi 计算）/ add Boundary Loss to lower HD95')
    ap.add_argument('--cosine_lr', action='store_true', help='余弦退火学习率 / cosine annealing LR')
    ap.add_argument('--freq_loss', action='store_true', help='频域解耦损失（LL Dice + HH MSE）/ freq-decoupled loss')
    ap.add_argument('--ch_attn', action='store_true', help='通道注意力（SE）/ channel attention')
    ap.add_argument('--diffusion_loss', action='store_true', help='扩散潜空间流形对齐损失（需4090+diffusers）/ SD latent manifold alignment loss')
    args = ap.parse_args()

    if args.model == 'peft_sam' and args.size != 1024:  # SAM 需 1024 输入 / SAM expects 1024
        print(f'[auto] --size {args.size}→1024 (SAM requires 1024)', flush=True)
        args.size = 1024

    set_seed(SEED)
    os.makedirs(args.out, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', device, '| model:', args.model, '| variant:', args.variant, '| size:', args.size, flush=True)

    tr, va, te = get_splits()
    if args.max_train > 0:
        tr = tr[:args.max_train]
    if args.max_val > 0:
        va = va[:args.max_val]
    print(f'split: train {len(tr)} / val {len(va)} / test-local {len(te)}', flush=True)

    build_cache(tr + va)  # 一次性预处理缓存，后续每轮直接读 / one-time preprocess cache

    tr_dl = DataLoader(Task1Dataset(tr, True, args.size), batch_size=args.batch,
                       shuffle=True, num_workers=args.num_workers, pin_memory=True)
    va_dl = DataLoader(Task1Dataset(va, False, args.size), batch_size=args.batch,
                       shuffle=False, num_workers=args.num_workers)

    model = build_model(args.model, args.variant).to(device)
    if args.ch_attn:  # 通道注意力 wrapper / channel attention wrapper
        model = AttnWrapper(model, 2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs) if args.cosine_lr else None
    diff_align = None
    if args.diffusion_loss:  # 潜空间流形对齐（4090+）/ latent manifold alignment
        from .diffusion_loss import DiffusionAlignLoss
        diff_align = DiffusionAlignLoss(device)

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
        model.train()
        opt.zero_grad()
        tot = 0.0
        pbar = tqdm(tr_dl, desc=f'ep {ep+1}/{args.epochs}', leave=False)
        for i, (img, mask, _) in enumerate(pbar):
            img = img.to(device)
            mask = mask.to(device)
            with torch.amp.autocast('cuda', enabled=(device == 'cuda')):
                # peft_sam 需要 gt_mask 做 box prompt / peft_sam needs gt_mask as box prompt
                if args.model == 'peft_sam':
                    out = model(pixel_values=img, gt_mask=mask).logits
                else:
                    out = model(pixel_values=img).logits
                out = F.interpolate(out, size=mask.shape[-2:], mode='bilinear', align_corners=False)
                ce = F.cross_entropy(out, mask)
                prob = F.softmax(out, dim=1)[:, 1:2]  # [B,1,H,W]
                m = mask.unsqueeze(1).float()
                loss = (ce + dice_loss(prob, m) + 0.5 * tversky_loss(prob, m)) / accum  # 梯度累积 / grad accum
                if args.boundary_loss:  # 加 Boundary Loss 降 HD95（alpha 前 10 轮线性 ramp）/ Boundary Loss, alpha ramps over 10 eps
                    phi = np.stack([signed_distance_np(mm) for mm in mask.cpu().numpy()]).astype(np.float32)
                    phi = torch.from_numpy(phi).unsqueeze(1).to(device)
                    alpha = min(1.0, (ep + 1) / 10.0)
                    loss = loss + alpha * boundary_loss(prob, phi) / accum
                if args.freq_loss:  # 频域解耦：LL Dice + HH MSE / freq decoupled loss
                    from .freq_utils import freq_loss as fl
                    loss = loss + fl(prob, m) * 0.3 / accum
                if diff_align is not None and ep >= 5 and i % 10 == 0:  # 每 10 batch 一次，避免 VAE+UNet 拖慢 / every 10 batches
                    loss = loss + diff_align(prob, m) * 0.1 / accum
            scaler.scale(loss).backward()
            if (i + 1) % accum == 0 or (i + 1) == len(tr_dl):
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
            tot += loss.item() * accum
            pbar.set_postfix({'loss': f'{loss.item()*accum:.3f}'})
        d, i, _ = evaluate(model, va_dl, device, compute_hd=False, model_type=args.model)  # 训练中不算 HD95（快）
        print(f'ep {ep+1}/{args.epochs}  loss={tot/len(tr_dl):.4f}  val Dice={d:.4f} IoU={i:.4f}', flush=True)
        if sched is not None:
            sched.step()  # 余弦退火 / cosine annealing
        if d > best:
            best = d
            no_improve = 0
            torch.save({'model': model.state_dict(), 'model_type': args.model, 'variant': args.variant, 'size': args.size,
                        'dice': d, 'iou': i, 'epoch': ep + 1}, best_path)
            json.dump({'dice': d, 'iou': i, 'epoch': ep + 1},
                      open(os.path.join(args.out, 'best_metrics.json'), 'w'), indent=2)
        else:
            no_improve += 1
        # 每轮存 last.pth，崩了能续 / save last.pth every epoch for crash recovery
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'epoch': ep + 1,
                    'best': best, 'no_improve': no_improve, 'model_type': args.model, 'variant': args.variant}, last_path)
        if args.patience > 0 and no_improve >= args.patience:
            print(f'early stopping at ep {ep+1} (no improve {no_improve} epochs)', flush=True)
            break
    print(f'best val Dice={best:.4f} -> {best_path}', flush=True)

    # 最终对最佳 checkpoint 在 val + test-local 上算 HD95（训练中省略了）/ final HD95 eval
    print('--- final eval (with HD95) on best checkpoint ---', flush=True)
    ck = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ck['model'])
    te_dl = DataLoader(Task1Dataset(te, False, args.size), batch_size=args.batch,
                       shuffle=False, num_workers=args.num_workers)
    for name, dl in [('val', va_dl), ('test-local', te_dl)]:
        d, i, h = evaluate(model, dl, device, compute_hd=True, model_type=args.model)
        print(f'{name}: Dice={d:.4f} IoU={i:.4f} HD95={h:.2f}', flush=True)
        if name == 'val':
            json.dump({'dice': d, 'iou': i, 'hd95': h}, open(os.path.join(args.out, 'final_val_metrics.json'), 'w'), indent=2)
        else:
            json.dump({'dice': d, 'iou': i, 'hd95': h}, open(os.path.join(args.out, 'final_testlocal_metrics.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
