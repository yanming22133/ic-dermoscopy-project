"""Task1 PEFT-SAM 病灶分割训练：冻结 SAM encoder，微调 mask_decoder + 可选 LoRA。
Task1 PEFT-SAM lesion segmentation training: freeze SAM encoder, fine-tune mask_decoder + optional LoRA.

PEFT-SAM 是 2026 SOTA：仅训练 mask_decoder（~4M 参数），4090 24GB 单卡可跑 batch=8 @ 1024×1024。
PEFT-SAM is 2026 SOTA: trains only mask_decoder (~4M params), runs batch=8 @ 1024×1024 on 4090.

用法（用正确环境跑）/ Usage (run with the correct env):
  f:\\anacondaenvs\\pytorch\\python.exe -m src.train_task1_sam --epochs 50 --batch 8 --size 1024 --lora --boundary_loss --cosine_lr --freq_loss --ch_attn --out outputs/task1_sam
冒烟测试 / Smoke test:
  f:\\anacondaenvs\\pytorch\\python.exe -m src.train_task1_sam --epochs 1 --batch 2 --size 1024 --max_train 20
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
from .model_sam import SamSegModel, build_sam
from .model import AttnWrapper
from .metrics import dice_score, iou_score, hausdorff95


# ============================================================
# 工具函数 / Utility Functions
# ============================================================

def set_seed(s):
    """固定随机种子 / Fix random seed."""
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def dice_loss(prob, target, smooth=1e-6):
    """Dice 损失：prob [B,1,H,W] 病灶概率，target [B,1,H,W] 0/1。
    Dice loss: prob [B,1,H,W] lesion probability, target [B,1,H,W] 0/1."""
    target = target.float()
    inter = (prob * target).sum(dim=(2, 3))
    denom = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return 1 - ((2 * inter + smooth) / (denom + smooth)).mean()


def tversky_loss(prob, target, alpha=0.3, beta=0.7, smooth=1e-6):
    """Tversky 边界损失：beta>alpha 偏重 recall（少漏病灶），降 HD95。
    Tversky boundary loss: beta>alpha emphasizes recall, lowers HD95."""
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
    dt_in = distance_transform_edt(mask01)       # 病灶内到边界 / inside-to-boundary
    dt_out = distance_transform_edt(1 - mask01)   # 病灶外到边界 / outside-to-boundary
    return (dt_out - dt_in).astype(np.float32)    # 内负外正 / neg inside, pos outside


def boundary_loss(prob, phi):
    """Boundary Loss（Kervadec）：prob [B,1,H,W]，phi [B,1,H,W]（内负外正）。
    Boundary Loss (Kervadec): prob [B,1,H,W], phi [B,1,H,W] (neg inside).
    直接优化边界距离，降 HD95。"""
    return (prob * phi).mean()


# ============================================================
# 评估 / Evaluation
# ============================================================

@torch.no_grad()
def evaluate(model, loader, device, compute_hd=True):
    """评估 PEFT-SAM：val 时也用 GT mask 做 box prompt（标准 prompt-based 分割协议）。
    Evaluate PEFT-SAM: use GT mask for box prompt during val (standard prompt-based protocol).
    compute_hd=False 跳过慢的 Hausdorff（训练中每轮用，快）。"""
    model.eval()
    dices, ious, hds = [], [], []
    for img, mask, _ in loader:
        img = img.to(device)
        mask_gpu = mask.to(device)

        # PEFT-SAM forward：传入 gt_mask 自动提取 bbox 做 prompt
        # PEFT-SAM forward: pass gt_mask to auto-extract bbox as prompt
        out = model(pixel_values=img, gt_mask=mask_gpu).logits
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


# ============================================================
# 主训练逻辑 / Main Training Loop
# ============================================================

def main():
    ap = argparse.ArgumentParser(description='PEFT-SAM Task1 病灶分割训练 / Lesion Segmentation Training')
    ap.add_argument('--epochs', type=int, default=50, help='训练轮数 / number of epochs')
    ap.add_argument('--batch', type=int, default=8, help='batch 大小，4090 上 1024×1024 最大约 8 / batch size, max ~8 on 4090 @1024')
    ap.add_argument('--lr', type=float, default=1e-4, help='学习率 / learning rate')
    ap.add_argument('--size', type=int, default=1024, help='输入分辨率，PEFT-SAM 必须 1024 / input resolution, must be 1024 for SAM')
    ap.add_argument('--lora', action='store_true', help='对 vision_encoder 加 LoRA adapter / add LoRA to vision_encoder')
    ap.add_argument('--lora_rank', type=int, default=4, help='LoRA 秩 / LoRA rank (default 4)')
    ap.add_argument('--patience', type=int, default=10, help='早停耐心；0=不早停 / early stop patience; 0=off')
    ap.add_argument('--accum_steps', type=int, default=1, help='梯度累积步数；实际 batch × accum = 等效 batch / grad accum steps')
    ap.add_argument('--resume', action='store_true', help='从 last.pth 断点续训 / resume from last.pth')
    ap.add_argument('--out', default=os.path.join(OUT_DIR, 'task1_sam'), help='输出目录 / output directory')
    ap.add_argument('--num_workers', type=int, default=0, help='DataLoader workers')
    ap.add_argument('--max_train', type=int, default=0, help='>0 时只取这么多训练图，用于冒烟测试 / if >0, use only this many train images (smoke test)')
    ap.add_argument('--max_val', type=int, default=0, help='>0 时只评估这么多 val 图 / if >0, evaluate only this many val images')
    ap.add_argument('--boundary_loss', action='store_true', help='加 Boundary Loss 降 HD95（慢，需 phi 计算）/ add Boundary Loss')
    ap.add_argument('--cosine_lr', action='store_true', help='余弦退火学习率 / cosine annealing LR')
    ap.add_argument('--freq_loss', action='store_true', help='频域解耦损失（LL Dice + HH MSE）/ freq-decoupled loss')
    ap.add_argument('--ch_attn', action='store_true', help='通道注意力（SE）/ channel attention')
    args = ap.parse_args()

    # PEFT-SAM 必须 1024×1024 / PEFT-SAM requires 1024×1024
    if args.size != 1024:
        print(f'[WARNING] PEFT-SAM expects size=1024, got {args.size}. Forcing 1024.', flush=True)
        args.size = 1024

    set_seed(SEED)
    os.makedirs(args.out, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('=' * 60, flush=True)
    print(f'PEFT-SAM Training | device: {device} | size: {args.size} | batch: {args.batch} '
          f'| lora: {args.lora} | lr: {args.lr}', flush=True)
    print('=' * 60, flush=True)

    # 数据划分 / Data split
    tr, va, te = get_splits()
    if args.max_train > 0:
        tr = tr[:args.max_train]
    if args.max_val > 0:
        va = va[:args.max_val]
    print(f'split: train {len(tr)} / val {len(va)} / test-local {len(te)}', flush=True)

    # 预处理缓存 / Preprocess cache
    build_cache(tr + va)

    # DataLoader（PEFT-SAM 用 1024×1024）/ DataLoader (PEFT-SAM uses 1024)
    tr_dl = DataLoader(Task1Dataset(tr, True, args.size), batch_size=args.batch,
                       shuffle=True, num_workers=args.num_workers, pin_memory=True)
    va_dl = DataLoader(Task1Dataset(va, False, args.size), batch_size=args.batch,
                       shuffle=False, num_workers=args.num_workers)

    # 构建模型 / Build model
    model = build_sam(num_labels=2, use_lora=args.lora, lora_rank=args.lora_rank).to(device)
    if args.ch_attn:
        model = AttnWrapper(model, 2).to(device)

    # 优化器（只优化可训练参数 / only trainable params）
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs) if args.cosine_lr else None

    # 训练状态 / Training state
    best = -1.0
    no_improve = 0
    start_ep = 0
    best_path = os.path.join(args.out, 'best.pth')
    last_path = os.path.join(args.out, 'last.pth')
    accum = max(1, args.accum_steps)

    # 断点续训 / Resume from crash
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        start_ep = ck.get('epoch', 0)
        best = ck.get('best', -1.0)
        no_improve = ck.get('no_improve', 0)
        print(f'resumed from epoch {start_ep}, best={best:.4f}', flush=True)

    # ============================================================
    # 训练循环 / Training Loop
    # ============================================================
    for ep in range(start_ep, args.epochs):
        model.train()
        opt.zero_grad()
        tot = 0.0

        for i, (img, mask, _) in enumerate(tr_dl):
            img = img.to(device)
            mask_gpu = mask.to(device)

            with torch.amp.autocast('cuda', enabled=(device == 'cuda')):
                # PEFT-SAM forward: 传入 gt_mask 自动提取 bbox 做 box prompt
                # PEFT-SAM forward: pass gt_mask to auto-extract bbox as box prompt
                out = model(pixel_values=img, gt_mask=mask_gpu).logits
                out = F.interpolate(out, size=mask_gpu.shape[-2:], mode='bilinear', align_corners=False)

                # 损失组件 / Loss components
                ce = F.cross_entropy(out, mask_gpu)
                prob = F.softmax(out, dim=1)[:, 1:2]  # [B, 1, H, W]
                m = mask_gpu.unsqueeze(1).float()      # [B, 1, H, W]
                loss = (ce + dice_loss(prob, m) + 0.5 * tversky_loss(prob, m)) / accum

                # Boundary Loss：直接优化边界距离，降 HD95
                # Boundary Loss: directly optimize boundary distance, lower HD95
                if args.boundary_loss:
                    phi = np.stack([signed_distance_np(mm) for mm in mask_gpu.cpu().numpy()]).astype(np.float32)
                    phi = torch.from_numpy(phi).unsqueeze(1).to(device)
                    alpha = min(1.0, (ep + 1) / 10.0)  # 前 10 轮线性 ramp / linear ramp over first 10 epochs
                    loss = loss + alpha * boundary_loss(prob, phi) / accum

                # 频域解耦损失：LL Dice + HH MSE / Freq-decoupled loss: LL Dice + HH MSE
                if args.freq_loss:
                    from .freq_utils import freq_loss as fl
                    loss = loss + fl(prob, m) * 0.3 / accum

            # 反向传播 + 梯度累积 / Backprop + gradient accumulation
            scaler.scale(loss).backward()
            if (i + 1) % accum == 0 or (i + 1) == len(tr_dl):
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()

            tot += loss.item() * accum

        # 验证（训练中不算 HD95，快）/ Validation (skip HD95 during training for speed)
        d, i, _ = evaluate(model, va_dl, device, compute_hd=False)
        print(f'ep {ep+1:2d}/{args.epochs}  loss={tot/len(tr_dl):.4f}  val Dice={d:.4f} IoU={i:.4f}', flush=True)

        if sched is not None:
            sched.step()

        # 保存最佳 / Save best
        if d > best:
            best = d
            no_improve = 0
            torch.save({
                'model': model.state_dict(), 'model_type': 'peft_sam', 'size': args.size,
                'dice': d, 'iou': i, 'epoch': ep + 1,
            }, best_path)
            json.dump({'dice': d, 'iou': i, 'epoch': ep + 1},
                      open(os.path.join(args.out, 'best_metrics.json'), 'w'), indent=2)
            print(f'  >> new best Dice={d:.4f}', flush=True)
        else:
            no_improve += 1

        # 每轮存 last.pth，崩了能续 / Save last.pth every epoch for crash recovery
        torch.save({
            'model': model.state_dict(), 'opt': opt.state_dict(), 'epoch': ep + 1,
            'best': best, 'no_improve': no_improve, 'model_type': 'peft_sam', 'size': args.size,
        }, last_path)

        # 早停 / Early stopping
        if args.patience > 0 and no_improve >= args.patience:
            print(f'early stopping at ep {ep+1} (no improve {no_improve} epochs)', flush=True)
            break

    print(f'\nbest val Dice={best:.4f} -> {best_path}', flush=True)

    # ============================================================
    # 最终评估（含 HD95）/ Final Evaluation (with HD95)
    # ============================================================
    print('--- final eval (with HD95) on best checkpoint ---', flush=True)
    ck = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ck['model'])
    te_dl = DataLoader(Task1Dataset(te, False, args.size), batch_size=args.batch,
                       shuffle=False, num_workers=args.num_workers)
    for name, dl in [('val', va_dl), ('test-local', te_dl)]:
        d, i, h = evaluate(model, dl, device, compute_hd=True)
        print(f'{name}: Dice={d:.4f} IoU={i:.4f} HD95={h:.2f}', flush=True)
        suffix = 'val' if name == 'val' else 'testlocal'
        json.dump({'dice': d, 'iou': i, 'hd95': h},
                  open(os.path.join(args.out, f'final_{suffix}_metrics.json'), 'w'), indent=2)

    print('\nDone!', flush=True)


if __name__ == '__main__':
    main()
