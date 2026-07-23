"""Task1 推理：加载 checkpoint，对一组图预测病灶 mask，存成 jpg L（照 example 格式）。
Task1 inference: load checkpoint, predict lesion masks for a set of images,
save as jpg L (matching the example format).

有 GT 时顺带算 Dice/IoU/HD95；没有（官方测试集）只存 mask。
With GT, also compute Dice/IoU/HD95; without GT (official test set) only save masks.

用法 / Usage:
  # 在 val/test-local 上评估 + 存 mask / eval on val/test-local + save masks
  f:\\anacondaenvs\\pytorch\\python.exe -m src.infer_task1 --ckpt outputs/task1_segformer/best.pth --split val
  # 对官方测试集（7/30）只存 mask / official test set (7/30), masks only
  f:\\anacondaenvs\\pytorch\\python.exe -m src.infer_task1 --ckpt outputs/task1_segformer/best.pth --image_dir <test img dir> --save_dir submit/task1
"""
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image

from .config import IMAGE_DIR, TASK1_GT_DIR, OUT_DIR, SEED
from .data import get_splits, load_image, load_image_pp, get_transforms, IMAGENET_MEAN, IMAGENET_STD, list_image_ids
from .preprocessing import preprocess
from .model import build_model
from .metrics import dice_score, iou_score, hausdorff95
from . import sam_refine


def postprocess(mask01):
    """后处理：形态学开闭（去噪点、填孔）+ 最大连通域（去假阳性）。零训练成本提 Dice/降 HD95。
    Postprocess: morphological open+close (denoise/fill) + largest CC (remove false positives).
    mask01: HxW 0/1，返回 0/1。/ returns 0/1."""
    m = (mask01 * 255).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num > 1:  # 保留最大连通域 / keep largest CC
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = (labels == largest).astype(np.uint8) * 255
    return (m > 127).astype(np.uint8)


@torch.no_grad()
def predict_prob(models, x, H, W, tta=False):
    """返回病灶概率 [H,W]。models: 模型列表（集成时平均各自概率）。
    Returns lesion prob [H,W]. models: list of models (ensemble averages their probs).
    TTA 时再对 4 个翻转方向平均。/ TTA further averages 4 flips."""
    def fwd(xx):
        probs = []
        for m in models:
            o = m(pixel_values=xx).logits
            o = F.interpolate(o, size=(H, W), mode='bilinear', align_corners=False)
            probs.append(F.softmax(o, dim=1)[:, 1:2])  # [1,1,H,W]
        return torch.stack(probs).mean(0)  # 集成平均 / ensemble mean
    if not tta:
        return fwd(x)[0, 0].cpu().numpy()
    probs = []
    for h in (False, True):
        for v in (False, True):
            xx = torch.flip(x, [3]) if h else x
            xx = torch.flip(xx, [2]) if v else xx
            p = fwd(xx)
            if h: p = torch.flip(p, [3])
            if v: p = torch.flip(p, [2])
            probs.append(p)
    return torch.stack(probs).mean(0)[0, 0].cpu().numpy()


@torch.no_grad()
def infer(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 集成：--ensemble 给多个 ckpt（逗号分隔），否则用 --ckpt 单模型
    # Ensemble: --ensemble gives multiple ckpts (comma-sep), else single --ckpt
    ckpt_paths = args.ensemble.split(',') if args.ensemble else [args.ckpt]
    models = []
    size = 512
    for cp in ckpt_paths:
        ck = torch.load(cp, map_location=device, weights_only=False)
        size = ck.get('size', 512)
        model_type = ck.get('model_type', 'segformer')
        variant = ck.get('variant', 'b2')
        m = build_model(model_type, variant).to(device)
        m.load_state_dict(ck['model'])
        m.eval()
        models.append(m)
    if len(models) > 1:
        print(f'ensemble of {len(models)} models', flush=True)

    sam_model, sam_proc = (None, None)
    if args.sam_refine:  # Tier1: SAM 边界精修 / SAM boundary refinement
        print('loading SAM for boundary refinement...', flush=True)
        sam_model, sam_proc = sam_refine.load_sam(device)
    sd_unet = None
    if args.diffusion_refine:  # 扩散边界精修（4090+）/ diffusion boundary refine
        from .diffusion_refine import load_sd_unet
        sd_unet = load_sd_unet(device)

    # 决定要跑哪些图 / decide which images to run
    if args.image_dir:
        ids = list_image_ids(args.image_dir)
    else:
        tr, va, te = get_splits()
        ids = {'train': tr, 'val': va, 'test-local': te}[args.split]

    save_dir = args.save_dir
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    tfm = get_transforms(False, size)
    dices, ious, hds = [], [], []
    for iid in ids:
        img = load_image(iid, args.image_dir or IMAGE_DIR)
        H, W = img.shape[:2]
        if args.do_preprocess:  # 走预处理缓存（快）；miss 则现算并存盘 / use cache (fast); compute+save on miss
            img_p = load_image_pp(iid, args.image_dir or IMAGE_DIR)
        else:
            img_p = img
        r = tfm(image=img_p, mask=np.zeros((H, W), np.uint8))
        x = r['image']
        x = (x.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(x.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        # 多尺度 TTA / multi-scale TTA
        if args.ms_tta:
            probs = []
            for sc in [0.8, 1.0, 1.2]:
                hs, ws = int(H * sc), int(W * sc)
                xs = F.interpolate(x, size=(hs, ws), mode='bilinear', align_corners=False)
                p = predict_prob(models, xs, hs, ws, tta=bool(args.tta))
                p = torch.from_numpy(p).float().unsqueeze(0).unsqueeze(0)  # [1,1,hs,ws]
                p = F.interpolate(p, size=(H, W), mode='bilinear', align_corners=False)[0, 0].numpy()
                probs.append(p)
            prob = np.stack(probs).mean(0)
        else:
            prob = predict_prob(models, x, H, W, tta=bool(args.tta))  # Tier1: TTA + 集成 / TTA + ensemble
        pred01_raw = (prob >= 0.5).astype(np.uint8)  # 原始预测，HD95 用它算（不受后处理影响）
        pred01 = pred01_raw.copy()
        if sam_model is not None:
            pred01 = sam_refine.refine_mask(sam_model, sam_proc, img_p, pred01, device)
        if args.postproc:
            pred01 = postprocess(pred01)
        if args.boundary_smooth:
            from .improvements.boundary_smooth import boundary_smooth
            pred01 = boundary_smooth(pred01)
        if sd_unet is not None:
            from .diffusion_refine import diffusion_refine_mask
            pred01 = diffusion_refine_mask(sd_unet, img_p, pred01, device)
        pred = pred01 * 255

        if save_dir:
            Image.fromarray(pred, mode='L').save(os.path.join(save_dir, iid + '.jpg'))

        gt_path = os.path.join(TASK1_GT_DIR, iid + '_segmentation.png')
        if os.path.exists(gt_path):
            gt = (np.array(Image.open(gt_path).convert('L')) > 127).astype(np.uint8)
            # 最终指标（提交版） / final metrics (submitted)
            pb = (pred > 127).astype(np.uint8)
            dices.append(dice_score(pb, gt))
            ious.append(iou_score(pb, gt))
            # HD95 用原始预测算（不被后处理干扰）/ HD95 on raw pred (not affected by postproc)
            hd = hausdorff95(pred01_raw, gt)
            if not np.isnan(hd):
                hds.append(hd)

    if dices:
        print(f'{len(ids)} imgs | Dice={np.mean(dices):.4f} IoU={np.mean(ious):.4f} '
              f'HD95={(np.mean(hds) if hds else float("nan")):.2f}')
    else:
        print(f'{len(ids)} imgs saved to {save_dir} (no GT, masks only)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=None, help='单模型 ckpt；与 --ensemble 二选一 / single ckpt; xor with --ensemble')
    ap.add_argument('--ensemble', default=None, help='Tier2: 多 ckpt 集成，逗号分隔 / multi-ckpt ensemble, comma-sep')
    ap.add_argument('--split', default='val', choices=['train', 'val', 'test-local'])
    ap.add_argument('--image_dir', default=None, help='官方测试集图目录；给了就忽略 --split / official test img dir; overrides --split if set')
    ap.add_argument('--save_dir', default=None)
    ap.add_argument('--do_preprocess', type=int, default=1)
    ap.add_argument('--tta', type=int, default=0, help='Tier1: 翻转 TTA 1/0 / flip TTA')
    ap.add_argument('--sam_refine', type=int, default=0, help='Tier1: SAM 边界精修 1/0 / SAM boundary refine')
    ap.add_argument('--postproc', type=int, default=0, help='后处理：形态学+最大连通域 1/0 / morphology + largest CC')
    ap.add_argument('--boundary_smooth', type=int, default=0, help='高斯平滑边界精修 1/0 / Gaussian boundary smooth')
    ap.add_argument('--ms_tta', type=int, default=0, help='多尺度 TTA：0.8x/1.0x/1.2x 平均 1/0 / multi-scale TTA')
    ap.add_argument('--diffusion_refine', type=int, default=0, help='扩散模型边界精修 1/0（需4090+diffusers）/ diffusion boundary refine')
    args = ap.parse_args()
    if not args.ckpt and not args.ensemble:
        ap.error('必须给 --ckpt 或 --ensemble / must give --ckpt or --ensemble')
    infer(args)


if __name__ == '__main__':
    main()
