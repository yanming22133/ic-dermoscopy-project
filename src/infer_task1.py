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
from PIL import Image

from .config import IMAGE_DIR, TASK1_GT_DIR, OUT_DIR, SEED
from .data import get_splits, load_image, get_transforms, IMAGENET_MEAN, IMAGENET_STD, list_image_ids
from .preprocessing import preprocess
from .model import build_segformer
from .metrics import dice_score, iou_score, hausdorff95


@torch.no_grad()
def infer(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    variant = ck.get('variant', 'b2')
    size = ck.get('size', 512)
    model = build_segformer(variant).to(device)
    model.load_state_dict(ck['model'])
    model.eval()

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
        if args.do_preprocess:
            img_p = preprocess(img)
        else:
            img_p = img
        r = tfm(image=img_p, mask=np.zeros((H, W), np.uint8))
        x = r['image']
        x = (x.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(x.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        out = model(pixel_values=x).logits
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        prob = F.softmax(out, dim=1)[0, 1].cpu().numpy()
        pred = (prob >= 0.5).astype(np.uint8) * 255  # 0/255，匹配 example / 0/255, matches example

        if save_dir:
            Image.fromarray(pred, mode='L').save(os.path.join(save_dir, iid + '.jpg'))

        gt_path = os.path.join(TASK1_GT_DIR, iid + '_segmentation.png')
        if os.path.exists(gt_path):
            gt = (np.array(Image.open(gt_path).convert('L')) > 127).astype(np.uint8)
            pb = (pred > 127).astype(np.uint8)
            dices.append(dice_score(pb, gt))
            ious.append(iou_score(pb, gt))
            hd = hausdorff95(pb, gt)
            if not np.isnan(hd):
                hds.append(hd)

    if dices:
        print(f'{len(ids)} imgs | Dice={np.mean(dices):.4f} IoU={np.mean(ious):.4f} '
              f'HD95={(np.mean(hds) if hds else float("nan")):.2f}')
    else:
        print(f'{len(ids)} imgs saved to {save_dir} (no GT, masks only)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--split', default='val', choices=['train', 'val', 'test-local'])
    ap.add_argument('--image_dir', default=None, help='官方测试集图目录；给了就忽略 --split / official test img dir; overrides --split if set')
    ap.add_argument('--save_dir', default=None)
    ap.add_argument('--do_preprocess', type=int, default=1)
    args = ap.parse_args()
    infer(args)


if __name__ == '__main__':
    main()
