"""Task2 推理：用 Task1 预测的病灶 mask 当 ROI，输出 5 个属性 mask + presence.json。
Task2 inference: use the Task1 predicted lesion mask as ROI, output 5 attribute masks + presence.json.

- 属性 mask 存成 {id}/{attr}.png，L，0/255，attr 用单数 milia_like_cyst（照 example）
  attribute masks saved as {id}/{attr}.png, L, 0/255, attr singular milia_like_cyst (per example)
- presence.json 用复数 milia_like_cysts（照 JSON schema），含 prob+status
  presence.json uses plural milia_like_cysts (per JSON schema), with prob+status
- p_attr = mean(sigmoid(logits_attr)) over lesion ROI；status: >=0.60 present / <=0.40 absent / 中间 uncertain

用法 / Usage:
  f:\\anacondaenvs\\pytorch\\python.exe -m src.infer_task2 --ckpt outputs/task2_segformer/best.pth \
      --task1_mask_dir outputs/task1_val_masks --split val --save_dir outputs/task2_val
  # 官方测试集 / official test set:
  ... --image_dir <test dir> --task1_mask_dir submit/task1 --save_dir submit/task2
"""
import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .config import (IMAGE_DIR, TASK1_GT_DIR, OUT_DIR, ATTRS_FILE, ATTRS_JSON,
                     STATUS_HI, STATUS_LO)
from .data import get_splits, load_image, load_image_pp, get_transforms, IMAGENET_MEAN, IMAGENET_STD, list_image_ids
from .preprocessing import preprocess
from .model import build_segformer


def load_pred_lesion_mask(iid, task1_mask_dir):
    """读 Task1 预测的病灶 mask（jpg 有损，>127 还原）。返回 0/1。
    Read Task1 predicted lesion mask (jpg lossy, >127 to recover). Returns 0/1."""
    for ext in ('.jpg', '.png'):
        p = os.path.join(task1_mask_dir, iid + ext)
        if os.path.exists(p):
            m = np.array(Image.open(p).convert('L'))
            return (m > 127).astype(np.uint8)
    return None


@torch.no_grad()
def predict_prob_multi(model, x, H, W, tta=False):
    """返回 5 通道属性概率 [5,H,W]。TTA 时 4 翻转平均。
    Returns 5-channel attribute prob [5,H,W]. TTA averages 4 flips."""
    def fwd(xx):
        o = model(pixel_values=xx).logits
        o = F.interpolate(o, size=(H, W), mode='bilinear', align_corners=False)
        return torch.sigmoid(o)  # [1,5,H,W]
    if not tta:
        return fwd(x)[0].cpu().numpy()
    probs = []
    for h in (False, True):
        for v in (False, True):
            xx = torch.flip(x, [3]) if h else x
            xx = torch.flip(xx, [2]) if v else xx
            p = fwd(xx)
            if h: p = torch.flip(p, [3])
            if v: p = torch.flip(p, [2])
            probs.append(p)
    return torch.stack(probs).mean(0)[0].cpu().numpy()


@torch.no_grad()
def infer(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    variant = ck.get('variant', 'b2'); size = ck.get('size', 512)
    model = build_segformer(variant, num_labels=5).to(device)
    model.load_state_dict(ck['model']); model.eval()

    if args.image_dir:
        ids = list_image_ids(args.image_dir)
    else:
        tr, va, te = get_splits()
        ids = {'train': tr, 'val': va, 'test-local': te}[args.split]

    os.makedirs(args.save_dir, exist_ok=True)
    tfm = get_transforms(False, size)
    presence = {}  # {id: {attr_json: {prob, status}}}
    json_attr_to_file = dict(zip(ATTRS_JSON, ATTRS_FILE))

    for iid in ids:
        img = load_image(iid, args.image_dir or IMAGE_DIR)
        H, W = img.shape[:2]
        img_p = load_image_pp(iid, args.image_dir or IMAGE_DIR) if args.do_preprocess else img
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
                p = predict_prob_multi(model, xs, hs, ws, tta=bool(args.tta))
                p = torch.from_numpy(p).float().unsqueeze(0)  # [1,5,hs,ws]
                p = F.interpolate(p, size=(H, W), mode='bilinear', align_corners=False)[0].numpy()  # [5,H,W]
                probs.append(p)
            prob = np.stack(probs).mean(0)
        else:
            prob = predict_prob_multi(model, x, H, W, tta=bool(args.tta))  # Tier1: TTA

        # ROI = Task1 预测病灶 mask / ROI = Task1 predicted lesion mask
        roi = load_pred_lesion_mask(iid, args.task1_mask_dir)
        if roi is None or not roi.any():
            roi = np.ones((H, W), dtype=bool)  # 没有/空 mask 退化为全图，避免 presence 全 absent
        else:                                  # missing/empty mask -> full image, avoid all-absent
            roi = roi.astype(bool)

        # 存 5 个属性 mask / save 5 attribute masks
        img_dir = os.path.join(args.save_dir, iid)
        os.makedirs(img_dir, exist_ok=True)
        pres = {}
        for c, attr_file in enumerate(ATTRS_FILE):
            mask = (prob[c] >= 0.5).astype(np.uint8) * 255
            Image.fromarray(mask, mode='L').save(os.path.join(img_dir, attr_file + '.png'))
            p_attr = float(prob[c][roi].mean()) if roi.any() else 0.0
            status = 'present' if p_attr >= STATUS_HI else ('absent' if p_attr <= STATUS_LO else 'uncertain')
            attr_json = ATTRS_JSON[c]  # 复数 milia_like_cysts / plural
            pres[attr_json] = {'prob': round(p_attr, 4), 'status': status}
        presence[iid] = pres

    json.dump(presence, open(os.path.join(args.save_dir, 'presence_raw.json'), 'w'), indent=2)
    if args.attr_rules:  # T1: 属性关系后处理 / attribute relationship correction
        from .task2_attr_graph import apply_attr_rules
        presence = apply_attr_rules(presence)

    json.dump(presence, open(os.path.join(args.save_dir, 'presence.json'), 'w'), indent=2)
    print(f'{len(ids)} imgs -> {args.save_dir} (masks + presence.json)', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--task1_mask_dir', required=True, help='Task1 预测病灶 mask 目录 / Task1 predicted lesion mask dir')
    ap.add_argument('--split', default='val', choices=['train', 'val', 'test-local'])
    ap.add_argument('--image_dir', default=None)
    ap.add_argument('--save_dir', required=True)
    ap.add_argument('--do_preprocess', type=int, default=1)
    ap.add_argument('--tta', type=int, default=0, help='Tier1: 翻转 TTA 1/0 / flip TTA')
    ap.add_argument('--ms_tta', type=int, default=0, help='多尺度 TTA：0.8x/1.0x/1.2x 平均 1/0 / multi-scale TTA')
    ap.add_argument('--attr_rules', type=int, default=0, help='T1: 属性关系图后处理 / attribute graph rules')
    args = ap.parse_args()
    infer(args)


if __name__ == '__main__':
    main()
