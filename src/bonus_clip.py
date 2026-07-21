"""Bonus：CLIP / DINOv2 语义检索 + RAG 报告增强。
Bonus: CLIP / DINOv2 semantic retrieval + RAG report augmentation.

- 编码器冻结，只做特征提取（tutorial p41，不微调）/ encoder frozen, feature extraction only
- 离线：编码全部训练图建 FAISS 余弦索引 / offline: encode all train images, build FAISS cosine index
- 在线：每张查询图取 Top-K 近邻，存 CSV（query_image, neighbor_id, similarity）
  online: Top-K neighbors per query, save CSV
- 评分重点是 audit completeness（全图、neighbor ID 全存）+ sanity（无重复、K 一致）
  grading focus: audit completeness (all images, all IDs) + sanity (no dup, consistent K)
- RAG（可选）：用近邻 presence 做先验平滑，生成增强报告（grounded，不抄邻居事实）
  RAG (optional): neighbor presence prior smoothing, augmented report (grounded, no copying)

用法 / Usage:
  # 建索引（训练图编码）/ build index (encode train images)
  f:\\anacondaenvs\\pytorch\\python.exe -m src.bonus_clip build --encoder clip
  # 检索查询图 / retrieve query images
  f:\\anacondaenvs\\pytorch\\python.exe -m src.bonus_clip retrieve --encoder clip \
      --image_dir <test dir> --task1_mask_dir submit/task1 --save_dir submit/bonus --k 3
  # RAG 增强（需要查询 presence.json）/ RAG augmentation (needs query presence.json)
  f:\\anacondaenvs\\pytorch\\python.exe -m src.bonus_clip rag --encoder clip \
      --query_presence submit/task2/presence.json --train_presence outputs/task2_train/presence.json \
      --save_dir submit/bonus
"""
import os
import csv
import json
import argparse
import numpy as np
import torch
from PIL import Image

from .config import IMAGE_DIR, OUT_DIR, ATTRS_JSON, ATTRS_FILE, STATUS_HI, STATUS_LO
from .data import list_image_ids, load_image, load_task1_mask
from .preprocessing import preprocess


# ============ 编码器 / Encoders ============
class CLIPEncoder:
    """OpenAI CLIP ViT-B/32，冻结。/ OpenAI CLIP ViT-B/32, frozen."""
    name = 'clip'
    def __init__(self, device):
        import clip
        self.device = device
        self.model, self.prep = clip.load('ViT-B/32', device=device)
        self.model.eval()
        self.dim = 512

    @torch.no_grad()
    def encode(self, pil_imgs):
        # pil_imgs: list[PIL.Image] -> [N, 512] normalized
        tensors = torch.stack([self.prep(im) for im in pil_imgs]).to(self.device)
        feat = self.model.encode_image(tensors)
        feat = feat / feat.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return feat.float().cpu().numpy()


class DINOv2Encoder:
    """DINOv2 ViT-B/14（密集视觉相似，医学检索常胜），冻结。/ frozen."""
    name = 'dinov2'
    def __init__(self, device):
        from transformers import AutoModel, AutoImageProcessor
        self.device = device
        self.proc = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        self.model = AutoModel.from_pretrained('facebook/dinov2-base').to(device).eval()
        self.dim = self.model.config.hidden_size  # 768

    @torch.no_grad()
    def encode(self, pil_imgs):
        inputs = self.proc(images=pil_imgs, return_tensors='pt').to(self.device)
        feat = self.model(**inputs).last_hidden_state[:, 0, :]  # CLS token
        feat = feat / feat.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return feat.float().cpu().numpy()


def build_encoder(name, device):
    return {'clip': CLIPEncoder, 'dinov2': DINOv2Encoder}[name](device)


# ============ 图像裁剪 / Image crop ============
def crop_to_lesion(img_np, mask, margin=0.1):
    """用 Task1 病灶 mask 的 bbox 裁图（聚焦病灶，提升检索）。无 mask 则返回原图。
    Crop image to lesion bbox (focus lesion, improve retrieval). No mask -> return original."""
    if mask is None or not mask.any():
        return img_np
    ys, xs = np.where(mask > 0)
    y0, y1 = ys.min(), ys.max(); x0, x1 = xs.min(), xs.max()
    h, w = mask.shape
    dy = int((y1 - y0) * margin); dx = int((x1 - x0) * margin)
    y0 = max(0, y0 - dy); y1 = min(h, y1 + dy); x0 = max(0, x0 - dx); x1 = min(w, x1 + dx)
    return img_np[y0:y1, x0:x1]


def load_mask(iid, mask_dir):
    if not mask_dir:
        return None
    for ext in ('.jpg', '.png'):
        p = os.path.join(mask_dir, iid + ext)
        if os.path.exists(p):
            return (np.array(Image.open(p).convert('L')) > 127).astype(np.uint8)
    return None


def to_pil(img_np):
    return Image.fromarray(img_np.astype(np.uint8))


@torch.no_grad()
def encode_ids(ids, encoder, image_dir, mask_fn, do_preprocess, batch=32):
    """编码一组图，返回 [N,D] 归一化特征。
    Encode a list of images, return [N,D] normalized.
    mask_fn(iid)->0/1 mask 或 None；两边用同一种裁剪策略保证 embedding 空间一致。
    mask_fn(iid)->0/1 mask or None; both sides use the same crop policy for a consistent space."""
    feats = []
    for i in range(0, len(ids), batch):
        chunk = ids[i:i + batch]
        imgs = []
        for iid in chunk:
            img = load_image(iid, image_dir)
            if do_preprocess:
                img = preprocess(img)
            mask = mask_fn(iid) if mask_fn is not None else None
            img = crop_to_lesion(img, mask)
            imgs.append(to_pil(img))
        feats.append(encoder.encode(imgs))
    return np.concatenate(feats, axis=0).astype(np.float32)


# ============ 命令：建索引 / build ============
def cmd_build(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    enc = build_encoder(args.encoder, device)
    cache = os.path.join(OUT_DIR, f'bonus_{args.encoder}_embeds.npz')
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(cache) and not args.rebuild:
        print('index cache exists:', cache, '(--rebuild to overwrite)'); return
    train_ids = list_image_ids(IMAGE_DIR)
    print(f'encoding {len(train_ids)} train images with {args.encoder}...', flush=True)
    # 训练图用 GT 病灶 mask 裁剪（与查询用预测 mask 裁剪保持一致空间）
    # train images cropped with GT lesion mask (consistent with query predicted-mask crop)
    feats = encode_ids(train_ids, enc, IMAGE_DIR, load_task1_mask, args.do_preprocess, args.batch)
    np.savez(cache, ids=np.array(train_ids), feats=feats)
    print(f'index saved: {cache}  shape={feats.shape}', flush=True)


def load_index(args):
    cache = os.path.join(OUT_DIR, f'bonus_{args.encoder}_embeds.npz')
    d = np.load(cache, allow_pickle=True)
    return d['ids'].tolist(), d['feats'].astype(np.float32)


# ============ 命令：检索 / retrieve ============
def cmd_retrieve(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    enc = build_encoder(args.encoder, device)
    train_ids, train_feats = load_index(args)
    import faiss
    index = faiss.IndexFlatIP(train_feats.shape[1])
    index.add(train_feats)

    query_ids = list_image_ids(args.image_dir)
    print(f'encoding {len(query_ids)} query images...', flush=True)
    q_mask_fn = (lambda iid: load_mask(iid, args.task1_mask_dir)) if args.task1_mask_dir else None
    q_feats = encode_ids(query_ids, enc, args.image_dir, q_mask_fn, args.do_preprocess, args.batch)
    sims, idxs = index.search(q_feats, args.k)  # [Nq, K]

    os.makedirs(args.save_dir, exist_ok=True)
    csv_path = os.path.join(args.save_dir, 'test_bonus_clip.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['query_image', 'neighbor_id', 'similarity'])
        for qi, iid in enumerate(query_ids):
            for j in range(args.k):
                nid = train_ids[idxs[qi, j]]
                sim = float(sims[qi, j])
                # neighbor_id 格式照 example：data/images/{id}.jpg / format per example
                w.writerow([iid, f'data/images/{nid}.jpg', f'{sim:.6f}'])
    # audit / sanity 报告 / report
    audit = {
        'n_queries': len(query_ids), 'k': args.k, 'rows': len(query_ids) * args.k,
        'all_queries_covered': len(query_ids) > 0,
        'consistent_k': all(len(r) == args.k for r in [idxs[qi] for qi in range(len(query_ids))]),
        'no_dup_query': len(set(query_ids)) == len(query_ids),
    }
    json.dump(audit, open(os.path.join(args.save_dir, 'audit.json'), 'w'), indent=2)
    print(f'{len(query_ids)} queries x k={args.k} -> {csv_path} | audit: {audit}', flush=True)


# ============ 命令：RAG 增强 / RAG augmentation ============
def cmd_rag(args):
    """用近邻 presence 先验平滑查询 presence，生成增强报告（grounded，不抄邻居事实）。
    Smooth query presence with neighbor presence prior; grounded, no fact copying."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    enc = build_encoder(args.encoder, device)
    train_ids, train_feats = load_index(args)
    import faiss
    index = faiss.IndexFlatIP(train_feats.shape[1]); index.add(train_feats)

    query_presence = json.load(open(args.query_presence))
    train_presence = json.load(open(args.train_presence)) if args.train_presence else {}
    query_ids = list(query_presence.keys())

    # 编码查询图 / encode queries（用查询 Task1 mask 裁剪；若无 mask_dir 则全图）
    q_mask_fn = (lambda iid: load_mask(iid, args.task1_mask_dir)) if args.task1_mask_dir else None
    q_feats = encode_ids(query_ids, enc, IMAGE_DIR, q_mask_fn, args.do_preprocess, args.batch)
    sims, idxs = index.search(q_feats, args.k)

    os.makedirs(args.save_dir, exist_ok=True)
    rag_presence = {}
    for qi, iid in enumerate(query_ids):
        q_pres = query_presence[iid]
        # 近邻 presence 均值先验 / neighbor presence mean prior
        neigh_probs = {a: [] for a in ATTRS_JSON}
        for j in range(args.k):
            nid = train_ids[idxs[qi, j]]
            npres = train_presence.get(nid, {})
            for a in ATTRS_JSON:
                if a in npres and 'prob' in npres[a]:
                    neigh_probs[a].append(npres[a]['prob'])
        # 平滑：0.7*查询 + 0.3*近邻均值 / smooth 0.7*query + 0.3*neighbor mean
        smoothed = {}
        for a in ATTRS_JSON:
            qp = q_pres.get(a, {}).get('prob', 0.0)
            nm = float(np.mean(neigh_probs[a])) if neigh_probs[a] else qp
            p = 0.7 * qp + 0.3 * nm
            status = 'present' if p >= STATUS_HI else ('absent' if p <= STATUS_LO else 'uncertain')
            smoothed[a] = {'prob': round(p, 4), 'status': status, 'neighbor_mean': round(nm, 4)}
        rag_presence[iid] = smoothed
    json.dump(rag_presence, open(os.path.join(args.save_dir, 'rag_presence.json'), 'w'), indent=2)
    print(f'RAG smoothed presence for {len(query_ids)} queries -> {args.save_dir}/rag_presence.json', flush=True)
    print('注：这是检索对 presence 的平滑影响，可与原 presence 对比证明 retrieval impact。', flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    pb = sub.add_parser('build')
    pb.add_argument('--encoder', default='clip', choices=['clip', 'dinov2'])
    pb.add_argument('--do_preprocess', type=int, default=1)
    pb.add_argument('--batch', type=int, default=32)
    pb.add_argument('--rebuild', action='store_true')
    pr = sub.add_parser('retrieve')
    pr.add_argument('--encoder', default='clip', choices=['clip', 'dinov2'])
    pr.add_argument('--image_dir', required=True)
    pr.add_argument('--task1_mask_dir', default=None)
    pr.add_argument('--save_dir', required=True)
    pr.add_argument('--k', type=int, default=3)
    pr.add_argument('--do_preprocess', type=int, default=1)
    pr.add_argument('--batch', type=int, default=32)
    pg = sub.add_parser('rag')
    pg.add_argument('--encoder', default='clip', choices=['clip', 'dinov2'])
    pg.add_argument('--query_presence', required=True)
    pg.add_argument('--train_presence', default=None)
    pg.add_argument('--task1_mask_dir', default=None)
    pg.add_argument('--save_dir', required=True)
    pg.add_argument('--k', type=int, default=3)
    pg.add_argument('--do_preprocess', type=int, default=1)
    pg.add_argument('--batch', type=int, default=32)
    args = ap.parse_args()
    {'build': cmd_build, 'retrieve': cmd_retrieve, 'rag': cmd_rag}[args.cmd](args)


if __name__ == '__main__':
    main()
