"""数据加载、确定性 80/10/10 划分、增强。
Data loading, deterministic 80/10/10 split, augmentation.

mask 值是 0/255，读进来统一转成 0/1。
Mask values are 0/255; converted to 0/1 on load.
"""
import os
import glob
import numpy as np
import torch
from PIL import Image
import cv2
import albumentations as A

from .config import IMAGE_DIR, TASK1_GT_DIR, TASK2_GT_DIR, ATTRS_FILE, SEED, IMG_SIZE, CACHE_DIR
from .preprocessing import preprocess

IMAGENET_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)


def list_image_ids(image_dir=IMAGE_DIR):
    files = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
    return [os.path.splitext(os.path.basename(f))[0] for f in files]


def get_splits(seed=SEED, ratios=(0.8, 0.1, 0.1)):
    """确定性 80/10/10 = train / val / test-local，固定种子（tutorial p56）。
    Deterministic 80/10/10 = train / val / test-local, fixed seed (tutorial p56)."""
    ids = list_image_ids()
    rng = np.random.RandomState(seed)
    ids = list(ids)
    rng.shuffle(ids)
    n = len(ids)
    n1 = int(n * ratios[0])
    n2 = int(n * (ratios[0] + ratios[1]))
    return ids[:n1], ids[n1:n2], ids[n2:]


def load_image(image_id, image_dir=IMAGE_DIR):
    return np.array(Image.open(os.path.join(image_dir, image_id + '.jpg')).convert('RGB'))


def load_image_pp(image_id, image_dir=IMAGE_DIR):
    """加载预处理好的图（DullRazor+ShadesGray）。有缓存读缓存，无则计算并存盘。
    Load preprocessed image (DullRazor+ShadesGray). Read cache if present, else compute & save."""
    cp = os.path.join(CACHE_DIR, image_id + '.jpg')
    if os.path.exists(cp):
        return np.array(Image.open(cp).convert('RGB'))
    img = preprocess(load_image(image_id, image_dir))
    os.makedirs(CACHE_DIR, exist_ok=True)
    Image.fromarray(img).save(cp, quality=95)
    return img


def _cache_one(iid, image_dir=IMAGE_DIR):
    """处理单张图并写缓存（供多进程调用）。/ Process one image and write cache (for multiprocessing)."""
    cp = os.path.join(CACHE_DIR, iid + '.jpg')
    if os.path.exists(cp):
        return 0
    img = preprocess(load_image(iid, image_dir))
    Image.fromarray(img).save(cp, quality=95)
    return 1


def build_cache(ids, image_dir=IMAGE_DIR, workers=4):
    """一次性预计算所有图的预处理缓存。多进程并行，失败退回单进程。
    Precompute the preprocess cache once. Multiprocessing, falls back to single process."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    from tqdm import tqdm
    n = 0
    try:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_cache_one, iid, image_dir) for iid in ids]
            for f in tqdm(futs, desc='building preprocess cache'):
                n += f.result()
    except Exception as e:
        print(f'parallel cache failed ({e}), falling back to single process', flush=True)
        for iid in tqdm(ids, desc='building preprocess cache (single)'):
            n += _cache_one(iid, image_dir)
    print(f'cache ready at {CACHE_DIR} (newly built: {n})', flush=True)


def load_task1_mask(image_id, gt_dir=TASK1_GT_DIR):
    m = np.array(Image.open(os.path.join(gt_dir, image_id + '_segmentation.png')).convert('L'))
    return (m > 127).astype(np.uint8)


def get_transforms(train=True, size=IMG_SIZE):
    """val 只 resize；train 加几何+颜色增强。mask 走最近邻插值保证不引入新值。
    val only resizes; train adds geometric + color aug. Mask uses nearest interp to avoid new values."""
    common = [A.Resize(size, size, interpolation=cv2.INTER_LINEAR)]
    if train:
        return A.Compose([
            A.Resize(size, size, interpolation=cv2.INTER_LINEAR),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=30,
                               border_mode=0, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.RandomGamma(p=0.2),          # Tier1: 更强增强 / stronger aug
            A.CLAHE(p=0.2),                # Tier1: 对比度受限直方图均衡 / CLAHE
            A.GaussianBlur(p=0.2),
        ])
    return A.Compose(common)


def compute_sample_weights(ids, rare_attrs=('negative_network', 'streaks'), rare_weight=3.0):
    """Tier1: Task2 稀疏类过采样。含稀有属性的样本权重提高。
    Tier1: Task2 sparse-class oversampling. Up-weight samples with rare attributes.
    返回每个 id 的采样权重列表。/ Returns a per-id sampling weight list."""
    from .config import ATTRS_FILE
    rare_idx = [ATTRS_FILE.index(a) for a in rare_attrs if a in ATTRS_FILE]
    weights = []
    for iid in ids:
        masks = load_task2_masks(iid)  # [H,W,5]
        has_rare = any(masks[..., i].any() for i in rare_idx)
        weights.append(rare_weight if has_rare else 1.0)
    return weights


class Task1Dataset(torch.utils.data.Dataset):
    def __init__(self, ids, train=True, size=IMG_SIZE, do_preprocess=True, use_cache=True):
        self.ids = ids
        self.train = train
        self.tfm = get_transforms(train, size)
        self.do_preprocess = do_preprocess
        self.use_cache = use_cache

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        if self.do_preprocess:
            img = load_image_pp(self.ids[i]) if self.use_cache else preprocess(load_image(self.ids[i]))
        else:
            img = load_image(self.ids[i])
        mask = load_task1_mask(self.ids[i])
        r = self.tfm(image=img, mask=mask)
        img, mask = r['image'], r['mask']
        img = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        mask = torch.from_numpy(mask.astype(np.int64))
        return img, mask, self.ids[i]


# ========== Task2: 5 通道多标签属性分割 / Task2: 5-channel multi-label attribute seg ==========
def load_task2_masks(image_id, gt_dir=TASK2_GT_DIR):
    """读 5 张属性 mask，返回 [H,W,5] 的 0/1 数组（多标签，允许重叠）。
    Read 5 attribute masks, return [H,W,5] 0/1 array (multi-label, overlap allowed)."""
    masks = []
    for attr in ATTRS_FILE:  # 文件名用单数 milia_like_cyst / filenames use singular
        p = os.path.join(gt_dir, f'{image_id}_attribute_{attr}.png')
        m = np.array(Image.open(p).convert('L')) > 127
        masks.append(m.astype(np.uint8))
    return np.stack(masks, axis=-1)  # [H,W,5]


class Task2Dataset(torch.utils.data.Dataset):
    """Task2 多标签分割数据集。返回 (img, mask[B,5,H,W], id)。
    Task2 multi-label segmentation dataset. Returns (img, mask[B,5,H,W], id)."""
    def __init__(self, ids, train=True, size=IMG_SIZE, do_preprocess=True, use_cache=True):
        self.ids = ids
        self.train = train
        self.tfm = get_transforms(train, size)
        self.do_preprocess = do_preprocess
        self.use_cache = use_cache

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        if self.do_preprocess:
            img = load_image_pp(self.ids[i]) if self.use_cache else preprocess(load_image(self.ids[i]))
        else:
            img = load_image(self.ids[i])
        masks = load_task2_masks(self.ids[i])  # [H,W,5]
        # albumentations 用 masks= 传多张 mask，同步变换 / pass multiple masks via masks=, transformed in sync
        r = self.tfm(image=img, masks=[masks[..., k] for k in range(masks.shape[-1])])
        img = r['image']
        masks = np.stack(r['masks'], axis=0)  # [5,H,W]
        img = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        mask = torch.from_numpy(masks.astype(np.float32))  # [5,H,W] 0/1
        return img, mask, self.ids[i]
