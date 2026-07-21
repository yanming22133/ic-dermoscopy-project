"""皮肤镜预处理：DullRazor 去毛 + Shades of Gray 颜色恒常（tutorial p35 要求）。
Dermoscopy preprocessing: DullRazor hair removal + Shades of Gray color constancy (tutorial p35).

输入输出都是 HxWx3 uint8 RGB。
Input/output are both HxWx3 uint8 RGB.
"""
import cv2
import numpy as np


def dullrazor(img, kernel_size=15, thresh=15, max_area_frac=0.10):
    """DullRazor：用 blackhat 找毛发，inpaint 抹掉。
    DullRazor: find hair with blackhat, remove it via inpainting.
    小核+高阈值+大面积跳过，保证速度（inpaint 是瓶颈）。
    Small kernel + higher thresh + skip-large-area for speed (inpaint is the bottleneck)."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, thresh, 255, cv2.THRESH_BINARY)
    if hair_mask.sum() / hair_mask.size > max_area_frac:
        return img  # 毛发/伪影太多，inpaint 会很慢，跳过 / too much, skip slow inpaint
    return cv2.inpaint(img, hair_mask, 1, cv2.INPAINT_TELEA)


def shades_of_gray(img, p=6):
    """Shades of Gray 颜色恒常：用 Minkowski p-范数估光照，逐通道归一。
    Shades of Gray color constancy: estimate illuminant via Minkowski p-norm, normalize per channel."""
    img_f = img.astype(np.float32)
    illum = np.power(np.mean(img_f ** p, axis=(0, 1)), 1.0 / p)  # 每通道光照估计 / per-channel illuminant
    illum = np.maximum(illum, 1e-6)
    img_f = img_f / illum[None, None, :]
    img_f = img_f / (img_f.max() + 1e-6) * 255.0
    return img_f.astype(np.uint8)


def preprocess(img):
    """完整预处理：先去毛，再颜色恒常。
    Full preprocessing: hair removal first, then color constancy."""
    img = dullrazor(img)
    img = shades_of_gray(img)
    return img
