"""推理侧边界精修（WA-NET + SAM-Adapter 形态学精修组合）。
Inference-side boundary refinement (WA-NET + SAM-Adapter morphological).

对二值 mask 做高斯平滑 → 重新阈值化 → 只改边界像素，不改内部/外部。
Gaussian smooth → re-threshold → only changes boundary pixels.
"""
import cv2
import numpy as np


def boundary_smooth(mask01, sigma=2.0, threshold=0.5, kernel_size=5):
    """边界精修：高斯平滑后重新阈值化。不改训练。
    Boundary refine: Gaussian blur → re-threshold. No training change.

    mask01: HxW 0/1 uint8. Returns refined 0/1 uint8."""
    if mask01.sum() == 0:
        return mask01
    # 高斯模糊
    blurred = cv2.GaussianBlur(mask01.astype(np.float32), (kernel_size, kernel_size), sigma)
    # 重新阈值化
    refined = (blurred >= threshold).astype(np.uint8)
    # 形态学后处理：开运算去噪点 → 闭运算填小孔
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, k)
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, k)
    return refined
