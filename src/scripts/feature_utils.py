"""
feature_utils.py — Đặt vào: scripts/feature_utils.py
Bộ trích xuất đặc trưng nâng cao: HSV + HOG + LBP + GrabCut mask
"""

import cv2
import numpy as np
from skimage.feature import local_binary_pattern


# ─────────────────────────────────────────────
# FOREGROUND MASK
# ─────────────────────────────────────────────

def get_foreground_mask(img: np.ndarray) -> np.ndarray:
    """
    GrabCut khi predict ảnh internet (nền phức tạp).
    Fallback về ellipse nếu GrabCut thất bại.
    img: RGB uint8
    """
    h, w = img.shape[:2]
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    margin_x, margin_y = int(w * 0.10), int(h * 0.10)
    rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)
    try:
        mask_gc = np.zeros((h, w), np.uint8)
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        cv2.grabCut(bgr, mask_gc, rect, bgd, fgd, iterCount=3,
                    mode=cv2.GC_INIT_WITH_RECT)
        fg = np.where((mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD),
                      255, 0).astype(np.uint8)
        if fg.sum() < h * w * 0.10 * 255:
            raise ValueError("mask quá nhỏ")
        return fg
    except Exception:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, (w // 2, h // 2),
                    (int(w * 0.42), int(h * 0.42)),
                    0, 0, 360, 255, -1)
        return mask


def get_ellipse_mask(img: np.ndarray) -> np.ndarray:
    """Ellipse mask nhanh — dùng khi train."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (w // 2, h // 2),
                (int(w * 0.42), int(h * 0.42)),
                0, 0, 360, 255, -1)
    return mask


# ─────────────────────────────────────────────
# AUGMENTATION
# ─────────────────────────────────────────────

def augment(img: np.ndarray) -> list:
    """
    3 biến thể từ 1 ảnh gốc (giảm từ 7 xuống 3 để tránh overfit).
    Giữ lại những augment có tác dụng nhất với ảnh rác thực tế.
    """
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2

    # Lật ngang — rác có thể nằm theo chiều nào cũng được
    flip = cv2.flip(img, 1)

    # Xoay 15° — góc chụp lệch nhẹ
    M = cv2.getRotationMatrix2D((cx, cy), 15, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    return [img, flip, rotated]  # x3 thay vì x7


# ─────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────

def _hsv_histogram(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    feats = []
    for ch, bins, rng in [(0, 32, [0, 180]),
                          (1, 32, [0, 256]),
                          (2, 32, [0, 256])]:
        h = cv2.calcHist([hsv], [ch], mask, [bins], rng)
        cv2.normalize(h, h)
        feats.append(h.flatten())
    return np.concatenate(feats)   # 96 dims


def _hog(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    img_m = cv2.bitwise_and(img, img, mask=mask)
    gray = cv2.cvtColor(cv2.resize(img_m, (64, 64)), cv2.COLOR_RGB2GRAY)
    hog = cv2.HOGDescriptor(
        _winSize=(64, 64), _blockSize=(16, 16),
        _blockStride=(8, 8), _cellSize=(8, 8), _nbins=9
    )
    return hog.compute(gray).flatten()   # 1764 dims


def _lbp(img: np.ndarray, mask: np.ndarray,
         P: int = 16, R: float = 2.0) -> np.ndarray:
    """LBP phân biệt texture bề mặt (nhựa/giấy/kim loại...)."""
    gray = cv2.cvtColor(cv2.resize(img, (64, 64)), cv2.COLOR_RGB2GRAY)
    mask64 = cv2.resize(mask, (64, 64))
    lbp = local_binary_pattern(gray, P=P, R=R, method='uniform')
    n_bins = P + 2
    hist, _ = np.histogram(lbp[mask64 > 0], bins=n_bins,
                           range=(0, n_bins), density=True)
    return hist.astype(np.float32)   # 18 dims


def extract_features(img: np.ndarray, use_grabcut: bool = False) -> np.ndarray:
    """
    Pipeline đầy đủ → vector ~1878 dims.

    Args:
        img: RGB uint8, đã resize về IMG_SIZE
        use_grabcut: True khi predict ảnh internet,
                     False khi train (ellipse, nhanh hơn)
    """
    mask = get_foreground_mask(img) if use_grabcut else get_ellipse_mask(img)
    return np.concatenate([
        _hsv_histogram(img, mask),
        _hog(img, mask),
        _lbp(img, mask),
    ]).astype(np.float32)