"""
feature_utils_v3.py
Sửa vấn đề: model học màu nền thay vì học vật thể
Chiến lược:
  1. Foreground mask (GrabCut / ellipse) → chỉ lấy features từ vùng VẬT THỂ
  2. Giảm trọng số HSV color (nền trắng gây bias)
  3. Tăng mạnh Texture features: LBP, Gabor, GLCM
  4. Edge features: Canny + contour complexity
  5. Augmentation thêm biến thể nền khác nhau
"""

import cv2
import numpy as np

try:
    from skimage.feature import local_binary_pattern
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

IMG_SIZE = (128, 128)


# ═══════════════════════════════════════════════════════════
# FOREGROUND MASKING — loại bỏ ảnh hưởng của nền
# ═══════════════════════════════════════════════════════════

def get_foreground_mask(img_rgb: np.ndarray, method='auto') -> np.ndarray:
    """
    Tách vật thể khỏi nền.
    method='grabcut' → chính xác hơn nhưng chậm hơn
    method='threshold' → nhanh, hiệu quả với nền trắng/đơn sắc
    method='auto' → thử threshold trước, fallback sang ellipse
    """
    h, w = img_rgb.shape[:2]

    if method == 'grabcut':
        return _grabcut_mask(img_rgb)

    if method in ('threshold', 'auto'):
        mask = _threshold_mask(img_rgb)
        fg_ratio = mask.sum() / (h * w * 255)
        # Nếu threshold cho quá ít hoặc quá nhiều vùng foreground → dùng ellipse
        if 0.10 < fg_ratio < 0.85:
            return mask

    # Fallback: ellipse crop (tránh 4 góc thường là nền)
    return _ellipse_mask(h, w)


def _threshold_mask(img_rgb: np.ndarray) -> np.ndarray:
    """
    Hiệu quả với nền TRẮNG, NỀN ĐƠN SẮC, NỀN SÁNG
    Loại bỏ pixel có saturation thấp (trắng/xám/đen thuần) ở viền
    """
    h, w = img_rgb.shape[:2]
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    # Nền trắng: V cao + S thấp
    white_bg = (hsv[:, :, 1] < 30) & (hsv[:, :, 2] > 200)
    # Nền đen: V thấp
    black_bg = hsv[:, :, 2] < 20

    bg_mask = (white_bg | black_bg).astype(np.uint8) * 255
    fg_mask = cv2.bitwise_not(bg_mask)

    # Morphology để làm mượt mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return fg_mask


def _grabcut_mask(img_rgb: np.ndarray) -> np.ndarray:
    """GrabCut — chính xác nhất nhưng ~5x chậm hơn"""
    h, w = img_rgb.shape[:2]
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    mx, my = int(w * 0.10), int(h * 0.10)
    rect = (mx, my, w - 2 * mx, h - 2 * my)
    try:
        mask_gc = np.zeros((h, w), np.uint8)
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        cv2.grabCut(bgr, mask_gc, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
        fg = np.where(
            (mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)
        if fg.sum() > h * w * 0.05 * 255:
            return fg
    except Exception:
        pass
    return _ellipse_mask(h, w)


def _ellipse_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (w // 2, h // 2),
                (int(w * 0.42), int(h * 0.42)), 0, 0, 360, 255, -1)
    return mask


# ═══════════════════════════════════════════════════════════
# FEATURE EXTRACTION — trọng tâm là TEXTURE & SHAPE
# ═══════════════════════════════════════════════════════════

def _hsv_histogram_masked(img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    HSV histogram CHỈ trên vùng foreground.
    Dùng ít bins hơn (24 thay vì 32) để giảm ảnh hưởng của màu sắc.
    Bỏ kênh V (brightness) vì nó bị ảnh hưởng nhiều bởi ánh sáng.
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    feats = []
    # Chỉ H (màu sắc thực) và S (độ bão hòa) — bỏ V
    for ch, bins, rng in [(0, 24, [0, 180]),   # Hue
                          (1, 24, [0, 256])]:   # Saturation
        h = cv2.calcHist([hsv], [ch], mask, [bins], rng).flatten()
        h /= (h.sum() + 1e-7)
        feats.append(h)
    return np.concatenate(feats)   # 48 dims (giảm từ 96)


def _lbp_texture_masked(img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    LBP — phân biệt bề mặt: nhựa bóng vs giấy nhám vs kim loại vs bìa cứng
    Dùng multi-scale LBP (R=1, R=2, R=3) để bắt texture ở nhiều mức độ
    """
    if not HAS_SKIMAGE:
        return np.array([], dtype=np.float32)

    gray = cv2.cvtColor(cv2.resize(img_rgb, (64, 64)), cv2.COLOR_RGB2GRAY)
    mask64 = cv2.resize(mask, (64, 64))
    feats = []
    for P, R in [(8, 1), (16, 2), (24, 3)]:
        lbp = local_binary_pattern(gray, P=P, R=R, method='uniform')
        n_bins = P + 2
        hist, _ = np.histogram(
            lbp[mask64 > 0], bins=n_bins, range=(0, n_bins), density=True
        )
        feats.append(hist.astype(np.float32))
    return np.concatenate(feats)   # 10+18+26 = 54 dims


def _gabor_texture(img_rgb: np.ndarray) -> np.ndarray:
    """
    Gabor filter bank — nhạy với texture định hướng
    Bìa cứng có vân song song, nhựa bóng không có vân
    """
    gray = cv2.cvtColor(
        cv2.resize(img_rgb, (64, 64)), cv2.COLOR_RGB2GRAY
    ).astype(np.float32)
    feats = []
    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        for sigma, freq in [(3, 0.1), (5, 0.25)]:
            k = cv2.getGaborKernel(
                (15, 15), sigma, theta, 1.0/freq, 0.5, 0, ktype=cv2.CV_32F
            )
            f = cv2.filter2D(gray, cv2.CV_32F, k)
            feats.extend([f.mean() / 255.0, f.std() / 255.0])
    return np.array(feats, dtype=np.float32)   # 16 dims


def _hog_shape(img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """HOG trên vùng foreground — hình dạng vật thể"""
    img_m = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
    gray = cv2.cvtColor(cv2.resize(img_m, (64, 64)), cv2.COLOR_RGB2GRAY)
    hog = cv2.HOGDescriptor(
        _winSize=(64, 64), _blockSize=(16, 16),
        _blockStride=(8, 8), _cellSize=(8, 8), _nbins=9
    )
    return hog.compute(gray).flatten()   # 1764 dims


def _edge_complexity(img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Đặc trưng cạnh/hình dạng:
    - Edge density: mật độ cạnh trong vùng FG
    - Contour complexity: độ phức tạp chu vi vật thể
    Giúp phân biệt: chai nhựa (cạnh cong đơn giản) vs rác hỗn hợp (cạnh phức tạp)
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask)

    edges = cv2.Canny(masked_gray, 50, 150)
    fg_pixels = mask.sum() / 255 + 1e-7
    edge_density = edges.sum() / (255 * fg_pixels)

    # Contour-based shape descriptors
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest) + 1e-7
        perimeter = cv2.arcLength(largest, True) + 1e-7
        # Circularity: hình tròn=1, hình phức tạp<1
        circularity = 4 * np.pi * area / (perimeter ** 2)
        # Extent: tỉ lệ diện tích vật / bounding box
        _, _, bw, bh = cv2.boundingRect(largest)
        extent = area / (bw * bh + 1e-7)
    else:
        circularity, extent = 0.5, 0.5

    return np.array([edge_density, circularity, extent], dtype=np.float32)   # 3 dims


def _glcm_features(img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    GLCM (Gray-Level Co-occurrence Matrix) — texture thống kê
    Contrast, Dissimilarity, Homogeneity, Energy, Correlation
    Nhựa bóng: homogeneity cao | Rác hữu cơ: contrast thấp, energy cao
    """
    try:
        from skimage.feature import graycomatrix, graycoprops
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        masked = cv2.bitwise_and(gray, gray, mask=mask)
        # Quantize về 32 levels để tính GLCM nhanh
        img_q = (masked // 8).astype(np.uint8)
        gcm = graycomatrix(img_q, distances=[1, 3], angles=[0, np.pi/4],
                           levels=32, symmetric=True, normed=True)
        feats = []
        for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
            feats.append(graycoprops(gcm, prop).flatten())
        return np.concatenate(feats).astype(np.float32)   # 5×4 = 20 dims
    except Exception:
        return np.zeros(20, dtype=np.float32)


# ═══════════════════════════════════════════════════════════
# MAIN FUNCTION
# ═══════════════════════════════════════════════════════════

def extract_features(img_rgb: np.ndarray, use_grabcut: bool = False) -> np.ndarray:
    """
    Pipeline đầy đủ — tối ưu chống color bias.

    Thành phần & trọng tâm:
      HSV (48)         ← GIẢM (nền trắng gây bias ở đây)
      LBP texture (54) ← TĂNG (texture bề mặt vật thể)
      Gabor (16)       ← TĂNG (vân texture định hướng)
      HOG shape (1764) ← GIỮ  (hình dạng vật thể)
      Edges (3)        ← MỚI  (độ phức tạp cạnh)
      GLCM (20)        ← MỚI  (thống kê texture)
    Tổng: ~1905 dims
    """
    mask = get_foreground_mask(
        img_rgb, method='grabcut' if use_grabcut else 'auto'
    )

    parts = [
        _hsv_histogram_masked(img_rgb, mask),     # 48
        _lbp_texture_masked(img_rgb, mask),        # 54
        _gabor_texture(img_rgb),                   # 16
        _hog_shape(img_rgb, mask),                 # 1764
        _edge_complexity(img_rgb, mask),           # 3
        _glcm_features(img_rgb, mask),             # 20
    ]
    return np.concatenate([p for p in parts if len(p) > 0]).astype(np.float32)


# ═══════════════════════════════════════════════════════════
# AUGMENTATION — thêm biến thể nền khác nhau
# ═══════════════════════════════════════════════════════════

def _replace_background(img_rgb: np.ndarray, mask: np.ndarray,
                         bg_color: tuple) -> np.ndarray:
    """Thay nền bằng màu khác để model không bị overfit màu nền"""
    bg = np.full_like(img_rgb, bg_color, dtype=np.uint8)
    fg = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
    inv_mask = cv2.bitwise_not(mask)
    bg_part = cv2.bitwise_and(bg, bg, mask=inv_mask)
    return cv2.add(fg, bg_part)


def augment(img_rgb: np.ndarray) -> list:
    """
    Augment gồm:
    - Geometric: flip, rotate ±15°
    - Background swap: nền trắng → nền xám, nền tối
      (giúp model không học màu nền)
    """
    h, w = img_rgb.shape[:2]
    cx, cy = w // 2, h // 2

    # Geometric augmentation
    flipped = cv2.flip(img_rgb, 1)
    M1 = cv2.getRotationMatrix2D((cx, cy), 15, 1.0)
    rot_p15 = cv2.warpAffine(img_rgb, M1, (w, h), borderMode=cv2.BORDER_REFLECT)
    M2 = cv2.getRotationMatrix2D((cx, cy), -15, 1.0)
    rot_m15 = cv2.warpAffine(img_rgb, M2, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Background augmentation — QUAN TRỌNG để chống bias nền trắng
    mask = get_foreground_mask(img_rgb, method='auto')
    gray_bg  = _replace_background(img_rgb, mask, (128, 128, 128))  # nền xám
    dark_bg  = _replace_background(img_rgb, mask, (40, 40, 40))     # nền tối
    beige_bg = _replace_background(img_rgb, mask, (210, 195, 170))  # nền be/gỗ

    return [img_rgb, flipped, rot_p15, rot_m15,
            gray_bg, dark_bg, beige_bg]   # x7 variants


# ═══════════════════════════════════════════════════════════
# QUICK TEST
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        feat = extract_features(img, use_grabcut=False)
        print(f"Feature vector shape: {feat.shape}")
        print(f"Non-zero: {(feat != 0).sum()}")
        mask = get_foreground_mask(img, method='auto')
        print(f"Foreground ratio: {mask.sum() / (255 * mask.size):.2%}")
    else:
        print("Usage: python feature_utils_v3.py <image_path>")
