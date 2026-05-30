"""
train_svm_pure_v2.py — Đặt vào: scripts/train_svm_pure_v2.py
Thay thế cho train_svm_pure.py
Yêu cầu: feature_utils.py phải nằm cùng thư mục scripts/
"""

import os
import sys
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.core.config import CLASSES, PATHS
from feature_utils import extract_features, augment

IMG_SIZE = (64, 64)


def log(msg: str, start: float = None):
    elapsed = f"  [{time.time() - start:.1f}s]" if start else ""
    print(f"{msg}{elapsed}", flush=True)


def load_data(data_dir: str, use_augment: bool = False):
    X, y = [], []
    label = "TRAIN (+ augment)" if use_augment else "VALIDATION"
    log(f"\n📂 [{label}] Tải từ: {data_dir}")

    total_files = 0
    for class_name in CLASSES:
        p = os.path.join(data_dir, class_name)
        if os.path.exists(p):
            total_files += len([f for f in os.listdir(p)
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    log(f"   Tổng ảnh gốc: {total_files}"
        + (f"  →  sau augment: ~{total_files * 7}" if use_augment else ""))

    t0 = time.time()
    for class_idx, class_name in enumerate(CLASSES):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.exists(class_path):
            log(f"  ⚠️  Bỏ qua (không tìm thấy): {class_path}")
            continue

        files = [f for f in os.listdir(class_path)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        t_cls = time.time()
        for f in tqdm(files, desc=f"  [{class_idx+1}/{len(CLASSES)}] {class_name:15s}", unit="img"):
            try:
                img = cv2.imread(os.path.join(class_path, f))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, IMG_SIZE)
                for aug_img in (augment(img) if use_augment else [img]):
                    X.append(extract_features(aug_img, use_grabcut=False))
                    y.append(class_idx)
            except Exception:
                continue

        n_class = y.count(class_idx)
        log(f"     ✓ {class_name}: {n_class} samples  [{time.time()-t_cls:.1f}s]")

    log(f"\n   ✅ Load xong: {len(X)} samples, {len(set(y))} classes", t0)
    return np.array(X, dtype=np.float32), np.array(y)


def train_svm():
    t_total = time.time()
    log("=" * 55)
    log("  🚀 BẮT ĐẦU TRAIN SVM v2")
    log("=" * 55)

    train_dir = os.path.join(project_root, 'dataset', 'train')
    val_dir   = os.path.join(project_root, 'dataset', 'validation')

    # 1. Load
    X_train, y_train = load_data(train_dir, use_augment=True)
    X_val,   y_val   = load_data(val_dir,   use_augment=False)

    # 2. Scale
    log("\n⚙️  [2/4] StandardScaler — chuẩn hóa features...")
    t = time.time()
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_vl = scaler.transform(X_val)
    log(f"   ✓ Xong  |  X_train: {X_tr.shape}  X_val: {X_vl.shape}", t)

    # 3. Train SVM
    log("\n🤖 [3/4] Huấn luyện SVM (RBF, C=10) — bước lâu nhất...")
    log(f"   Input: {X_tr.shape[0]} samples × {X_tr.shape[1]} features")
    log("   ⏳ Đang fit... (dấu *. bên dưới là đang chạy bình thường)")
    t = time.time()
    model = SVC(
        kernel='rbf',
        C=10.0,
        gamma='scale',
        class_weight='balanced',
        probability=True,
        random_state=42,
        verbose=True
    )
    model.fit(X_tr, y_train)
    log(f"   ✓ Train xong!", t)

    # 4. Đánh giá
    log("\n📊 [4/4] Đánh giá trên validation set...")
    t = time.time()
    y_pred = model.predict(X_vl)
    acc = accuracy_score(y_val, y_pred)
    log(f"   ✓ Predict xong", t)

    log(f"\n{'='*55}")
    log(f"  ⭐ ACCURACY: {acc*100:.2f}%")
    log(f"{'='*55}\n")
    print(classification_report(y_val, y_pred, target_names=CLASSES))

    # 5. Confusion matrix
    output_dir = os.path.join(project_root, 'outputs', 'svm_results')
    os.makedirs(output_dir, exist_ok=True)

    log("💾 Lưu confusion matrix...")
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(max(8, len(CLASSES)), max(6, len(CLASSES) - 1)))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=CLASSES, yticklabels=CLASSES, cmap='YlOrRd')
    plt.title(f'SVM v2 — Acc: {acc*100:.2f}%')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'svm_v2_confusion_matrix.png'), dpi=120)
    log(f"   → outputs/svm_results/svm_v2_confusion_matrix.png")

    # 6. Lưu model
    log("💾 Lưu model...")
    model_path = os.path.join(output_dir, 'svm_v2_model.pkl')
    joblib.dump({
        'model':        model,
        'scaler':       scaler,
        'img_size':     IMG_SIZE,
        'feature_type': 'hsv+hog+lbp_v2',
        'classes':      CLASSES,
        'use_grabcut_at_predict': True,
    }, model_path)
    log(f"   → {model_path}")

    log(f"\n🏁 HOÀN TẤT!", t_total)


if __name__ == "__main__":
    train_svm()