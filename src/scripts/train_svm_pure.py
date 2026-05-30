"""
train_svm_pure_v2.py — Đặt vào: scripts/train_svm_pure_v2.py
Thay thế cho train_svm_pure.py
Yêu cầu: feature_utils.py phải nằm cùng thư mục scripts/
"""

import os
import sys

# Thêm project root vào sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.core.config import CLASSES, PATHS
from feature_utils import extract_features, augment   # cùng thư mục scripts/

IMG_SIZE = (128, 128)


def load_data(data_dir: str, use_augment: bool = False):
    X, y = [], []
    print(f"\n📂 Tải từ: {data_dir}  (augment={use_augment})")
    for class_idx, class_name in enumerate(CLASSES):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.exists(class_path):
            print(f"  ⚠️  Bỏ qua (không tìm thấy): {class_path}")
            continue
        files = [f for f in os.listdir(class_path)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for f in tqdm(files, desc=f"  {class_name:15s}"):
            try:
                img = cv2.imread(os.path.join(class_path, f))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, IMG_SIZE)
                for aug_img in (augment(img) if use_augment else [img]):
                    X.append(extract_features(aug_img, use_grabcut=False))
                    y.append(class_idx)
            except Exception:
                continue
    print(f"  → {len(X)} samples, {len(set(y))} classes")
    return np.array(X, dtype=np.float32), np.array(y)


def train_svm():
    train_dir = os.path.join(project_root, 'dataset', 'train')
    val_dir   = os.path.join(project_root, 'dataset', 'validation')

    # 1. Load
    X_train, y_train = load_data(train_dir, use_augment=True)
    X_val,   y_val   = load_data(val_dir,   use_augment=False)

    # 2. Scale
    print("\n⚙️  StandardScaler...")
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_vl = scaler.transform(X_val)

    # 3. Train SVM + calibration
    print("\n🤖 Huấn luyện SVM (RBF, C=15, calibrated)...")
    base = SVC(kernel='rbf', C=15.0, gamma='scale',
               class_weight='balanced', probability=True,
               random_state=42)
    model = CalibratedClassifierCV(base, cv=3, method='isotonic')
    model.fit(X_tr, y_train)

    # 4. Đánh giá
    y_pred = model.predict(X_vl)
    acc = accuracy_score(y_val, y_pred)
    print(f"\n✅ Accuracy: {acc*100:.2f}%\n")
    print(classification_report(y_val, y_pred, target_names=CLASSES))

    # 5. Confusion matrix
    output_dir = os.path.join(project_root, 'outputs', 'svm_results')
    os.makedirs(output_dir, exist_ok=True)
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(max(8, len(CLASSES)), max(6, len(CLASSES) - 1)))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=CLASSES, yticklabels=CLASSES, cmap='YlOrRd')
    plt.title(f'SVM v2 — Acc: {acc*100:.2f}%')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'svm_v2_confusion_matrix.png'), dpi=120)
    print(f"📈 Confusion matrix → outputs/svm_results/svm_v2_confusion_matrix.png")

    # 6. Lưu model
    model_path = os.path.join(output_dir, 'svm_v2_model.pkl')
    joblib.dump({
        'model':        model,
        'scaler':       scaler,
        'img_size':     IMG_SIZE,
        'feature_type': 'hsv+hog+lbp_v2',
        'classes':      CLASSES,
        'use_grabcut_at_predict': True,
    }, model_path)
    print(f"💾 Model → {model_path}")


if __name__ == "__main__":
    train_svm()
