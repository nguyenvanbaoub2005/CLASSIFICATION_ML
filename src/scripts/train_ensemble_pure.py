"""
train_ensemble_pure_v2.py — Đặt vào: scripts/train_ensemble_pure_v2.py
Thay thế cho train_ensemble_pure.py
Yêu cầu: feature_utils.py phải nằm cùng thư mục scripts/
"""

import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import cv2
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

from src.core.config import CLASSES
from feature_utils import extract_features, augment   # cùng thư mục scripts/

IMG_SIZE = (128, 128)
CONFIDENCE_THRESHOLD = 0.55


def load_data(data_dir: str, use_augment: bool = False):
    X, y = [], []
    print(f"\n📂 Tải từ: {data_dir}  (augment={use_augment})")
    for class_idx, class_name in enumerate(CLASSES):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.exists(class_path):
            print(f"  ⚠️  Bỏ qua: {class_path}")
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
    print(f"  → {len(X)} samples")
    return np.array(X, dtype=np.float32), np.array(y)


def train_ensemble():
    print("""
    ============================================================
      🚀 ENSEMBLE V2 — HSV + HOG + LBP + Augmentation
    ============================================================
    """)
    train_dir = os.path.join(project_root, 'dataset', 'train')
    val_dir   = os.path.join(project_root, 'dataset', 'validation')

    X_train, y_train = load_data(train_dir, use_augment=True)
    X_val,   y_val   = load_data(val_dir,   use_augment=False)

    print("\n⚙️  StandardScaler...")
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_vl = scaler.transform(X_val)

    # Base models
    knn = KNeighborsClassifier(n_neighbors=7, weights='distance', metric='cosine')
    rf  = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    svm = SVC(kernel='rbf', C=15.0, gamma='scale',
              class_weight='balanced', probability=True, random_state=42)

    # Ensemble models
    voting = VotingClassifier(
        estimators=[('knn', knn), ('rf', rf), ('svm', svm)],
        voting='soft',
        weights=[1, 2, 3]   # SVM ưu tiên cao hơn
    )
    stacking = StackingClassifier(
        estimators=[('knn', knn), ('rf', rf), ('svm', svm)],
        final_estimator=LogisticRegression(C=1.0, max_iter=1000,
                                           class_weight='balanced'),
        passthrough=True    # meta-learner thấy cả features gốc
    )

    models = {
        'KNN (cosine, k=7)':         knn,
        'Random Forest (200 trees)': rf,
        'SVM (RBF, C=15)':           svm,
        'Soft Voting (w=1:2:3)':     voting,
        'Stacking + passthrough':    stacking,
    }

    results   = []
    best_acc  = 0
    best_name = ""
    best_mdl  = None

    print("\n🤖 Huấn luyện & đánh giá...\n")
    for name, mdl in models.items():
        print(f"  ▶ {name:32s}", end="  ", flush=True)
        mdl.fit(X_tr, y_train)
        y_pred = mdl.predict(X_vl)
        acc    = accuracy_score(y_val, y_pred)
        print(f"Accuracy: {acc*100:.2f}%")
        results.append({'Model': name, 'Accuracy (%)': round(acc * 100, 2)})
        if acc > best_acc:
            best_acc, best_name, best_mdl = acc, name, mdl

    # Bảng kết quả
    df = pd.DataFrame(results).sort_values('Accuracy (%)', ascending=False)
    print("\n📊 BẢNG SO SÁNH:")
    print("=" * 55)
    print(df.to_string(index=False))
    print("=" * 55)
    print(f"🏆 Model tốt nhất: {best_name}  ({best_acc*100:.2f}%)")

    # Confidence check
    if hasattr(best_mdl, 'predict_proba'):
        proba = best_mdl.predict_proba(X_vl)
        uncertain = (proba.max(axis=1) < CONFIDENCE_THRESHOLD).sum()
        print(f"\n🔍 Ảnh uncertain (conf < {CONFIDENCE_THRESHOLD}): "
              f"{uncertain}/{len(X_vl)} ({uncertain/len(X_vl)*100:.1f}%)")

    # Lưu output
    output_dir = os.path.join(project_root, 'outputs', 'ensemble_results')
    os.makedirs(output_dir, exist_ok=True)

    # Confusion matrix
    y_pred_best = best_mdl.predict(X_vl)
    cm = confusion_matrix(y_val, y_pred_best)
    n = len(CLASSES)
    plt.figure(figsize=(max(8, n), max(6, n - 1)))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=CLASSES, yticklabels=CLASSES, cmap='Blues')
    plt.title(f'{best_name} — Acc: {best_acc*100:.2f}%')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ensemble_v2_confusion_matrix.png'), dpi=120)

    # Bar chart so sánh models
    plt.figure(figsize=(10, 5))
    colors = ['#F44336' if n == df.iloc[0]['Model'] else '#90CAF9'
              for n in df['Model']]
    bars = plt.barh(df['Model'][::-1], df['Accuracy (%)'][::-1],
                    color=colors[::-1])
    plt.xlabel('Accuracy (%)')
    plt.title('So sánh Models — Ensemble v2')
    plt.xlim(0, 105)
    for bar, val in zip(bars, df['Accuracy (%)'][::-1]):
        plt.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                 f'{val:.1f}%', va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ensemble_v2_model_comparison.png'), dpi=120)

    # Lưu model
    model_path = os.path.join(output_dir, 'best_ensemble_v2_model.pkl')
    joblib.dump({
        'model':        best_mdl,
        'scaler':       scaler,
        'img_size':     IMG_SIZE,
        'feature_type': 'hsv+hog+lbp_v2',
        'classes':      CLASSES,
        'model_name':   best_name,
        'confidence_threshold': CONFIDENCE_THRESHOLD,
        'use_grabcut_at_predict': True,
    }, model_path)
    print(f"\n💾 Model → {model_path}")
    print(f"📈 Charts → {output_dir}/")


if __name__ == "__main__":
    train_ensemble()
