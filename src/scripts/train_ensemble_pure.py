import os
import sys
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import joblib
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import concurrent.futures

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from src.core.config import CLASSES
from src.scripts.feature_utils import extract_features, augment

IMG_SIZE = (64, 64)
CONFIDENCE_THRESHOLD = 0.55

def _process_single_image(file_path, class_idx, use_augment):
    X_local, y_local = [], []
    try:
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        for aug_img in (augment(img) if use_augment else [img]):
            X_local.append(extract_features(aug_img, use_grabcut=False))
            y_local.append(class_idx)
    except Exception:
        pass
    return X_local, y_local

def load_data(data_dir: str, use_augment: bool = False):
    X, y = [], []
    print(f"\n📂 Tải từ: {data_dir} (Ensemble V2 Đa luồng - Augment={use_augment})")
    for class_idx, class_name in enumerate(CLASSES):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.exists(class_path):
            continue
        
        files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        file_paths = [os.path.join(class_path, f) for f in files]
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(_process_single_image, fp, class_idx, use_augment) for fp in file_paths]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"  {class_name:15s}", unit="img"):
                rx, ry = future.result()
                X.extend(rx)
                y.extend(ry)
                    
    print(f"  → {len(X)} samples")
    return np.array(X, dtype=np.float32), np.array(y)

def train_ensemble():
    print("=" * 60)
    print("   ENSEMBLE V2 OPTIMIZED (HSV + HOG + LBP + ĐA LUỒNG)")
    print("=" * 60)
    
    train_dir = os.path.join(project_root, 'dataset', 'train')
    val_dir   = os.path.join(project_root, 'dataset', 'validation')

    # Bật Augment (nhân 3 data) cho tập train giống hệ V2
    X_train, y_train = load_data(train_dir, use_augment=True)
    X_val,   y_val   = load_data(val_dir, use_augment=False)

    print("\  Chuẩn hóa StandardScaler...")
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_vl = scaler.transform(X_val)

    # Base models tối ưu tốc độ
    knn = KNeighborsClassifier(n_neighbors=7, weights='distance', metric='cosine', n_jobs=-1)
    rf  = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
    # Lấy đúng SVM C=10 của bản V2
    svm = SVC(kernel='rbf', C=10.0, gamma='scale', class_weight='balanced', probability=True, cache_size=2000, random_state=42)

    voting = VotingClassifier(
        estimators=[('knn', knn), ('rf', rf), ('svm', svm)],
        voting='soft',
        weights=[1, 4, 2],  
        n_jobs=None
    )

    models = {
        'KNN (cosine, k=7)': knn, #nhược điểm là khi dữ liệu lớn thì dự đoán chậm và dễ bị ảnh hưởng bởi các mẫu nhiễu.
        'Random Forest (200 trees)': rf, #Khó giải thích chi tiết từng quyết định.
        'SVM (RBF, C=10)': svm, #dùng kernel='rbf' để học ranh giới phi tuyến .. Khó giải thích chi tiết từng quyết định.
        'Soft Voting Ensemble': voting
    }

    results = []
    best_acc = 0
    best_name = ""
    best_mdl = None

    print("\n🤖 Huấn luyện & đánh giá...\n")
    for name, mdl in models.items():
        print(f"  ▶ {name:30s}", end="  ", flush=True)
        mdl.fit(X_tr, y_train)
        y_pred = mdl.predict(X_vl)
        acc = accuracy_score(y_val, y_pred)
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

    # Lưu output
    output_dir = os.path.join(project_root, 'outputs', 'ensemble_results')
    os.makedirs(output_dir, exist_ok=True)

    y_pred_best = best_mdl.predict(X_vl)
    cm = confusion_matrix(y_val, y_pred_best)
    n = len(CLASSES)
    plt.figure(figsize=(max(8, n), max(6, n - 1)))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASSES, yticklabels=CLASSES, cmap='YlOrRd')
    plt.title(f'{best_name} — Acc: {best_acc*100:.2f}%')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ensemble_v2_confusion_matrix.png'), dpi=120)
    print(f"   → outputs/ensemble_results/ensemble_v2_confusion_matrix.png")

    # Biểu đồ thanh ngang so sánh độ chính xác các mô hình
    df_sorted = df.sort_values('Accuracy (%)', ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(df_sorted['Model'], df_sorted['Accuracy (%)'], color='steelblue', height=0.5)
    for bar, val in zip(bars, df_sorted['Accuracy (%)']):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                f'{val:.2f}%', va='center', fontsize=11, fontweight='bold')
    ax.set_xlabel('Accuracy (%)', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    ax.set_title('So sánh độ chính xác giữa các mô hình', fontsize=13)
    ax.set_xlim(0, 100)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    chart_path = os.path.join(output_dir, 'ensemble_comparison_chart.png')
    plt.savefig(chart_path, dpi=120)
    print(f"   → outputs/ensemble_results/ensemble_comparison_chart.png")

    # Lưu với metadata hsv+hog+lbp_v2
    model_path = os.path.join(output_dir, 'best_ensemble_model.pkl')
    joblib.dump({
    'model': best_mdl,
    'scaler': scaler,
    'img_size': IMG_SIZE,
    'feature_type': 'hsv+hog+lbp_v2',
    'classes': CLASSES,
    'model_name': best_name,
    'confidence_threshold': CONFIDENCE_THRESHOLD,
    'validation_accuracy': best_acc
}, model_path, compress=3)
    
    print(f"\n💾 Đã lưu Model vào → {model_path}")

if __name__ == "__main__":
    train_ensemble()
