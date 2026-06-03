import os
import sys

# Thêm thư mục gốc vào sys.path để có thể import module 'src'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
# pyrefly: ignore [missing-import]
import joblib
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Import cấu hình từ project
from src.core.config import CLASSES, PATHS

def extract_color_features(img):
    """
    Trích xuất đặc trưng: Màu sắc (HSV Histogram) + Hình khối (HOG)
    Đây là phương pháp nâng cao, rất mạnh mẽ của Machine Learning truyền thống.
    """
    # 1. Color Histogram (HSV)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    # Tính toán histogram
    h_hist = cv2.calcHist([hsv], [0], None, [64], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], None, [64], [0, 256])
    v_hist = cv2.calcHist([hsv], [2], None, [64], [0, 256])
    
    # Nối và chuẩn hóa
    color_features = np.concatenate([h_hist, s_hist, v_hist]).flatten()
    cv2.normalize(color_features, color_features)
    
    # 2. HOG Features (Trích xuất hình dáng, góc cạnh, chống chịu góc chụp xoay)
    img_resized = cv2.resize(img, (64, 64))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    
    hog = cv2.HOGDescriptor(
        _winSize=(64, 64),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9
    )
    hog_features = hog.compute(gray).flatten()
    
    return np.concatenate([color_features, hog_features])

import concurrent.futures

def _process_single_image(file_path, class_idx, img_size):
    """Hàm chạy độc lập trên từng luồng (Core) của CPU"""
    try:
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        features = extract_color_features(img)
        return features, class_idx
    except Exception:
        return None, None

def load_data_svm(data_dir, img_size=(64, 64)):
    X = []
    y = []
    
    print(f"\n Đang tải dữ liệu và trích xuất Color Features (ĐA LUỒNG) từ: {data_dir}")
    for class_idx, class_name in enumerate(CLASSES):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.exists(class_path): continue
            
        files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        file_paths = [os.path.join(class_path, f) for f in files]
        
        # Mở ProcessPoolExecutor để chạy song song
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(_process_single_image, fp, class_idx, img_size) for fp in file_paths]
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Processing {class_name:15s}", unit="img"):
                feat, label = future.result()
                if feat is not None:
                    X.append(feat)
                    y.append(label)
                    
    return np.array(X), np.array(y)

def train_svm_color():
    # 1. Load dữ liệu
    img_size = (64, 64)
    train_dir = os.path.join(project_root, 'dataset/train')
    val_dir = os.path.join(project_root, 'dataset/validation')
    
    X_train, y_train = load_data_svm(train_dir, img_size)
    X_val, y_val = load_data_svm(val_dir, img_size)
    
    # 2. Chuẩn hóa dữ liệu
    print("\n Đang chuẩn hóa dữ liệu...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # 3. Huấn luyện SVM
    print(" Đang huấn luyện SVM (Color Features - Mở rộng RAM 2000MB)...")
    # Thêm cache_size=2000 để chạy siêu tốc
    svm_model = SVC(kernel='rbf', C=15.0, gamma='scale', probability=True, cache_size=2000, verbose=True)
    svm_model.fit(X_train_scaled, y_train)
    
    # 4. Đánh giá
    print("\n ĐÁNH GIÁ MÔ HÌNH SVM MÀU SẮC")
    y_pred = svm_model.predict(X_val_scaled)
    acc = accuracy_score(y_val, y_pred)
    print(f" Accuracy: {acc*100:.2f}%")
    print(classification_report(y_val, y_pred, target_names=CLASSES))
    
    # 5. Lưu kết quả
    output_dir = "outputs/svm_results"
    os.makedirs(output_dir, exist_ok=True)
    
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASSES, yticklabels=CLASSES, cmap='YlGnBu')
    plt.title(f'SVM Color Model Confusion Matrix (Acc: {acc*100:.2f}%)')
    plt.savefig(os.path.join(output_dir, 'svm_color_confusion_matrix.png'))
    
    # 6. Lưu model
    model_path = os.path.join(output_dir, 'svm_model.pkl')
    joblib.dump({
        'model': svm_model, 
        'scaler': scaler, 
        'img_size': img_size,
        'feature_type': 'color_hist'
    }, model_path)
    print(f" Đã lưu model tại: {model_path}")

if __name__ == "__main__":
    train_svm_color()