import os
import sys

# Thêm thư mục gốc vào sys.path để có thể import module 'src'
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

# Import cấu hình từ project
from src.core.config import CLASSES, PATHS

def extract_color_features(img):
    """
    Trích xuất đặc trưng Màu sắc (HSV Histogram) + Cấu trúc (Small Pixels)
    Đây là phương pháp hiệu quả nhất cho SVM thuần trong bài toán này.
    """
    # 1. Chuyển sang không gian màu HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    # 2. Tính toán histogram chi tiết hơn (64 bins thay vì 32)
    h_hist = cv2.calcHist([hsv], [0], None, [64], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], None, [64], [0, 256])
    v_hist = cv2.calcHist([hsv], [2], None, [64], [0, 256])
    
    # Nối và chuẩn hóa
    color_features = np.concatenate([h_hist, s_hist, v_hist]).flatten()
    cv2.normalize(color_features, color_features)
    
    # 3. Thêm thông tin cấu trúc bằng ảnh siêu nhỏ (8x8 pixels)
    # Giúp SVM biết được phân bổ màu sắc theo vị trí
    small_img = cv2.resize(img, (8, 8)).flatten() / 255.0
    
    return np.concatenate([color_features, small_img])

def load_data_svm(data_dir, img_size=(64, 64)):
    X = []
    y = []
    
    print(f"📂 Đang tải dữ liệu và trích xuất Color Features từ: {data_dir}")
    for class_idx, class_name in enumerate(CLASSES):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.exists(class_path): continue
            
        files = os.listdir(class_path)
        for f in tqdm(files, desc=f"Processing {class_name}"):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    img = cv2.imread(os.path.join(class_path, f))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, img_size)
                    
                    features = extract_color_features(img)
                    X.append(features)
                    y.append(class_idx)
                except:
                    continue
                    
    return np.array(X), np.array(y)

def train_svm_color():
    # 1. Load dữ liệu
    img_size = (64, 64)
    train_dir = os.path.join(project_root, 'dataset/train')
    val_dir = os.path.join(project_root, 'dataset/validation')
    
    X_train, y_train = load_data_svm(train_dir, img_size)
    X_val, y_val = load_data_svm(val_dir, img_size)
    
    # 2. Chuẩn hóa dữ liệu
    print("⚙️ Đang chuẩn hóa dữ liệu...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # 3. Huấn luyện SVM
    print("🤖 Đang huấn luyện SVM (Color Features)...")
    # Tăng C để model học mạnh hơn
    svm_model = SVC(kernel='rbf', C=15.0, gamma='scale', probability=True, verbose=True)
    svm_model.fit(X_train_scaled, y_train)
    
    # 4. Đánh giá
    print("\n✅ ĐÁNH GIÁ MÔ HÌNH SVM MÀU SẮC")
    y_pred = svm_model.predict(X_val_scaled)
    acc = accuracy_score(y_val, y_pred)
    print(f"⭐ Accuracy: {acc*100:.2f}%")
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
    model_path = os.path.join(output_dir, 'svm_color_model.pkl')
    joblib.dump({
        'model': svm_model, 
        'scaler': scaler, 
        'img_size': img_size,
        'feature_type': 'color_hist'
    }, model_path)
    print(f"💾 Đã lưu model tại: {model_path}")

if __name__ == "__main__":
    train_svm_color()
