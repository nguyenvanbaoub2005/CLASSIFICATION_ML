# master_pipeline.py
"""
Master ML Pipeline: Tự động chạy toàn bộ quy trình EDA, Training 3 mô hình, và Đánh giá so sánh.
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

from src.core.config import MODEL_CONFIG, PATHS, AUGMENTATION_CONFIG, CLASSES, CLASS_INFO
from src.models.model import create_waste_classifier_model, create_transfer_learning_model, get_model_summary

# Tạo các thư mục output
OUTPUT_DIR = "outputs"
EDA_DIR = os.path.join(OUTPUT_DIR, "eda")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")

for d in [EDA_DIR, MODELS_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ==========================================
# PHASE 1: EDA & Phân tích trực quan
# ==========================================
def run_eda(train_dir):
    print("\n" + "="*70)
    print("📊 PHASE 1: EDA & PHÂN TÍCH DỮ LIỆU")
    print("="*70)
    
    if not os.path.exists(train_dir):
        print(f"❌ Không tìm thấy dữ liệu tại {train_dir}")
        return False
        
    counts = {}
    sample_images = {}
    
    for cls in CLASSES:
        cls_dir = os.path.join(train_dir, cls)
        if os.path.exists(cls_dir):
            images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            counts[cls] = len(images)
            if len(images) > 0:
                sample_images[cls] = os.path.join(cls_dir, random.choice(images))
        else:
            counts[cls] = 0
            
    # 1. Vẽ Bar Chart phân bố dữ liệu
    plt.figure(figsize=(10, 6))
    colors = [CLASS_INFO[cls]['color'].replace('\033[', '').replace('m', '') for cls in CLASSES]
    # Fallback to standard matplotlib colors if terminal ANSI codes were used
    safe_colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'gray']
    
    sns.barplot(x=list(counts.keys()), y=list(counts.values()), palette=safe_colors[:len(CLASSES)])
    plt.title("Phân bố số lượng ảnh trong tập Training", fontsize=15, fontweight='bold')
    plt.ylabel("Số lượng ảnh")
    plt.xlabel("Loại rác")
    plt.xticks(rotation=45)
    
    for i, v in enumerate(counts.values()):
        plt.text(i, v + 5, str(v), ha='center', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "class_distribution.png"), dpi=300)
    plt.close()
    
    # 2. Vẽ Grid Image (3x3)
    if len(sample_images) > 0:
        plt.figure(figsize=(12, 10))
        plt.suptitle("Một số hình ảnh mẫu từ Dataset", fontsize=16, fontweight='bold')
        
        for i, (cls, img_path) in enumerate(list(sample_images.items())[:9]):
            plt.subplot(3, 3, i+1)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.imshow(img)
                plt.title(f"{CLASS_INFO[cls]['name_vi']} ({cls})")
                plt.axis('off')
                
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_DIR, "sample_grid.png"), dpi=300)
        plt.close()
        
    print("✅ Đã tạo biểu đồ EDA thành công tại thư mục 'outputs/eda'")
    return counts


# ==========================================
# PHASE 2: Tiền xử lý & Cân bằng dữ liệu
# ==========================================
def create_generators_and_weights(train_dir, val_dir):
    print("\n" + "="*70)
    print("⚙️ PHASE 2: TIỀN XỬ LÝ & TÍNH CLASS WEIGHTS")
    print("="*70)
    
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=AUGMENTATION_CONFIG['rotation_range'],
        width_shift_range=AUGMENTATION_CONFIG['width_shift_range'],
        height_shift_range=AUGMENTATION_CONFIG['height_shift_range'],
        horizontal_flip=AUGMENTATION_CONFIG['horizontal_flip'],
        zoom_range=AUGMENTATION_CONFIG['zoom_range'],
        shear_range=AUGMENTATION_CONFIG['shear_range'],
        fill_mode=AUGMENTATION_CONFIG['fill_mode']
    )
    
    val_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=MODEL_CONFIG['input_shape'][:2],
        batch_size=MODEL_CONFIG['batch_size'],
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=MODEL_CONFIG['input_shape'][:2],
        batch_size=MODEL_CONFIG['batch_size'],
        class_mode='categorical',
        shuffle=False
    )
    
    # Tính Class Weights để cân bằng dữ liệu
    labels = train_generator.classes
    classes = np.unique(labels)
    class_weights_array = compute_class_weight('balanced', classes=classes, y=labels)
    class_weights = {i: weight for i, weight in enumerate(class_weights_array)}
    
    print(f"✅ Đã tải {train_generator.samples} ảnh train và {val_generator.samples} ảnh val.")
    print("⚖️ Trọng số tự động (Class Weights):")
    for i, w in class_weights.items():
        print(f"   - {list(train_generator.class_indices.keys())[i]}: {w:.2f}")
        
    return train_generator, val_generator, class_weights


# ==========================================
# PHASE 3: Training 3 Mô hình
# ==========================================
def get_callbacks(model_name):
    model_path = os.path.join(MODELS_DIR, f"{model_name}_best.h5")
    return [
        keras.callbacks.ModelCheckpoint(model_path, save_best_only=True, monitor='val_accuracy', mode='max', verbose=0),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=0),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=0)
    ]

def train_multiple_models(train_gen, val_gen, class_weights, epochs):
    print("\n" + "="*70)
    print("🧠 PHASE 3: HUẤN LUYỆN 3 MÔ HÌNH")
    print("="*70)
    
    # Định nghĩa 3 mô hình
    models_dict = {
        'Custom_CNN': create_waste_classifier_model(),
        'MobileNetV2': create_transfer_learning_model('MobileNetV2'),
        'ResNet50': create_transfer_learning_model('ResNet50')
    }
    
    histories = {}
    trained_models = {}
    
    for name, model in models_dict.items():
        print(f"\n🚀 Đang huấn luyện: {name}")
        print("-" * 30)
        
        callbacks = get_callbacks(name)
        history = model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        histories[name] = history
        trained_models[name] = model
        
        # Lưu model cuối cùng
        model.save(os.path.join(MODELS_DIR, f"{name}_final.h5"))
        print(f"✅ Đã hoàn thành {name}")
        
    return trained_models, histories


# ==========================================
# PHASE 4: Đánh giá & So sánh
# ==========================================
def evaluate_and_compare(trained_models, histories, val_gen):
    print("\n" + "="*70)
    print("🏆 PHASE 4: ĐÁNH GIÁ & SO SÁNH KẾT QUẢ")
    print("="*70)
    
    model_names = list(trained_models.keys())
    accuracies = []
    losses = []
    reports_data = []
    
    class_labels = list(val_gen.class_indices.keys())
    
    for name, model in trained_models.items():
        print(f"🔍 Đang đánh giá {name}...")
        val_gen.reset()
        
        # Lấy loss/acc cơ bản
        loss, acc = model.evaluate(val_gen, verbose=0)
        accuracies.append(acc)
        losses.append(loss)
        
        # Lấy dự đoán chi tiết
        val_gen.reset()
        preds = model.predict(val_gen, verbose=0)
        y_pred = np.argmax(preds, axis=1)
        y_true = val_gen.classes
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
        plt.title(f"Confusion Matrix - {name}")
        plt.ylabel('Thực tế')
        plt.xlabel('Dự đoán')
        plt.tight_layout()
        plt.savefig(os.path.join(REPORTS_DIR, f"cm_{name}.png"), dpi=300)
        plt.close()
        
        # 2. Classification Report
        rep = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
        rep_df = pd.DataFrame(rep).transpose()
        rep_df.to_csv(os.path.join(REPORTS_DIR, f"report_{name}.csv"))
        
        reports_data.append({
            'Model': name,
            'Accuracy': f"{acc*100:.2f}%",
            'F1-Score (Macro)': f"{rep['macro avg']['f1-score']:.4f}",
            'Loss': f"{loss:.4f}"
        })
        
    # 3. Vẽ biểu đồ so sánh Accuracy 3 mô hình
    plt.figure(figsize=(10, 5))
    sns.barplot(x=model_names, y=accuracies, palette="viridis")
    plt.title("So sánh Accuracy giữa các Model", fontweight='bold')
    plt.ylim(0, 1.1)
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.02, f"{v*100:.2f}%", ha='center', fontweight='bold')
    plt.savefig(os.path.join(REPORTS_DIR, "accuracy_comparison.png"), dpi=300)
    plt.close()
    
    # 4. Lưu bảng tổng hợp ra CSV
    final_report_df = pd.DataFrame(reports_data)
    final_report_df.to_csv(os.path.join(REPORTS_DIR, "final_comparison_report.csv"), index=False)
    
    print("\n✅ TỔNG HỢP KẾT QUẢ:")
    print(final_report_df.to_string(index=False))
    print(f"\n📂 Đã xuất toàn bộ biểu đồ, ma trận và báo cáo ra thư mục: {OUTPUT_DIR}")


# ==========================================
# HÀM CHÍNH (MAIN)
# ==========================================
def main():
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║            🤖 MASTER ML PIPELINE - TỰ ĐỘNG HÓA            ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    train_dir = input("📁 Đường dẫn thư mục Training  (Ví dụ: dataset/train): ").strip()
    val_dir   = input("📁 Đường dẫn thư mục Validation (Ví dụ: dataset/validation): ").strip()
    epochs    = input("⏱️  Số epochs chạy cho mỗi Model (Mặc định=15): ").strip()
    
    epochs = int(epochs) if epochs else 15
    
    # Phase 1
    counts = run_eda(train_dir)
    if not counts: return
    
    # Phase 2
    train_gen, val_gen, class_weights = create_generators_and_weights(train_dir, val_dir)
    
    # Phase 3
    trained_models, histories = train_multiple_models(train_gen, val_gen, class_weights, epochs)
    
    # Phase 4
    evaluate_and_compare(trained_models, histories, val_gen)
    
    print("\n🎉 XUẤT SẮC! QUY TRÌNH ML ĐÃ HOÀN TẤT!")


if __name__ == "__main__":
    main()
