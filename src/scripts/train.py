# train.py
"""
File huấn luyện model phân loại rác thải
"""

import os
import matplotlib.pyplot as plt
from tensorflow import keras
from src.models.model import create_waste_classifier_model, create_transfer_learning_model, get_model_summary
from src.core.config import MODEL_CONFIG, PATHS, AUGMENTATION_CONFIG


def create_data_generators(train_dir, val_dir, batch_size=None):
    """
    Tạo data generators cho training và validation
    
    Args:
        train_dir: Đường dẫn thư mục training data
        val_dir: Đường dẫn thư mục validation data
        batch_size: Kích thước batch
    
    Returns:
        train_generator, val_generator
    """
    if batch_size is None:
        batch_size = MODEL_CONFIG['batch_size']
    
    # Data Augmentation cho training
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
    
    # Chỉ rescale cho validation
    val_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=MODEL_CONFIG['input_shape'][:2],
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=MODEL_CONFIG['input_shape'][:2],
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator


def get_callbacks():
    """Tạo callbacks cho training"""
    callbacks = [
        # Lưu model tốt nhất
        keras.callbacks.ModelCheckpoint(
            PATHS['best_model'],
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        
        # Early stopping
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Giảm learning rate khi plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        
        # TensorBoard logging
        keras.callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1,
            write_graph=True
        )
    ]
    
    return callbacks


def train_model(train_dir, val_dir, epochs=None, batch_size=None, use_transfer_learning=False):
    """
    Huấn luyện model
    
    Args:
        train_dir: Đường dẫn thư mục training
        val_dir: Đường dẫn thư mục validation
        epochs: Số epochs
        batch_size: Kích thước batch
        use_transfer_learning: Sử dụng transfer learning hay không
    
    Returns:
        model, history
    """
    if epochs is None:
        epochs = MODEL_CONFIG['epochs']
    if batch_size is None:
        batch_size = MODEL_CONFIG['batch_size']
    
    print("\n🚀 BẮT ĐẦU HUẤN LUYỆN MODEL")
    print("="*70)
    
    # Tạo data generators
    print("\n📁 Đang tải dữ liệu...")
    train_generator, val_generator = create_data_generators(train_dir, val_dir, batch_size)
    
    print(f"\n✓ Số lượng ảnh training: {train_generator.samples}")
    print(f"✓ Số lượng ảnh validation: {val_generator.samples}")
    print(f"✓ Số classes: {len(train_generator.class_indices)}")
    print(f"✓ Classes: {list(train_generator.class_indices.keys())}")
    
    # Tạo model
    print("\n🏗️  Đang xây dựng model...")
    if use_transfer_learning:
        print("   Sử dụng Transfer Learning (MobileNetV2)")
        model = create_transfer_learning_model('MobileNetV2')
    else:
        print("   Sử dụng CNN từ đầu")
        model = create_waste_classifier_model()
    
    get_model_summary(model)
    
    # Callbacks
    callbacks = get_callbacks()
    
    # Training
    print("\n🎯 Bắt đầu training...")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Steps per epoch: {train_generator.samples // batch_size}")
    print("="*70 + "\n")
    
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Lưu model cuối cùng
    model.save(PATHS['model_save'])
    print(f"\n✅ Model đã được lưu tại: {PATHS['model_save']}")
    
    return model, history


def plot_training_history(history):
    """
    Vẽ biểu đồ quá trình training
    
    Args:
        history: History object từ model.fit()
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train', linewidth=2)
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train', linewidth=2)
    axes[0, 1].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Accuracy comparison
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    axes[1, 0].bar(['Training', 'Validation'], [final_train_acc, final_val_acc], 
                   color=['#2ecc71', '#3498db'])
    axes[1, 0].set_title('Final Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_ylim([0, 1])
    for i, v in enumerate([final_train_acc, final_val_acc]):
        axes[1, 0].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
    
    # Loss comparison
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    axes[1, 1].bar(['Training', 'Validation'], [final_train_loss, final_val_loss],
                   color=['#e74c3c', '#f39c12'])
    axes[1, 1].set_title('Final Loss Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Loss')
    for i, v in enumerate([final_train_loss, final_val_loss]):
        axes[1, 1].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(PATHS['training_plot'], dpi=300, bbox_inches='tight')
    print(f"📊 Biểu đồ training đã lưu tại: {PATHS['training_plot']}")
    plt.show()


def main():
    """Main function cho training"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║           🎓 HUẤN LUYỆN MODEL PHÂN LOẠI RÁC THẢI         ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Nhập thông tin
    train_dir = input("📁 Đường dẫn thư mục training: ").strip()
    val_dir = input("📁 Đường dẫn thư mục validation: ").strip()
    
    if not os.path.exists(train_dir):
        print(f"❌ Không tìm thấy thư mục: {train_dir}")
        return
    
    if not os.path.exists(val_dir):
        print(f"❌ Không tìm thấy thư mục: {val_dir}")
        return
    
    use_transfer = input("\n🔄 Sử dụng Transfer Learning? (y/n): ").strip().lower() == 'y'
    
    epochs_input = input(f"⏱️  Số epochs (mặc định {MODEL_CONFIG['epochs']}): ").strip()
    epochs = int(epochs_input) if epochs_input else MODEL_CONFIG['epochs']
    
    # Training
    try:
        model, history = train_model(
            train_dir, 
            val_dir, 
            epochs=epochs,
            use_transfer_learning=use_transfer
        )
        
        # Vẽ biểu đồ
        plot_training_history(history)
        
        print("\n" + "="*70)
        print("✅ HUẤN LUYỆN HOÀN TẤT!")
        print("="*70)
        print(f"Model cuối cùng: {PATHS['model_save']}")
        print(f"Model tốt nhất: {PATHS['best_model']}")
        print(f"Biểu đồ training: {PATHS['training_plot']}")
        
    except Exception as e:
        print(f"\n❌ Lỗi trong quá trình training: {str(e)}")


if __name__ == "__main__":
    main()