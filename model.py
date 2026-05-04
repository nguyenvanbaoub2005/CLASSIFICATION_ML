# model.py
"""
File chứa kiến trúc model CNN cho phân loại rác thải
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from config import MODEL_CONFIG


def create_waste_classifier_model(input_shape=None, num_classes=None):
    """
    Tạo model CNN để phân loại rác thải
    
    Args:
        input_shape: Kích thước input (height, width, channels)
        num_classes: Số lượng classes cần phân loại
    
    Returns:
        model: Keras model đã compile
    """
    if input_shape is None:
        input_shape = MODEL_CONFIG['input_shape']
    if num_classes is None:
        num_classes = MODEL_CONFIG['num_classes']
    
    model = keras.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        
        # Block 4
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        
        # Flatten và Dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=MODEL_CONFIG['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_transfer_learning_model(base_model_name='MobileNetV2', input_shape=None, num_classes=None):
    """
    Tạo model sử dụng transfer learning
    
    Args:
        base_model_name: Tên model gốc ('MobileNetV2', 'VGG16', 'ResNet50')
        input_shape: Kích thước input
        num_classes: Số lượng classes
    
    Returns:
        model: Keras model với transfer learning
    """
    if input_shape is None:
        input_shape = MODEL_CONFIG['input_shape']
    if num_classes is None:
        num_classes = MODEL_CONFIG['num_classes']
    
    # Chọn base model
    if base_model_name == 'MobileNetV2':
        base_model = keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
    elif base_model_name == 'VGG16':
        base_model = keras.applications.VGG16(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
    elif base_model_name == 'ResNet50':
        base_model = keras.applications.ResNet50(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet'
        )
    else:
        raise ValueError(f"Không hỗ trợ base model: {base_model_name}")
    
    # Đóng băng các layer của base model
    base_model.trainable = False
    
    # Thêm custom layers
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=MODEL_CONFIG['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def get_model_summary(model):
    """In thông tin tóm tắt về model"""
    print("\n" + "="*70)
    print("📊 THÔNG TIN MODEL")
    print("="*70)
    model.summary()
    print("="*70 + "\n")
    
    total_params = model.count_params()
    print(f"Tổng số parameters: {total_params:,}")
    return total_params