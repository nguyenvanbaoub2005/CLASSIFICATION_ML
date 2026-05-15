# classifier.py
"""
Class WasteClassifier để phân loại rác thải
"""

import os
import cv2
import numpy as np
from PIL import Image
from tensorflow import keras
import joblib
from src.models.model import create_waste_classifier_model
from src.core.config import CLASSES, CLASS_INFO, MODEL_CONFIG, PATHS, CONFIDENCE_THRESHOLD, COLORS


class WasteClassifier:
    """Class chính để phân loại rác thải"""
    
    def __init__(self, model_path=None):
        """
        Khởi tạo classifier
        
        Args:
            model_path: Đường dẫn đến file model (.h5)
        """
        self.classes = CLASSES
        self.class_info = CLASS_INFO
        self.confidence_threshold = CONFIDENCE_THRESHOLD
        
        # Tự động tìm model nếu không chỉ định path
        if model_path is None:
            if os.path.exists(PATHS['model_save']):
                model_path = PATHS['model_save']
            elif os.path.exists(PATHS['best_model']):
                model_path = PATHS['best_model']
        
        # Load model
        self.model_type = 'cnn' # Mặc định là CNN
        if model_path and os.path.exists(model_path):
            print(f"📂 Đang load model từ: {model_path}")
            if model_path.endswith('.pkl'):
                self.model_data = joblib.load(model_path)
                self.model = self.model_data['model']
                self.scaler = self.model_data['scaler']
                self.model_type = 'svm'
                print("✅ Đã load model SVM (.pkl) thành công!")
            else:
                self.model = keras.models.load_model(model_path)
                self.model_type = 'cnn'
                print("✅ Đã load model CNN (.h5) thành công!")
        else:
            print("⚠️  Model chưa được huấn luyện - Sử dụng model mới")
            self.model = create_waste_classifier_model()
    
    def preprocess_image(self, image_path):
        """
        Tiền xử lý ảnh trước khi đưa vào model
        
        Args:
            image_path: Đường dẫn hoặc ảnh PIL/numpy array
        
        Returns:
            img_array: Numpy array đã được xử lý
        """
        # Xử lý nhiều loại input
        if isinstance(image_path, str):
            img = Image.open(image_path)
        elif isinstance(image_path, Image.Image):
            img = image_path
        elif isinstance(image_path, np.ndarray):
            img = Image.fromarray(image_path)
        else:
            raise ValueError("Input phải là đường dẫn, PIL Image, hoặc numpy array")
        
        # Chuyển sang RGB nếu cần
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize
        img = img.resize(MODEL_CONFIG['input_shape'][:2])
        
        # Convert sang array và normalize
        img_array = np.array(img) / 255.0
        
        # Thêm batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def extract_svm_features(self, img_pil):
        """Trích xuất đặc trưng cho SVM (Color Hist 64 bins + 8x8 Pixels)"""
        img = np.array(img_pil)
        # 1. Color Histogram (HSV) - 64 bins để tăng độ chính xác
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h_hist = cv2.calcHist([hsv], [0], None, [64], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [64], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [64], [0, 256])
        hist_features = np.concatenate([h_hist, s_hist, v_hist]).flatten()
        cv2.normalize(hist_features, hist_features)
        
        # 2. Small Pixels (8x8)
        small_img = cv2.resize(img, (8, 8)).flatten() / 255.0
        
        features = np.concatenate([hist_features, small_img])
        return self.scaler.transform([features])

    def predict(self, image_path, return_all=True):
        """
        Dự đoán loại rác từ ảnh
        
        Args:
            image_path: Đường dẫn đến ảnh
            return_all: Trả về tất cả predictions hay không
        
        Returns:
            dict: Kết quả dự đoán
        """
        # Xử lý input ảnh
        if isinstance(image_path, str):
            img_pil = Image.open(image_path).convert('RGB')
        elif isinstance(image_path, Image.Image):
            img_pil = image_path.convert('RGB')
        elif isinstance(image_path, np.ndarray):
            img_pil = Image.fromarray(image_path).convert('RGB')
            
        if self.model_type == 'svm':
            # Dự đoán bằng SVM
            features = self.extract_svm_features(img_pil)
            predictions = self.model.predict_proba(features)
        else:
            # Dự đoán bằng CNN
            img_array = self.preprocess_image(img_pil)
            predictions = self.model.predict(img_array, verbose=0)
        
        # Lấy class có xác suất cao nhất
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx] * 100)
        predicted_class = self.classes[class_idx]
        
        result = {
            'class': predicted_class,
            'class_name_vi': self.class_info[predicted_class]['name_vi'],
            'confidence': confidence,
            'is_confident': bool(confidence >= self.confidence_threshold)
        }
        
        if return_all:
            result['all_predictions'] = {
                self.classes[i]: float(predictions[0][i] * 100) 
                for i in range(len(self.classes))
            }
        
        return result
    
    def display_result(self, image_path, result):
        """
        Hiển thị kết quả phân loại đẹp mắt
        
        Args:
            image_path: Đường dẫn ảnh
            result: Kết quả từ hàm predict()
        """
        predicted_class = result['class']
        confidence = result['confidence']
        info = self.class_info[predicted_class]
        
        reset = COLORS['reset']
        color = info['color']
        
        print("\n" + "="*70)
        print(f"{color}{'🎯 KẾT QUẢ PHÂN LOẠI':^70}{reset}")
        print("="*70)
        
        print(f"\n{info['icon']}  Loại rác: {color}{info['name_vi'].upper()}{reset}")
        print(f"{'':3}Class: {predicted_class}")
        
        # Confidence với màu sắc
        conf_color = COLORS['green'] if result['is_confident'] else COLORS['yellow']
        print(f"{'':3}Độ tin cậy: {conf_color}{confidence:.2f}%{reset}", end="")
        
        if not result['is_confident']:
            print(f" {COLORS['yellow']}⚠️  (Thấp){reset}")
        else:
            print()
        
        print(f"\n📌 Cách xử lý:")
        print(f"{'':3}{info['disposal']}")
        
        print(f"\n📝 Ví dụ:")
        print(f"{'':3}{', '.join(info['examples'])}")
        
        print(f"\n♻️  Giá trị tái chế: {info['recycling_value']}")
        
        # Hiển thị tất cả xác suất
        if 'all_predictions' in result:
            print(f"\n📊 Chi tiết các xác suất:")
            sorted_preds = sorted(
                result['all_predictions'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            for cls, prob in sorted_preds:
                bar_length = int(prob / 3)
                bar = "█" * bar_length
                icon = self.class_info[cls]['icon']
                print(f"   {icon} {cls:12s}: {bar:33s} {prob:6.2f}%")
        
        print("\n" + "="*70 + "\n")
    
    def predict_batch(self, image_paths):
        """
        Dự đoán nhiều ảnh cùng lúc
        
        Args:
            image_paths: List các đường dẫn ảnh
        
        Returns:
            list: Danh sách kết quả
        """
        results = []
        for img_path in image_paths:
            try:
                result = self.predict(img_path)
                results.append({
                    'image': img_path,
                    'result': result
                })
            except Exception as e:
                print(f"❌ Lỗi xử lý {img_path}: {str(e)}")
        
        return results
    
    def get_statistics(self, results):
        """
        Thống kê kết quả phân loại
        
        Args:
            results: Danh sách kết quả từ predict_batch
        
        Returns:
            dict: Thống kê
        """
        stats = {cls: 0 for cls in self.classes}
        total = len(results)
        
        for item in results:
            predicted_class = item['result']['class']
            stats[predicted_class] += 1
        
        print("\n📊 THỐNG KÊ PHÂN LOẠI")
        print("="*50)
        for cls in self.classes:
            count = stats[cls]
            percentage = (count / total * 100) if total > 0 else 0
            icon = self.class_info[cls]['icon']
            name = self.class_info[cls]['name_vi']
            print(f"{icon} {name:20s}: {count:3d} ({percentage:5.1f}%)")
        print("="*50)
        print(f"Tổng số ảnh: {total}")
        
        return stats


def test_classifier():
    """Function test classifier"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║              🧪 TEST WASTE CLASSIFIER                     ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Khởi tạo
    model_path = PATHS['model_save']
    if not os.path.exists(model_path):
        model_path = PATHS['best_model']
    
    classifier = WasteClassifier(model_path)
    
    # Test với ảnh
    image_path = input("\n📷 Nhập đường dẫn ảnh để test: ").strip()
    
    if os.path.exists(image_path):
        print("\n🔍 Đang phân tích...")
        result = classifier.predict(image_path)
        classifier.display_result(image_path, result)
    else:
        print("❌ File không tồn tại!")


if __name__ == "__main__":
    test_classifier()