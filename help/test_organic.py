# test_organic.py
"""
Script kiểm tra model có phân loại organic tốt không
"""

import os
import numpy as np
from PIL import Image
from tensorflow import keras
from config import CLASSES, CLASS_INFO, PATHS

def test_model_prediction_distribution():
    """Kiểm tra phân bố prediction của model"""
    print("\n" + "="*70)
    print("🧪 KIỂM TRA KHẢ NĂNG PHÂN LOẠI CỦA MODEL")
    print("="*70)
    
    # Load model
    model_path = PATHS['model_save']
    if not os.path.exists(model_path):
        model_path = PATHS['best_model']
    
    try:
        model = keras.models.load_model(model_path)
        print(f"\n✅ Đã load model: {model_path}")
    except Exception as e:
        print(f"\n❌ Không thể load model: {e}")
        return
    
    # Tạo ảnh test ngẫu nhiên
    print("\n🔬 Test với 100 ảnh ngẫu nhiên...")
    
    predictions_count = {cls: 0 for cls in CLASSES}
    num_tests = 100
    
    for _ in range(num_tests):
        # Tạo ảnh random
        random_img = np.random.rand(1, 224, 224, 3)
        
        # Predict
        pred = model.predict(random_img, verbose=0)
        predicted_idx = np.argmax(pred[0])
        predicted_class = CLASSES[predicted_idx]
        
        predictions_count[predicted_class] += 1
    
    # Hiển thị kết quả
    print("\n📊 Phân bố predictions trên ảnh ngẫu nhiên:")
    print("-" * 70)
    
    for cls in CLASSES:
        count = predictions_count[cls]
        percentage = (count / num_tests) * 100
        bar = "█" * int(percentage / 2)
        icon = CLASS_INFO[cls]['icon']
        
        print(f"{icon} {cls:12s}: {bar:50s} {count:3d} ({percentage:5.1f}%)")
    
    # Phân tích
    print("\n" + "="*70)
    print("📈 PHÂN TÍCH:")
    print("="*70)
    
    organic_count = predictions_count['organic']
    
    if organic_count == 0:
        print("\n❌ CẢNH BÁO: Model KHÔNG BAO GIỜ predict class 'organic'!")
        print("\n   Nguyên nhân có thể:")
        print("   • Model được train trước khi thêm organic")
        print("   • Dataset train thiếu hoặc không có ảnh organic")
        print("   • Class organic bị lỗi trong quá trình train")
        print("\n   ⚠️  CẦN RETRAIN với dataset có đủ ảnh organic!")
        return False
    
    elif organic_count < 5:
        print("\n⚠️  CẢNH BÁO: Model predict organic RẤT ÍT!")
        print(f"   Chỉ {organic_count}/{num_tests} lần ({organic_count}%)")
        print("\n   Model có thể bias về các class khác.")
        print("   Khuyến nghị: Retrain với nhiều ảnh organic hơn")
        return False
    
    else:
        print("\n✅ Model có khả năng predict organic!")
        print(f"   Tỷ lệ: {organic_count}/{num_tests} lần ({organic_count}%)")
        
        # Check balance
        max_count = max(predictions_count.values())
        min_count = min(predictions_count.values())
        ratio = max_count / min_count if min_count > 0 else float('inf')
        
        if ratio > 5:
            print(f"\n⚠️  Cảnh báo: Model bị IMBALANCE!")
            print(f"   Tỷ lệ max/min: {ratio:.1f}x")
            print("   Class nào đó bị predict quá nhiều.")
            return False
        else:
            print(f"\n✅ Model tương đối cân bằng (ratio: {ratio:.1f}x)")
            return True


def suggest_next_steps(is_ok):
    """Gợi ý bước tiếp theo"""
    print("\n" + "="*70)
    print("🎯 BƯỚC TIẾP THEO")
    print("="*70)
    
    if is_ok:
        print("""
✅ Model đang hoạt động tốt!

Bạn có thể:
1. Test với ảnh organic thật:
   python classifier.py
   → Nhập ảnh: rau củ, vỏ trái cây, thức ăn thừa

2. Chạy GUI để test real-time:
   python gui_app.py

3. Nếu accuracy trên ảnh thật thấp, có thể cần:
   • Thu thập thêm ảnh organic
   • Chạy incremental training (fine-tune)
""")
    
    else:
        print("""
❌ Model CÓ VẤN ĐỀ với class organic!

GIẢI PHÁP:

🔄 Option 1: RETRAIN hoàn toàn (Khuyến nghị)
───────────────────────────────────────────
   • Chuẩn bị dataset đủ 7 classes
   • Mỗi class ≥ 500 ảnh (train) + 100 ảnh (val)
   • Chạy: python train.py
   • Thời gian: 30-60 phút

📦 Option 2: Download dataset có sẵn
───────────────────────────────────────────
   Dataset Kaggle có organic:
   • "RealWaste" dataset
   • "Organic Waste Classification"
   • "Food Waste Recognition"

🎯 Option 3: Fine-tune (nếu đã có ít ảnh)
───────────────────────────────────────────
   • Scan ≥20 ảnh organic qua GUI
   • Chọn "🔄 Fine-tune" trong menu
   • Model sẽ học thêm từ ảnh mới

💡 TẠM THỜI: Có thể dùng model hiện tại
───────────────────────────────────────────
   Nhưng nó sẽ classify sai organic thành trash/paper/...
""")


def main():
    """Main function"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║           🧪 KIỂM TRA CLASS ORGANIC                       ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    is_ok = test_model_prediction_distribution()
    suggest_next_steps(is_ok)
    
    print("\n" + "="*70)
    print("Hoàn tất kiểm tra!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()