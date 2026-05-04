# verify_config.py
"""
Script kiểm tra config đã đúng và hướng dẫn retrain
"""

from config import CLASSES, CLASS_INFO, MODEL_CONFIG
import os

def verify_config():
    """Kiểm tra config đã đúng chưa"""
    print("\n" + "="*70)
    print("🔍 KIỂM TRA CẤU HÌNH")
    print("="*70)
    
    # Kiểm tra số lượng
    classes_count = len(CLASSES)
    class_info_count = len(CLASS_INFO)
    
    print(f"\n📊 Thống kê:")
    print(f"   CLASSES list: {classes_count} classes")
    print(f"   CLASS_INFO dict: {class_info_count} classes")
    print(f"   MODEL_CONFIG num_classes: {MODEL_CONFIG['num_classes']}")
    
    # Kiểm tra match
    if classes_count == class_info_count == MODEL_CONFIG['num_classes']:
        print(f"\n✅ PASS: Tất cả đều có {classes_count} classes")
    else:
        print(f"\n❌ FAIL: Số lượng không khớp!")
        return False
    
    # Kiểm tra từng class
    print(f"\n📋 Chi tiết từng class:")
    all_ok = True
    for i, cls in enumerate(CLASSES):
        if cls in CLASS_INFO:
            info = CLASS_INFO[cls]
            print(f"   [{i}] {info['icon']} {cls:12s} → {info['name_vi']:25s} ✅")
        else:
            print(f"   [{i}] ❌ {cls:12s} → THIẾU TRONG CLASS_INFO!")
            all_ok = False
    
    # Kiểm tra ngược lại
    print(f"\n🔄 Kiểm tra ngược CLASS_INFO:")
    for cls in CLASS_INFO.keys():
        if cls not in CLASSES:
            print(f"   ⚠️  '{cls}' có trong CLASS_INFO nhưng KHÔNG có trong CLASSES list")
            all_ok = False
    
    if all_ok:
        print(f"\n✅ CẤU HÌNH HOÀN TOÀN ĐÚNG!")
    else:
        print(f"\n❌ VẪN CÒN LỖI TRONG CẤU HÌNH!")
    
    print("="*70)
    return all_ok


def check_model_status():
    """Kiểm tra trạng thái model hiện tại"""
    print("\n" + "="*70)
    print("🤖 KIỂM TRA MODEL HIỆN TẠI")
    print("="*70)
    
    model_files = [
        ('waste_classifier_final.h5', 'Model chính'),
        ('waste_classifier_best.h5', 'Model tốt nhất'),
    ]
    
    need_retrain = False
    
    for filename, description in model_files:
        if os.path.exists(filename):
            size_mb = os.path.getsize(filename) / (1024 * 1024)
            print(f"\n✅ {description}: {filename}")
            print(f"   Kích thước: {size_mb:.2f} MB")
            
            # Kiểm tra số classes trong model
            try:
                from tensorflow import keras
                model = keras.models.load_model(filename)
                output_shape = model.output_shape
                num_classes_in_model = output_shape[-1]
                
                print(f"   Số classes trong model: {num_classes_in_model}")
                
                if num_classes_in_model != len(CLASSES):
                    print(f"   ⚠️  BẤT KHỚP! Config có {len(CLASSES)} classes")
                    need_retrain = True
                else:
                    print(f"   ✅ KHỚP với config ({len(CLASSES)} classes)")
                
            except Exception as e:
                print(f"   ⚠️  Không thể load model: {str(e)}")
                need_retrain = True
        else:
            print(f"\n❌ {description}: Không tồn tại")
            need_retrain = True
    
    print("\n" + "="*70)
    
    if need_retrain:
        print("\n⚠️  CẦN RETRAIN MODEL!")
        print("\nLý do:")
        print("   • Model chưa tồn tại HOẶC")
        print("   • Số classes trong model cũ không khớp với config mới")
        print("   • Config đã thêm class 'organic' mới")
    else:
        print("\n✅ Model hiện tại đã đúng với config!")
    
    return need_retrain


def show_retrain_guide():
    """Hiển thị hướng dẫn retrain"""
    print("\n" + "="*70)
    print("🎓 HƯỚNG DẪN RETRAIN MODEL")
    print("="*70)
    
    print("""
📁 BƯỚC 1: Chuẩn bị dữ liệu
─────────────────────────────────────────────────────
Đảm bảo bạn có cấu trúc thư mục:

dataset/
├── train/
│   ├── cardboard/     (nhiều ảnh)
│   ├── glass/         (nhiều ảnh)
│   ├── metal/         (nhiều ảnh)
│   ├── paper/         (nhiều ảnh)
│   ├── plastic/       (nhiều ảnh)
│   ├── trash/         (nhiều ảnh)
│   └── organic/       (nhiều ảnh) ← QUAN TRỌNG!
└── validation/
    ├── cardboard/     (ít ảnh hơn)
    ├── glass/
    ├── metal/
    ├── paper/
    ├── plastic/
    ├── trash/
    └── organic/       ← QUAN TRỌNG!

💡 Lưu ý: 
   • Mỗi thư mục cần ít nhất 100-200 ảnh (train)
   • Validation nên có 20-30% số lượng train
   • Class 'organic' GỒM: thức ăn thừa, vỏ trái cây, rau củ, lá cây


🚀 BƯỚC 2: Chạy training
─────────────────────────────────────────────────────
Chọn 1 trong 3 cách:

A. GUI Mode (Dễ nhất):
   python gui_app.py
   → Chọn "🎓 Training" 
   → Chọn thư mục train và validation
   → Bắt đầu

B. Menu Mode:
   python main.py
   → Chọn option 5
   → Nhập đường dẫn train và validation

C. Direct Training:
   python train.py


⚙️ BƯỚC 3: Cấu hình training
─────────────────────────────────────────────────────
Khuyến nghị:

• Transfer Learning: YES (nếu dataset < 5000 ảnh/class)
• Epochs: 50-100 
• Batch size: 32 (mặc định)
• Augmentation: Bật (mặc định)


⏱️ BƯỚC 4: Đợi training
─────────────────────────────────────────────────────
• Training sẽ mất 30-60 phút (tùy GPU/CPU)
• Theo dõi accuracy và loss
• Model tốt nhất được lưu tự động


✅ BƯỚC 5: Kiểm tra kết quả
─────────────────────────────────────────────────────
Sau training, test với:
   python classifier.py
   → Nhập ảnh organic để test


🎯 NHANH CHÓNG: Nếu bạn đã có dataset
─────────────────────────────────────────────────────
python train.py

Rồi nhập:
   📁 Thư mục training: dataset/train
   📁 Thư mục validation: dataset/validation
   🔄 Transfer Learning: y
   ⏱️  Epochs: 50
""")
    
    print("="*70)


def main():
    """Main function"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║        🔧 CÔNG CỤ KIỂM TRA & HƯỚNG DẪN RETRAIN           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Bước 1: Verify config
    config_ok = verify_config()
    
    if not config_ok:
        print("\n❌ Vui lòng sửa lỗi config trước!")
        return
    
    # Bước 2: Check model
    need_retrain = check_model_status()
    
    # Bước 3: Hướng dẫn
    if need_retrain:
        show_retrain_guide()
        
        print("\n" + "="*70)
        choice = input("Bạn có muốn bắt đầu training ngay không? (y/n): ").strip().lower()
        
        if choice == 'y':
            print("\n🚀 Đang khởi động training...")
            import train
            train.main()
    else:
        print("\n✨ Mọi thứ đã sẵn sàng! Bạn có thể:")
        print("   • Chạy GUI: python gui_app.py")
        print("   • Chạy classifier: python classifier.py")
        print("   • Chạy camera: python camera.py")


if __name__ == "__main__":
    main()