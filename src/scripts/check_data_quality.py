import os
import cv2
import numpy as np
import shutil
from tqdm import tqdm

from src.core.config import CLASSES

def check_image_quality(dataset_dir="dataset/train", issues_dir="dataset_issues"):
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║           🔍 AI KIỂM ĐỊNH CHẤT LƯỢNG DATASET              ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    if not os.path.exists(dataset_dir):
        print(f"❌ Không tìm thấy thư mục {dataset_dir}!")
        return

    # Tạo thư mục chứa các ảnh có vấn đề để bạn review lại
    os.makedirs(issues_dir, exist_ok=True)
    
    stats = {
        'total': 0,
        'corrupted': 0,
        'blurry': 0,
        'too_dark': 0,
        'too_bright': 0,
        'good': 0
    }

    # Các ngưỡng (Thresholds)
    BLUR_THRESHOLD = 80.0     # Dưới 80 là mờ
    DARK_THRESHOLD = 40.0     # Dưới 40 là quá tối
    BRIGHT_THRESHOLD = 230.0  # Trên 230 là quá sáng (chói)

    for cls in CLASSES:
        cls_dir = os.path.join(dataset_dir, cls)
        if not os.path.exists(cls_dir):
            continue
            
        images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        stats['total'] += len(images)
        
        print(f"\n📂 Đang quét thư mục: {cls} ({len(images)} ảnh)")
        
        # Tạo thư mục con trong dataset_issues
        issue_cls_dir = os.path.join(issues_dir, cls)
        os.makedirs(issue_cls_dir, exist_ok=True)

        for img_name in tqdm(images, desc=f"Scanning {cls}"):
            img_path = os.path.join(cls_dir, img_name)
            
            # 1. Kiểm tra file lỗi (Corrupted)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    stats['corrupted'] += 1
                    shutil.copy(img_path, os.path.join(issue_cls_dir, f"CORRUPT_{img_name}"))
                    continue
            except:
                stats['corrupted'] += 1
                shutil.copy(img_path, os.path.join(issue_cls_dir, f"CORRUPT_{img_name}"))
                continue

            # Chuyển sang ảnh xám để phân tích
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 2. Kiểm tra độ mờ (Blur) bằng phương sai Laplacian
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # 3. Kiểm tra độ sáng (Brightness) bằng trung bình cường độ pixel
            brightness_score = np.mean(gray)

            is_bad = False
            issue_prefix = ""

            if blur_score < BLUR_THRESHOLD:
                stats['blurry'] += 1
                is_bad = True
                issue_prefix += "BLURRY_"
                
            if brightness_score < DARK_THRESHOLD:
                stats['too_dark'] += 1
                is_bad = True
                issue_prefix += "DARK_"
                
            if brightness_score > BRIGHT_THRESHOLD:
                stats['too_bright'] += 1
                is_bad = True
                issue_prefix += "BRIGHT_"

            if is_bad:
                # Copy ảnh có vấn đề ra thư mục riêng để bạn review
                shutil.copy(img_path, os.path.join(issue_cls_dir, f"{issue_prefix}{img_name}"))
            else:
                stats['good'] += 1

    # In Báo cáo
    print("\n" + "="*50)
    print("📋 BÁO CÁO CHẤT LƯỢNG DATASET")
    print("="*50)
    print(f"Tổng số ảnh đã quét : {stats['total']}")
    print(f"✅ Ảnh đạt chuẩn    : {stats['good']} ({stats['good']/stats['total']*100:.1f}%)")
    print("-" * 50)
    print("⚠️  CÁC LỖI PHÁT HIỆN:")
    print(f"❌ File hỏng (Corrupt): {stats['corrupted']}")
    print(f"🌫️ Mờ/Nhòe (Blurry)   : {stats['blurry']}")
    print(f"🌑 Quá tối (Dark)     : {stats['too_dark']}")
    print(f"☀️ Quá chói (Bright)  : {stats['too_bright']}")
    print("="*50)
    print(f"\n💡 Các ảnh có vấn đề đã được COPY vào thư mục: '{issues_dir}'")
    print("   Bạn có thể vào đó xem lại. Nếu thấy ảnh nào quá tệ, hãy xóa nó khỏi thư mục 'dataset/train' gốc để AI không học cái sai nhé!")

if __name__ == "__main__":
    train_folder = input("Nhập đường dẫn dataset cần kiểm tra (Enter = 'dataset/train'): ").strip()
    if not train_folder:
        train_folder = "dataset/train"
    check_image_quality(train_folder)
