import os
import cv2
import numpy as np
import collections
from tqdm import tqdm

from src.core.config import CLASSES

def dhash(image, hash_size=8):
    """Thuật toán mã băm ảnh dHash để tìm ảnh trùng lặp"""
    # Resize thành 9x8 và chuyển sang ảnh xám
    resized = cv2.resize(image, (hash_size + 1, hash_size))
    if len(resized.shape) == 3:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    # Tính sự khác biệt giữa các pixel liền kề
    diff = resized[:, 1:] > resized[:, :-1]
    # Trả về hash số nguyên
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

def hamming_distance(hash1, hash2):
    """Tính khoảng cách Hamming giữa 2 mã băm"""
    return bin(hash1 ^ hash2).count('1')

def evaluate_dataset_for_ml(dataset_dir="dataset/train"):
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║       🧠 ML DATASET EVALUATOR (Chấm điểm cho AI)          ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    if not os.path.exists(dataset_dir):
        print(f"❌ Không tìm thấy thư mục {dataset_dir}!")
        return

    # Khởi tạo các biến lưu trữ
    class_counts = {}
    total_images = 0
    all_hashes = {}  # {hash: path}
    duplicates_found = 0
    bg_colors = collections.defaultdict(list)

    # 1. Thu thập dữ liệu
    for cls in CLASSES:
        cls_dir = os.path.join(dataset_dir, cls)
        if not os.path.exists(cls_dir):
            continue
            
        images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        class_counts[cls] = len(images)
        total_images += len(images)
        
        print(f"🔍 Đang phân tích class '{cls}'...")
        
        for img_name in tqdm(images, leave=False):
            img_path = os.path.join(cls_dir, img_name)
            img = cv2.imread(img_path)
            
            if img is None:
                continue

            # a) Tính dHash cho ảnh để tìm trùng lặp
            h = dhash(img)
            
            # Tìm xem hash này có bị trùng hoặc cực kỳ giống với ảnh nào trước đó không (Hamming distance < 3)
            # (Để tăng tốc, tôi chỉ check exact match hoặc giống 95% trên hash dictionary)
            is_duplicate = False
            for prev_hash in all_hashes.keys():
                if hamming_distance(h, prev_hash) <= 3:
                    duplicates_found += 1
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                all_hashes[h] = img_path

            # b) Lấy trung bình màu sắc của đường viền (giả định là background) để kiểm tra độ đa dạng nền
            # Cắt 10 pixel ở các viền để lấy màu nền
            h_img, w_img = img.shape[:2]
            if h_img > 20 and w_img > 20:
                top = img[0:10, :]
                bottom = img[h_img-10:h_img, :]
                left = img[10:h_img-10, 0:10]
                right = img[10:h_img-10, w_img-10:w_img]
                
                bg_pixels = np.concatenate([top.flatten(), bottom.flatten(), left.flatten(), right.flatten()])
                bg_mean_color = np.mean(bg_pixels)
                bg_colors[cls].append(bg_mean_color)

    # ==========================================
    # KẾT QUẢ ĐÁNH GIÁ (SCORING)
    # ==========================================
    print("\n" + "="*60)
    print("📊 KẾT QUẢ ĐÁNH GIÁ ĐỘ SẴN SÀNG CHO MACHINE LEARNING")
    print("="*60)

    # 1. Đánh giá Cân bằng dữ liệu (Class Balance)
    if not class_counts:
        print("Dataset trống!")
        return
        
    max_cls = max(class_counts, key=class_counts.get)
    min_cls = min([c for c in class_counts if class_counts[c] > 0], key=class_counts.get)
    max_count = class_counts[max_cls]
    min_count = class_counts[min_cls]
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

    print("\n1️⃣ ĐỘ CÂN BẰNG DỮ LIỆU (CLASS BALANCE):")
    print(f"   - Nhiều nhất: {max_cls} ({max_count} ảnh)")
    print(f"   - Ít nhất   : {min_cls} ({min_count} ảnh)")
    if imbalance_ratio <= 1.5:
        print("   ✅ Rất tốt: Dữ liệu phân bổ đều, AI sẽ không bị thiên vị.")
    elif imbalance_ratio <= 3:
        print("   ⚠️ Khá: Dữ liệu hơi lệch (Tỷ lệ 1:%.1f). Nên bổ sung thêm '%s'." % (imbalance_ratio, min_cls))
    else:
        print("   ❌ Tồi tệ: Dữ liệu cực kỳ mất cân đối (Tỷ lệ 1:%.1f)! AI sẽ bị Overfitting và chuyên đoán bừa là '%s'." % (imbalance_ratio, max_cls))

    # 2. Đánh giá Rò rỉ dữ liệu (Data Leakage/Duplicates)
    dup_percent = (duplicates_found / total_images) * 100 if total_images > 0 else 0
    print("\n2️⃣ TRÙNG LẶP DỮ LIỆU (DUPLICATE DETECTION):")
    print(f"   - Phát hiện : {duplicates_found} ảnh trùng lặp/giống hệt nhau.")
    if dup_percent < 5:
        print("   ✅ Rất tốt: Dữ liệu đa dạng, không bị lặp lại nhiều.")
    elif dup_percent < 15:
        print("   ⚠️ Cảnh báo: Gần %.1f%% ảnh bị trùng. Có vẻ bạn cắt video thành ảnh hơi sát nhau. AI sẽ bị học vẹt." % dup_percent)
    else:
        print("   ❌ Nguy hiểm: Lên tới %.1f%% ảnh y hệt nhau! Độ chính xác khi train sẽ rất cao nhưng test thực tế sẽ thất bại thảm hại." % dup_percent)

    # 3. Đánh giá đa dạng nền (Background Diversity)
    print("\n3️⃣ ĐA DẠNG BỐI CẢNH (BACKGROUND DIVERSITY):")
    bg_warnings = []
    for cls, colors in bg_colors.items():
        if len(colors) > 0:
            variance = np.var(colors)
            if variance < 500: # Ngưỡng tự do, phương sai màu nền thấp
                bg_warnings.append(cls)

    if not bg_warnings:
        print("   ✅ Rất tốt: Các loại rác được chụp ở nhiều phông nền/ánh sáng khác nhau.")
    else:
        print(f"   ⚠️ Cảnh báo: Các lớp {bg_warnings} có vẻ bị chụp trên cùng MỘT phông nền (phương sai màu thấp).")
        print("      -> Hậu quả: AI sẽ học 'cái bàn/phông nền' thay vì học đặc điểm của cục rác!")

    print("\n" + "="*60)
    print("💡 LỜI KHUYÊN DÀNH CHO BẠN:")
    if imbalance_ratio > 3:
        print(f"  👉 Chụp thêm rác loại: {min_cls}")
    if dup_percent >= 5:
        print("  👉 Lọc và xóa bớt các bức ảnh chụp liên tiếp y hệt nhau để tăng độ tổng quát.")
    if bg_warnings:
        print(f"  👉 Khi thu thập {bg_warnings}, hãy mang rác ra nhiều chỗ khác nhau (sàn nhà, mặt đường, bãi cỏ) để chụp.")
    print("============================================================")

if __name__ == "__main__":
    train_folder = input("Nhập đường dẫn dataset cần chấm điểm ML (Enter = 'dataset/train'): ").strip()
    if not train_folder:
        train_folder = "dataset/train"
    evaluate_dataset_for_ml(train_folder)
