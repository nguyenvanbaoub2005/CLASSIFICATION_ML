import os
import math

# ============================
# CẤU HÌNH CHO BẠN
# ============================
TRAIN_DIR = "dataset/train"
DELETE_PERCENT = 0.70   # Xóa 73% mỗi folder

def delete_73_percent_per_class():
    print("🗑️ BẮT ĐẦU XÓA 73% ẢNH TRONG MỖI CLASS...\n")

    # Duyệt qua từng class trong dataset/train/
    for cls in sorted(os.listdir(TRAIN_DIR)):
        cls_path = os.path.join(TRAIN_DIR, cls)

        # Bỏ qua nếu không phải thư mục
        if not os.path.isdir(cls_path):
            continue

        # Lấy ảnh JPG/JPEG/PNG
        images = [
            f for f in os.listdir(cls_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if len(images) == 0:
            print(f"⚠️ Class '{cls}' không có ảnh – bỏ qua.\n")
            continue

        # Sắp xếp theo thời gian tạo file → ảnh cũ lên trên, mới ở dưới
        images.sort(key=lambda f: os.path.getmtime(os.path.join(cls_path, f)))

        total = len(images)
        delete_count = math.ceil(total * DELETE_PERCENT)

        # Chọn ảnh mới nhất (73%) từ dưới lên
        to_delete = images[-delete_count:]

        print(f"🔹 {cls}: tổng {total} ảnh → xóa {delete_count} ảnh (73%)")

        # Xóa ảnh và JSON đi kèm (nếu có)
        for img in to_delete:
            img_path = os.path.join(cls_path, img)
            json_path = img_path.rsplit('.', 1)[0] + ".json"

            if os.path.exists(img_path):
                os.remove(img_path)

            if os.path.exists(json_path):
                os.remove(json_path)

        print(f"   → Đã xóa {len(to_delete)} ảnh của class '{cls}'\n")

    print("🎉 HOÀN TẤT – ĐÃ XÓA 73% MỖI CLASS!\n")


if __name__ == "__main__":
    delete_73_percent_per_class()
