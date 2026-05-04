import os

# ----------------------------
# CONFIG
# ----------------------------
DATASET_PATH = "dataset/train"   # Thư mục chứa các class
KEEP_RATIO = 1500 / 8000         # Tỉ lệ ảnh muốn giữ (VD: 18.75%)
# ----------------------------

def clean_dataset_interleave(dataset_path, keep_ratio):
    """
    Giữ ảnh theo tỉ lệ nhưng *lấy theo kiểu xen kẽ đều*.
    
    Args:
        dataset_path: thư mục chứa các class con
        keep_ratio: phần trăm ảnh cần giữ lại (0.0 → 1.0)
    """
    if keep_ratio <= 0 or keep_ratio > 1:
        print("❌ KEEP_RATIO không hợp lệ!")
        return
    
    folders = [f for f in os.listdir(dataset_path) 
               if os.path.isdir(os.path.join(dataset_path, f))]

    print(f"\n📁 Đang xử lý dataset: {dataset_path}")
    print(f"📌 Tỉ lệ giữ lại: {keep_ratio*100:.2f}%\n")

    for folder in folders:
        folder_path = os.path.join(dataset_path, folder)

        # Lọc các file ảnh
        files = [f for f in os.listdir(folder_path) 
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        files.sort()  # Sắp xếp để xen kẽ đúng thứ tự

        total = len(files)
        keep = int(total * keep_ratio)

        print(f"\n📂 Folder: {folder}")
        print(f"   Tổng: {total} ảnh")
        print(f"   Giữ lại: {keep} ảnh")

        if keep >= total:
            print("   → Không cần xoá ảnh.")
            continue

        # Tính step để lấy ảnh xen kẽ đều
        step = total / keep

        keep_files = set()
        index = 0.0

        for _ in range(keep):
            keep_files.add(files[int(index)])
            index += step

        # Ảnh còn lại bị xoá
        delete_files = set(files) - keep_files

        deleted = 0
        for fname in delete_files:
            fpath = os.path.join(folder_path, fname)
            os.remove(fpath)
            deleted += 1

        print(f"   ✓ Đã xoá {deleted} ảnh dư.")

    print("\n🎉 Hoàn tất dọn dataset theo tỉ lệ + xen kẽ!")


# ----------------------------
# RUN
# ----------------------------
clean_dataset_interleave(DATASET_PATH, KEEP_RATIO)
