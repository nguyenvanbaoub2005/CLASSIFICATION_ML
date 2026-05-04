# quick_fix_dataset.py
"""
Script sửa nhanh dataset:
- Gộp tất cả ảnh từ train + validation
- Cân bằng classes
- Chia lại 80/20 KHÔNG TRÙNG
"""

import os
import shutil
import random
from pathlib import Path
from config import CLASSES

def main():
    print("""
╔═══════════════════════════════════════════════════════════╗
║        🔧 SỬA NHANH DATASET - TÁCH TRAIN/VAL              ║
╚═══════════════════════════════════════════════════════════╝
    """)
    
    dataset_dir = "dataset"
    output_dir = "dataset_fixed"
    
    # Set random seed
    random.seed(42)
    
    print("📋 CÁC BƯỚC:")
    print("1. Backup dataset gốc")
    print("2. Gộp tất cả ảnh (train + validation)")
    print("3. Cân bằng classes")
    print("4. Chia lại 80/20 KHÔNG TRÙNG")
    print("5. Tạo dataset mới\n")
    
    confirm = input("Tiếp tục? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Đã hủy.")
        return
    
    # Backup
    print("\n📦 BƯỚC 1: Backup dataset gốc...")
    backup_dir = f"{dataset_dir}_backup"
    if not os.path.exists(backup_dir):
        shutil.copytree(dataset_dir, backup_dir)
        print(f"✅ Đã backup tại: {backup_dir}")
    else:
        print(f"⚠️  Backup đã tồn tại: {backup_dir}")
    
    # Thu thập ảnh
    print("\n📊 BƯỚC 2: Thu thập tất cả ảnh...")
    all_images = {cls: [] for cls in CLASSES}
    
    for split in ['train', 'validation']:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.exists(split_dir):
            continue
        
        for cls in CLASSES:
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.exists(cls_dir):
                continue
            
            images = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            all_images[cls].extend(images)
    
    print("\n📈 Số ảnh mỗi class (trước cân bằng):")
    for cls in CLASSES:
        print(f"  {cls:12s}: {len(all_images[cls])} ảnh")
    
    # Cân bằng
    print("\n⚖️  BƯỚC 3: Cân bằng classes...")
    
    # Lấy số lượng của class ít nhất
    min_count = min(len(imgs) for imgs in all_images.values())
    print(f"   Target: {min_count} ảnh/class (undersample)")
    
    balanced_images = {}
    for cls in CLASSES:
        images = all_images[cls]
        if len(images) >= min_count:
            # Sample ngẫu nhiên
            balanced_images[cls] = random.sample(images, min_count)
            print(f"  {cls:12s}: {len(images)} → {min_count}")
        else:
            balanced_images[cls] = images
            print(f"  {cls:12s}: {len(images)} → {len(images)} (giữ nguyên)")
    
    # Chia train/val
    print("\n✂️  BƯỚC 4: Chia train/validation (80/20)...")
    
    train_images = {cls: [] for cls in CLASSES}
    val_images = {cls: [] for cls in CLASSES}
    
    for cls in CLASSES:
        images = balanced_images[cls].copy()
        random.shuffle(images)
        
        split_idx = int(len(images) * 0.8)
        
        train_images[cls] = images[:split_idx]
        val_images[cls] = images[split_idx:]
        
        print(f"  {cls:12s}: Train={len(train_images[cls])}, Val={len(val_images[cls])}")
    
    # Tạo dataset mới
    print(f"\n📁 BƯỚC 5: Tạo dataset mới tại: {output_dir}")
    
    # Xóa thư mục cũ nếu tồn tại
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    # Tạo cấu trúc
    for split in ['train', 'validation']:
        for cls in CLASSES:
            os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)
    
    # Copy ảnh train
    print("\n📦 Copy ảnh training...")
    for cls in CLASSES:
        for i, src_path in enumerate(train_images[cls]):
            ext = os.path.splitext(src_path)[1]
            dst_path = os.path.join(output_dir, 'train', cls, f"{cls}_{i:04d}{ext}")
            shutil.copy2(src_path, dst_path)
        print(f"  {cls:12s}: {len(train_images[cls])} ảnh")
    
    # Copy ảnh validation
    print("\n📦 Copy ảnh validation...")
    for cls in CLASSES:
        for i, src_path in enumerate(val_images[cls]):
            ext = os.path.splitext(src_path)[1]
            dst_path = os.path.join(output_dir, 'validation', cls, f"{cls}_val_{i:04d}{ext}")
            shutil.copy2(src_path, dst_path)
        print(f"  {cls:12s}: {len(val_images[cls])} ảnh")
    
    # Tóm tắt
    print("\n" + "="*70)
    print("✅ HOÀN TẤT!")
    print("="*70)
    
    total_train = sum(len(imgs) for imgs in train_images.values())
    total_val = sum(len(imgs) for imgs in val_images.values())
    
    print(f"\n📊 Thống kê:")
    print(f"  Train:      {total_train} ảnh ({len(CLASSES)} classes)")
    print(f"  Validation: {total_val} ảnh ({len(CLASSES)} classes)")
    print(f"  Tổng:       {total_train + total_val} ảnh")
    
    # Verify không trùng
    print(f"\n🔍 Verify: Train và Val KHÔNG TRÙNG...")
    all_train_files = set()
    all_val_files = set()
    
    for cls in CLASSES:
        for img in train_images[cls]:
            all_train_files.add(os.path.basename(img))
        for img in val_images[cls]:
            all_val_files.add(os.path.basename(img))
    
    overlap = all_train_files.intersection(all_val_files)
    if overlap:
        print(f"❌ CÓ {len(overlap)} ảnh trùng!")
    else:
        print("✅ KHÔNG CÓ ảnh trùng lặp!")
    
    print(f"\n📂 Files:")
    print(f"  Dataset gốc: {dataset_dir}/ (đã backup)")
    print(f"  Dataset mới: {output_dir}/")
    print(f"  Backup:      {backup_dir}/")
    
    print("\n🚀 Bước tiếp theo:")
    print("1. Xóa model cũ:")
    print("   rm waste_classifier_*.h5")
    print("   rm -rf logs/")
    print("")
    print("2. Train với dataset mới:")
    print("   python train.py")
    print(f"   → Train dir:      {output_dir}/train")
    print(f"   → Validation dir: {output_dir}/validation")
    print("   → Transfer Learning: y")
    print("   → Epochs: 50")

if __name__ == "__main__":
    main()