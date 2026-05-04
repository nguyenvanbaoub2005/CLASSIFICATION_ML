# data_manager.py
"""
Quản lý và tổ chức dữ liệu đã scan để training
"""

import os
import json
import shutil
from datetime import datetime
from config import CLASSES
import random


class DataManager:
    """Class quản lý dữ liệu training"""
    
    def __init__(self, scanned_dir="scanned_data", dataset_dir="dataset"):
        self.scanned_dir = scanned_dir
        self.dataset_dir = dataset_dir
        self.train_ratio = 0.8  # 80% train, 20% validation
        
    def get_scanned_stats(self):
        """Thống kê dữ liệu đã scan"""
        stats = {}
        total = 0
        high_conf_count = 0
        
        for cls in CLASSES:
            cls_dir = os.path.join(self.scanned_dir, cls)
            if os.path.exists(cls_dir):
                images = [f for f in os.listdir(cls_dir) if f.endswith('.jpg')]
                stats[cls] = {
                    'count': len(images),
                    'high_confidence': 0,
                    'low_confidence': 0
                }
                
                # Đếm theo confidence
                for img in images:
                    json_path = os.path.join(cls_dir, img.replace('.jpg', '.json'))
                    if os.path.exists(json_path):
                        with open(json_path, 'r') as f:
                            data = json.load(f)
                            if data['confidence'] >= 80:
                                stats[cls]['high_confidence'] += 1
                                high_conf_count += 1
                            else:
                                stats[cls]['low_confidence'] += 1
                
                total += stats[cls]['count']
            else:
                stats[cls] = {'count': 0, 'high_confidence': 0, 'low_confidence': 0}
        
        return {
            'total': total,
            'high_confidence': high_conf_count,
            'by_class': stats
        }
    
    def prepare_training_data(self, min_confidence=80, use_all=False):
        """
        Chuẩn bị dữ liệu từ scanned_data vào dataset cho training
        
        Args:
            min_confidence: Độ tin cậy tối thiểu
            use_all: Sử dụng tất cả dữ liệu (không phân chia train/val)
        
        Returns:
            dict: Thông tin về dữ liệu đã chuẩn bị
        """
        print("\n" + "="*70)
        print("📦 CHUẨN BỊ DỮ LIỆU TRAINING")
        print("="*70)
        
        # Tạo thư mục dataset
        train_dir = os.path.join(self.dataset_dir, 'train')
        val_dir = os.path.join(self.dataset_dir, 'validation')
        
        for cls in CLASSES:
            os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
            os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
        
        stats = {'train': {}, 'val': {}}
        
        for cls in CLASSES:
            cls_scanned = os.path.join(self.scanned_dir, cls)
            cls_train = os.path.join(train_dir, cls)
            cls_val = os.path.join(val_dir, cls)
            
            if not os.path.exists(cls_scanned):
                continue
            
            # Lấy tất cả ảnh đủ điều kiện
            valid_images = []
            for img_file in os.listdir(cls_scanned):
                if not img_file.endswith('.jpg'):
                    continue
                
                json_path = os.path.join(cls_scanned, img_file.replace('.jpg', '.json'))
                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                        if data['confidence'] >= min_confidence:
                            valid_images.append(img_file)
            
            if not valid_images:
                print(f"⚠️  {cls}: Không có ảnh nào đủ điều kiện")
                continue
            
            # Shuffle
            random.shuffle(valid_images)
            
            # Phân chia train/val
            split_idx = int(len(valid_images) * self.train_ratio)
            train_images = valid_images[:split_idx]
            val_images = valid_images[split_idx:]
            
            # Copy files
            for img in train_images:
                src = os.path.join(cls_scanned, img)
                dst = os.path.join(cls_train, img)
                shutil.copy2(src, dst)
            
            for img in val_images:
                src = os.path.join(cls_scanned, img)
                dst = os.path.join(cls_val, img)
                shutil.copy2(src, dst)
            
            stats['train'][cls] = len(train_images)
            stats['val'][cls] = len(val_images)
            
            print(f"✓ {cls:12s}: {len(train_images)} train, {len(val_images)} val")
        
        print("="*70)
        print(f"✅ Hoàn tất! Dataset sẵn sàng tại: {self.dataset_dir}")
        
        return stats
    
    def merge_with_existing_dataset(self, existing_train_dir, existing_val_dir):
        """
        Merge dữ liệu mới với dataset cũ
        
        Args:
            existing_train_dir: Thư mục train hiện có
            existing_val_dir: Thư mục validation hiện có
        """
        print("\n🔀 MERGE DỮ LIỆU MỚI VỚI DATASET CŨ...")
        
        for cls in CLASSES:
            # Train
            src_train = os.path.join(self.dataset_dir, 'train', cls)
            dst_train = os.path.join(existing_train_dir, cls)
            
            if os.path.exists(src_train):
                os.makedirs(dst_train, exist_ok=True)
                for img in os.listdir(src_train):
                    if img.endswith('.jpg'):
                        shutil.copy2(
                            os.path.join(src_train, img),
                            os.path.join(dst_train, img)
                        )
            
            # Validation
            src_val = os.path.join(self.dataset_dir, 'validation', cls)
            dst_val = os.path.join(existing_val_dir, cls)
            
            if os.path.exists(src_val):
                os.makedirs(dst_val, exist_ok=True)
                for img in os.listdir(src_val):
                    if img.endswith('.jpg'):
                        shutil.copy2(
                            os.path.join(src_val, img),
                            os.path.join(dst_val, img)
                        )
        
        print("✅ Đã merge xong!")
    
    def export_high_quality_data(self, output_dir, min_confidence=90):
        """
        Export dữ liệu chất lượng cao (confidence >= 90%)
        
        Args:
            output_dir: Thư mục output
            min_confidence: Độ tin cậy tối thiểu
        """
        print(f"\n📤 EXPORT DỮ LIỆU CHẤT LƯỢNG CAO (>={min_confidence}%)...")
        
        for cls in CLASSES:
            cls_dir = os.path.join(self.scanned_dir, cls)
            output_cls_dir = os.path.join(output_dir, cls)
            os.makedirs(output_cls_dir, exist_ok=True)
            
            if not os.path.exists(cls_dir):
                continue
            
            count = 0
            for img_file in os.listdir(cls_dir):
                if not img_file.endswith('.jpg'):
                    continue
                
                json_path = os.path.join(cls_dir, img_file.replace('.jpg', '.json'))
                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                        if data['confidence'] >= min_confidence:
                            shutil.copy2(
                                os.path.join(cls_dir, img_file),
                                os.path.join(output_cls_dir, img_file)
                            )
                            count += 1
            
            print(f"✓ {cls:12s}: {count} ảnh")
        
        print(f"✅ Đã export vào: {output_dir}")
    
    def clean_low_quality_data(self, max_confidence=60):
        """
        Xóa dữ liệu chất lượng thấp
        
        Args:
            max_confidence: Xóa ảnh có confidence <= giá trị này
        """
        print(f"\n🗑️  XÓA DỮ LIỆU CHẤT LƯỢNG THẤP (<={max_confidence}%)...")
        
        total_removed = 0
        
        for cls in CLASSES:
            cls_dir = os.path.join(self.scanned_dir, cls)
            if not os.path.exists(cls_dir):
                continue
            
            removed = 0
            for img_file in os.listdir(cls_dir):
                if not img_file.endswith('.jpg'):
                    continue
                
                json_path = os.path.join(cls_dir, img_file.replace('.jpg', '.json'))
                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                        if data['confidence'] <= max_confidence:
                            os.remove(os.path.join(cls_dir, img_file))
                            os.remove(json_path)
                            removed += 1
            
            if removed > 0:
                print(f"✓ {cls:12s}: Đã xóa {removed} ảnh")
                total_removed += removed
        
        print(f"✅ Tổng đã xóa: {total_removed} ảnh")
    
    def generate_report(self):
        """Tạo báo cáo chi tiết"""
        print("\n" + "="*70)
        print("📊 BÁO CÁO DỮ LIỆU")
        print("="*70)
        
        stats = self.get_scanned_stats()
        
        print(f"\nTổng số mẫu: {stats['total']}")
        print(f"Chất lượng cao (≥80%): {stats['high_confidence']}")
        print(f"Tỷ lệ chất lượng: {stats['high_confidence']/stats['total']*100:.1f}%")
        
        print("\n" + "-"*70)
        print(f"{'Class':12s} | {'Tổng':>6s} | {'Cao':>6s} | {'Thấp':>6s} | {'% Cao':>8s}")
        print("-"*70)
        
        for cls in CLASSES:
            data = stats['by_class'][cls]
            total = data['count']
            high = data['high_confidence']
            low = data['low_confidence']
            pct = (high/total*100) if total > 0 else 0
            
            print(f"{cls:12s} | {total:6d} | {high:6d} | {low:6d} | {pct:7.1f}%")
        
        print("="*70)
        
        # Khuyến nghị
        print("\n💡 KHUYẾN NGHỊ:")
        min_samples = min(stats['by_class'][cls]['count'] for cls in CLASSES)
        
        if stats['total'] < 500:
            print("⚠️  Cần thêm dữ liệu (tối thiểu 500 mẫu)")
        elif min_samples < 50:
            print("⚠️  Một số class thiếu dữ liệu (cần ít nhất 50 mẫu/class)")
        elif stats['high_confidence'] / stats['total'] < 0.7:
            print("⚠️  Nhiều mẫu chất lượng thấp - nên review lại")
        else:
            print("✅ Dữ liệu đủ điều kiện để training!")


def main():
    """Main function"""
    manager = DataManager()
    
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║              📊 QUẢN LÝ DỮ LIỆU TRAINING                  ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    while True:
        print("\n" + "="*70)
        print("MENU:")
        print("1. Xem thống kê dữ liệu")
        print("2. Chuẩn bị dữ liệu cho training")
        print("3. Export dữ liệu chất lượng cao")
        print("4. Xóa dữ liệu chất lượng thấp")
        print("5. Tạo báo cáo chi tiết")
        print("0. Thoát")
        print("="*70)
        
        choice = input("\nNhập lựa chọn: ").strip()
        
        if choice == '1':
            stats = manager.get_scanned_stats()
            print(f"\nTổng: {stats['total']} mẫu")
            print(f"Chất lượng cao: {stats['high_confidence']}")
            for cls, data in stats['by_class'].items():
                print(f"  {cls:12s}: {data['count']:4d} (High: {data['high_confidence']}, Low: {data['low_confidence']})")
        
        elif choice == '2':
            min_conf = input("Độ tin cậy tối thiểu (Enter=80): ").strip()
            min_conf = int(min_conf) if min_conf else 80
            manager.prepare_training_data(min_confidence=min_conf)
        
        elif choice == '3':
            output = input("Thư mục output (Enter=high_quality_data): ").strip()
            output = output if output else "high_quality_data"
            min_conf = input("Độ tin cậy tối thiểu (Enter=90): ").strip()
            min_conf = int(min_conf) if min_conf else 90
            manager.export_high_quality_data(output, min_conf)
        
        elif choice == '4':
            confirm = input("⚠️  Bạn có chắc muốn xóa? (yes/no): ").strip().lower()
            if confirm == 'yes':
                max_conf = input("Xóa ảnh có confidence <= (Enter=60): ").strip()
                max_conf = int(max_conf) if max_conf else 60
                manager.clean_low_quality_data(max_conf)
        
        elif choice == '5':
            manager.generate_report()
        
        elif choice == '0':
            break
        
        input("\nNhấn Enter để tiếp tục...")


if __name__ == "__main__":
    main()