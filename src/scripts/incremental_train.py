# incremental_train.py
"""
Training tăng dần với dữ liệu mới từ scan
"""

import os
from tensorflow import keras
from src.data.manager import DataManager
from src.scripts.train import create_data_generators, get_callbacks, plot_training_history
from src.core.config import PATHS, MODEL_CONFIG


class IncrementalTrainer:
    """Class training tăng dần"""
    
    def __init__(self, existing_model_path=None):
        """
        Khởi tạo incremental trainer
        
        Args:
            existing_model_path: Đường dẫn model hiện có
        """
        self.data_manager = DataManager()
        self.existing_model_path = existing_model_path or PATHS['model_save']
        
    def check_data_ready(self, min_samples_per_class=20):
        """
        Kiểm tra dữ liệu có đủ để training không
        
        Args:
            min_samples_per_class: Số mẫu tối thiểu mỗi class
        
        Returns:
            bool, dict: Sẵn sàng hay không và thống kê
        """
        stats = self.data_manager.get_scanned_stats()
        
        ready = True
        for cls, data in stats['by_class'].items():
            if data['high_confidence'] < min_samples_per_class:
                ready = False
                break
        
        return ready, stats
    
    def prepare_incremental_data(self):
        """Chuẩn bị dữ liệu cho incremental training"""
        print("\n" + "="*70)
        print("📦 CHUẨN BỊ DỮ LIỆU CHO INCREMENTAL TRAINING")
        print("="*70)
        
        # Kiểm tra
        ready, stats = self.check_data_ready()
        
        if not ready:
            print("\n⚠️  DỮ LIỆU CHƯA ĐỦ!")
            print("Cần ít nhất 20 mẫu chất lượng cao (≥80%) cho mỗi class")
            print("\nThống kê hiện tại:")
            for cls, data in stats['by_class'].items():
                print(f"  {cls:12s}: {data['high_confidence']} mẫu")
            return False
        
        # Chuẩn bị dữ liệu
        self.data_manager.prepare_training_data(min_confidence=80)
        
        return True
    
    def train_incremental(self, epochs=20, fine_tune=True):
        """
        Training tăng dần từ model cũ
        
        Args:
            epochs: Số epochs
            fine_tune: Fine-tune hay train lại hoàn toàn
        
        Returns:
            model, history
        """
        print("\n" + "="*70)
        print("🎓 INCREMENTAL TRAINING")
        print("="*70)
        
        # Load model cũ
        if os.path.exists(self.existing_model_path):
            print(f"\n📂 Loading model cũ: {self.existing_model_path}")
            model = keras.models.load_model(self.existing_model_path)
            print("✅ Đã load model!")
            
            if fine_tune:
                # Giảm learning rate cho fine-tuning
                print("\n🔧 Chuyển sang chế độ Fine-tuning...")
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
        else:
            print("\n⚠️  Không tìm thấy model cũ!")
            print("   Sẽ training từ đầu...")
            from src.models.model import create_waste_classifier_model
            model = create_waste_classifier_model()
        
        # Load data
        train_dir = os.path.join(self.data_manager.dataset_dir, 'train')
        val_dir = os.path.join(self.data_manager.dataset_dir, 'validation')
        
        print("\n📁 Loading dữ liệu...")
        train_gen, val_gen = create_data_generators(
            train_dir, 
            val_dir,
            batch_size=MODEL_CONFIG['batch_size']
        )
        
        print(f"\n✓ Train samples: {train_gen.samples}")
        print(f"✓ Validation samples: {val_gen.samples}")
        
        # Callbacks
        callbacks = get_callbacks()
        
        # Training
        print(f"\n🎯 Bắt đầu training ({epochs} epochs)...")
        print("="*70 + "\n")
        
        history = model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        # Lưu model
        save_path = PATHS['model_save']
        model.save(save_path)
        print(f"\n✅ Model đã lưu: {save_path}")
        
        # Vẽ biểu đồ
        plot_training_history(history)
        
        return model, history
    
    def evaluate_improvement(self, old_model_path, new_model_path, test_dir):
        """
        So sánh model cũ và mới
        
        Args:
            old_model_path: Model cũ
            new_model_path: Model mới
            test_dir: Thư mục test
        """
        print("\n" + "="*70)
        print("📊 SO SÁNH MODEL CŨ VÀ MỚI")
        print("="*70)
        
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_gen = test_datagen.flow_from_directory(
            test_dir,
            target_size=MODEL_CONFIG['input_shape'][:2],
            batch_size=32,
            class_mode='categorical',
            shuffle=False
        )
        
        # Evaluate old model
        print("\n📈 Đánh giá model CŨ...")
        old_model = keras.models.load_model(old_model_path)
        old_loss, old_acc = old_model.evaluate(test_gen, verbose=0)
        print(f"   Accuracy: {old_acc*100:.2f}%")
        print(f"   Loss: {old_loss:.4f}")
        
        # Evaluate new model
        print("\n📈 Đánh giá model MỚI...")
        new_model = keras.models.load_model(new_model_path)
        new_loss, new_acc = new_model.evaluate(test_gen, verbose=0)
        print(f"   Accuracy: {new_acc*100:.2f}%")
        print(f"   Loss: {new_loss:.4f}")
        
        # So sánh
        acc_improvement = (new_acc - old_acc) * 100
        
        print("\n" + "="*70)
        print("📊 KẾT QUẢ:")
        print("="*70)
        
        if acc_improvement > 0:
            print(f"✅ Model MỚI TốT HƠN: +{acc_improvement:.2f}%")
        elif acc_improvement < -1:
            print(f"⚠️  Model MỚI KÉM HƠN: {acc_improvement:.2f}%")
        else:
            print(f"➡️  Model TƯƠNG ĐƯƠNG: {acc_improvement:.2f}%")
        
        print("="*70)


def main():
    """Main function"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║         🔄 INCREMENTAL TRAINING VỚI DỮ LIỆU MỚI          ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    trainer = IncrementalTrainer()
    
    print("\n1️⃣  BƯỚC 1: Kiểm tra dữ liệu")
    ready, stats = trainer.check_data_ready()
    
    if not ready:
        print("\n❌ Dữ liệu chưa đủ để training!")
        print("   Hãy scan thêm ảnh qua GUI hoặc camera")
        return
    
    print("\n✅ Dữ liệu đã sẵn sàng!")
    print(f"   Tổng: {stats['total']} mẫu")
    print(f"   Chất lượng cao: {stats['high_confidence']}")
    
    input("\nNhấn Enter để tiếp tục...")
    
    print("\n2️⃣  BƯỚC 2: Chuẩn bị dữ liệu")
    if not trainer.prepare_incremental_data():
        return
    
    input("\nNhấn Enter để bắt đầu training...")
    
    print("\n3️⃣  BƯỚC 3: Training")
    epochs = input("Số epochs (Enter=20): ").strip()
    epochs = int(epochs) if epochs else 20
    
    fine_tune = input("Fine-tune model cũ? (y/n, Enter=y): ").strip().lower()
    fine_tune = fine_tune != 'n'
    
    model, history = trainer.train_incremental(epochs=epochs, fine_tune=fine_tune)
    
    print("\n" + "="*70)
    print("✅ HOÀN TẤT INCREMENTAL TRAINING!")
    print("="*70)
    print("\n💡 Gợi ý tiếp theo:")
    print("   - Test model mới với camera/ảnh")
    print("   - Tiếp tục scan thêm dữ liệu")
    print("   - So sánh với model cũ nếu có test set")


if __name__ == "__main__":
    main()