# 🤖 Hệ Thống Phân Loại Rác Thải Bằng AI

Hệ thống phân loại rác thải tự động sử dụng Deep Learning (CNN) để nhận diện các loại rác và hỗ trợ inference từ ảnh, webcam (real-time) và file video. Hệ thống hỗ trợ huấn luyện (train) với Data Augmentation và Transfer Learning.

## 📌 Tóm tắt quan trọng (theo mã nguồn hiện tại)
- Số lớp (classes): 7 — cardboard, glass, metal, organic, paper, plastic, trash (được định nghĩa trong config.py)
- File model mặc định (PATHS trong config.py):
  - waste_classifier_final.h5
  - waste_classifier_best.h5
  - temp_capture.jpg (ảnh tạm cho camera)
  - training_history.png
- Ngưỡng confidence: 70.0% (CONFIDENCE_THRESHOLD)
- Transfer Learning: hỗ trợ MobileNetV2, VGG16, ResNet50 (create_transfer_learning_model)
- Learning rate mặc định: 0.001 (MODEL_CONFIG)
- Input image size mặc định: 224x224x3

## ✨ Tính năng
- Phân loại 7 loại rác: plastic, paper, glass, metal, cardboard, trash, organic.
- Phân loại từ ảnh đơn lẻ.
- Phân loại real-time từ webcam (gui điều khiển: SPACE, C, S, Q).
- Phân loại từ file video (classify mỗi N frame).
- Batch predict (predict nhiều ảnh).
- Hiển thị độ tin cậy (confidence) và hướng dẫn xử lý theo cấu hình CLASS_INFO.
- Hỗ trợ Transfer Learning và Data Augmentation tự động.

## 💻 Yêu cầu hệ thống
- Python 3.7 - 3.10
- CPU: Intel i5 hoặc tương đương (GPU NVIDIA + CUDA khuyến nghị cho training)
- RAM: >=8GB (16GB khuyến nghị)

## 📦 Cài đặt
1. Clone repo:
```bash
git clone https://github.com/nguyenvanbaoub2005/CLASSIFICATION.git
cd CLASSIFICATION
```

2. Tạo virtual environment (khuyến nghị):
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Cài đặt phụ thuộc:
```bash
pip install -r requirements.txt
```

## 📁 Cấu trúc dự án (quan trọng)
Root repository chứa:
- config.py            # cấu hình (CLASSES, CLASS_INFO, MODEL_CONFIG, PATHS,...)
- model.py             # định nghĩa model CNN và hàm transfer learning
- train.py             # script huấn luyện (dùng ImageDataGenerator)
- classifier.py        # lớp WasteClassifier để load model và predict/display
- camera.py            # xử lý camera, chế độ real-time và xử lý video
- requirements.txt
- dataset/             # (tự tạo) train/validation theo từng class
- models/              # (tùy) nơi bạn lưu model

## 🗂️ Dataset (cách tổ chức)
Tổ chức thư mục dataset theo cấu trúc:
```
dataset/
├── train/
│   ├── cardboard/
│   ├── glass/
│   ├── metal/
│   ├── organic/
│   ├── paper/
│   ├── plastic/
│   └── trash/
└── validation/
    ├── cardboard/
    ├── glass/
    ├── metal/
    ├── organic/
    ├── paper/
    ├── plastic/
    └── trash/
```
Gợi ý số lượng ảnh: train: 400–1000 ảnh/class, validation: 50–200 ảnh/class (tùy dataset).

## 🚀 Huấn luyện model
Chạy:
```bash
python train.py
```
Script sẽ hỏi:
- Đường dẫn thư mục training
- Đường dẫn thư mục validation
- Sử dụng Transfer Learning? (y/n) — nếu chọn y, mặc định code sử dụng MobileNetV2 (có thể thay trong model.py)
- Số epochs (mặc định MODEL_CONFIG['epochs'] = 50)

Chi tiết:
- Data augmentation được cấu hình trong AUGMENTATION_CONFIG (config.py).
- Callbacks: ModelCheckpoint (lưu best model theo val_accuracy), EarlyStopping, ReduceLROnPlateau, TensorBoard.
- Sau training, model cuối cùng được lưu theo PATHS['model_save'] (waste_classifier_final.h5) và best model theo PATHS['best_model'].

## 🧪 Sử dụng (Inference)

- Phân loại ảnh đơn:
```bash
python classifier.py
# Sau đó nhập đường dẫn ảnh khi được hỏi.
```
Hoặc dùng lớp trực tiếp:
```python
from classifier import WasteClassifier
clf = WasteClassifier('waste_classifier_final.h5')
result = clf.predict('test.jpg')
clf.display_result('test.jpg', result)
```

- Camera real-time:
```bash
python camera.py
# Chọn option 1 (Camera real-time)
```
Phím điều khiển:
- SPACE: chụp & phân loại
- C: toggle phân loại liên tục
- S: lưu ảnh
- Q / ESC: thoát

- Xử lý video file:
```bash
python camera.py
# Chọn option 2 và nhập đường dẫn video
```
Hàm classify_video_file phân loại mỗi N frames (mặc định tham số interval).

## ℹ️ Chi tiết kỹ thuật hữu ích (theo code)
- Model custom: create_waste_classifier_model() — một CNN tuần tự với nhiều block Conv2D + MaxPool + BatchNorm + Dropout, cuối cùng softmax theo num_classes.
- Transfer Learning: create_transfer_learning_model(base_model_name) — base_model (MobileNetV2 / VGG16 / ResNet50) với lớp top custom; base_model.trainable = False.
- Preprocessing: classifier.preprocess_image() resize về MODEL_CONFIG['input_shape'][:2] (224x224) và rescale /255.
- Loss: categorical_crossentropy, optimizer Adam (learning_rate từ MODEL_CONFIG).
- Requirements (requirements.txt): tensorflow>=2.10.0, opencv-python, pillow, matplotlib, numpy.

## 📈 Kết quả kỳ vọng (ước lượng)
Các con số này phụ thuộc dataset và training; README cũ có ước lượng, bạn có thể điều chỉnh sau khi thử nghiệm:
- Training Accuracy: cao tùy dataset (ví dụ 90%+ trên dataset tốt)
- Validation Accuracy: biến động theo dataset (mục tiêu tối ưu hóa bằng augmentation + transfer learning)
- Inference time: phụ thuộc phần cứng (GPU/CPU)

## 🛠️ Lưu ý và khuyến nghị
- README cũ đề cập 6 lớp; hiện repo có 7 lớp (thêm 'organic') — nếu bạn muốn chỉ 6 lớp, cần sửa CLASSES trong config.py và điều chỉnh dataset, retrain.
- Nếu muốn fine-tune base model (transfer learning), bạn cần thay base_model.trainable = True cho một số layer và giảm learning rate.
- Để giảm sử dụng bộ nhớ khi training, giảm batch_size trong MODEL_CONFIG.
- Có thể export model sang ONNX/TF-Lite nếu cần deploy trên thiết bị di động/edge.

---

Nếu bạn đồng ý, tôi có thể:
- Tạo PR để thay thế README.md bằng phiên bản này.
- Hoặc cập nhật README ngay (tôi sẽ tạo PR nếu bạn cho phép).

Bạn muốn tôi tạo PR cập nhật README không?

Nếu bạn chỉ muốn dùng app trên máy tính (có giao diện cửa sổ):

bash
python -m src.gui.app
2. Lệnh chạy Website (Cần chạy cả BE và FE song song)
Để Website hoạt động và quét được ảnh, bạn bắt buộc phải mở 2 tab Terminal và chạy 2 lệnh này cùng lúc:

Tab Terminal 1 - Chạy Backend (API Server): Đây là bộ não AI nhận ảnh từ web và trả về kết quả.

bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
Tab Terminal 2 - Chạy Frontend (Giao diện Web): Đây là phần giao diện siêu đẹp hiển thị trên trình duyệt.

bash
cd frontend
npm run dev




# CLASSIFICATION_GREEN
