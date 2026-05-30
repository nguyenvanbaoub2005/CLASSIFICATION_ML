import cv2
import numpy as np

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except Exception:
    YOLO = None
    _YOLO_AVAILABLE = False

try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:
    torch = None
    _TORCH_AVAILABLE = False

class YOLODetector:
    """Wrapper YOLOv8 dùng chung cho toàn bộ hệ thống."""
    def __init__(self, model_name="yolov8n.pt", conf_threshold=0.35):
        self.enabled = False
        self.model = None
        self.device = "cpu"
        self.conf_threshold = conf_threshold

        if not _YOLO_AVAILABLE:
            print("⚠️ Thư viện ultralytics chưa được cài đặt - YOLO bị vô hiệu hóa.")
            return

        # Chọn device
        if _TORCH_AVAILABLE and torch is not None:
            try:
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self.device = "mps"
                elif torch.cuda.is_available():
                    self.device = "cuda"
            except:
                self.device = "cpu"

        try:
            print(f"📦 Load YOLO model: {model_name} (device={self.device})")
            self.model = YOLO(model_name)
            self.enabled = True
        except Exception as e:
            print(f"❌ Lỗi load YOLO: {e}")
            self.enabled = False

    def detect_single_object(self, frame):
        """Trả về bounding box tốt nhất (x, y, w, h)"""
        if not self.enabled or self.model is None:
            return None

        try:
            results = self.model.predict(
                frame, imgsz=640, conf=self.conf_threshold,
                device=self.device, verbose=False
            )
            if not results:
                return None

            res = results[0]
            if res.boxes is None or len(res.boxes) == 0:
                return None

            boxes = res.boxes.xyxy.cpu().numpy()
            scores = res.boxes.conf.cpu().numpy()
            best_idx = scores.argmax()

            x1, y1, x2, y2 = boxes[best_idx]
            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)

            if w <= 0 or h <= 0:
                return None
            return (x, y, w, h)

        except Exception as e:
            print(f"❌ YOLO detect lỗi: {e}")
            return None

    def crop_object(self, frame):
        """
        Dùng YOLO tìm vật thể.
        Nếu thấy -> Cắt (Crop) phần ảnh chứa vật thể.
        Nếu không thấy -> Trả về ảnh gốc.
        """
        bbox = self.detect_single_object(frame)
        if bbox:
            x, y, w, h = bbox
            
            # Mở rộng bounding box ra một chút (padding 10%) để không bị cắt lẹm
            pad_w = int(w * 0.1)
            pad_h = int(h * 0.1)
            
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(frame.shape[1], x + w + pad_w)
            y2 = min(frame.shape[0], y + h + pad_h)
            
            cropped = frame[y1:y2, x1:x2]
            
            # Đảm bảo ảnh không bị rỗng
            if cropped.size > 0:
                return cropped
                
        return frame
