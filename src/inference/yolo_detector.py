import cv2
import numpy as np

class YOLODetector:
    def __init__(self, model_path="yolov5s.onnx", conf_threshold=0.45, iou_threshold=0.45):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Load YOLO ONNX
        self.net = cv2.dnn.readNetFromONNX(model_path)

        # GPU nếu có
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self.input_width = 640
        self.input_height = 640

    def detect(self, frame):
        """Trả về bounding box lớn nhất tìm được"""
        blob = cv2.dnn.blobFromImage(
            frame, 1/255.0, (self.input_width, self.input_height),
            swapRB=True, crop=False
        )
        self.net.setInput(blob)
        preds = self.net.forward()

        class_ids = []
        confs = []
        boxes = []

        h, w = frame.shape[:2]

        for det in preds[0]:
            confidence = det[4]
            if confidence < self.conf_threshold:
                continue

            scores = det[5:]
            class_id = np.argmax(scores)
            class_conf = scores[class_id]

            if class_conf > self.conf_threshold:
                cx, cy, bw, bh = det[0:4]

                x = int((cx - bw/2) * w)
                y = int((cy - bh/2) * h)
                bw = int(bw * w)
                bh = int(bh * h)

                class_ids.append(class_id)
                confs.append(float(class_conf))
                boxes.append([x, y, bw, bh])

        if len(boxes) == 0:
            return None

        areas = [(b[2] * b[3]) for b in boxes]
        max_idx = np.argmax(areas)
        return boxes[max_idx]
