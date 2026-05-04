"""
GUI hiện đại cho hệ thống phân loại rác – hỗ trợ YOLO (optional)
File này gồm 6 phần (bạn đang xem PART 1)
"""

# ==============================
# 🚀 PART 1 – IMPORT + YOLO WRAPPER + CAMERA CLASSIFIER + ModernButton
# ==============================

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import cv2
from PIL import Image, ImageTk
import threading
import os
import json
import csv
from datetime import datetime
import numpy as np

# Import classifier & training modules
from classifier import WasteClassifier
from train import train_model, plot_training_history
from data_manager import DataManager
from incremental_train import IncrementalTrainer
from config import PATHS, CLASS_INFO, CLASSES, MODEL_CONFIG


# ======================
# Optional YOLOv8 Detector
# ======================

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
    """Wrapper YOLOv8 đơn giản – tự động tắt nếu không dùng được."""
    def __init__(self, model_name="yolov8n.pt", conf_threshold=0.35):
        self.enabled = False
        self.model = None
        self.device = "cpu"
        self.conf_threshold = conf_threshold

        if not _YOLO_AVAILABLE:
            print("⚠️ ultralytics chưa được cài – fallback MOG2.")
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
            print("✅ YOLO đã sẵn sàng!")
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


# ======================
# CAMERA CLASSIFIER – dùng để xử lý video
# ======================

class CameraClassifier:
    def __init__(self, model_path):
        self.classifier = WasteClassifier(model_path)

    def classify_video_file(self, video_path, output_path=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Không thể mở video!")

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        frame_idx = 0
        temp_file = "_tmp_frame.jpg"

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            if frame_idx % 5 == 0:
                cv2.imwrite(temp_file, frame)
                try:
                    result = self.classifier.predict(temp_file, return_all=True)
                    label = f"{result['class']} {result['confidence']:.1f}%"
                    cv2.putText(frame, label, (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                except:
                    pass

            if writer is not None:
                writer.write(frame)

        cap.release()
        if writer is not None:
            writer.release()
        if os.path.exists(temp_file):
            os.remove(temp_file)


# ======================
# Modern Button – nút UI đẹp kiểu macOS
# ======================

class ModernButton(tk.Button):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.config(
            font=('SF Pro Text', 11, 'bold'),  # Tăng size từ 10→11
            relief='flat', bd=0,
            padx=20, pady=10,
            cursor='hand2',
            activebackground=kwargs.get('bg', '#007aff'),
            activeforeground='white',
            highlightthickness=0,
            fg='white'  # ← THÊM DÒNG NÀY
        )
        self.default_bg = kwargs.get('bg', '#007aff')
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)

    def on_enter(self, e):
        self['background'] = self._light(self.default_bg)

    def on_leave(self, e):
        self['background'] = self.default_bg

    def _light(self, color):
        lighten = {
            '#007aff': '#259bff',
            '#34c759': '#5cd96f',
            '#ff3b30': '#ff5e57',
            '#ffcc00': '#ffdd33',
            '#8e8e93': '#a3a3aa',
            '#5ac8fa': '#7fd8ff',
        }
        return lighten.get(color, color)


# ======================
# BẮT ĐẦU CLASS CHÍNH GUI
# ======================

class WasteClassifierGUIAdvanced:
    """GUI chính – PART 1 kết thúc ngay trước setup_ui()"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("🌿 Hệ Thống Phân Loại Rác Thải AI")

        # Kích thước window
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f"{sw}x{sh}")
        self.root.state("zoomed")

        # Màu sắc theme macOS
        self.colors = {
            'bg': '#f5f5f7',
            'sidebar': '#ffffff',
            'header': '#ffffff',
            'card': '#ffffff',
            'primary': '#007aff',
            'success': '#34c759',
            'danger': '#ff3b30',
            'warning': '#ffcc00',
            'info': '#5ac8fa',
            'secondary': '#8e8e93',
            'text': '#1c1c1e',
            'text_secondary': '#8e8e93',
            'border': '#d1d1d6'
        }

        self.root.configure(bg=self.colors['bg'])

        # Load model
        self.load_model()

        # Camera states
        self.cap = None
        self.camera_running = False
        self.auto_scan = False
        self.current_frame = None
        self.detected_bbox = None
        self.last_scan_time = 0
        self.scan_cooldown = 2.0

        # YOLO detector
        self.yolo = YOLODetector()

        # fallback detector
        self.object_detector = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True
        )

        # Data manager
        self.data_manager = DataManager()

        # Folder scan save
        self.data_save_dir = "scanned_data"
        os.makedirs(self.data_save_dir, exist_ok=True)
        for cls in CLASSES:
            os.makedirs(os.path.join(self.data_save_dir, cls), exist_ok=True)

        # Chuẩn bị giao diện
        self.setup_ui()

        # Lịch sử
        self.scan_history = []
        self.load_scan_history()


    # ======================
    # Load model
    # ======================
    def load_model(self):
        model_path = PATHS['model_save']
        print(f"📂 Load model: {model_path}")
        if not os.path.exists(model_path):
            model_path = PATHS['best_model']

        try:
            self.classifier = WasteClassifier(model_path)
            self.model_loaded = True
            print("✅ Model đã load!")
        except Exception as e:
            print(f"❌ Lỗi load model: {e}")
            self.model_loaded = False
    # ======================
    # Thiết lập giao diện chính
    # ======================
    def setup_ui(self):
        """Tạo header, sidebar, camera panel và result panel"""

        # ----- Header -----
        header = tk.Frame(self.root, bg=self.colors['header'], height=80)
        header.pack(fill='x')
        header.pack_propagate(False)

        # Line shadow dưới header
        shadow = tk.Frame(self.root, height=1, bg=self.colors['border'])
        shadow.pack(fill='x')

        # Nội dung header
        header_content = tk.Frame(header, bg=self.colors['header'])
        header_content.pack(fill='both', expand=True, padx=30)

        # Tiêu đề bên trái
        title_frame = tk.Frame(header_content, bg=self.colors['header'])
        title_frame.pack(side='left', pady=20)

        title = tk.Label(
            title_frame,
            text="🌿 Phân Loại Rác Thải Thông Minh",
            font=('SF Pro Display', 24, 'bold'),
            bg=self.colors['header'],
            fg=self.colors['primary']
        )
        title.pack(side='left')

        subtitle = tk.Label(
            title_frame,
            text="AI-Powered Waste Classification",
            font=('SF Pro Text', 11),
            bg=self.colors['header'],
            fg=self.colors['text_secondary']
        )
        subtitle.pack(side='left', padx=(15, 0))

        # Trạng thái model bên phải
        status_frame = tk.Frame(header_content, bg=self.colors['header'])
        status_frame.pack(side='right', pady=20)

        if getattr(self, "model_loaded", False):
            dot_color = self.colors['success']
            status_text = "Model Ready"
            status_color = self.colors['success']
        else:
            dot_color = self.colors['danger']
            status_text = "Model Not Found"
            status_color = self.colors['danger']

        status_dot = tk.Label(
            status_frame,
            text="●",
            font=('Arial', 20),
            bg=self.colors['header'],
            fg=dot_color
        )
        status_dot.pack(side='left')

        self.status_label = tk.Label(
            status_frame,
            text=status_text,
            font=('SF Pro Text', 12, 'bold'),
            bg=self.colors['header'],
            fg=status_color
        )
        self.status_label.pack(side='left', padx=(5, 0))

        # ----- Main container -----
        main = tk.Frame(self.root, bg=self.colors['bg'])
        main.pack(fill='both', expand=True, padx=20, pady=20)

        # Sidebar trái
        self.setup_left_sidebar(main)

        # Khung camera giữa
        self.setup_center_panel(main)

        # Khung kết quả bên phải
        self.setup_right_panel(main)

        # Cập nhật thống kê ban đầu
        self.update_statistics()


    # ======================
    # Sidebar menu bên trái
    # ======================
    def setup_left_sidebar(self, parent):
        sidebar_outer = tk.Frame(parent, bg=self.colors['bg'])
        sidebar_outer.pack(side='left', fill='y', padx=(0, 15))

        sidebar = tk.Frame(
            sidebar_outer,
            bg=self.colors['card'],
            highlightthickness=1,
            highlightbackground=self.colors['border'],
            bd=0,
            relief='flat'
        )
        sidebar.pack(fill='y', expand=False)
        sidebar.config(width=230)
        sidebar.pack_propagate(False)

        # Tiêu đề menu
        menu_title = tk.Label(
            sidebar,
            text="📋 MENU",
            font=('SF Pro Text', 14, 'bold'),
            bg=self.colors['card'],
            fg=self.colors['text']
        )
        menu_title.pack(pady=(20, 15))

        # Các nút menu
        buttons = [
            ("📷 Camera",        self.show_camera_mode,   self.colors['primary']),
            ("📸 Upload Ảnh",    self.upload_image,       self.colors['info']),
            ("📹 Xử Lý Video",   self.process_video,      '#6f42c1'),
            ("📁 Batch",         self.batch_classify,     self.colors['success']),
            ("🎓 Training",      self.show_training_panel,'#fd7e14'),
            ("🔄 Fine-tune",     self.incremental_training, '#6f42c1'),
            ("📊 Quản Lý Data",  self.show_data_management, self.colors['warning']),
            ("📈 Thống Kê",      self.show_statistics,    self.colors['info']),
            ("ℹ️ Hướng Dẫn",     self.show_guide,         self.colors['secondary']),
        ]

        for text, cmd, color in buttons:
            btn = ModernButton(
                sidebar,
                text=text,
                bg=color,
                fg='white',
                command=cmd,
                width=16
            )
            btn.pack(pady=6, padx=15)

        # Đệm ở dưới
        tk.Frame(sidebar, bg=self.colors['card']).pack(expand=True)

        # Nút Thoát
        btn_exit = ModernButton(
            sidebar,
            text="🚪 Thoát",
            bg=self.colors['secondary'],
            fg='white',
            command=self.on_closing,
            width=16
        )
        btn_exit.pack(pady=20, padx=15)


    # ======================
    # Khung camera ở giữa
    # ======================
    def setup_center_panel(self, parent):
        center_outer = tk.Frame(parent, bg=self.colors['bg'])
        center_outer.pack(side='left', fill='both', expand=True, padx=(0, 15))

        center = tk.Frame(
            center_outer,
            bg=self.colors['card'],
            highlightthickness=1,
            highlightbackground=self.colors['border'],
            bd=0,
            relief='flat'
        )
        center.pack(fill='both', expand=True)

        # Title
        title_label = tk.Label(
            center,
            text="📷 Camera Phát Hiện & Phân Loại",
            font=('SF Pro Text', 14, 'bold'),
            bg=self.colors['card'],
            fg=self.colors['text']
        )
        title_label.pack(pady=(15, 10), padx=20, anchor='w')

        separator = tk.Frame(center, height=1, bg=self.colors['border'])
        separator.pack(fill='x', padx=20)

        # Toggle Auto scan
        toggle_frame = tk.Frame(center, bg=self.colors['card'])
        toggle_frame.pack(fill='x', padx=20, pady=(10, 0))

        self.auto_scan_var = tk.BooleanVar(value=False)
        auto_check = tk.Checkbutton(
            toggle_frame,
            text="🤖 Tự động quét",
            variable=self.auto_scan_var,
            font=('SF Pro Text', 11, 'bold'),
            bg=self.colors['card'],
            fg=self.colors['text'],
            selectcolor=self.colors['card'],
            activebackground=self.colors['card'],
            activeforeground=self.colors['primary'],
            command=self.toggle_auto_scan
        )
        auto_check.pack(side='right')

        # Khung video
        video_container = tk.Frame(
            center,
            bg=self.colors['border'],
            bd=1,
            relief='flat'
        )
        video_container.pack(padx=20, pady=15, fill='both', expand=True)

        self.video_frame = tk.Label(video_container, bg='#000000')
        self.video_frame.pack(fill='both', expand=True, padx=1, pady=1)

        # Nút điều khiển camera
        control_frame = tk.Frame(center, bg=self.colors['card'])
        control_frame.pack(pady=(0, 20))

        self.btn_start_camera = ModernButton(
            control_frame,
            text="▶️ Bật Camera",
            bg=self.colors['success'],
            fg='white',
            width=13,
            command=self.toggle_camera
        )
        self.btn_start_camera.pack(side='left', padx=5)

        self.btn_scan = ModernButton(
            control_frame,
            text="📸 Scan",
            bg=self.colors['primary'],
            fg='white',
            width=13,
            command=self.manual_scan,
            state='disabled'
        )
        self.btn_scan.pack(side='left', padx=5)

        self.btn_save_frame = ModernButton(
            control_frame,
            text="💾 Lưu Frame",
            bg='#6f42c1',
            fg='white',
            width=13,
            command=self.save_current_frame,
            state='disabled'
        )
        self.btn_save_frame.pack(side='left', padx=5)


    # ======================
    # Khung kết quả bên phải
    # ======================
    def setup_right_panel(self, parent):
        right_outer = tk.Frame(parent, bg=self.colors['bg'])
        right_outer.pack(side='right', fill='both')

        right = tk.Frame(
            right_outer,
            bg=self.colors['card'],
            highlightthickness=1,
            highlightbackground=self.colors['border'],
            bd=0,
            relief='flat'
        )
        right.pack(fill='both', expand=True)
        right.config(width=520)
        right.pack_propagate(False)

        # Title
        title_label = tk.Label(
            right,
            text="📊 Kết Quả Phân Loại",
            font=('SF Pro Text', 14, 'bold'),
            bg=self.colors['card'],
            fg=self.colors['text']
        )
        title_label.pack(pady=(15, 10), padx=20, anchor='w')

        separator = tk.Frame(right, height=1, bg=self.colors['border'])
        separator.pack(fill='x', padx=20)

        # Khung text kết quả
        result_container = tk.Frame(
            right,
            bg=self.colors['border'],
            bd=1,
            relief='flat'
        )
        result_container.pack(fill='both', expand=True, padx=20, pady=(10, 15))

        self.result_text = scrolledtext.ScrolledText(
            result_container,
            font=('SF Mono', 12),  # Tăng từ 11→12
            bg='#ffffff',  # Đổi từ '#f5f5f7' → trắng
            fg='#1c1c1e',  # Màu chữ đậm hơn
            wrap='word',
            relief='flat',
            bd=0,
            state='disabled',
            padx=15,
            pady=15
        )
        self.result_text.pack(fill='both', expand=True, padx=1, pady=1)

        # Tag style
        self.result_text.tag_config('header', font=('SF Pro Text', 13, 'bold'),
                                    foreground=self.colors['primary'])
        self.result_text.tag_config('success', foreground=self.colors['success'])
        self.result_text.tag_config('warning', foreground=self.colors['warning'])
        self.result_text.tag_config('info', foreground=self.colors['info'])
        self.result_text.tag_config('bold', font=('SF Mono', 11, 'bold'))

       # Nút action
        action_frame = tk.Frame(right, bg=self.colors['card'])
        action_frame.pack(pady=(0, 15))

        self.btn_save = ModernButton(
            action_frame,
            text="💾 Lưu Kết Quả",
            bg='#6f42c1',
            fg='white',
            width=16,
            command=self.save_scan_result,
            state='disabled'
        )
        self.btn_save.pack(side='left', padx=5)

        btn_history = ModernButton(
            action_frame,
            text="📜 Lịch Sử",
            bg=self.colors['secondary'],
            fg='white',
            width=16,
            command=self.show_history
        )
        btn_history.pack(side='left', padx=5)

        # Thống kê nhanh
        stats_card = tk.Frame(right, bg='#e7f0ff', bd=0, relief='flat')
        stats_card.pack(fill='x', padx=20, pady=(0, 20))

        stats_title = tk.Label(
            stats_card,
            text="📈 Thống Kê Nhanh",
            font=('SF Pro Text', 12, 'bold'),
            bg='#e7f0ff',
            fg=self.colors['primary']
        )
        stats_title.pack(pady=(12, 8), padx=15, anchor='w')

        self.stats_label = tk.Label(
            stats_card,
            text="Chưa có dữ liệu",
            font=('SF Pro Text', 10),
            bg='#e7f0ff',
            fg=self.colors['text'],
            justify='left',
            anchor='w'
        )
        self.stats_label.pack(padx=15, pady=(0, 12), anchor='w')
    # =========================================================
    # CAMERA CONTROL
    # =========================================================
    def toggle_camera(self):
        """Bật hoặc tắt camera"""
        if not self.camera_running:
            self.start_camera()
        else:
            self.stop_camera()

    def start_camera(self):
        """Khởi động camera"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Lỗi", "Không thể mở camera!")
            return

        # Set độ phân giải cao
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.camera_running = True
        self.btn_start_camera.config(text="⏹️ Tắt Camera", bg=self.colors['danger'])
        self.btn_scan.config(state='normal')
        self.btn_save_frame.config(state='normal')

        self.update_camera()

    def stop_camera(self):
        """Dừng camera"""
        self.camera_running = False
        self.auto_scan = False
        self.auto_scan_var.set(False)

        if self.cap:
            self.cap.release()

        self.btn_start_camera.config(text="▶️ Bật Camera", bg=self.colors['success'])
        self.btn_scan.config(state='disabled')
        self.btn_save_frame.config(state='disabled')
        self.video_frame.config(image='')


    # =========================================================
    # AUTO SCAN TOGGLE
    # =========================================================
    def toggle_auto_scan(self):
        self.auto_scan = self.auto_scan_var.get()
        if self.auto_scan:
            print("🤖 Bật auto scan")
        else:
            print("⏸️ Tắt auto scan")


    # =========================================================
    # OBJECT DETECTION (YOLO → MOG2 fallback)
    # =========================================================
    def detect_object(self, frame):
        """
        1️⃣ Nếu có YOLO → dùng YOLO
        2️⃣ Nếu YOLO fail → dùng MOG2 như bản gốc
        """

        # Thử YOLO trước
        if hasattr(self, "yolo") and getattr(self.yolo, "enabled", False):
            bbox = self.yolo.detect_single_object(frame)
            if bbox:
                return bbox

        # ----- Fallback: MOG2 -----
        fg_mask = self.object_detector.apply(frame)
        fg_mask[fg_mask == 127] = 0  # remove shadows

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area < 5000:
            return None

        x, y, w, h = cv2.boundingRect(largest)
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio < 0.2 or aspect_ratio > 5:
            return None

        return (x, y, w, h)


    # =========================================================
    # DRAW BOUNDING BOX
    # =========================================================
    def draw_detection_box(self, frame, bbox):
        """Vẽ khung xanh kiểu iOS/macOS"""
        if bbox is None:
            return frame

        x, y, w, h = bbox
        color = (0, 200, 100)  # xanh đẹp
        t = 3
        L = 35

        # 4 góc
        # Top-left
        cv2.line(frame, (x, y), (x + L, y), color, t)
        cv2.line(frame, (x, y), (x, y + L), color, t)

        # Top-right
        cv2.line(frame, (x + w, y), (x + w - L, y), color, t)
        cv2.line(frame, (x + w, y), (x + w, y + L), color, t)

        # Bottom-left
        cv2.line(frame, (x, y + h), (x + L, y + h), color, t)
        cv2.line(frame, (x, y + h), (x, y + h - L), color, t)

        # Bottom-right
        cv2.line(frame, (x + w, y + h), (x + w - L, y + h), color, t)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - L), color, t)

        return frame


    # =========================================================
    # HIỂN THỊ FRAME LÊN GUI
    # =========================================================
    def _display_frame(self, frame):
        """Resize + Convert → Tkinter"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        display_h = 480
        ratio = img.width / img.height
        display_w = int(display_h * ratio)

        img = img.resize((display_w, display_h), Image.Resampling.LANCZOS)

        imgtk = ImageTk.PhotoImage(image=img)
        self.video_frame.imgtk = imgtk
        self.video_frame.configure(image=imgtk)


    # =========================================================
    # UPDATE CAMERA FRAME
    # =========================================================
    def update_camera(self):
        if not self.camera_running:
            return

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            self.current_frame = frame.copy()

            # 1. Detect
            bbox = self.detect_object(frame)

            if bbox:
                self.detected_bbox = bbox
                frame = self.draw_detection_box(frame, bbox)

                # 2. Auto scan
                if self.auto_scan and self.model_loaded:
                    now = datetime.now().timestamp()
                    if now - self.last_scan_time > self.scan_cooldown:
                        self.auto_classify(bbox)
                        self.last_scan_time = now

            else:
                self.detected_bbox = None

            # 3. Status overlay
            status_text = "AUTO SCAN: ON" if self.auto_scan else "MANUAL MODE"
            color = (0, 200, 100) if self.auto_scan else (130, 130, 130)

            (tw, th), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (15, 15), (tw + 35, th + 35), color, -1)
            cv2.putText(frame, status_text, (25, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)

            # 4. Hiển thị frame
            self._display_frame(frame)

        self.root.after(10, self.update_camera)


    # =========================================================
    # SCAN THỦ CÔNG
    # =========================================================
    def manual_scan(self):
        if self.current_frame is None:
            return

        if self.detected_bbox:
            x, y, w, h = self.detected_bbox
            cropped = self.current_frame[y:y+h, x:x+w]
        else:
            # Không thấy vật → cắt khung giữa
            h, w = self.current_frame.shape[:2]
            cx, cy = w // 2, h // 2
            box = 350
            cropped = self.current_frame[
                cy - box//2 : cy + box//2,
                cx - box//2 : cx + box//2
            ]

        temp_path = "temp_manual.jpg"
        cv2.imwrite(temp_path, cropped)
        self.classify_image(temp_path, cropped)


    # =========================================================
    # AUTO SCAN → CLASSIFY
    # =========================================================
    def auto_classify(self, bbox):
        if self.current_frame is None:
            return

        x, y, w, h = bbox
        cropped = self.current_frame[y:y+h, x:x+w]

        if cropped.size == 0:
            return

        temp_path = "temp_auto.jpg"
        cv2.imwrite(temp_path, cropped)

        # chạy thread để không lag UI
        threading.Thread(
            target=self.classify_image_async,
            args=(temp_path, cropped, True),
            daemon=True
        ).start()


    # =========================================================
    # SAVE CURRENT FRAME
    # =========================================================
    def save_current_frame(self):
        if self.current_frame is None:
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"frame_{ts}.jpg"
        cv2.imwrite(filename, self.current_frame)
        messagebox.showinfo("Thành công", f"Đã lưu: {filename}")
    # =========================================================
    # CLASSIFY ASYNC (AUTO SCAN)
    # =========================================================
    def classify_image_async(self, image_path, original_image, is_auto):
        """Phân loại trong thread để UI không bị đứng"""
        try:
            result = self.classifier.predict(image_path, return_all=True)

            self.current_result = {
                'image_path': image_path,
                'image': original_image,
                'result': result,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'is_auto': is_auto
            }

            self.root.after(0, lambda: self.display_result(result))
            self.root.after(0, lambda: self.btn_save.config(state='normal'))

        except Exception as e:
            print(f"❌ AUTO classify lỗi: {e}")


    # =========================================================
    # CLASSIFY MANUAL
    # =========================================================
    def classify_image(self, image_path, original_image):
        """Phân loại thủ công"""
        if self.classifier is None:
            messagebox.showerror("Lỗi", "Model chưa load!")
            return

        try:
            result = self.classifier.predict(image_path, return_all=True)

            self.current_result = {
                'image_path': image_path,
                'image': original_image,
                'result': result,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'is_auto': False
            }

            self.display_result(result)
            self.btn_save.config(state='normal')

        except Exception as e:
            messagebox.showerror("Lỗi", f"Phân loại lỗi: {str(e)}")


    # =========================================================
    # DISPLAY RESULT TO RIGHT PANEL
    # =========================================================
    def display_result(self, result):
        """Hiển thị kết quả phân loại với UI đẹp"""

        predicted_class = result['class']
        confidence = result['confidence']
        info = CLASS_INFO[predicted_class]

        # Reset text
        self.result_text.config(state='normal')
        self.result_text.delete(1.0, tk.END)

        # ---- Header ----
        self.result_text.insert(tk.END, f"\n{info['icon']}  ", 'header')
        self.result_text.insert(tk.END, f"{info['name_vi'].upper()}\n", 'header')
        self.result_text.insert(tk.END, f"({predicted_class})\n\n", 'info')

        # ---- Confidence ----
        self.result_text.insert(tk.END, "🎯 Độ Tin Cậy: ", 'bold')

        if result['is_confident']:
            self.result_text.insert(tk.END, f"{confidence:.1f}%  ✅\n", 'success')
        else:
            self.result_text.insert(tk.END, f"{confidence:.1f}%  ⚠️\n", 'warning')

        # Progress bar (ASCII)
        bar_len = int(confidence / 2)
        bar = "█" * bar_len + "░" * (50 - bar_len)
        self.result_text.insert(tk.END, f"{bar}\n\n")

        # ---- Hướng dẫn xử lý ----
        self.result_text.insert(tk.END, "♻️  Cách xử lý:\n", 'bold')
        self.result_text.insert(tk.END, f"   {info['disposal']}\n\n")

        # ---- Ví dụ ----
        self.result_text.insert(tk.END, "📝 Ví dụ:\n", 'bold')
        self.result_text.insert(tk.END, f"   {', '.join(info['examples'])}\n\n")

        # ---- Giá trị tái chế ----
        self.result_text.insert(tk.END, "💰 Giá trị tái chế: ", 'bold')
        self.result_text.insert(tk.END, f"{info['recycling_value']}\n\n")

        # ---- Separator ----
        self.result_text.insert(tk.END, "─" * 55 + "\n\n")

        # ---- Detailed Probabilities ----
        self.result_text.insert(tk.END, "📊 Chi Tiết Các Xác Suất:\n\n", 'bold')

        sorted_preds = sorted(
            result['all_predictions'].items(),
            key=lambda x: x[1],
            reverse=True
        )

        for cls, prob in sorted_preds:
            icon = CLASS_INFO[cls]['icon']
            nr_len = int(prob / 3)
            bar = "█" * nr_len
            self.result_text.insert(tk.END, f"{icon} {cls:11s} ")
            self.result_text.insert(tk.END, f"{bar:33s} {prob:5.1f}%\n")

        self.result_text.config(state='disabled')


    # =========================================================
    # UPLOAD IMAGE
    # =========================================================
    def upload_image(self):
        if not self.model_loaded:
            messagebox.showerror("Lỗi", "Model chưa load!")
            return

        file_path = filedialog.askopenfilename(
            title="Chọn ảnh",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )

        if not file_path:
            return

        img = cv2.imread(file_path)
        if img is None:
            messagebox.showerror("Lỗi", "Không thể đọc ảnh!")
            return

        # Hiển thị lên khung camera
        self.current_frame = img.copy()
        self.detected_bbox = None
        self._display_frame(img)

        # Phân loại ảnh
        self.classify_image(file_path, img)


    # =========================================================
    # PROCESS VIDEO
    # =========================================================
    def process_video(self):
        if not self.model_loaded:
            messagebox.showerror("Lỗi", "Model chưa load!")
            return

        video_path = filedialog.askopenfilename(
            title="Chọn video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )

        if not video_path:
            return

        # Hiển thị frame đầu tiên
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                self.current_frame = frame.copy()
                self.detected_bbox = None
                self._display_frame(frame)
            cap.release()

        save_output = messagebox.askyesno("Lưu video?", "Bạn muốn lưu video output không?")
        output_path = None

        if save_output:
            output_path = filedialog.asksaveasfilename(
                title="Lưu video",
                defaultextension=".mp4",
                filetypes=[("MP4 files", "*.mp4")]
            )

        threading.Thread(
            target=self.process_video_thread,
            args=(video_path, output_path),
            daemon=True
        ).start()


    def process_video_thread(self, video_path, output_path):
        """Chạy xử lý video thread"""
        try:
            cam_classifier = CameraClassifier(PATHS['model_save'])
            cam_classifier.classify_video_file(video_path, output_path)

            self.root.after(0, lambda:
                messagebox.showinfo("Thành công", "✔ Đã xử lý video!")
            )
        except Exception as e:
            self.root.after(0, lambda:
                messagebox.showerror("Lỗi", f"Lỗi xử lý video: {str(e)}")
            )


    # =========================================================
    # BATCH CLASSIFY
    # =========================================================
    def batch_classify(self):
        if not self.model_loaded:
            messagebox.showerror("Lỗi", "Model chưa load!")
            return

        folder = filedialog.askdirectory(title="Chọn thư mục chứa ảnh")

        if not folder:
            return

        exts = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        image_files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in exts
        ]

        if not image_files:
            messagebox.showwarning("Không có ảnh", "Thư mục không chứa ảnh!")
            return

        threading.Thread(
            target=self.batch_classify_thread,
            args=(image_files,),
            daemon=True
        ).start()


    def batch_classify_thread(self, image_files):
        try:
            results = self.classifier.predict_batch(image_files)
            self.root.after(0, lambda: self.show_batch_results(results))
        except Exception as e:
            self.root.after(0, lambda:
                messagebox.showerror("Lỗi", f"Batch classify lỗi: {str(e)}")
            )
    # =========================================================
    # SAVE CURRENT FRAME
    # =========================================================
    def save_current_frame(self):
        """Lưu frame hiện tại (ảnh camera/ảnh video đang hiển thị)"""
        if self.current_frame is None:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"frame_{timestamp}.jpg"
        cv2.imwrite(filename, self.current_frame)
        messagebox.showinfo("Thành công", f"✅ Đã lưu: {filename}")

    # =========================================================
    # SAVE SCAN RESULT
    # =========================================================
    def save_scan_result(self):
        """Lưu ảnh đã scan + metadata JSON + update history"""
        if not hasattr(self, 'current_result'):
            return

        result = self.current_result['result']
        predicted_class = result['class']
        confidence = result['confidence']

        # Cảnh báo nếu độ tin cậy thấp
        if confidence < 80:
            ok = messagebox.askyesno(
                "Xác nhận",
                f"Độ tin cậy thấp ({confidence:.1f}%).\nBạn có chắc muốn lưu không?"
            )
            if not ok:
                return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{predicted_class}_{timestamp}_{confidence:.0f}.jpg"
        save_path = os.path.join(self.data_save_dir, predicted_class, filename)

        # Lưu ảnh
        cv2.imwrite(save_path, self.current_result['image'])

        # Lưu metadata
        metadata = {
            'class': predicted_class,
            'confidence': confidence,
            'timestamp': self.current_result['timestamp'],
            'all_predictions': result['all_predictions'],
            'is_auto_scan': self.current_result.get('is_auto', False)
        }

        json_path = save_path.replace('.jpg', '.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # Cập nhật history + thống kê
        self.scan_history.append(metadata)
        self.save_scan_history()
        self.update_statistics()

        messagebox.showinfo("Thành công", f"✅ Đã lưu kết quả!\n\n{save_path}")
        self.btn_save.config(state='disabled')

    # =========================================================
    # BATCH RESULT UI + SAVE CSV
    # =========================================================
    def show_batch_results(self, results):
        """Cửa sổ hiển thị kết quả batch (nhiều ảnh)"""
        window = tk.Toplevel(self.root)
        window.title("📁 Kết Quả Phân Loại Batch")
        window.geometry("1100x750")
        window.configure(bg=self.colors['bg'])

        # Header
        header = tk.Frame(window, bg=self.colors['card'], height=70)
        header.pack(fill='x')
        header.pack_propagate(False)

        tk.Label(
            header,
            text="📊 Kết Quả Phân Loại Batch",
            font=('SF Pro Display', 20, 'bold'),
            bg=self.colors['card'],
            fg=self.colors['primary']
        ).pack(pady=20)

        # Treeview
        frame = tk.Frame(window, bg=self.colors['bg'])
        frame.pack(fill='both', expand=True, padx=30, pady=20)

        columns = ('STT', 'File', 'Loại', 'Confidence', 'Status')
        tree = ttk.Treeview(frame, columns=columns, show='headings', height=22)

        for col in columns:
            tree.heading(col, text=col)

        tree.column('STT', width=60)
        tree.column('File', width=400)
        tree.column('Loại', width=250)
        tree.column('Confidence', width=130)
        tree.column('Status', width=100)

        for i, item in enumerate(results, 1):
            result = item['result']
            filename = os.path.basename(item['image'])
            icon = CLASS_INFO[result['class']]['icon']
            status = "✅ Cao" if result['is_confident'] else "⚠️ Thấp"

            tree.insert('', 'end', values=(
                i,
                filename,
                f"{icon} {result['class_name_vi']}",
                f"{result['confidence']:.1f}%",
                status
            ))

        scrollbar = ttk.Scrollbar(frame, orient='vertical', command=tree.yview)
        tree.configure(yscroll=scrollbar.set)

        tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        # Buttons
        btn_frame = tk.Frame(window, bg=self.colors['bg'])
        btn_frame.pack(pady=20)

        ModernButton(
            btn_frame,
            text="💾 Lưu CSV",
            bg=self.colors['primary'],
            fg='white',
            command=lambda: self.save_batch_csv(results)
        ).pack(side='left', padx=10)

        ModernButton(
            btn_frame,
            text="🚪 Đóng",
            bg=self.colors['secondary'],
            fg='white',
            command=window.destroy
        ).pack(side='left', padx=10)

    def save_batch_csv(self, results):
        """Lưu kết quả batch ra file CSV"""
        file_path = filedialog.asksaveasfilename(
            title="Lưu CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )

        if not file_path:
            return

        try:
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['STT', 'File', 'Class', 'Class_VI', 'Confidence', 'Status'])

                for i, item in enumerate(results, 1):
                    result = item['result']
                    status = "High" if result['is_confident'] else "Low"
                    writer.writerow([
                        i,
                        os.path.basename(item['image']),
                        result['class'],
                        result['class_name_vi'],
                        f"{result['confidence']:.2f}",
                        status
                    ])

            messagebox.showinfo("Thành công", f"✅ Đã lưu: {file_path}")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi lưu CSV: {str(e)}")

    # =========================================================
    # TRAINING PANEL
    # =========================================================
    def show_training_panel(self):
        """Cửa sổ cấu hình training model mới"""
        window = tk.Toplevel(self.root)
        window.title("🎓 Training Model")
        window.geometry("900x700")
        window.configure(bg=self.colors['bg'])

        # Header
        header = tk.Frame(window, bg=self.colors['card'], height=80)
        header.pack(fill='x')
        header.pack_propagate(False)

        tk.Label(
            header,
            text="🎓 Training Model Mới",
            font=('SF Pro Display', 22, 'bold'),
            bg=self.colors['card'],
            fg=self.colors['primary']
        ).pack(pady=25)

        # Form
        form_outer = tk.Frame(window, bg=self.colors['bg'])
        form_outer.pack(fill='both', expand=True, padx=30, pady=20)

        form = tk.Frame(
            form_outer,
            bg=self.colors['card'],
            relief='flat',
            bd=0,
            highlightthickness=1,
            highlightbackground=self.colors['border']
        )
        form.pack(fill='both', expand=True)

        # Train dir
        tk.Label(
            form,
            text="📁 Thư mục Training:",
            font=('SF Pro Text', 12),
            bg=self.colors['card'],
            fg=self.colors['text']
        ).pack(pady=(20, 5), anchor='w', padx=30)

        train_frame = tk.Frame(form, bg=self.colors['card'])
        train_frame.pack(fill='x', padx=30, pady=5)

        train_entry = tk.Entry(
            train_frame,
            font=('SF Pro Text', 11),
            width=60,
            relief='solid',
            bd=1
        )
        train_entry.pack(side='left', ipady=8, padx=(0, 10))

        ModernButton(
            train_frame,
            text="Browse",
            bg=self.colors['info'],
            fg='white',
            command=lambda: train_entry.insert(0, filedialog.askdirectory())
        ).pack()

        # Val dir
        tk.Label(
            form,
            text="📁 Thư mục Validation:",
            font=('SF Pro Text', 12),
            bg=self.colors['card'],
            fg=self.colors['text']
        ).pack(pady=(15, 5), anchor='w', padx=30)

        val_frame = tk.Frame(form, bg=self.colors['card'])
        val_frame.pack(fill='x', padx=30, pady=5)

        val_entry = tk.Entry(
            val_frame,
            font=('SF Pro Text', 11),
            width=60,
            relief='solid',
            bd=1
        )
        val_entry.pack(side='left', ipady=8, padx=(0, 10))

        ModernButton(
            val_frame,
            text="Browse",
            bg=self.colors['info'],
            fg='white',
            command=lambda: val_entry.insert(0, filedialog.askdirectory())
        ).pack()

        # Epochs
        tk.Label(
            form,
            text="⏱️ Số Epochs:",
            font=('SF Pro Text', 12),
            bg=self.colors['card'],
            fg=self.colors['text']
        ).pack(pady=(15, 5), anchor='w', padx=30)

        epochs_entry = tk.Entry(
            form,
            font=('SF Pro Text', 11),
            width=20,
            relief='solid',
            bd=1
        )
        epochs_entry.insert(0, "50")
        epochs_entry.pack(anchor='w', padx=30, pady=5, ipady=8)

        # Transfer learning
        transfer_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            form,
            text="🔄 Sử dụng Transfer Learning",
            variable=transfer_var,
            font=('SF Pro Text', 12),
            bg=self.colors['card'],
            fg=self.colors['text'],
            selectcolor=self.colors['card'],
            activebackground=self.colors['card']
        ).pack(pady=20, anchor='w', padx=30)

        # Button start
        ModernButton(
            form,
            text="🚀 Bắt Đầu Training",
            bg=self.colors['success'],
            fg='white',
            width=25,
            command=lambda: self.start_training(
                train_entry.get(),
                val_entry.get(),
                int(epochs_entry.get()),
                transfer_var.get(),
                window
            )
        ).pack(pady=30)

    def start_training(self, train_dir, val_dir, epochs, use_transfer, window):
        """Validate folder và start thread training"""
        if not os.path.exists(train_dir) or not os.path.exists(val_dir):
            messagebox.showerror("Lỗi", "Thư mục không tồn tại!")
            return

        window.destroy()

        threading.Thread(
            target=self.training_thread,
            args=(train_dir, val_dir, epochs, use_transfer),
            daemon=True
        ).start()

        messagebox.showinfo("Training", "Training đã bắt đầu!\nXem log ở console.")

    def training_thread(self, train_dir, val_dir, epochs, use_transfer):
        """Thread chạy training"""
        try:
            model, history = train_model(
                train_dir,
                val_dir,
                epochs=epochs,
                use_transfer_learning=use_transfer
            )
            plot_training_history(history)
            self.load_model()

            self.root.after(
                0,
                lambda: messagebox.showinfo("Thành công", "✅ Training hoàn tất!")
            )
        except Exception as e:
            self.root.after(
                0,
                lambda: messagebox.showerror("Lỗi", f"Lỗi training: {str(e)}")
            )

    # =========================================================
    # INCREMENTAL TRAINING
    # =========================================================
    def incremental_training(self):
        """Kiểm tra dữ liệu và bắt đầu incremental training (fine-tune)"""
        trainer = IncrementalTrainer()
        ready, stats = trainer.check_data_ready()

        if not ready:
            msg = "❌ Dữ liệu chưa đủ!\n\nCần ít nhất 20 mẫu chất lượng cao/class.\n\n"
            for cls, data in stats['by_class'].items():
                msg += f"{cls}: {data['high_confidence']} mẫu\n"
            messagebox.showwarning("Cảnh báo", msg)
            return

        if messagebox.askyesno(
            "Xác nhận",
            f"✅ Dữ liệu sẵn sàng!\n\nTổng: {stats['total']} mẫu\nBắt đầu training?"
        ):
            threading.Thread(
                target=self.incremental_training_thread,
                args=(trainer,),
                daemon=True
            ).start()
            messagebox.showinfo("Training", "Incremental training đã bắt đầu!")

    def incremental_training_thread(self, trainer):
        """Thread incremental training"""
        try:
            trainer.prepare_incremental_data()
            model, history = trainer.train_incremental(epochs=20, fine_tune=True)
            self.load_model()

            self.root.after(
                0,
                lambda: messagebox.showinfo("Thành công", "✅ Incremental training hoàn tất!")
            )
        except Exception as e:
            self.root.after(
                0,
                lambda: messagebox.showerror("Lỗi", f"Lỗi incremental training: {e}")
            )
    # =========================================================
    # SCAN HISTORY
    # =========================================================
    def load_scan_history(self):
        self.history_file = "scan_history.json"
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r', encoding='utf-8') as f:
                self.scan_history = json.load(f)
        else:
            self.scan_history = []

    def save_scan_history(self):
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(self.scan_history, f, indent=2, ensure_ascii=False)

    def show_history(self):
        """Hiển thị lịch sử scan"""
        if not self.scan_history:
            messagebox.showinfo("Lịch sử", "Chưa có dữ liệu scan nào!")
            return
        
        window = tk.Toplevel(self.root)
        window.title("📜 Lịch Sử Scan")
        window.geometry("900x600")
        window.configure(bg=self.colors['bg'])
        
        tk.Label(
            window,
            text="📜 Lịch Sử Scan",
            font=('SF Pro Display', 22, 'bold'),
            bg=self.colors['bg'],
            fg=self.colors['primary']
        ).pack(pady=20)
        
        # Treeview
        frame = tk.Frame(window, bg=self.colors['bg'])
        frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        columns = ('STT', 'Thời gian', 'Loại', 'Confidence', 'Mode')
        tree = ttk.Treeview(frame, columns=columns, show='headings', height=20)
        
        tree.heading('STT', text='STT')
        tree.heading('Thời gian', text='Thời gian')
        tree.heading('Loại', text='Loại rác')
        tree.heading('Confidence', text='Độ tin cậy')
        tree.heading('Mode', text='Chế độ')
        
        tree.column('STT', width=60)
        tree.column('Thời gian', width=180)
        tree.column('Loại', width=250)
        tree.column('Confidence', width=120)
        tree.column('Mode', width=120)
        
        for i, item in enumerate(reversed(self.scan_history[-50:]), 1):
            cls = item['class']
            icon = CLASS_INFO[cls]['icon']
            mode = "🤖 Auto" if item.get('is_auto_scan', False) else "👤 Manual"
            
            tree.insert('', 'end', values=(
                i,
                item['timestamp'],
                f"{icon} {cls}",
                f"{item['confidence']:.1f}%",
                mode
            ))
        
        scrollbar = ttk.Scrollbar(frame, orient='vertical', command=tree.yview)
        tree.configure(yscroll=scrollbar.set)
        tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        ModernButton(
            window,
            text="🚪 Đóng",
            bg=self.colors['secondary'],
            fg='white',
            command=window.destroy
        ).pack(pady=20)

    
    # =========================================================
    # STATISTICS PANEL
    # =========================================================
    def show_statistics(self):
        """Thống kê số lượng ảnh theo từng loại"""
        window = tk.Toplevel(self.root)
        window.title("📊 Thống Kê")
        window.geometry("800x600")
        window.configure(bg=self.colors['bg'])

        # header
        tk.Label(
            window,
            text="📊 Thống kê dữ liệu đã lưu",
            font=("SF Pro Display", 22, "bold"),
            bg=self.colors['bg']
        ).pack(pady=20)

        counts = {cls: 0 for cls in CLASSES}

        for item in self.scan_history:
            c = item['class']
            if c in counts:
                counts[c] += 1

        frame = tk.Frame(window, bg=self.colors['bg'])
        frame.pack(pady=10)

        for cls in CLASSES:
            info = CLASS_INFO[cls]
            text = f"{info['icon']} {cls}: {counts[cls]} ảnh"
            tk.Label(
                frame,
                text=text,
                font=("SF Pro Text", 14),
                bg=self.colors['bg'],
                fg=self.colors['text']
            ).pack(anchor='w', pady=4)

    # =========================================================
    # DATA MANAGER PANEL
    # =========================================================
    def show_data_manager(self):
        """Quản lý thư mục dữ liệu"""
        window = tk.Toplevel(self.root)
        window.title("🗂️ Quản Lý Dữ Liệu")
        window.geometry("900x650")
        window.configure(bg=self.colors['bg'])

        tk.Label(
            window,
            text="🗂️ Quản lý dữ liệu đã lưu",
            font=("SF Pro Display", 22, "bold"),
            bg=self.colors['bg']
        ).pack(pady=25)

        listbox = tk.Listbox(
            window,
            font=("SF Pro Text", 13),
            width=70,
            height=20
        )
        listbox.pack(pady=10)

        # load file
        dataset_dir = self.data_save_dir
        all_files = []

        for cls in CLASSES:
            cls_path = os.path.join(dataset_dir, cls)
            for f in os.listdir(cls_path):
                if f.endswith(".jpg"):
                    full = os.path.join(cls_path, f)
                    all_files.append(full)
                    listbox.insert(tk.END, full)

        # nút xóa
        def delete_selected():
            sel = listbox.curselection()
            if not sel:
                return
            path = listbox.get(sel[0])

            if messagebox.askyesno("Xóa?", f"Bạn có chắc muốn xóa?\n{path}"):
                os.remove(path)
                json_path = path.replace(".jpg", ".json")
                if os.path.exists(json_path):
                    os.remove(json_path)

                listbox.delete(sel[0])
                messagebox.showinfo("Thành công", "Đã xóa file.")

        ModernButton(
            window,
            text="🗑️ Xóa File",
            bg=self.colors['danger'],
            fg='white',
            command=delete_selected
        ).pack(pady=15)

    # =========================================================
    # SETUP UI (SIDEBAR + MAIN LAYOUT)
    # =========================================================
    def setup_ui(self):
        """Khởi tạo layout tổng thể"""

        main = tk.Frame(self.root, bg=self.colors['bg'])
        main.pack(fill='both', expand=True)

        self.setup_left_sidebar(main)
        self.setup_center_panel(main)
        self.setup_right_panel(main)
    def show_data_management(self):
        """Quản lý dữ liệu"""
        window = tk.Toplevel(self.root)
        window.title("📊 Quản Lý Dữ Liệu")
        window.geometry("1000x750")
        window.configure(bg=self.colors['bg'])
        
        # Header
        header = tk.Frame(window, bg=self.colors['card'], height=80)
        header.pack(fill='x')
        header.pack_propagate(False)
        
        tk.Label(header, text="📊 Quản Lý Dữ Liệu Training",
                font=('Segoe UI', 22, 'bold'),
                bg=self.colors['card'], fg=self.colors['primary']).pack(pady=25)
        
        # Stats
        stats = self.data_manager.get_scanned_stats()
        
        overview = f"""
📈 TỔNG QUAN
{'─'*70}
Tổng số mẫu: {stats['total']}
Chất lượng cao (≥80%): {stats['high_confidence']}
Tỷ lệ: {stats['high_confidence']/stats['total']*100 if stats['total'] > 0 else 0:.1f}%

📋 CHI TIẾT THEO CLASS
{'─'*70}
"""
        
        for cls in CLASSES:
            data = stats['by_class'][cls]
            icon = CLASS_INFO[cls]['icon']
            overview += f"{icon} {cls:12s}: {data['count']:4d} (Cao: {data['high_confidence']}, Thấp: {data['low_confidence']})\n"
        
        text = scrolledtext.ScrolledText(window, font=('Consolas', 11),
                                        bg='#f8f9fa', fg=self.colors['text'],
                                        wrap='word', height=20, relief='solid', bd=1)
        text.pack(fill='both', expand=True, padx=30, pady=20)
        text.insert(1.0, overview)
        text.config(state='disabled')
        
        # Buttons
        btn_frame = tk.Frame(window, bg=self.colors['bg'])
        btn_frame.pack(pady=20)
        
        ModernButton(btn_frame, text="📦 Chuẩn Bị", bg=self.colors['primary'], fg='white',
                    command=self.prepare_dataset).pack(side='left', padx=8)
        ModernButton(btn_frame, text="📤 Export", bg=self.colors['success'], fg='white',
                    command=self.export_high_quality).pack(side='left', padx=8)
        ModernButton(btn_frame, text="🗑️ Xóa", bg=self.colors['danger'], fg='white',
                    command=self.clean_low_quality).pack(side='left', padx=8)
    # =========================================================
    # LEFT SIDEBAR
    # =========================================================
    def setup_left_sidebar(self, parent):
        sidebar = tk.Frame(parent, bg=self.colors['sidebar'], width=240)
        sidebar.pack(side='left', fill='y')
        sidebar.pack_propagate(False)

        tk.Label(
            sidebar,
            text="♻️ AI Waste System",
            font=("SF Pro Display", 17, "bold"),
            fg=self.colors['primary'],
            bg=self.colors['sidebar']
        ).pack(pady=25)

        menu_items = [
            ("📷 Camera", self.show_camera_mode, self.colors['primary']),
            ("🖼️ Ảnh", self.upload_image, self.colors['info']),
            ("🎥 Video", self.process_video, self.colors['warning']),
            ("📁 Batch", self.batch_classify, self.colors['success']),
            ("📊 Thống Kê", self.show_statistics, self.colors['info']),
            ("🗂️ Data", self.show_data_manager, self.colors['secondary']),
            ("🎓 Train", self.show_training_panel, self.colors['primary']),
            ("🔧 Fine-Tune", self.incremental_training, self.colors['success']),
            ("🚪 Thoát", self.on_closing, self.colors['danger']),
        ]

        for (text, cmd, color) in menu_items:
            ModernButton(
                sidebar,
                text=text,
                bg=color,
                fg="white",
                width=20,
                command=cmd
            ).pack(pady=10)

    def show_camera_mode(self):
        """Chuyển UI sang chế độ camera"""
        # Không làm gì thêm vì camera panel luôn ở giữa
        pass

    # =========================================================
    # CENTER PANEL – CAMERA VIEW + BUTTONS
    # =========================================================
    def setup_center_panel(self, parent):
        center = tk.Frame(parent, bg=self.colors['bg'])
        center.pack(side='left', fill='both', expand=True)

        # ảnh camera
        self.video_frame = tk.Label(center, bg=self.colors['bg'])
        self.video_frame.pack(pady=20)

        # nút điều khiển
        btn_frame = tk.Frame(center, bg=self.colors['bg'])
        btn_frame.pack()

        self.btn_start_camera = ModernButton(
            btn_frame,
            text="▶️ Bật Camera",
            bg=self.colors['success'],
            fg='white',
            width=20,
            command=self.toggle_camera
        )
        self.btn_start_camera.pack(side='left', padx=10)

        self.btn_scan = ModernButton(
            btn_frame,
            text="🔍 Scan",
            bg=self.colors['primary'],
            fg='white',
            width=20,
            command=self.manual_scan
        )
        self.btn_scan.pack(side='left', padx=10)
        self.btn_scan.config(state='disabled')

        self.btn_save_frame = ModernButton(
            btn_frame,
            text="💾 Lưu Frame",
            bg=self.colors['info'],
            fg='white',
            width=20,
            command=self.save_current_frame
        )
        self.btn_save_frame.pack(side='left', padx=10)
        self.btn_save_frame.config(state='disabled')

        # auto scan
        self.auto_scan_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            center,
            text="🤖 Auto Scan",
            variable=self.auto_scan_var,
            command=self.toggle_auto_scan,
            font=("SF Pro Text", 12),
            bg=self.colors['bg']
        ).pack(pady=10)

    def show_guide(self):
        """Hướng dẫn"""
        window = tk.Toplevel(self.root)
        window.title("ℹ️ Hướng Dẫn")
        window.geometry("1000x750")
        window.configure(bg=self.colors['bg'])
        
        # Header
        header = tk.Frame(window, bg=self.colors['card'], height=80)
        header.pack(fill='x')
        header.pack_propagate(False)
        
        tk.Label(header, text="📖 Hướng Dẫn Sử Dụng",
                font=('Segoe UI', 22, 'bold'),
                bg=self.colors['card'], fg=self.colors['primary']).pack(pady=25)
        
        guide = """
╔═══════════════════════════════════════════════════════════════╗
║                    🎯 HƯỚNG DẪN SỬ DỤNG                        ║
╚═══════════════════════════════════════════════════════════════╝

📷 CAMERA SCAN
──────────────────────────────────────────────────────────────
1. Nhấn "▶️ Bật Camera"
2. Đặt vật phẩm vào khung
3. Hệ thống tự động phát hiện và DI CHUYỂN KHUNG XANH
4. Bật "🤖 Tự động quét" để scan liên tục (mỗi 2 giây)
5. Hoặc nhấn "📸 Scan" để scan thủ công
6. Xem kết quả bên phải và lưu nếu cần

📸 UPLOAD & BATCH
──────────────────────────────────────────────────────────────
• Upload: Chọn 1 ảnh để phân loại
• Batch: Chọn thư mục nhiều ảnh, xem kết quả bảng, lưu CSV

🎓 TRAINING
──────────────────────────────────────────────────────────────
• Training: Train model mới từ dataset có sẵn
• Fine-tune: Cập nhật model với dữ liệu đã scan (≥20 mẫu/class)

📊 QUẢN LÝ DỮ LIỆU
──────────────────────────────────────────────────────────────
• Xem thống kê dữ liệu đã scan
• Chuẩn bị dataset (auto chia 80/20)
• Export dữ liệu chất lượng cao (≥90%)
• Xóa dữ liệu kém (≤60%)

💡 TIPS
──────────────────────────────────────────────────────────────
✓ Khung xanh tự động theo dõi vật thể
✓ Chỉ lưu ảnh confidence ≥80%
✓ Dùng Fine-tune để cải thiện model liên tục
✓ Auto scan cooldown 2 giây tránh spam

⚙️ YÊU CẦU HỆ THỐNG
──────────────────────────────────────────────────────────────
• Python 3.7+
• TensorFlow 2.x
• OpenCV
• Camera (cho real-time)

──────────────────────────────────────────────────────────────
Happy Classifying! 🌿
──────────────────────────────────────────────────────────────
"""
        
        text = scrolledtext.ScrolledText(
            window,
            font=('Consolas', 10),
            bg='#f8f9fa',
            fg=self.colors['text'],
            wrap='word',
            relief='solid',
            bd=1
        )
        text.pack(fill='both', expand=True, padx=30, pady=(0, 20))
        text.insert(1.0, guide)
        text.config(state='disabled')
        
        ModernButton(
            window,
            text="🚪 Đóng",
            bg=self.colors['secondary'],
            fg='white',
            command=window.destroy
        ).pack(pady=20)

    def update_statistics(self):
        """Cập nhật thống kê"""
        stats = {cls: 0 for cls in CLASSES}
        high_conf_count = 0
        auto_count = 0
        
        for item in self.scan_history:
            stats[item['class']] += 1
            if item['confidence'] >= 80:
                high_conf_count += 1
            if item.get('is_auto_scan', False):
                auto_count += 1
        
        total = len(self.scan_history)
        
        if total == 0:
            self.stats_label.config(text="Chưa có dữ liệu")
            return
        
        text = f"📊 Tổng: {total} lần scan\n"
        text += f"✅ Tin cậy cao: {high_conf_count}/{total}\n"
        text += f"🤖 Auto scan: {auto_count}/{total}\n\n"
        
        # Top 3 classes
        sorted_stats = sorted(stats.items(), key=lambda x: x[1], reverse=True)[:3]
        for cls, count in sorted_stats:
            if count > 0:
                icon = CLASS_INFO[cls]['icon']
                pct = (count / total * 100)
                text += f"{icon} {cls}: {count} ({pct:.0f}%)\n"
        
        self.stats_label.config(text=text)
    # =========================================================
    # RIGHT PANEL – RESULT
    # =========================================================
    def setup_right_panel(self, parent):
        right = tk.Frame(parent, bg=self.colors['card'], width=380)
        right.pack(side='right', fill='y')
        right.pack_propagate(False)

        tk.Label(
            right,
            text="📄 Kết Quả",
            font=("SF Pro Display", 18, "bold"),
            bg=self.colors['card']
        ).pack(pady=15)

        self.result_text = scrolledtext.ScrolledText(
            right,
            width=45,
            height=30,
            font=("SF Pro Text", 12),
            bg=self.colors['card']
        )
        self.result_text.pack(padx=15, pady=10)
        self.result_text.config(state='disabled')

    # =========================================================
    # CLOSE WINDOW
    # =========================================================
    def on_closing(self):
        self.stop_camera()
        self.save_scan_history()
        self.root.destroy()


# =========================================================
# MAIN FUNCTION
# =========================================================
def main():
    root = tk.Tk()
    app = WasteClassifierGUIAdvanced(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
