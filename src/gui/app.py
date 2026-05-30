"""
GUI hiện đại cho hệ thống phân loại rác – FIXED VERSION
File này gồm 6 phần (PART 1/6)
"""

# ==============================
# PART 1 – IMPORT + WRAPPER + CAMERA CLASSIFIER + ModernButton
# ==============================

import sys
import os

# Thêm thư mục gốc vào sys.path để có thể import module 'src'
# Giúp chạy được lệnh: python src/gui/app.py
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import cv2
from PIL import Image, ImageTk
import threading
import json
import csv
from datetime import datetime
import numpy as np

# Import classifier & training modules
from src.inference.classifier import WasteClassifier
from src.scripts.train import train_model, plot_training_history
from src.data.manager import DataManager
from src.scripts.incremental_train import IncrementalTrainer
from src.core.config import PATHS, CLASS_INFO, CLASSES, MODEL_CONFIG


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
            print("ultralytics chưa được cài – fallback MOG2.")
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
            print(f"Load YOLO model: {model_name} (device={self.device})")
            self.model = YOLO(model_name)
            self.enabled = True
            print("YOLO đã sẵn sàng!")
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
        # Xử lý màu chữ trắng mặc định
        if 'fg' not in kwargs:
            kwargs['fg'] = 'white'
        
        super().__init__(parent, **kwargs)
        self.config(
            font=('SF Pro Text', 11, 'bold'),
            relief='flat', 
            bd=0,
            padx=20, 
            pady=10,
            cursor='hand2',
            activebackground=kwargs.get('bg', '#007aff'),
            activeforeground='white',
            highlightthickness=0
        )
        self.default_bg = kwargs.get('bg', '#007aff')
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)

    def on_enter(self, e):
        self['background'] = self._lighten(self.default_bg)

    def on_leave(self, e):
        self['background'] = self.default_bg

    def _lighten(self, color):
        """Làm sáng màu khi hover"""
        lighten_map = {
            '#007aff': '#3395ff',
            '#34c759': '#5cd96f',
            '#ff3b30': '#ff5e57',
            '#ffcc00': '#ffdd33',
            '#8e8e93': '#a3a3aa',
            '#5ac8fa': '#7fd8ff',
            '#6f42c1': '#8c5fd8',
            '#fd7e14': '#ff983d',
        }
        return lighten_map.get(color, color)
    # ==============================
# PART 2 – CLASS INIT + LOAD MODEL + SETUP UI
# ==============================

class WasteClassifierGUIAdvanced:
    """GUI chính – ĐÃ SỬA HOÀN TOÀN"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Hệ Thống Phân Loại Rác Thải AI")

        # Kích thước window
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f"{int(sw*0.9)}x{int(sh*0.9)}")
        
        # Màu sắc theme macOS (ĐÃ CẢI THIỆN)
        self.colors = {
            'bg': '#f5f5f7',
            'sidebar': '#ffffff',
            'header': '#ffffff',
            'card': '#ffffff',
            'primary': '#007aff',      # Xanh dương Apple
            'success': '#34c759',      # Xanh lá
            'danger': '#ff3b30',       # Đỏ
            'warning': '#ffcc00',      # Vàng
            'info': '#5ac8fa',         # Xanh nhạt
            'secondary': '#8e8e93',    # Xám
            'purple': '#6f42c1',       # Tím
            'orange': '#fd7e14',       # Cam
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

        # Lịch sử
        self.scan_history = []
        self.load_scan_history()
        
        # Khởi tạo result rỗng
        self.current_result = None

        # Chuẩn bị giao diện
        self.setup_ui()

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
            print("Model đã load!")
        except Exception as e:
            print(f"❌ Lỗi load model: {e}")
            self.model_loaded = False
            self.classifier = None

    def setup_ui(self):
        """Tạo header + layout 3 panel bằng GRID (chuẩn nhất)"""

        # ===== HEADER =====
        header = tk.Frame(self.root, bg=self.colors['header'], height=80)
        header.grid(row=0, column=0, sticky='ew')
        header.grid_propagate(False)

        shadow = tk.Frame(self.root, height=1, bg=self.colors['border'])
        shadow.grid(row=1, column=0, sticky='ew')
        
        self.root.grid_rowconfigure(2, weight=1)   # hàng chứa self.main co giãn theo chiều dọc
        self.root.grid_columnconfigure(0, weight=1)  # cột 0 (self.main) co giãn theo chiều ngang

        # Nội dung header
        header_content = tk.Frame(header, bg=self.colors['header'])
        header_content.pack(fill='both', expand=True, padx=30)

        # Title trái
        title_frame = tk.Frame(header_content, bg=self.colors['header'])
        title_frame.pack(side='left', pady=20)

        title = tk.Label(
            title_frame,
            text="Phân Loại Rác Thải Thông Minh",
            font=('Arial', 24, 'bold'),
            bg=self.colors['header'],
            fg=self.colors['primary']
        )
        title.pack(side='left')

        subtitle = tk.Label(
            title_frame,
            text="AI-Powered Waste Classification",
            font=('Arial', 11),
            bg=self.colors['header'],
            fg=self.colors['text_secondary']
        )
        subtitle.pack(side='left', padx=(15, 0))

        # Status phải
        status_frame = tk.Frame(header_content, bg=self.colors['header'])
        status_frame.pack(side='right', pady=20)

        dot_color = self.colors['success'] if getattr(self, "model_loaded", False) else self.colors['danger']
        status_text = "Model Ready" if getattr(self, "model_loaded", False) else "Model Not Found"

        status_dot = tk.Label(status_frame, text="●", font=('Arial', 20), bg=self.colors['header'], fg=dot_color)
        status_dot.pack(side='left')

        self.status_label = tk.Label(
            status_frame,
            text=status_text,
            font=('Arial', 12, 'bold'),
            bg=self.colors['header'],
            fg=dot_color
        )
        self.status_label.pack(side='left', padx=(5, 0))

        # ===== MAIN LAYOUT (CHỈ 1 DÒNG GRID) =====
        self.main = tk.Frame(self.root, bg=self.colors['bg'])
        self.main.grid(row=2, column=0, sticky='nsew')

        # GRID CONFIG CHO MAIN
        self.main.columnconfigure(0, weight=0, minsize=230)   # Sidebar — cố định
        self.main.columnconfigure(1, weight=1)                # Center — co giãn
        self.main.columnconfigure(2, weight=0, minsize=520)   # Right — cố định
        self.main.rowconfigure(0, weight=1)

        # ===== 3 PANEL =====
        left = self.setup_left_sidebar_grid(self.main)
        left.grid(row=0, column=0, sticky='nsew', padx=(20, 10), pady=20)

        center = self.setup_center_panel_grid(self.main)
        center.grid(row=0, column=1, sticky='nsew', padx=10, pady=20)

        right = self.setup_right_panel_grid(self.main)
        right.grid(row=0, column=2, sticky='nsew', padx=(10, 20), pady=20)

        self.update_statistics()


    # =========================================================
    # SIDEBAR - DÙNG GRID
    # =========================================================
    def setup_left_sidebar_grid(self, parent):
        """Sidebar menu bên trái - GRID"""

        sidebar = tk.Frame(
            parent,
            bg=self.colors['card'],
            width=230,
            highlightthickness=1,
            highlightbackground=self.colors['border']
        )
        sidebar.grid_propagate(False)

        # Title
        menu_title = tk.Label(
            sidebar,
            text="MENU",
            font=('Arial', 14, 'bold'),
            bg=self.colors['card'],
            fg=self.colors['text']
        )
        menu_title.pack(pady=(20, 15))

        # Buttons
        buttons = [
            ("Camera",        self.show_camera_mode,      self.colors['primary']),
            ("Upload Ảnh",    self.upload_image,          self.colors['info']),
            # ("Xử Lý Video",   self.process_video,         self.colors['purple']),
            (" Batch",         self.batch_classify,        self.colors['success']),
            (" Training",      self.show_training_panel,   self.colors['orange']),
            (" Fine-tune",     self.incremental_training,  self.colors['purple']),
            ("Đổi Mô Hình",  self.change_model,          self.colors['danger']),
            (" Quản Lý Data",  self.show_data_management,  self.colors['warning']),
            (" Thống Kê",      self.show_statistics,       self.colors['info']),
            ("Hướng Dẫn",     self.show_guide,            self.colors['secondary']),
        ]

        for text, cmd, color in buttons:
            tk.Button(
                sidebar,
                text=text,
                bg=color,
                fg='black',
                font=('Arial', 11, 'bold'),
                relief='flat',
                bd=0,
                padx=20,
                pady=10,
                cursor='hand2',
                width=16,
                command=cmd
            ).pack(pady=6, padx=15)

        # Spacer
        tk.Frame(sidebar, bg=self.colors['card']).pack(expand=True)

        # Exit
        tk.Button(
            sidebar,
            text="Thoát",
            bg=self.colors['secondary'],
            fg='black',
            font=('Arial', 11, 'bold'),
            relief='flat',
            bd=0,
            padx=20,
            pady=10,
            cursor='hand2',
            width=16,
            command=self.on_closing
        ).pack(pady=20, padx=15)

        return sidebar


    # =========================================================
    # CENTER PANEL - DÙNG GRID
    # =========================================================
    def setup_center_panel_grid(self, parent):
        """Khung camera ở giữa - GRID"""

        center = tk.Frame(
            parent,
            bg=self.colors['card'],
            highlightthickness=1,
            highlightbackground=self.colors['border']
        )

        # Title
        title_label = tk.Label(
            center,
            text="Camera Phát Hiện & Phân Loại",
            font=('Arial', 14, 'bold'),
            bg=self.colors['card'],
            fg=self.colors['text']
        )
        title_label.pack(pady=(15, 10), padx=20, anchor='w')

        separator = tk.Frame(center, height=1, bg=self.colors['border'])
        separator.pack(fill='x', padx=20)

        # Toggle
        toggle_frame = tk.Frame(center, bg=self.colors['card'])
        toggle_frame.pack(fill='x', padx=20, pady=(10, 0))

        self.auto_scan_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            toggle_frame,
            text="Tự động quét",
            variable=self.auto_scan_var,
            font=('Arial', 11, 'bold'),
            bg=self.colors['card'],
            fg=self.colors['text'],
            selectcolor=self.colors['card'],
            command=self.toggle_auto_scan
        ).pack(side='right')

        # Video container
        video_container = tk.Frame(center, bg=self.colors['border'], bd=1)
        video_container.pack(padx=20, pady=15, fill='both', expand=True)

        self.video_frame = tk.Label(video_container, bg='#000000')
        self.video_frame.pack(fill='both', expand=True, padx=1, pady=1)

        # Buttons
        control = tk.Frame(center, bg=self.colors['card'])
        control.pack(pady=(0, 20))

        self.btn_start_camera = tk.Button(
            control, text="Bật Camera",
            bg=self.colors['success'], fg='black', width=15,
            font=('Arial', 11, 'bold'), relief='flat',
            command=self.toggle_camera
        )
        self.btn_start_camera.pack(side='left', padx=5)

        self.btn_scan = tk.Button(
            control, text="Scan",
            bg=self.colors['primary'], fg='black', width=15,
            font=('Arial', 11, 'bold'), relief='flat',
            state='disabled', command=self.manual_scan
        )
        self.btn_scan.pack(side='left', padx=5)

        self.btn_save_frame = tk.Button(
            control, text="Lưu Frame",
            bg=self.colors['purple'], fg='black', width=15,
            font=('Arial', 11, 'bold'), relief='flat',
            state='disabled', command=self.save_current_frame
        )
        self.btn_save_frame.pack(side='left', padx=5)

        return center


    # =========================================================
    # RIGHT PANEL - DÙNG GRID
    # =========================================================
    def setup_right_panel_grid(self, parent):
        """Khung kết quả bên phải - GRID"""

        right = tk.Frame(
            parent,
            bg=self.colors['card'],
            width=520,
            highlightthickness=1,
            highlightbackground=self.colors['border']
        )
        right.grid_propagate(False)

        # Title
        title_label = tk.Label(
            right,
            text="Kết Quả Phân Loại",
            font=('Arial', 20, 'bold'),
            bg=self.colors['card'],
            fg=self.colors['text']
        )
        title_label.pack(pady=(15, 10), padx=20, anchor='w')

        separator = tk.Frame(right, height=1, bg=self.colors['border'])
        separator.pack(fill='x', padx=20)

        # Result text
        result_container = tk.Frame(right, bg=self.colors['border'], bd=1)
        result_container.pack(fill='both', expand=True, padx=20, pady=(10, 15))

        self.result_text = scrolledtext.ScrolledText(
            result_container,
            font=('Courier', 12),
            bg='#ffffff',
            fg='#1c1c1e',
            wrap='word',
            relief='flat',
            state='disabled',
            padx=15,
            pady=15
        )
        self.result_text.pack(fill='both', expand=True, padx=1, pady=1)

        # Buttons
        action = tk.Frame(right, bg=self.colors['card'])
        action.pack(pady=(0, 15))

        self.btn_save = tk.Button(
            action, text="Lưu Kết Quả",
            bg=self.colors['purple'], fg='black', width=16,
            font=('Arial', 11, 'bold'), relief='flat',
            state='disabled', command=self.save_scan_result
        )
        self.btn_save.pack(side='left', padx=5)

        tk.Button(
            action, text="📜 Lịch Sử",
            bg=self.colors['secondary'], fg='black', width=16,
            font=('Arial', 11, 'bold'), relief='flat',
            command=self.show_history
        ).pack(side='left', padx=5)

        # Stats
        stats_card = tk.Frame(right, bg='#e7f0ff')
        stats_card.pack(fill='x', padx=20, pady=(0, 20))

        tk.Label(
            stats_card, text="Thống Kê Nhanh",
            font=('Arial', 12, 'bold'),
            bg='#e7f0ff', fg=self.colors['primary']
        ).pack(pady=(12, 8), padx=15, anchor='w')

        self.stats_label = tk.Label(
            stats_card,
            text="Chưa có dữ liệu",
            font=('Arial', 10),
            bg='#e7f0ff',
            fg=self.colors['text'],
            justify='left'
        )
        self.stats_label.pack(padx=15, pady=(0, 12), anchor='w')

        return right

    # ==============================
# PART 4 – CAMERA CONTROL + DETECTION + CLASSIFICATION
# ==============================

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
        self.btn_start_camera.config(text="Tắt Camera", bg=self.colors['danger'])
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

        self.btn_start_camera.config(text="Bật Camera", bg=self.colors['success'])
        self.btn_scan.config(state='disabled')
        self.btn_save_frame.config(state='disabled')
        self.video_frame.config(image='')

    def toggle_auto_scan(self):
        self.auto_scan = self.auto_scan_var.get()
        if self.auto_scan:
            print("Bật auto scan")
        else:
            print("Tắt auto scan")

    # =========================================================
    # OBJECT DETECTION (YOLO → MOG2 fallback)
    # =========================================================
    def detect_object(self, frame):
        """
        1️⃣ Nếu có YOLO → dùng YOLO
        2️⃣ Nếu YOLO fail → dùng MOG2
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
    def change_model(self):
        """Cho phép người dùng chọn model khác để test"""
        file_path = filedialog.askopenfilename(
            title="Chọn file model (.h5, .keras hoặc .pkl)",
            filetypes=[
                ("All Models", "*.h5 *.keras *.pkl"),
                ("Keras Model", "*.h5 *.keras"),
                ("SVM Model", "*.pkl"),
                ("All Files", "*.*")
            ]
        )
        if file_path:
            try:
                self.classifier = WasteClassifier(file_path)
                self.model_loaded = True
                self.status_label.config(text=f"Model: {os.path.basename(file_path)}", fg=self.colors['success'])
                messagebox.showinfo("Thành công", f"Đã chuyển sang model: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể load model này:\n{e}")

    def show_camera_mode(self):
        """Hiển thị thông báo camera mode"""
        messagebox.showinfo(
            "Camera Mode",
            "Chế độ camera đang hiển thị ở màn hình chính!\n\n"
            "• Nhấn 'Bật Camera' để bắt đầu\n"
            "• Bật 'Tự động quét' để scan liên tục\n"
            "• Khung xanh tự động theo dõi vật thể"
        )

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
        self.result_text.insert(tk.END, "Độ Tin Cậy: ", 'bold')

        if result['is_confident']:
            self.result_text.insert(tk.END, f"{confidence:.1f}%  \n", 'success')
        else:
            self.result_text.insert(tk.END, f"{confidence:.1f}%  \n", 'warning')

        # Progress bar (ASCII)
        bar_len = int(confidence / 2)
        bar = "█" * bar_len + "░" * (50 - bar_len)
        self.result_text.insert(tk.END, f"{bar}\n\n")

        # ---- Hướng dẫn xử lý ----
        self.result_text.insert(tk.END, " Cách xử lý:\n", 'bold')
        self.result_text.insert(tk.END, f"   {info['disposal']}\n\n")

        # ---- Ví dụ ----
        self.result_text.insert(tk.END, "Ví dụ:\n", 'bold')
        self.result_text.insert(tk.END, f"   {', '.join(info['examples'])}\n\n")

        # ---- Giá trị tái chế ----
        self.result_text.insert(tk.END, "💰 Giá trị tái chế: ", 'bold')
        self.result_text.insert(tk.END, f"{info['recycling_value']}\n\n")

        # ---- Separator ----
        self.result_text.insert(tk.END, "─" * 55 + "\n\n")

        # ---- Detailed Probabilities ----
        self.result_text.insert(tk.END, "Chi Tiết Các Xác Suất:\n\n", 'bold')

        sorted_preds = sorted(
            result['all_predictions'].items(),
            key=lambda x: x[1],
            reverse=True
        )

        for cls, prob in sorted_preds:
            icon = CLASS_INFO[cls]['icon']
            bar_len = int(prob / 3)
            bar = "█" * bar_len
            self.result_text.insert(tk.END, f"{icon} {cls:11s} ")
            self.result_text.insert(tk.END, f"{bar:33s} {prob:5.1f}%\n")

        self.result_text.config(state='disabled')
    # ==============================
# PART 5 – IMAGE/VIDEO/BATCH + TRAINING + DATA MANAGEMENT
# ==============================

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
                messagebox.showinfo("Thành công", "Đã xử lý video!")
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
    # SAVE SCAN RESULT
    # =========================================================
    def save_scan_result(self):
        """Lưu ảnh đã scan + metadata JSON + update history"""
        if not hasattr(self, 'current_result') or self.current_result is None:
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

        messagebox.showinfo("Thành công", f"Đã lưu kết quả!\n\n{save_path}")
        self.btn_save.config(state='disabled')

    # =========================================================
    # BATCH RESULT UI + SAVE CSV
    # =========================================================
    def show_batch_results(self, results):
        """Cửa sổ hiển thị kết quả batch dạng Grid hình ảnh"""
        window = tk.Toplevel(self.root)
        window.title("Kết Quả Phân Loại Batch")
        window.geometry("1100x750")
        window.configure(bg=self.colors['bg'])

        # Header
        header = tk.Frame(window, bg=self.colors['card'], height=70)
        header.pack(fill='x')
        header.pack_propagate(False)

        tk.Label(
            header,
            text="Kết Quả Phân Loại Batch",
            font=('SF Pro Display', 20, 'bold'),
            bg=self.colors['card'],
            fg=self.colors['primary']
        ).pack(pady=20)

        # Khung cuộn (Scrollable Grid)
        frame = tk.Frame(window, bg=self.colors['bg'])
        frame.pack(fill='both', expand=True, padx=30, pady=20)

        canvas = tk.Canvas(frame, bg=self.colors['bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.colors['bg'])

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Bắt sự kiện lăn chuột an toàn cho macOS/Windows
        def _on_mousewheel(event):
            # Với macOS event.delta thường nhỏ, với Windows thường là bội số của 120
            delta = -1 if event.delta < 0 else 1
            if sys.platform == 'darwin':
                canvas.yview_scroll(int(-1 * event.delta), "units")
            else:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
                
        def _bind_mouse(e):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
        def _unbind_mouse(e):
            canvas.unbind_all("<MouseWheel>")
            
        canvas.bind("<Enter>", _bind_mouse)
        canvas.bind("<Leave>", _unbind_mouse)

        # Cần biến này để giữ hình ảnh không bị bộ thu gom rác (GC) xoá
        self._batch_images = []

        columns_count = 4
        for i, item in enumerate(results):
            row = i // columns_count
            col = i % columns_count

            card = tk.Frame(scrollable_frame, bg=self.colors['card'], highlightbackground=self.colors['border'], highlightthickness=1)
            card.grid(row=row, column=col, padx=15, pady=15, sticky="nsew")

            result = item['result']
            icon = CLASS_INFO[result['class']]['icon']

            try:
                img = Image.open(item['image'])
                # Cắt/Thu nhỏ ảnh về hình vuông 200x200
                img.thumbnail((220, 220), Image.Resampling.LANCZOS)
                imgtk = ImageTk.PhotoImage(img)
                self._batch_images.append(imgtk)
                
                img_lbl = tk.Label(card, image=imgtk, bg=self.colors['card'])
                img_lbl.pack(padx=10, pady=(10, 5))
            except Exception as e:
                tk.Label(card, text="Lỗi hiển thị ảnh", bg=self.colors['card']).pack(pady=20)

            # Text Nhãn và Độ tin cậy
            tk.Label(card, text=f"{icon} {result['class_name_vi']}", font=('Arial', 12, 'bold'), bg=self.colors['card'], fg=self.colors['text']).pack()
            tk.Label(card, text=f"{result['confidence']:.1f}%", font=('Arial', 10), bg=self.colors['card'], fg=self.colors['primary']).pack(pady=(0, 10))

        # Buttons
        btn_frame = tk.Frame(window, bg=self.colors['bg'])
        btn_frame.pack(pady=20)

        ModernButton(
            btn_frame,
            text="Lưu CSV",
            bg=self.colors['primary'],
            fg='white',
            command=lambda: self.save_batch_csv(results)
        ).pack(side='left', padx=10)

        ModernButton(
            btn_frame,
            text="Đóng",
            bg=self.colors['secondary'],
            fg='white',
            command=window.destroy
        ).pack(side='left', padx=10)

        def _on_closing():
            try:
                canvas.unbind_all("<MouseWheel>")
            except:
                pass
            window.destroy()
        window.protocol("WM_DELETE_WINDOW", _on_closing)

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

            messagebox.showinfo("Thành công", f"Đã lưu: {file_path}")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi lưu CSV: {str(e)}")

    # =========================================================
    # TRAINING PANEL
    # =========================================================
    def show_training_panel(self):
        """Cửa sổ cấu hình training model mới"""
        window = tk.Toplevel(self.root)
        window.title("Training Model")
        window.geometry("900x700")
        window.configure(bg=self.colors['bg'])

        # Header
        header = tk.Frame(window, bg=self.colors['card'], height=80)
        header.pack(fill='x')
        header.pack_propagate(False)

        tk.Label(
            header,
            text="Training Model Mới",
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
            text="Thư mục Training:",
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
            text="Thư mục Validation:",
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
            text="Số Epochs:",
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
            text="Sử dụng Transfer Learning",
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
            text="Bắt Đầu Training",
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
            self.root.after(0, self.load_model)

            self.root.after(
                0,
                lambda: messagebox.showinfo("Thành công", "Training hoàn tất!")
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
        """Kiểm tra dữ liệu và bắt đầu incremental training"""
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
            f"Dữ liệu sẵn sàng!\n\nTổng: {stats['total']} mẫu\nBắt đầu training?"
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
            self.root.after(0, self.load_model)

            self.root.after(
                0,
                lambda: messagebox.showinfo("Thành công", "Incremental training hoàn tất!")
            )
        except Exception as e:
            self.root.after(
                0,
                lambda: messagebox.showerror("Lỗi", f"Lỗi incremental training: {e}")
            )
    # ==============================
# PART 6 – DATA MANAGEMENT + HISTORY + STATS + GUIDE + MAIN (FINAL)
# ==============================

    # =========================================================
    # DATA MANAGEMENT (ĐÃ SỬA - THÊM CÁC METHOD THIẾU)
    # =========================================================
    def show_data_management(self):
        """Quản lý dữ liệu"""
        window = tk.Toplevel(self.root)
        window.title("Quản Lý Dữ Liệu")
        window.geometry("1000x750")
        window.configure(bg=self.colors['bg'])
        
        # Header
        header = tk.Frame(window, bg=self.colors['card'], height=80)
        header.pack(fill='x')
        header.pack_propagate(False)
        
        tk.Label(header, text=" Quản Lý Dữ Liệu Training",
                font=('SF Pro Display', 30, 'bold'),
                bg=self.colors['card'], fg=self.colors['primary']).pack(pady=25)
        
        # Stats
        stats = self.data_manager.get_scanned_stats()
        
        overview = f"""
 TỔNG QUAN
{'─'*70}
Tổng số mẫu: {stats['total']}
Chất lượng cao (≥80%): {stats['high_confidence']}
Tỷ lệ: {stats['high_confidence']/stats['total']*100 if stats['total'] > 0 else 0:.1f}%

 CHI TIẾT THEO CLASS
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
        
        ModernButton(btn_frame, text="Chuẩn Bị", bg=self.colors['primary'], fg='white',
                    command=self.prepare_dataset).pack(side='left', padx=8)
        ModernButton(btn_frame, text="Export", bg=self.colors['success'], fg='white',
                    command=self.export_high_quality).pack(side='left', padx=8)
        ModernButton(btn_frame, text="Xóa", bg=self.colors['danger'], fg='white',
                    command=self.clean_low_quality).pack(side='left', padx=8)

    def prepare_dataset(self):
        """Chuẩn bị dataset"""
        if messagebox.askyesno("Xác nhận", "Chuẩn bị dữ liệu cho training?"):
            try:
                self.data_manager.prepare_training_data(min_confidence=80)
                messagebox.showinfo("Thành công", "Đã chuẩn bị dataset!")
            except Exception as e:
                messagebox.showerror("Lỗi", f"{e}")
    
    def export_high_quality(self):
        """Export chất lượng cao"""
        output_dir = filedialog.askdirectory(title="Chọn thư mục lưu")
        if output_dir:
            try:
                self.data_manager.export_high_quality_data(output_dir, min_confidence=90)
                messagebox.showinfo("Thành công", f"Đã export!")
            except Exception as e:
                messagebox.showerror("Lỗi", f"{e}")
    
    def clean_low_quality(self):
        """Xóa chất lượng thấp"""
        if messagebox.askyesno("Cảnh báo", "Xóa ảnh ≤60%? Không thể hoàn tác!"):
            try:
                self.data_manager.clean_low_quality_data(max_confidence=60)
                messagebox.showinfo("Thành công", "Đã xóa!")
                self.update_statistics()
            except Exception as e:
                messagebox.showerror("Lỗi", f"{e}")

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
        window.title("Lịch Sử Scan")
        window.geometry("900x600")
        window.configure(bg=self.colors['bg'])
        
        tk.Label(
            window,
            text="Lịch Sử Scan",
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
            mode = "Auto" if item.get('is_auto_scan', False) else "👤 Manual"
            
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
            text="Đóng",
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
        window.title("Thống Kê")
        window.geometry("800x600")
        window.configure(bg=self.colors['bg'])

        # header
        tk.Label(
            window,
            text="Thống kê dữ liệu đã lưu",
            font=("SF Pro Display", 22, "bold"),
            bg=self.colors['bg'],
            fg=self.colors['primary']
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

        ModernButton(
            window,
            text="Đóng",
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
        
        text = f"Tổng: {total} lần scan\n"
        text += f"Tin cậy cao: {high_conf_count}/{total}\n"
        text += f"Auto scan: {auto_count}/{total}\n\n"
        
        # Top 3 classes
        sorted_stats = sorted(stats.items(), key=lambda x: x[1], reverse=True)[:3]
        for cls, count in sorted_stats:
            if count > 0:
                icon = CLASS_INFO[cls]['icon']
                pct = (count / total * 100)
                text += f"{icon} {cls}: {count} ({pct:.0f}%)\n"
        
        self.stats_label.config(text=text)

    # =========================================================
    # GUIDE
    # =========================================================
    def show_guide(self):
        """Hướng dẫn"""
        window = tk.Toplevel(self.root)
        window.title("Hướng Dẫn")
        window.geometry("1000x750")
        window.configure(bg=self.colors['bg'])
        
        # Header
        header = tk.Frame(window, bg=self.colors['card'], height=80)
        header.pack(fill='x')
        header.pack_propagate(False)
        
        tk.Label(header, text="📖 Hướng Dẫn Sử Dụng",
                font=('SF Pro Display', 30, 'bold'),
                bg=self.colors['card'], fg=self.colors['primary']).pack(pady=25)
        
        guide = """
╔═══════════════════════════════════════════════════════════════╗
║                    HƯỚNG DẪN SỬ DỤNG                        ║
╚═══════════════════════════════════════════════════════════════╝

CAMERA SCAN
──────────────────────────────────────────────────────────────
1. Nhấn "Bật Camera"
2. Đặt vật phẩm vào khung
3. Hệ thống tự động phát hiện và DI CHUYỂN KHUNG XANH
4. Bật "Tự động quét" để scan liên tục (mỗi 2 giây)
5. Hoặc nhấn "Scan" để scan thủ công
6. Xem kết quả bên phải và lưu nếu cần

UPLOAD & BATCH
──────────────────────────────────────────────────────────────
• Upload: Chọn 1 ảnh để phân loại
• Batch: Chọn thư mục nhiều ảnh, xem kết quả bảng, lưu CSV

TRAINING
──────────────────────────────────────────────────────────────
• Training: Train model mới từ dataset có sẵn
• Fine-tune: Cập nhật model với dữ liệu đã scan (≥20 mẫu/class)

QUẢN LÝ DỮ LIỆU
──────────────────────────────────────────────────────────────
• Xem thống kê dữ liệu đã scan
• Chuẩn bị dataset (auto chia 80/20)
• Export dữ liệu chất lượng cao (≥90%)
• Xóa dữ liệu kém (≤60%)

TIPS
──────────────────────────────────────────────────────────────
✓ Khung xanh tự động theo dõi vật thể
✓ Chỉ lưu ảnh confidence ≥80%
✓ Dùng Fine-tune để cải thiện model liên tục
✓ Auto scan cooldown 2 giây tránh spam

YÊU CẦU HỆ THỐNG
──────────────────────────────────────────────────────────────
• Python 3.7+
• TensorFlow 2.x
• OpenCV
• Camera (cho real-time)

──────────────────────────────────────────────────────────────
Happy Classifying! 
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
            text="Đóng",
            bg=self.colors['secondary'],
            fg='white',
            command=window.destroy
        ).pack(pady=20)

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