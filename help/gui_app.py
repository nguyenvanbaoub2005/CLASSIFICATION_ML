# gui_app.py
"""
Giao diện GUI hiện đại với theme sáng
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import cv2
from PIL import Image, ImageTk
import threading
import os
import json
import csv
from datetime import datetime
from classifier import WasteClassifier

from train import train_model, plot_training_history
from data_manager import DataManager
from incremental_train import IncrementalTrainer
from config import PATHS, CLASS_INFO, CLASSES, MODEL_CONFIG
import numpy as np


class ModernButton(tk.Button):
    """Custom modern button với shadow effect"""
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.config(
            font=('Segoe UI', 10, 'bold'),
            relief='flat',
            bd=0,
            padx=20,
            pady=12,
            cursor='hand2',
            activebackground=kwargs.get('bg', '#0066cc')
        )
        
        # Hover effect
        self.bind('<Enter>', self.on_enter)
        self.bind('<Leave>', self.on_leave)
        self.default_bg = kwargs.get('bg', '#0066cc')
    
    def on_enter(self, e):
        self['background'] = self.lighten_color(self.default_bg)
    
    def on_leave(self, e):
        self['background'] = self.default_bg
    
    def lighten_color(self, color):
        """Làm sáng màu khi hover"""
        color_map = {
            '#0066cc': '#0077ee',
            '#28a745': '#32d956',
            '#dc3545': '#ff4757',
            '#ffc107': '#ffd43b',
            '#6c757d': '#868e96',
            '#17a2b8': '#1ac9e6',
            '#6f42c1': '#8357d8',
            '#fd7e14': '#ff922b',
        }
        return color_map.get(color, color)


class WasteClassifierGUIAdvanced:
    """Class GUI nâng cao với theme sáng"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("🌿 Hệ Thống Phân Loại Rác Thải AI")
        
        # Set window size
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{screen_width}x{screen_height}")
        self.root.state('zoomed')
        
        # Theme colors - Sáng và hiện đại
        self.colors = {
            'bg': '#f8f9fa',           # Background chính - xám sáng
            'sidebar': '#ffffff',       # Sidebar - trắng
            'header': '#ffffff',        # Header - trắng
            'card': '#ffffff',          # Card - trắng
            'primary': '#0066cc',       # Primary - xanh dương
            'success': '#28a745',       # Success - xanh lá
            'danger': '#dc3545',        # Danger - đỏ
            'warning': '#ffc107',       # Warning - vàng
            'info': '#17a2b8',          # Info - xanh ngọc
            'secondary': '#6c757d',     # Secondary - xám
            'text': '#212529',          # Text chính - đen
            'text_secondary': '#6c757d', # Text phụ - xám
            'border': '#dee2e6',        # Border - xám nhạt
            'shadow': '#00000010',      # Shadow nhẹ
        }
        
        self.root.configure(bg=self.colors['bg'])
        
        # Load model
        self.load_model()
        
        # Camera
        self.cap = None
        self.camera_running = False
        self.auto_scan = False
        self.current_frame = None
        self.scan_history = []
        self.last_scan_time = 0
        self.scan_cooldown = 2.0
        
        # Object detection
        self.object_detector = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True
        )
        self.detected_bbox = None
        
        # Data manager
        self.data_manager = DataManager()
        
        # Tạo thư mục lưu dữ liệu
        self.data_save_dir = "scanned_data"
        os.makedirs(self.data_save_dir, exist_ok=True)
        for cls in CLASSES:
            os.makedirs(os.path.join(self.data_save_dir, cls), exist_ok=True)
        
        self.setup_ui()
        self.load_scan_history()
    
    def load_model(self):
        """Load model"""
        model_path = PATHS['model_save']
        if not os.path.exists(model_path):
            model_path = PATHS['best_model']
        
        try:
            self.classifier = WasteClassifier(model_path)
            self.model_loaded = True
        except:
            self.classifier = None
            self.model_loaded = False
    
    def create_card(self, parent, title=None):
        """Tạo card với shadow effect"""
        card = tk.Frame(
            parent,
            bg=self.colors['card'],
            relief='flat',
            bd=0
        )
        
        # Shadow effect (frame phía sau)
        shadow = tk.Frame(
            parent,
            bg=self.colors['border'],
            relief='flat'
        )
        
        if title:
            title_label = tk.Label(
                card,
                text=title,
                font=('Segoe UI', 14, 'bold'),
                bg=self.colors['card'],
                fg=self.colors['text']
            )
            title_label.pack(pady=(15, 10), padx=20, anchor='w')
            
            # Separator line
            separator = tk.Frame(card, height=2, bg=self.colors['border'])
            separator.pack(fill='x', padx=20)
        
        return card
    
    def setup_ui(self):
        """Thiết lập giao diện"""
        # Header với gradient effect
        header = tk.Frame(self.root, bg=self.colors['header'], height=80)
        header.pack(fill='x')
        header.pack_propagate(False)
        
        # Shadow cho header
        shadow = tk.Frame(self.root, height=3, bg=self.colors['border'])
        shadow.pack(fill='x')
        
        # Header content
        header_content = tk.Frame(header, bg=self.colors['header'])
        header_content.pack(fill='both', expand=True, padx=30)
        
        # Logo and title
        title_frame = tk.Frame(header_content, bg=self.colors['header'])
        title_frame.pack(side='left', pady=20)
        
        title = tk.Label(
            title_frame,
            text="🌿 Phân Loại Rác Thải Thông Minh",
            font=('Segoe UI', 26, 'bold'),
            bg=self.colors['header'],
            fg=self.colors['primary']
        )
        title.pack(side='left')
        
        subtitle = tk.Label(
            title_frame,
            text="AI-Powered Waste Classification",
            font=('Segoe UI', 11),
            bg=self.colors['header'],
            fg=self.colors['text_secondary']
        )
        subtitle.pack(side='left', padx=(15, 0))
        
        # Status indicator
        status_frame = tk.Frame(header_content, bg=self.colors['header'])
        status_frame.pack(side='right', pady=20)
        
        if self.model_loaded:
            status_dot = tk.Label(
                status_frame,
                text="●",
                font=('Arial', 20),
                bg=self.colors['header'],
                fg=self.colors['success']
            )
            status_text = "Model Ready"
            status_color = self.colors['success']
        else:
            status_dot = tk.Label(
                status_frame,
                text="●",
                font=('Arial', 20),
                bg=self.colors['header'],
                fg=self.colors['danger']
            )
            status_text = "Model Not Found"
            status_color = self.colors['danger']
        
        status_dot.pack(side='left')
        
        self.status_label = tk.Label(
            status_frame,
            text=status_text,
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['header'],
            fg=status_color
        )
        self.status_label.pack(side='left', padx=(5, 0))
        
        # Main container
        main = tk.Frame(self.root, bg=self.colors['bg'])
        main.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Left sidebar - Menu
        self.setup_left_sidebar(main)
        
        # Center panel - Camera (nhỏ hơn)
        self.setup_center_panel(main)
        
        # Right panel - Results (rộng hơn)
        self.setup_right_panel(main)
    
    def setup_left_sidebar(self, parent):
        """Setup sidebar menu"""
        sidebar = self.create_card(parent)
        sidebar.pack(side='left', fill='y', padx=(0, 15))
        sidebar.config(width=220)
        sidebar.pack_propagate(False)
        
        # Menu title
        menu_title = tk.Label(
            sidebar,
            text="📋 MENU",
            font=('Segoe UI', 14, 'bold'),
            bg=self.colors['card'],
            fg=self.colors['text']
        )
        menu_title.pack(pady=(20, 15))
        
        # Menu buttons
        buttons = [
            ("📷 Camera", self.show_camera_mode, self.colors['primary']),
            ("📸 Upload Ảnh", self.upload_image, self.colors['info']),
            ("📹 Xử Lý Video", self.process_video, '#6f42c1'),
            ("📁 Batch", self.batch_classify, self.colors['success']),
            ("🎓 Training", self.show_training_panel, '#fd7e14'),
            ("🔄 Fine-tune", self.incremental_training, '#6f42c1'),
            ("📊 Quản Lý Data", self.show_data_management, self.colors['warning']),
            ("📈 Thống Kê", self.show_statistics, self.colors['info']),
            ("ℹ️ Hướng Dẫn", self.show_guide, self.colors['secondary']),
        ]
        
        for text, command, color in buttons:
            btn = ModernButton(
                sidebar,
                text=text,
                bg=color,
                fg='white',
                command=command,
                width=16
            )
            btn.pack(pady=6, padx=15)
        
        # Spacer
        tk.Frame(sidebar, bg=self.colors['card']).pack(expand=True)
        
        # Exit button
        btn_exit = ModernButton(
            sidebar,
            text="🚪 Thoát",
            bg=self.colors['secondary'],
            fg='white',
            command=self.on_closing,
            width=16
        )
        btn_exit.pack(pady=20, padx=15)
    
    def setup_center_panel(self, parent):
        """Setup center camera panel - Nhỏ hơn"""
        center_panel = self.create_card(parent, "📷 Camera Phát Hiện & Phân Loại")
        center_panel.pack(side='left', fill='both', expand=True, padx=(0, 15))
        
        # Auto scan toggle
        toggle_frame = tk.Frame(center_panel, bg=self.colors['card'])
        toggle_frame.pack(fill='x', padx=20, pady=(10, 0))
        
        self.auto_scan_var = tk.BooleanVar()
        auto_check = tk.Checkbutton(
            toggle_frame,
            text="🤖 Tự động quét",
            variable=self.auto_scan_var,
            font=('Segoe UI', 11, 'bold'),
            bg=self.colors['card'],
            fg=self.colors['text'],
            selectcolor=self.colors['card'],
            activebackground=self.colors['card'],
            activeforeground=self.colors['primary'],
            command=self.toggle_auto_scan
        )
        auto_check.pack(side='right')
        
        # Video frame container với border
        video_container = tk.Frame(
            center_panel,
            bg=self.colors['border'],
            relief='flat',
            bd=2
        )
        video_container.pack(padx=20, pady=15, fill='both', expand=True)
        
        self.video_frame = tk.Label(video_container, bg='#000000')
        self.video_frame.pack(fill='both', expand=True, padx=2, pady=2)
        
        # Camera controls
        control_frame = tk.Frame(center_panel, bg=self.colors['card'])
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
            text="💾 Lưu",
            bg='#6f42c1',
            fg='white',
            width=13,
            command=self.save_current_frame,
            state='disabled'
        )
        self.btn_save_frame.pack(side='left', padx=5)
    
    def setup_right_panel(self, parent):
        """Setup right results panel - Rộng hơn"""
        right_panel = self.create_card(parent, "📊 Kết Quả Phân Loại")
        right_panel.pack(side='right', fill='both')
        right_panel.config(width=520)
        right_panel.pack_propagate(False)
        
        # Result display với custom styling
        result_container = tk.Frame(
            right_panel,
            bg=self.colors['border'],
            relief='flat',
            bd=1
        )
        result_container.pack(fill='both', expand=True, padx=20, pady=(10, 15))
        
        self.result_text = scrolledtext.ScrolledText(
            result_container,
            font=('Consolas', 11),
            bg='#f8f9fa',
            fg=self.colors['text'],
            wrap='word',
            relief='flat',
            bd=0,
            state='disabled',
            padx=15,
            pady=15
        )
        self.result_text.pack(fill='both', expand=True, padx=1, pady=1)
        
        # Configure text tags for colored output
        self.result_text.tag_config('header', font=('Segoe UI', 13, 'bold'), foreground=self.colors['primary'])
        self.result_text.tag_config('success', foreground=self.colors['success'])
        self.result_text.tag_config('warning', foreground=self.colors['warning'])
        self.result_text.tag_config('info', foreground=self.colors['info'])
        self.result_text.tag_config('bold', font=('Consolas', 11, 'bold'))
        
        # Action buttons
        action_frame = tk.Frame(right_panel, bg=self.colors['card'])
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
        
        # Statistics card
        stats_card = tk.Frame(
            right_panel,
            bg='#e7f3ff',
            relief='flat',
            bd=0
        )
        stats_card.pack(fill='x', padx=20, pady=(0, 20))
        
        stats_title = tk.Label(
            stats_card,
            text="📈 Thống Kê Nhanh",
            font=('Segoe UI', 12, 'bold'),
            bg='#e7f3ff',
            fg=self.colors['primary']
        )
        stats_title.pack(pady=(12, 8), padx=15, anchor='w')
        
        self.stats_label = tk.Label(
            stats_card,
            text="Chưa có dữ liệu",
            font=('Segoe UI', 10),
            bg='#e7f3ff',
            fg=self.colors['text'],
            justify='left',
            anchor='w'
        )
        self.stats_label.pack(padx=15, pady=(0, 12), anchor='w')
        
        self.update_statistics()
    
    def toggle_camera(self):
        """Bật/tắt camera"""
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
    
    def toggle_auto_scan(self):
        """Toggle auto scan mode"""
        self.auto_scan = self.auto_scan_var.get()
        if self.auto_scan:
            print("✅ Bật chế độ tự động quét")
        else:
            print("⏸️ Tắt chế độ tự động quét")
    
    def detect_object(self, frame):
        """Phát hiện vật thể"""
        fg_mask = self.object_detector.apply(frame)
        fg_mask[fg_mask == 127] = 0
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area < 5000:
            return None
        
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = w / h if h > 0 else 0
        
        if aspect_ratio < 0.2 or aspect_ratio > 5:
            return None
        
        return (x, y, w, h)
    
    def draw_detection_box(self, frame, bbox):
        """Vẽ khung phát hiện"""
        if bbox is None:
            return frame
        
        x, y, w, h = bbox
        
        # Màu xanh lá gradient
        color = (0, 200, 100)
        thickness = 3
        corner_length = 35
        
        # Vẽ 4 góc bo tròn
        cv2.line(frame, (x, y), (x + corner_length, y), color, thickness)
        cv2.line(frame, (x, y), (x, y + corner_length), color, thickness)
        
        cv2.line(frame, (x + w, y), (x + w - corner_length, y), color, thickness)
        cv2.line(frame, (x + w, y), (x + w, y + corner_length), color, thickness)
        
        cv2.line(frame, (x, y + h), (x + corner_length, y + h), color, thickness)
        cv2.line(frame, (x, y + h), (x, y + h - corner_length), color, thickness)
        
        cv2.line(frame, (x + w, y + h), (x + w - corner_length, y + h), color, thickness)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_length), color, thickness)
        
        # Label với background
        label = "VAT THE PHAT HIEN"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        cv2.rectangle(frame, (x, y - label_h - 15), (x + label_w + 10, y), color, -1)
        cv2.putText(frame, label, (x + 5, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Size info
        size_text = f"{w}x{h}px"
        cv2.putText(frame, size_text, (x, y + h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame
    
    def update_camera(self):
        """Cập nhật camera frame"""
        if self.camera_running:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                self.current_frame = frame.copy()
                
                # Phát hiện vật thể
                bbox = self.detect_object(frame)
                
                if bbox:
                    self.detected_bbox = bbox
                    frame = self.draw_detection_box(frame, bbox)
                    
                    # Auto scan
                    if self.auto_scan and self.model_loaded:
                        current_time = datetime.now().timestamp()
                        if current_time - self.last_scan_time > self.scan_cooldown:
                            self.auto_classify(bbox)
                            self.last_scan_time = current_time
                else:
                    self.detected_bbox = None
                    # Khung mờ ở giữa
                    h, w = frame.shape[:2]
                    center_x, center_y = w // 2, h // 2
                    box_size = 350
                    
                    x1 = center_x - box_size // 2
                    y1 = center_y - box_size // 2
                    x2 = center_x + box_size // 2
                    y2 = center_y + box_size // 2
                    
                    color = (180, 180, 180)
                    thickness = 2
                    corner_length = 30
                    
                    cv2.line(frame, (x1, y1), (x1 + corner_length, y1), color, thickness)
                    cv2.line(frame, (x1, y1), (x1, y1 + corner_length), color, thickness)
                    cv2.line(frame, (x2, y1), (x2 - corner_length, y1), color, thickness)
                    cv2.line(frame, (x2, y1), (x2, y1 + corner_length), color, thickness)
                    cv2.line(frame, (x1, y2), (x1 + corner_length, y2), color, thickness)
                    cv2.line(frame, (x1, y2), (x1, y2 - corner_length), color, thickness)
                    cv2.line(frame, (x2, y2), (x2 - corner_length, y2), color, thickness)
                    cv2.line(frame, (x2, y2), (x2, y2 - corner_length), color, thickness)
                    
                    cv2.putText(frame, "Dat vat pham vao khung", (center_x - 130, y1 - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Status indicator
                status_text = "AUTO SCAN: ON" if self.auto_scan else "MANUAL MODE"
                status_color = (0, 200, 100) if self.auto_scan else (100, 100, 100)
                
                # Background cho status
                (text_w, text_h), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (15, 15), (text_w + 35, text_h + 35), status_color, -1)
                cv2.putText(frame, status_text, (25, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Convert và resize để fit
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                
                # Resize nhỏ hơn để chừa chỗ cho results
                display_height = 480  # Giảm từ 700 xuống 480
                aspect_ratio = img.width / img.height
                display_width = int(display_height * aspect_ratio)
                img = img.resize((display_width, display_height), Image.Resampling.LANCZOS)
                
                imgtk = ImageTk.PhotoImage(image=img)
                
                self.video_frame.imgtk = imgtk
                self.video_frame.configure(image=imgtk)
            
            self.root.after(10, self.update_camera)
    
    def manual_scan(self):
        """Scan thủ công"""
        if self.current_frame is None:
            return
        
        if self.detected_bbox:
            x, y, w, h = self.detected_bbox
            cropped = self.current_frame[y:y+h, x:x+w]
        else:
            h, w = self.current_frame.shape[:2]
            center_x, center_y = w // 2, h // 2
            box_size = 350
            
            x1 = center_x - box_size // 2
            y1 = center_y - box_size // 2
            x2 = center_x + box_size // 2
            y2 = center_y + box_size // 2
            
            cropped = self.current_frame[y1:y2, x1:x2]
        
        temp_path = "temp_manual_scan.jpg"
        cv2.imwrite(temp_path, cropped)
        
        self.classify_image(temp_path, cropped)
    
    def auto_classify(self, bbox):
        """Tự động phân loại"""
        if self.current_frame is None:
            return
        
        x, y, w, h = bbox
        cropped = self.current_frame[y:y+h, x:x+w]
        
        if cropped.size == 0:
            return
        
        temp_path = "temp_auto_scan.jpg"
        cv2.imwrite(temp_path, cropped)
        
        threading.Thread(
            target=self.classify_image_async,
            args=(temp_path, cropped, True),
            daemon=True
        ).start()
    
    def classify_image_async(self, image_path, original_image, is_auto):
        """Phân loại async"""
        try:
            result = self.classifier.predict(image_path, return_all=True)
            
            if result['confidence'] >= 70:
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
            print(f"❌ Lỗi auto classify: {e}")
    
    def classify_image(self, image_path, original_image):
        """Phân loại ảnh"""
        if self.classifier is None:
            messagebox.showerror("Lỗi", "Model chưa được load!")
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
            messagebox.showerror("Lỗi", f"Lỗi phân loại: {str(e)}")
    
    def display_result(self, result):
        """Hiển thị kết quả với styling đẹp"""
        predicted_class = result['class']
        confidence = result['confidence']
        info = CLASS_INFO[predicted_class]
        
        self.result_text.config(state='normal')
        self.result_text.delete(1.0, tk.END)
        
        # Icon lớn và tên
        self.result_text.insert(tk.END, f"\n{info['icon']}  ", 'header')
        self.result_text.insert(tk.END, f"{info['name_vi'].upper()}\n", 'header')
        self.result_text.insert(tk.END, f"({predicted_class})\n\n", 'info')
        
        # Độ tin cậy với progress bar
        self.result_text.insert(tk.END, "🎯 Độ Tin Cậy: ", 'bold')
        
        if result['is_confident']:
            self.result_text.insert(tk.END, f"{confidence:.1f}% ✅\n", 'success')
        else:
            self.result_text.insert(tk.END, f"{confidence:.1f}% ⚠️\n", 'warning')
        
        # Progress bar
        bar_length = int(confidence / 2)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        self.result_text.insert(tk.END, f"{bar}\n\n")
        
        # Hướng dẫn xử lý
        self.result_text.insert(tk.END, "♻️  Cách Xử Lý:\n", 'bold')
        self.result_text.insert(tk.END, f"   {info['disposal']}\n\n")
        
        # Ví dụ
        self.result_text.insert(tk.END, "📝 Ví Dụ:\n", 'bold')
        self.result_text.insert(tk.END, f"   {', '.join(info['examples'])}\n\n")
        
        # Giá trị tái chế
        self.result_text.insert(tk.END, "💰 Giá Trị Tái Chế: ", 'bold')
        self.result_text.insert(tk.END, f"{info['recycling_value']}\n\n")
        
        # Separator
        self.result_text.insert(tk.END, "─" * 55 + "\n\n")
        
        # Chi tiết xác suất
        self.result_text.insert(tk.END, "📊 Chi Tiết Các Xác Suất:\n\n", 'bold')
        
        sorted_preds = sorted(
            result['all_predictions'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for cls, prob in sorted_preds:
            icon = CLASS_INFO[cls]['icon']
            bar_length = int(prob / 3)
            bar = "█" * bar_length
            self.result_text.insert(tk.END, f"{icon} {cls:11s} ")
            self.result_text.insert(tk.END, f"{bar:33s} {prob:5.1f}%\n")
        
        self.result_text.config(state='disabled')
    
    def save_current_frame(self):
        """Lưu frame hiện tại"""
        if self.current_frame is None:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"frame_{timestamp}.jpg"
        cv2.imwrite(filename, self.current_frame)
        messagebox.showinfo("Thành công", f"✅ Đã lưu: {filename}")
    
    def save_scan_result(self):
        """Lưu kết quả scan"""
        if not hasattr(self, 'current_result'):
            return
        
        result = self.current_result['result']
        predicted_class = result['class']
        confidence = result['confidence']
        
        if confidence < 80:
            response = messagebox.askyesno(
                "Xác nhận",
                f"Độ tin cậy thấp ({confidence:.1f}%).\nBạn có chắc muốn lưu?"
            )
            if not response:
                return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{predicted_class}_{timestamp}_{confidence:.0f}.jpg"
        save_path = os.path.join(self.data_save_dir, predicted_class, filename)
        
        cv2.imwrite(save_path, self.current_result['image'])
        
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
        
        self.scan_history.append(metadata)
        self.save_scan_history()
        self.update_statistics()
        
        messagebox.showinfo("Thành công", f"✅ Đã lưu kết quả!\n\n{save_path}")
        self.btn_save.config(state='disabled')
    
    def upload_image(self):
        """Upload ảnh"""
        if not self.model_loaded:
            messagebox.showerror("Lỗi", "Model chưa được load!")
            return
        
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        
        if file_path:
            img = cv2.imread(file_path)
            self.classify_image(file_path, img)
    
    def process_video(self):
        """Xử lý video"""
        if not self.model_loaded:
            messagebox.showerror("Lỗi", "Model chưa được load!")
            return
        
        video_path = filedialog.askopenfilename(
            title="Chọn video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        
        if not video_path:
            return
        
        save_output = messagebox.askyesno("Lưu video?", "Bạn có muốn lưu video kết quả không?")
        
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
        """Xử lý video thread"""
        try:
            cam_classifier = CameraClassifier(PATHS['model_save'])
            cam_classifier.classify_video_file(video_path, output_path)
            
            self.root.after(0, lambda: messagebox.showinfo("Thành công", "✅ Đã xử lý video!"))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Lỗi", f"Lỗi xử lý video: {str(e)}"))
    
    def batch_classify(self):
        """Phân loại batch"""
        if not self.model_loaded:
            messagebox.showerror("Lỗi", "Model chưa được load!")
            return
        
        folder_path = filedialog.askdirectory(title="Chọn thư mục chứa ảnh")
        
        if not folder_path:
            return
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        image_files = []
        
        for file in os.listdir(folder_path):
            ext = os.path.splitext(file)[1].lower()
            if ext in image_extensions:
                image_files.append(os.path.join(folder_path, file))
        
        if not image_files:
            messagebox.showwarning("Cảnh báo", "Không tìm thấy ảnh nào!")
            return
        
        threading.Thread(target=self.batch_classify_thread, args=(image_files,), daemon=True).start()
    
    def batch_classify_thread(self, image_files):
        """Batch classify thread"""
        try:
            results = self.classifier.predict_batch(image_files)
            self.root.after(0, lambda: self.show_batch_results(results))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Lỗi", f"Lỗi batch: {str(e)}"))
    
    def show_batch_results(self, results):
        """Hiển thị kết quả batch"""
        window = tk.Toplevel(self.root)
        window.title("📁 Kết Quả Batch")
        window.geometry("1100x750")
        window.configure(bg=self.colors['bg'])
        
        # Header
        header = tk.Frame(window, bg=self.colors['card'], height=70)
        header.pack(fill='x')
        header.pack_propagate(False)
        
        tk.Label(
            header,
            text="📊 Kết Quả Phân Loại Batch",
            font=('Segoe UI', 20, 'bold'),
            bg=self.colors['card'],
            fg=self.colors['primary']
        ).pack(pady=20)
        
        # Treeview
        tree_frame = tk.Frame(window, bg=self.colors['bg'])
        tree_frame.pack(fill='both', expand=True, padx=30, pady=20)
        
        columns = ('STT', 'File', 'Loại', 'Confidence', 'Status')
        tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=22)
        
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
                i, filename,
                f"{icon} {result['class_name_vi']}",
                f"{result['confidence']:.1f}%",
                status
            ))
        
        scrollbar = ttk.Scrollbar(tree_frame, orient='vertical', command=tree.yview)
        tree.configure(yscroll=scrollbar.set)
        
        tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Buttons
        btn_frame = tk.Frame(window, bg=self.colors['bg'])
        btn_frame.pack(pady=20)
        
        ModernButton(
            btn_frame, text="💾 Lưu CSV", bg=self.colors['primary'], fg='white',
            command=lambda: self.save_batch_csv(results)
        ).pack(side='left', padx=10)
        
        ModernButton(
            btn_frame, text="🚪 Đóng", bg=self.colors['secondary'], fg='white',
            command=window.destroy
        ).pack(side='left', padx=10)
    
    def save_batch_csv(self, results):
        """Lưu CSV"""
        file_path = filedialog.asksaveasfilename(
            title="Lưu CSV", defaultextension=".csv", filetypes=[("CSV files", "*.csv")]
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
                        i, os.path.basename(item['image']),
                        result['class'], result['class_name_vi'],
                        f"{result['confidence']:.2f}", status
                    ])
            
            messagebox.showinfo("Thành công", f"✅ Đã lưu: {file_path}")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi lưu CSV: {str(e)}")
    
    def show_training_panel(self):
        """Panel training"""
        window = tk.Toplevel(self.root)
        window.title("🎓 Training Model")
        window.geometry("900x700")
        window.configure(bg=self.colors['bg'])
        
        # Header
        header = tk.Frame(window, bg=self.colors['card'], height=80)
        header.pack(fill='x')
        header.pack_propagate(False)
        
        tk.Label(
            header, text="🎓 Training Model Mới",
            font=('Segoe UI', 22, 'bold'),
            bg=self.colors['card'], fg=self.colors['primary']
        ).pack(pady=25)
        
        # Form
        form = self.create_card(window)
        form.pack(fill='both', expand=True, padx=30, pady=20)
        
        # Train dir
        tk.Label(form, text="📁 Thư mục Training:", font=('Segoe UI', 12),
                bg=self.colors['card'], fg=self.colors['text']).pack(pady=(20, 5), anchor='w', padx=30)
        
        train_frame = tk.Frame(form, bg=self.colors['card'])
        train_frame.pack(fill='x', padx=30, pady=5)
        
        train_entry = tk.Entry(train_frame, font=('Segoe UI', 11), width=60,
                              relief='solid', bd=1)
        train_entry.pack(side='left', ipady=8, padx=(0, 10))
        
        ModernButton(train_frame, text="Browse", bg=self.colors['info'], fg='white',
                    command=lambda: train_entry.insert(0, filedialog.askdirectory())).pack()
        
        # Val dir
        tk.Label(form, text="📁 Thư mục Validation:", font=('Segoe UI', 12),
                bg=self.colors['card'], fg=self.colors['text']).pack(pady=(15, 5), anchor='w', padx=30)
        
        val_frame = tk.Frame(form, bg=self.colors['card'])
        val_frame.pack(fill='x', padx=30, pady=5)
        
        val_entry = tk.Entry(val_frame, font=('Segoe UI', 11), width=60,
                            relief='solid', bd=1)
        val_entry.pack(side='left', ipady=8, padx=(0, 10))
        
        ModernButton(val_frame, text="Browse", bg=self.colors['info'], fg='white',
                    command=lambda: val_entry.insert(0, filedialog.askdirectory())).pack()
        
        # Epochs
        tk.Label(form, text="⏱️ Số Epochs:", font=('Segoe UI', 12),
                bg=self.colors['card'], fg=self.colors['text']).pack(pady=(15, 5), anchor='w', padx=30)
        
        epochs_entry = tk.Entry(form, font=('Segoe UI', 11), width=20, relief='solid', bd=1)
        epochs_entry.insert(0, "50")
        epochs_entry.pack(anchor='w', padx=30, pady=5, ipady=8)
        
        # Transfer learning
        transfer_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            form, text="🔄 Sử dụng Transfer Learning",
            variable=transfer_var, font=('Segoe UI', 12),
            bg=self.colors['card'], fg=self.colors['text'],
            selectcolor=self.colors['card'], activebackground=self.colors['card']
        ).pack(pady=20, anchor='w', padx=30)
        
        # Button
        ModernButton(
            form, text="🚀 Bắt Đầu Training",
            bg=self.colors['success'], fg='white', width=25,
            command=lambda: self.start_training(
                train_entry.get(), val_entry.get(),
                int(epochs_entry.get()), transfer_var.get(), window
            )
        ).pack(pady=30)
    
    def start_training(self, train_dir, val_dir, epochs, use_transfer, window):
        """Bắt đầu training"""
        if not os.path.exists(train_dir) or not os.path.exists(val_dir):
            messagebox.showerror("Lỗi", "Thư mục không tồn tại!")
            return
        
        window.destroy()
        threading.Thread(
            target=self.training_thread,
            args=(train_dir, val_dir, epochs, use_transfer),
            daemon=True
        ).start()
        
        messagebox.showinfo("Training", "Training đã bắt đầu!\nKiểm tra console.")
    
    def training_thread(self, train_dir, val_dir, epochs, use_transfer):
        """Training thread"""
        try:
            model, history = train_model(train_dir, val_dir, epochs=epochs,
                                        use_transfer_learning=use_transfer)
            plot_training_history(history)
            self.load_model()
            
            self.root.after(0, lambda: messagebox.showinfo(
                "Thành công", "✅ Training hoàn tất!"))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Lỗi", f"Lỗi: {str(e)}"))
    
    def incremental_training(self):
        """Incremental training"""
        trainer = IncrementalTrainer()
        ready, stats = trainer.check_data_ready()
        
        if not ready:
            msg = "❌ Dữ liệu chưa đủ!\n\nCần ít nhất 20 mẫu chất lượng cao/class.\n\n"
            for cls, data in stats['by_class'].items():
                msg += f"{cls}: {data['high_confidence']} mẫu\n"
            
            messagebox.showwarning("Cảnh báo", msg)
            return
        
        if messagebox.askyesno("Xác nhận", f"✅ Dữ liệu sẵn sàng!\n\nTổng: {stats['total']}\nBắt đầu training?"):
            threading.Thread(target=self.incremental_training_thread,
                           args=(trainer,), daemon=True).start()
            messagebox.showinfo("Training", "Incremental training đã bắt đầu!")
    
    def incremental_training_thread(self, trainer):
        """Incremental training thread"""
        try:
            trainer.prepare_incremental_data()
            model, history = trainer.train_incremental(epochs=20, fine_tune=True)
            self.load_model()
            
            self.root.after(0, lambda: messagebox.showinfo("Thành công", "✅ Hoàn tất!"))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Lỗi", f"Lỗi: {str(e)}"))
    
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
    
    def prepare_dataset(self):
        """Chuẩn bị dataset"""
        if messagebox.askyesno("Xác nhận", "Chuẩn bị dữ liệu cho training?"):
            try:
                self.data_manager.prepare_training_data(min_confidence=80)
                messagebox.showinfo("Thành công", "✅ Đã chuẩn bị dataset!")
            except Exception as e:
                messagebox.showerror("Lỗi", f"{e}")
    
    def export_high_quality(self):
        """Export chất lượng cao"""
        output_dir = filedialog.askdirectory(title="Chọn thư mục lưu")
        if output_dir:
            try:
                self.data_manager.export_high_quality_data(output_dir, min_confidence=90)
                messagebox.showinfo("Thành công", f"✅ Đã export!")
            except Exception as e:
                messagebox.showerror("Lỗi", f"{e}")
    
    def clean_low_quality(self):
        """Xóa chất lượng thấp"""
        if messagebox.askyesno("Cảnh báo", "⚠️ Xóa ảnh ≤60%? Không thể hoàn tác!"):
            try:
                self.data_manager.clean_low_quality_data(max_confidence=60)
                messagebox.showinfo("Thành công", "✅ Đã xóa!")
                self.update_statistics()
            except Exception as e:
                messagebox.showerror("Lỗi", f"{e}")
    
    def show_statistics(self):
        """Thống kê"""
        stats = self.data_manager.get_scanned_stats()
        
        msg = f"""📊 THỐNG KÊ CHI TIẾT

{'═'*50}
TỔNG QUAN
{'═'*50}
• Tổng mẫu: {stats['total']}
• Chất lượng cao: {stats['high_confidence']}
• Tỷ lệ: {stats['high_confidence']/stats['total']*100 if stats['total'] > 0 else 0:.1f}%

{'═'*50}
CHI TIẾT
{'═'*50}
"""
        
        for cls in CLASSES:
            data = stats['by_class'][cls]
            icon = CLASS_INFO[cls]['icon']
            msg += f"\n{icon} {CLASS_INFO[cls]['name_vi']}:\n"
            msg += f"   Tổng: {data['count']}, Cao: {data['high_confidence']}, Thấp: {data['low_confidence']}\n"
        
        messagebox.showinfo("Thống Kê", msg)
    
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
    
    def show_camera_mode(self):
        """Chuyển về camera mode"""
        messagebox.showinfo(
            "Camera Mode",
            "📷 Chế độ camera đang hiển thị ở màn hình chính!\n\n" +
            "• Nhấn '▶️ Bật Camera' để bắt đầu\n" +
            "• Bật 'Tự động quét' để scan liên tục\n" +
            "• Khung xanh tự động theo dõi vật thể"
        )
    
    def show_history(self):
        """Hiển thị lịch sử"""
        window = tk.Toplevel(self.root)
        window.title("📜 Lịch Sử Scan")
        window.geometry("1100x750")
        window.configure(bg=self.colors['bg'])
        
        # Header
        header = tk.Frame(window, bg=self.colors['card'], height=80)
        header.pack(fill='x')
        header.pack_propagate(False)
        
        tk.Label(
            header,
            text="📜 Lịch Sử Phân Loại",
            font=('Segoe UI', 22, 'bold'),
            bg=self.colors['card'],
            fg=self.colors['primary']
        ).pack(pady=25)
        
        # Treeview
        tree_frame = tk.Frame(window, bg=self.colors['bg'])
        tree_frame.pack(fill='both', expand=True, padx=30, pady=20)
        
        columns = ('STT', 'Loại', 'Confidence', 'Thời gian', 'Mode')
        tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=25)
        
        tree.heading('STT', text='STT')
        tree.heading('Loại', text='Loại Rác')
        tree.heading('Confidence', text='Độ Tin Cậy')
        tree.heading('Thời gian', text='Thời Gian')
        tree.heading('Mode', text='Chế Độ')
        
        tree.column('STT', width=60)
        tree.column('Loại', width=280)
        tree.column('Confidence', width=130)
        tree.column('Thời gian', width=180)
        tree.column('Mode', width=120)
        
        # Thêm dữ liệu
        for i, item in enumerate(reversed(self.scan_history), 1):
            icon = CLASS_INFO[item['class']]['icon']
            mode = "🤖 Auto" if item.get('is_auto_scan', False) else "👤 Manual"
            
            tree.insert('', 'end', values=(
                i,
                f"{icon} {CLASS_INFO[item['class']]['name_vi']}",
                f"{item['confidence']:.1f}%",
                item['timestamp'],
                mode
            ))
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(tree_frame, orient='vertical', command=tree.yview)
        tree.configure(yscroll=scrollbar.set)
        
        tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Close button
        ModernButton(
            window,
            text="🚪 Đóng",
            bg=self.colors['secondary'],
            fg='white',
            command=window.destroy
        ).pack(pady=20)
    
    def save_scan_history(self):
        """Lưu lịch sử"""
        history_path = os.path.join(self.data_save_dir, 'scan_history.json')
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.scan_history, f, indent=2, ensure_ascii=False)
    
    def load_scan_history(self):
        """Load lịch sử"""
        history_path = os.path.join(self.data_save_dir, 'scan_history.json')
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r', encoding='utf-8') as f:
                    self.scan_history = json.load(f)
            except:
                self.scan_history = []
        else:
            self.scan_history = []
    
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
    
    def on_closing(self):
        """Xử lý đóng cửa sổ"""
        if self.camera_running:
            self.stop_camera()
        
        if messagebox.askokcancel("Thoát", "Bạn có chắc muốn thoát?"):
            self.root.destroy()


def main():
    """Main function"""
    root = tk.Tk()
    
    # Configure ttk style
    style = ttk.Style()
    style.theme_use('clam')
    
    # Treeview styling
    style.configure(
        "Treeview",
        background="#ffffff",
        foreground="#212529",
        fieldbackground="#ffffff",
        borderwidth=1,
        relief='solid',
        rowheight=30
    )
    
    style.configure(
        "Treeview.Heading",
        background="#f8f9fa",
        foreground="#0066cc",
        borderwidth=1,
        relief='solid',
        font=('Segoe UI', 10, 'bold')
    )
    
    style.map(
        'Treeview',
        background=[('selected', '#e3f2fd')],
        foreground=[('selected', '#0066cc')]
    )
    
    # Scrollbar styling
    style.configure(
        "Vertical.TScrollbar",
        background="#dee2e6",
        troughcolor="#f8f9fa",
        borderwidth=0,
        arrowsize=15
    )
    
    app = WasteClassifierGUIAdvanced(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()