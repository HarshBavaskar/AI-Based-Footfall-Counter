# -*- coding: utf-8 -*-
"""
AI Footfall Counter - Harsh Bavaskar
"""

import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
import threading
from footfall_counter import FootfallCounter
import os
from tkinter import filedialog, messagebox
import time

use_gpu=True
half_precision=False
skip_frames=2

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class FootfallApp(ctk.CTk):
    """AI FOOTFALL COUNTER"""

    def __init__(self):
        super().__init__()

        # Window configuration
        self.title("AI Footfall Counter - Harsh Bavaskar")
        self.geometry("1800x1000")

        # Application state
        self.counter = None
        self.processing = False
        self.current_source = None
        self.video_thread = None
        self.current_frame = None

        # Variables
        self.show_heatmap = ctk.BooleanVar(value=True)
        self.show_trajectories = ctk.BooleanVar(value=True)
        self.entry_count = 0
        self.exit_count = 0
        self.total_count = 0
        self.current_fps = 0.0

        self.setup_ui()

    def setup_ui(self):
        """Setup the perfect UI"""

        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Main container
        self.main_container = ctk.CTkFrame(self, corner_radius=0)
        self.main_container.grid(row=0, column=0, sticky="nsew")
        self.main_container.grid_columnconfigure(0, weight=1)
        self.main_container.grid_rowconfigure(1, weight=1)

        # Top bar
        self.top_bar = ctk.CTkFrame(self.main_container, height=100, corner_radius=0, fg_color=("gray85", "gray20"))
        self.top_bar.grid(row=0, column=0, sticky="ew")
        self.top_bar.grid_columnconfigure(1, weight=1)

        # Title
        self.title_frame = ctk.CTkFrame(self.top_bar, fg_color="transparent")
        self.title_frame.grid(row=0, column=0, padx=30, pady=20, sticky="w")

        ctk.CTkLabel(
            self.title_frame,
            text="AI FOOTFALL COUNTER",
            font=ctk.CTkFont(family="agency fb",size=32, weight="bold")
        ).pack(side="top", padx=(0, 20))

        ctk.CTkLabel(
            self.title_frame,
            text="Models used: YOLO v8 and DeepSORT",
            font=ctk.CTkFont(size=14),
            text_color="grey"
        ).pack(side="left")

        # Top stats
        self.top_stats_frame = ctk.CTkFrame(self.top_bar, fg_color="transparent")
        self.top_stats_frame.grid(row=0, column=1, padx=60, pady=10, sticky="e")

        self.create_mini_stat("👥", "0", "Entries", "forestgreen", 0)
        self.create_mini_stat("🚪", "0", "Exits", "purple", 1)
        self.create_mini_stat("📊", "0", "Total", "royalblue", 2)
        self.create_mini_stat("⚡", "0.0", "FPS", "orangered", 3)

        # Theme toggle
        self.theme_frame = ctk.CTkFrame(self.top_bar, fg_color="transparent")
        self.theme_frame.grid(row=0, column=2, padx=30, pady=20, sticky="e")

        ctk.CTkLabel(self.theme_frame, text="Theme:", font=ctk.CTkFont(size=12)).pack(side="left", padx=(0, 10))

        self.theme_menu = ctk.CTkOptionMenu(
            self.theme_frame,
            values=["Dark", "Light", "System"],
            command=self.change_theme,
            width=100,
            font=ctk.CTkFont(size=12)
        )
        self.theme_menu.pack(side="left")
        self.theme_menu.set("Dark")

        # Tabbed view
        self.tabview = ctk.CTkTabview(self.main_container, corner_radius=10)
        self.tabview.grid(row=1, column=0, sticky="nsew", padx=20, pady=(10, 20))

        # Create tabs
        self.tab_webcam = self.tabview.add("🎥 Webcam")
        self.tab_rtsp = self.tabview.add("🌐 RTSP Stream")
        self.tab_file = self.tabview.add("📁 Video File")
        self.tab_settings = self.tabview.add("⚙️ Settings")

        # Setup tabs
        self.setup_webcam_tab()
        self.setup_rtsp_tab()
        self.setup_file_tab()
        self.setup_settings_tab()

        # Status bar
        self.status_bar = ctk.CTkFrame(self.main_container, height=40, corner_radius=0, fg_color=("gray80", "gray25"))
        self.status_bar.grid(row=2, column=0, sticky="ew")

        self.status_label = ctk.CTkLabel(
            self.status_bar,
            text="Ready",
            fg_color= "grey",
            padx=60, pady=10,
            font=ctk.CTkFont(size=12),
            anchor="w"
        )
        self.status_label.pack(side="left", padx=20, pady=10)

        self.info_label = ctk.CTkLabel(
            self.status_bar,
            text="Made by Harsh Bavaskar",
            font=ctk.CTkFont(size=12),
            anchor="w"
        )
        self.info_label.pack(side="right", padx=20, pady=10)

        self.time_label = ctk.CTkLabel(
            self.status_bar,
            text=self.get_current_time(),
            font=ctk.CTkFont(size=12)
        )
        self.time_label.pack(side="right", padx=20, pady=10)

        self.update_time()

    def create_mini_stat(self, icon, value, label, color, column):
        """Create mini stat card"""
        card = ctk.CTkFrame(self.top_stats_frame, fg_color=color, corner_radius=8, width=120, height=60)
        card.grid(row=0, column=column, padx=5, sticky="ew")
        card.grid_propagate(False)

        content_frame = ctk.CTkFrame(card, fg_color="transparent")
        content_frame.pack(expand=True, fill="both", padx=10, pady=5)

        ctk.CTkLabel(
            content_frame,
            text=f"{icon} {value}",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="white"
        ).pack()

        label_widget = ctk.CTkLabel(
            content_frame,
            text=label,
            font=ctk.CTkFont(size=10),
            text_color="white"
        )
        label_widget.pack()

        if "Entries" in label:
            self.mini_entry_label = content_frame.winfo_children()[0]
        elif "Exits" in label:
            self.mini_exit_label = content_frame.winfo_children()[0]
        elif "Total" in label:
            self.mini_total_label = content_frame.winfo_children()[0]
        elif "FPS" in label:
            self.mini_fps_label = content_frame.winfo_children()[0]

    def setup_webcam_tab(self):
        """Setup webcam tab with scrollable full resolution display"""
        self.tab_webcam.grid_columnconfigure(0, weight=1)
        self.tab_webcam.grid_rowconfigure(1, weight=1)

        # Control panel
        control_frame = ctk.CTkFrame(self.tab_webcam, corner_radius=10)
        control_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

        ctk.CTkLabel(
            control_frame,
            text="Webcam Real-time Processing",
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(pady=(15, 10))

        # Buttons
        btn_frame = ctk.CTkFrame(control_frame, fg_color="transparent")
        btn_frame.pack(pady=(0, 15))

        self.webcam_start_btn = ctk.CTkButton(
            btn_frame,
            text="🎥 Start Webcam",
            command=self.start_webcam,
            width=200,
            height=45,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="green",
            hover_color="darkgreen"
        )
        self.webcam_start_btn.pack(side="left", padx=5)

        self.webcam_stop_btn = ctk.CTkButton(
            btn_frame,
            text="⏹️ Stop",
            command=self.stop_processing,
            width=150,
            height=45,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="red",
            hover_color="darkred",
            state="disabled"
        )
        self.webcam_stop_btn.pack(side="left", padx=5)

        # Options
        options_frame = ctk.CTkFrame(control_frame, fg_color="transparent")
        options_frame.pack(pady=(0, 15))

        ctk.CTkSwitch(
            options_frame,
            text="Show Heatmap",
            variable=self.show_heatmap,
            font=ctk.CTkFont(size=13)
        ).pack(side="left", padx=15)

        ctk.CTkSwitch(
            options_frame,
            text="Show Trajectories",
            variable=self.show_trajectories,
            font=ctk.CTkFont(size=13)
        ).pack(side="left", padx=15)

        # Scrollable video display - FULL RESOLUTION
        self.webcam_scroll = ctk.CTkScrollableFrame(
            self.tab_webcam,
            corner_radius=10,
            label_text="📹 Video Feed"
        )
        self.webcam_scroll.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.webcam_scroll.grid_columnconfigure(0, weight=1)

        self.webcam_video_label = ctk.CTkLabel(
            self.webcam_scroll,
            text="📹\n\nClick 'Start Webcam' to begin processing",
            font=ctk.CTkFont(size=20)
        )
        self.webcam_video_label.grid(row=0, column=0, padx=20, pady=20)

    def setup_rtsp_tab(self):
        """Setup RTSP tab with full resolution"""
        self.tab_rtsp.grid_columnconfigure(0, weight=1)
        self.tab_rtsp.grid_rowconfigure(2, weight=1)

        # Control panel
        control_frame = ctk.CTkFrame(self.tab_rtsp, corner_radius=10)
        control_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

        ctk.CTkLabel(
            control_frame,
            text="RTSP Stream",
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(pady=(15, 10))

        # URL input
        url_frame = ctk.CTkFrame(control_frame, fg_color="transparent")
        url_frame.pack(pady=(0, 15))

        ctk.CTkLabel(url_frame, text="RTSP URL:", font=ctk.CTkFont(size=14)).pack(side="left", padx=(0, 10))

        self.rtsp_url_entry = ctk.CTkEntry(
            url_frame,
            width=500,
            height=40,
            placeholder_text="rtsp://username:password@192.168.1.100:554/stream",
            font=ctk.CTkFont(size=13)
        )
        self.rtsp_url_entry.pack(side="left", padx=5)

        # Buttons
        btn_frame = ctk.CTkFrame(control_frame, fg_color="transparent")
        btn_frame.pack(pady=(0, 15))

        self.rtsp_start_btn = ctk.CTkButton(
            btn_frame,
            text="🌐 Connect & Start",
            command=self.start_rtsp,
            width=200,
            height=45,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="green",
            hover_color="darkgreen"
        )
        self.rtsp_start_btn.pack(side="left", padx=5)

        self.rtsp_stop_btn = ctk.CTkButton(
            btn_frame,
            text="⏹️ Stop",
            command=self.stop_processing,
            width=150,
            height=45,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="red",
            hover_color="darkred",
            state="disabled"
        )
        self.rtsp_stop_btn.pack(side="left", padx=5)

        # Examples
        examples_frame = ctk.CTkFrame(self.tab_rtsp, corner_radius=10, fg_color=("gray90", "gray17"))
        examples_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))

        ctk.CTkLabel(
            examples_frame,
            text="📝 Common RTSP URL Formats:",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(10, 5), anchor="w", padx=15)

        examples = [
            "Generic: rtsp://[username]:[password]@[ip]:[port]/[stream]",
            "Hikvision: rtsp://admin:password@192.168.1.64:554/Streaming/Channels/101",
            "Simple: rtsp://192.168.1.100:554/stream"
        ]

        for example in examples:
            ctk.CTkLabel(
                examples_frame,
                text=f"  • {example}",
                font=ctk.CTkFont(size=11),
                text_color="gray",
                anchor="w"
            ).pack(pady=2, anchor="w", padx=30)

        ctk.CTkLabel(examples_frame, text="").pack(pady=5)

        # Scrollable video display
        self.rtsp_scroll = ctk.CTkScrollableFrame(
            self.tab_rtsp,
            corner_radius=10,
            label_text="🌐 Stream Feed"
        )
        self.rtsp_scroll.grid(row=2, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.rtsp_scroll.grid_columnconfigure(0, weight=1)

        self.rtsp_video_label = ctk.CTkLabel(
            self.rtsp_scroll,
            text="🌐\n\nEnter RTSP URL and click 'Connect & Start'",
            font=ctk.CTkFont(size=20)
        )
        self.rtsp_video_label.grid(row=0, column=0, padx=20, pady=20)

    def setup_file_tab(self):
        """Setup file tab with full resolution and progress"""
        self.tab_file.grid_columnconfigure(0, weight=1)
        self.tab_file.grid_rowconfigure(2, weight=1)

        # Control panel
        control_frame = ctk.CTkFrame(self.tab_file, corner_radius=10)
        control_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

        ctk.CTkLabel(
            control_frame,
            text="Video File Processing",
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(pady=(15, 10))

        # File selection
        file_frame = ctk.CTkFrame(control_frame, fg_color="transparent")
        file_frame.pack(pady=(0, 15))

        self.file_input_label = ctk.CTkLabel(
            file_frame,
            text="No file selected",
            font=ctk.CTkFont(size=13),
            text_color="gray"
        )
        self.file_input_label.pack(side="left", padx=10)

        ctk.CTkButton(
            file_frame,
            text="📂 Select Video File",
            command=self.select_input_file,
            width=180,
            height=40,
            font=ctk.CTkFont(size=14)
        ).pack(side="left", padx=5)

        # Process button
        btn_frame = ctk.CTkFrame(control_frame, fg_color="transparent")
        btn_frame.pack(pady=(0, 15))

        self.file_process_btn = ctk.CTkButton(
            btn_frame,
            text="▶️ Process Video",
            command=self.process_file,
            width=200,
            height=45,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="green",
            hover_color="darkgreen",
            state="disabled"
        )
        self.file_process_btn.pack(side="left", padx=5)

        # Progress section
        self.progress_frame = ctk.CTkFrame(self.tab_file, corner_radius=10, fg_color=("gray90", "gray17"))
        self.progress_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))

        ctk.CTkLabel(
            self.progress_frame,
            text="Processing Progress",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(15, 10))

        self.file_progress = ctk.CTkProgressBar(
            self.progress_frame,
            width=1500,
            height=25,
            corner_radius=10
        )
        self.file_progress.pack(pady=10, padx=30)
        self.file_progress.set(0)

        self.file_progress_label = ctk.CTkLabel(
            self.progress_frame,
            text="0% - Ready to process",
            font=ctk.CTkFont(size=14)
        )
        self.file_progress_label.pack(pady=(0, 15))

        # Scrollable preview
        self.file_scroll = ctk.CTkScrollableFrame(
            self.tab_file,
            corner_radius=10,
            label_text="📁 Video Preview"
        )
        self.file_scroll.grid(row=2, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.file_scroll.grid_columnconfigure(0, weight=1)

        self.file_video_label = ctk.CTkLabel(
            self.file_scroll,
            text="📁\n\nSelect a video file to begin",
            font=ctk.CTkFont(size=20)
        )
        self.file_video_label.grid(row=0, column=0, padx=20, pady=20)

    def setup_settings_tab(self):
        """Setup settings tab"""
        self.tab_settings.grid_columnconfigure(0, weight=1)

        header = ctk.CTkFrame(self.tab_settings, corner_radius=10, fg_color=("gray90", "gray17"))
        header.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

        ctk.CTkLabel(
            header,
            text="⚙️ Settings & Actions",
            font=ctk.CTkFont(size=24, weight="bold")
        ).pack(pady=20)

        settings_container = ctk.CTkFrame(self.tab_settings, fg_color="transparent")
        settings_container.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        settings_container.grid_columnconfigure((0, 1), weight=1)

        # Visualization
        viz_frame = ctk.CTkFrame(settings_container, corner_radius=10)
        viz_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=5)

        ctk.CTkLabel(
            viz_frame,
            text="🎨 Visualization Options",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=(20, 15))

        ctk.CTkSwitch(
            viz_frame,
            text="Enable Heatmap Overlay",
            variable=self.show_heatmap,
            font=ctk.CTkFont(size=14)
        ).pack(pady=10, padx=30, anchor="w")

        ctk.CTkLabel(
            viz_frame,
            text="Shows traffic density with color gradient",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        ).pack(pady=(0, 15), padx=50, anchor="w")

        ctk.CTkSwitch(
            viz_frame,
            text="Enable Trajectory Paths",
            variable=self.show_trajectories,
            font=ctk.CTkFont(size=14)
        ).pack(pady=10, padx=30, anchor="w")

        ctk.CTkLabel(
            viz_frame,
            text="Displays movement paths with gradient trails",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        ).pack(pady=(0, 20), padx=50, anchor="w")

        # Actions
        actions_frame = ctk.CTkFrame(settings_container, corner_radius=10)
        actions_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=5)

        ctk.CTkLabel(
            actions_frame,
            text="⚡ Actions",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=(20, 20))

        ctk.CTkButton(
            actions_frame,
            text="🔄 Reset All Counts",
            command=self.reset_counts,
            width=300,
            height=50,
            font=ctk.CTkFont(size=15),
            fg_color="orange",
            hover_color="darkorange"
        ).pack(pady=10, padx=30)

        ctk.CTkButton(
            actions_frame,
            text="📸 Take Screenshot",
            command=self.take_screenshot,
            width=300,
            height=50,
            font=ctk.CTkFont(size=15)
        ).pack(pady=10, padx=30)

        ctk.CTkButton(
            actions_frame,
            text="💾 Export Report",
            command=self.export_report,
            width=300,
            height=50,
            font=ctk.CTkFont(size=15)
        ).pack(pady=10, padx=30)

        # Info
        info_frame = ctk.CTkFrame(settings_container, corner_radius=10, fg_color=("gray90", "gray17"))
        info_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=0, pady=(10, 5))

        ctk.CTkLabel(
            info_frame,
            text="ℹ️ Made by Harsh Bavaskar",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(15, 10))

        info_text = """
        Features:
        • Real-time webcam processing with full frame visible
        • RTSP stream support with complete video
        • File processing with progress tracking
        • Heatmap and trajectory visualization
        """

        ctk.CTkLabel(
            info_frame,
            text=info_text,
            font=ctk.CTkFont(size=12),
            justify="left"
        ).pack(pady=(0, 15), padx=30)

    def select_input_file(self):
        """Select input file"""
        filepath = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"),
                ("All files", "*.*")
            ]
        )

        if filepath:
            self.selected_input_file = filepath
            filename = os.path.basename(filepath)
            self.file_input_label.configure(text=f"Selected: {filename}")
            self.file_process_btn.configure(state="normal")

    def start_webcam(self):
        """Start webcam"""
        if self.processing:
            messagebox.showwarning("Warning", "Already processing!", weight = "bold")
            return

        self.counter = FootfallCounter(model_path='yolov8n.pt')
        self.current_source = 0
        self.processing = True
        self.status_label.configure(text="Processing • Webcam", fg_color= "green", padx=60, pady=10)
        self.webcam_start_btn.configure(state="disabled")
        self.webcam_stop_btn.configure(state="normal")

        self.video_thread = threading.Thread(target=self._process_webcam, daemon=True)
        self.video_thread.start()

    def start_rtsp(self):
        """Start RTSP"""
        if self.processing:
            messagebox.showwarning("Warning", "Already processing!")
            return

        rtsp_url = self.rtsp_url_entry.get().strip()

        if not rtsp_url:
            messagebox.showwarning("Warning", "Please enter an RTSP URL")
            return

        self.counter = FootfallCounter(model_path='yolov8n.pt')
        self.current_source = rtsp_url
        self.processing = True
        self.status_label.configure(text=f"Processing • RTSP", fg_color= "green", padx=60, pady=10)

        self.rtsp_start_btn.configure(state="disabled")
        self.rtsp_stop_btn.configure(state="normal")

        self.video_thread = threading.Thread(target=self._process_rtsp, daemon=True)
        self.video_thread.start()

    def process_file(self):
        """Process file"""
        if self.processing:
            messagebox.showwarning("Warning", "Already processing!")
            return

        if not hasattr(self, 'selected_input_file'):
            messagebox.showwarning("Warning", "Please select a video file first")
            return

        output_path = filedialog.asksaveasfilename(
            title="Save Processed Video",
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
        )

        if not output_path:
            return

        self.counter = FootfallCounter(model_path='yolov8n.pt')
        self.processing = True
        self.file_process_btn.configure(state="disabled")
        self.file_progress.set(0)
        self.file_progress_label.configure(text="0% - Starting...")
        self.status_label.configure(text="Processing • File", fg_color= "green", padx=60, pady=10)

        threading.Thread(
            target=self._process_file,
            args=(self.selected_input_file, output_path),
            daemon=True
        ).start()

    def _process_webcam(self):
        """Process webcam"""
        cap = cv2.VideoCapture(self.current_source)

        if not cap.isOpened():
            messagebox.showerror("Error", "Cannot open webcam")
            self.processing = False
            return

        try:
            while self.processing:
                ret, frame = cap.read()
                if not ret:
                    continue

                processed_frame = self.counter.process_frame(
                    frame,
                    show_heatmap=self.show_heatmap.get(),
                    show_trajectories=self.show_trajectories.get()
                )

                self.current_frame = processed_frame
                self.update_statistics()
                self.display_frame_full_res(processed_frame, self.webcam_video_label)

        finally:
            cap.release()
            self.processing = False
            self.webcam_start_btn.configure(state="normal")
            self.webcam_stop_btn.configure(state="disabled")
            self.status_label.configure(text="Ready" , fg_color= "gray", padx=60, pady=10)

    def _process_rtsp(self):
        """Process RTSP"""
        cap = cv2.VideoCapture(self.current_source)

        if not cap.isOpened():
            messagebox.showerror("Error", f"Cannot connect to RTSP stream")
            self.processing = False
            return

        try:
            while self.processing:
                ret, frame = cap.read()
                if not ret:
                    continue

                processed_frame = self.counter.process_frame(
                    frame,
                    show_heatmap=self.show_heatmap.get(),
                    show_trajectories=self.show_trajectories.get()
                )

                self.current_frame = processed_frame
                self.update_statistics()
                self.display_frame_full_res(processed_frame, self.rtsp_video_label)

        finally:
            cap.release()
            self.processing = False
            self.rtsp_start_btn.configure(state="normal")
            self.rtsp_stop_btn.configure(state="disabled")
            self.status_label.configure(text="Ready", fg_color= "gray", padx=60, pady=10)

    def _process_file(self, input_path, output_path):
        """Process file"""
        try:
            cap = cv2.VideoCapture(input_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                processed_frame = self.counter.process_frame(
                    frame,
                    show_heatmap=self.show_heatmap.get(),
                    show_trajectories=self.show_trajectories.get()
                )

                out.write(processed_frame)

                frame_count += 1
                progress = frame_count / total_frames
                percent = int(progress * 100)

                self.file_progress.set(progress)
                self.file_progress_label.configure(
                    text=f"{percent}% - Processing frame {frame_count}/{total_frames}"
                )

                self.update_statistics()

                if frame_count % 10 == 0:
                    self.display_frame_full_res(processed_frame, self.file_video_label)

            cap.release()
            out.release()

            self.file_progress.set(1.0)
            self.file_progress_label.configure(text="100% - Complete!")

            result = {
                'entry_count': self.counter.entry_count,
                'exit_count': self.counter.exit_count,
                'total_count': self.counter.entry_count + self.counter.exit_count
            }

            messagebox.showinfo(
                "Success",
                f"Processing complete!\n\n"
                f"Entries: {result['entry_count']}\n"
                f"Exits: {result['exit_count']}\n"
                f"Total: {result['total_count']}\n\n"
                f"Saved to: {output_path}"
            )

        except Exception as e:
            messagebox.showerror("Error", f"Processing failed: {str(e)}")

        finally:
            self.processing = False
            self.file_process_btn.configure(state="normal")
            self.status_label.configure(text="Ready", fg_color= "gray", padx=60, pady=10)

    def display_frame_full_res(self, frame, label_widget):
        """Display frame in FULL RESOLUTION with proper aspect ratio - NO CROPPING"""
        # Get original dimensions
        height, width = frame.shape[:2]

        # Calculate aspect ratio
        aspect_ratio = width / height

        # Max display size (can be scrolled if larger)
        max_width = 1600
        max_height = 800

        # Scale to fit while maintaining aspect ratio
        if width > max_width or height > max_height:
            if width / max_width > height / max_height:
                new_width = max_width
                new_height = int(max_width / aspect_ratio)
            else:
                new_height = max_height
                new_width = int(max_height * aspect_ratio)
        else:
            new_width = width
            new_height = height

        # Resize maintaining aspect ratio
        frame_resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        # Update label - NO CROPPING!
        label_widget.configure(image=imgtk, text="")
        label_widget.image = imgtk

    def update_statistics(self):
        """Update statistics"""
        if self.counter:
            self.entry_count = self.counter.entry_count
            self.exit_count = self.counter.exit_count
            self.total_count = self.entry_count + self.exit_count

            if len(self.counter.fps_history) > 0:
                self.current_fps = sum(self.counter.fps_history) / len(self.counter.fps_history)

            self.mini_entry_label.configure(text=f"👥 {self.entry_count}")
            self.mini_exit_label.configure(text=f"🚪 {self.exit_count}")
            self.mini_total_label.configure(text=f"📊 {self.total_count}")
            self.mini_fps_label.configure(text=f"⚡ {self.current_fps:.1f}")

    def stop_processing(self):
        """Stop processing"""
        if self.processing:
            self.processing = False
            self.status_label.configure(text="Stopping...", fg_color= "red", padx=60, pady=10)

    def reset_counts(self):
        """Reset counts"""
        if self.counter:
            self.counter.entry_count = 0
            self.counter.exit_count = 0
            self.counter.counted_ids.clear()
            self.update_statistics()
            messagebox.showinfo("Success", "All counts reset")

    def take_screenshot(self):
        """Take screenshot"""
        if not self.processing or self.current_frame is None:
            messagebox.showinfo("Info", "No active video stream")
            return

        save_path = filedialog.asksaveasfilename(
            title="Save Screenshot",
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png")]
        )

        if save_path:
            cv2.imwrite(save_path, self.current_frame)
            messagebox.showinfo("Success", f"Screenshot saved to:\n{save_path}")

    def export_report(self):
        """Export report"""
        if not self.counter:
            messagebox.showinfo("Info", "No data to export")
            return

        save_path = filedialog.asksaveasfilename(
            title="Export Report",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv")]
        )

        if save_path:
            try:
                with open(save_path, 'w') as f:
                    f.write("="*60 + "\n")
                    f.write("AI Footfall Counter - Statistics Report\n")
                    f.write("="*60 + "\n\n")
                    f.write(f"Generated: {self.get_current_time()}\n")
                    f.write(f"Source: {self.current_source}\n\n")
                    f.write(f"Total Entries: {self.entry_count}\n")
                    f.write(f"Total Exits: {self.exit_count}\n")
                    f.write(f"Total Count: {self.total_count}\n")
                    f.write(f"Average FPS: {self.current_fps:.2f}\n")
                    f.write("\n" + "="*60 + "\n")

                messagebox.showinfo("Success", f"Report exported to:\n{save_path}")

            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {str(e)}")

    def change_theme(self, choice):
        """Change theme"""
        ctk.set_appearance_mode(choice.lower())

    def get_current_time(self):
        """Get current time"""
        return time.strftime("%I:%M %p • %B %d, %Y")

    def update_time(self):
        """Update time"""
        self.time_label.configure(text=self.get_current_time())
        self.after(1000, self.update_time)


def main():
    """Main entry point"""
    app = FootfallApp()
    app.mainloop()


if __name__ == '__main__':
    main()
