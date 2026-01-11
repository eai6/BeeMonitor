#!/usr/bin/env python3
"""
BeeMonitor Configuration GUI v3.0 - Video Player Edition
========================================================

Enhanced with:
- Play/Pause video controls
- Results visualization overlay
- Simplified, intuitive interface
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QTextEdit, QFileDialog,
    QMessageBox, QTabWidget, QGroupBox, QGridLayout, QSplitter,
    QScrollArea, QProgressDialog, QComboBox, QCheckBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QPoint
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QAction

try:
    from beemonitor import BeeMonitor
    from beemonitor.core.config import Config
    from beemonitor.detection import BlobDetector
except ImportError:
    QMessageBox.critical(None, "Import Error",
                        "Cannot import beemonitor. Please install the package.")
    sys.exit(1)


class AnalysisThread(QThread):
    """Background thread for running video analysis."""
    
    progress = pyqtSignal(str)
    finished = pyqtSignal(object, str)
    error = pyqtSignal(str)
    
    def __init__(self, monitor, video_path, output_folder):
        super().__init__()
        self.monitor = monitor
        self.video_path = video_path
        self.output_folder = output_folder
    
    def run(self):
        try:
            self.progress.emit("Initializing analysis...")
            self.progress.emit("Nest detector will automatically detect hotel ROI...")
            
            # Check if analyze_video accepts these parameters
            import inspect
            sig = inspect.signature(self.monitor.analyze_video)
            
            # Build kwargs based on what's accepted
            kwargs = {
                'video_path': self.video_path,
            }
            
            # Add optional parameters if supported
            if 'output_folder' in sig.parameters:
                kwargs['output_folder'] = self.output_folder
            if 'visualize' in sig.parameters:
                kwargs['visualize'] = True
            
            self.progress.emit("Running analysis...")
            result = self.monitor.analyze_video(**kwargs)
            
            # Save results
            csv_path = os.path.join(self.output_folder, 'tracking_results.csv')
            
            try:
                import pandas as pd
                
                if hasattr(result, 'to_csv') and callable(result.to_csv):
                    import inspect
                    sig = inspect.signature(result.to_csv)
                    if 'index' in sig.parameters:
                        result.to_csv(csv_path, index=False)
                    else:
                        result.to_csv(csv_path)
                        
                elif hasattr(result, 'tracks'):
                    tracks = result.tracks
                    if isinstance(tracks, list):
                        pd.DataFrame(tracks).to_csv(csv_path, index=False)
                    elif isinstance(tracks, pd.DataFrame):
                        tracks.to_csv(csv_path, index=False)
                    else:
                        pd.DataFrame(tracks).to_csv(csv_path, index=False)
                else:
                    if isinstance(result, dict):
                        if 'tracks' in result:
                            pd.DataFrame(result['tracks']).to_csv(csv_path, index=False)
                        else:
                            pd.DataFrame(result).to_csv(csv_path, index=False)
                    elif isinstance(result, list):
                        pd.DataFrame(result).to_csv(csv_path, index=False)
                    else:
                        pd.DataFrame(result).to_csv(csv_path, index=False)
                        
            except Exception as e:
                self.progress.emit(f"Warning: Could not save CSV: {e}")
            
            self.finished.emit(result, csv_path)
            
        except Exception as e:
            import traceback
            error_details = f"{str(e)}\n\n{traceback.format_exc()}"
            self.error.emit(error_details)


class VideoCanvas(QLabel):
    """Video display widget with overlay support."""
    
    roi_changed = pyqtSignal(tuple)
    
    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 480)
        self.setStyleSheet("background-color: black;")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.current_frame = None
        self.original_pixmap = None
        
        # Visualization overlays
        self.show_detections = False
        self.show_tracks = False
        self.detections = []
        self.tracks = {}  # {track_id: [(x, y), ...]}
        self.roi = None
        
        # ROI drawing
        self.drawing_roi = False
        self.roi_start = None
        self.roi_current = None
        self.setMouseTracking(True)
    
    def set_frame(self, frame, detections=None, tracks=None, roi=None):
        """Update displayed frame with optional overlays."""
        self.current_frame = frame.copy()
        
        if detections is not None:
            self.detections = detections
            print(f"VideoCanvas: Received {len(detections)} detections")
        if tracks is not None:
            self.tracks = tracks
            print(f"VideoCanvas: Received {len(tracks)} tracks")
        if roi is not None:
            self.roi = roi
        
        print(f"VideoCanvas state: show_detections={self.show_detections}, show_tracks={self.show_tracks}")
        self._draw_frame()
    
    def _draw_frame(self):
        """Draw frame with all enabled overlays."""
        if self.current_frame is None:
            return
        
        frame = self.current_frame.copy()
        
        # Draw ROI
        if self.roi is not None:
            x1, y1, x2, y2 = self.roi
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "ROI", (x1+5, y1+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw temporary ROI during drawing
        if self.drawing_roi and self.roi_start and self.roi_current:
            x1, y1 = self.roi_start
            x2, y2 = self.roi_current
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        
        # Draw detections
        detections_drawn = 0
        if self.show_detections and self.detections:
            if len(self.detections) > 0:
                print(f"Drawing {len(self.detections)} detections")
            for det in self.detections:
                x1, y1, x2, y2 = [int(c) for c in det.bbox]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                area = (x2-x1) * (y2-y1)
                cv2.putText(frame, f"{area:.0f}", (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                detections_drawn += 1
        
        # Draw tracks
        tracks_drawn = 0
        if self.show_tracks and self.tracks:
            if len(self.tracks) > 0:
                print(f"Drawing {len(self.tracks)} tracks")
            colors = [(255, 0, 0), (0, 255, 255), (255, 0, 255), 
                     (255, 255, 0), (128, 0, 255), (255, 128, 0)]
            
            for i, (track_id, trajectory) in enumerate(self.tracks.items()):
                color = colors[i % len(colors)]
                
                # Draw trajectory
                if len(trajectory) > 1:
                    points = np.array(trajectory, dtype=np.int32)
                    cv2.polylines(frame, [points], False, color, 2)
                
                # Draw current position
                if trajectory:
                    x, y = trajectory[-1]
                    cv2.circle(frame, (int(x), int(y)), 5, color, -1)
                    cv2.putText(frame, f"ID:{track_id}", (int(x)+10, int(y)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    tracks_drawn += 1
        
        # Add status overlay
        status_lines = []
        if self.show_detections:
            status_lines.append(f"Detections: {len(self.detections)} ({detections_drawn} drawn)")
        if self.show_tracks:
            status_lines.append(f"Tracks: {len(self.tracks)} ({tracks_drawn} drawn)")
        
        if status_lines:
            y_pos = 30
            for line in status_lines:
                cv2.putText(frame, line, (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_pos += 30
        
        # Convert to QPixmap
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        self.original_pixmap = pixmap
        self.setPixmap(scaled_pixmap)
    
    def toggle_detections(self, enabled):
        """Toggle detection overlay."""
        self.show_detections = enabled
        print(f"Detections overlay: {'ON' if enabled else 'OFF'}")
        self._draw_frame()
    
    def toggle_tracks(self, enabled):
        """Toggle track overlay."""
        self.show_tracks = enabled
        print(f"Tracks overlay: {'ON' if enabled else 'OFF'}")
        self._draw_frame()
    
    def start_roi_drawing(self):
        """Start ROI drawing mode."""
        self.drawing_roi = True
        self.roi_start = None
        self.roi_current = None
        self.setCursor(Qt.CursorShape.CrossCursor)
    
    def clear_roi(self):
        """Clear ROI."""
        self.roi = None
        self._draw_frame()
    
    def mousePressEvent(self, event):
        """Handle mouse press for ROI drawing."""
        if not self.drawing_roi or self.current_frame is None:
            return
        
        pos = self._widget_to_image_coords(event.pos())
        if pos:
            self.roi_start = pos
    
    def mouseMoveEvent(self, event):
        """Handle mouse move for ROI drawing."""
        if not self.drawing_roi or not self.roi_start:
            return
        
        pos = self._widget_to_image_coords(event.pos())
        if pos:
            self.roi_current = pos
            self._draw_frame()
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release for ROI drawing."""
        if not self.drawing_roi or not self.roi_start:
            return
        
        pos = self._widget_to_image_coords(event.pos())
        if pos:
            x1, y1 = self.roi_start
            x2, y2 = pos
            
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            h, w = self.current_frame.shape[:2]
            x1 = max(0, min(w, x1))
            x2 = max(0, min(w, x2))
            y1 = max(0, min(h, y1))
            y2 = max(0, min(h, y2))
            
            self.roi = (x1, y1, x2, y2)
            self.roi_changed.emit(self.roi)
        
        self.drawing_roi = False
        self.roi_start = None
        self.roi_current = None
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self._draw_frame()
    
    def _widget_to_image_coords(self, widget_pos):
        """Convert widget coordinates to image coordinates."""
        if self.original_pixmap is None or self.current_frame is None:
            return None
        
        pixmap = self.pixmap()
        if pixmap is None:
            return None
        
        x_offset = (self.width() - pixmap.width()) // 2
        y_offset = (self.height() - pixmap.height()) // 2
        
        x = widget_pos.x() - x_offset
        y = widget_pos.y() - y_offset
        
        if x < 0 or y < 0 or x >= pixmap.width() or y >= pixmap.height():
            return None
        
        scale_x = self.current_frame.shape[1] / pixmap.width()
        scale_y = self.current_frame.shape[0] / pixmap.height()
        
        img_x = int(x * scale_x)
        img_y = int(y * scale_y)
        
        return (img_x, img_y)
    
    def resizeEvent(self, event):
        """Handle resize."""
        super().resizeEvent(event)
        if self.current_frame is not None:
            self._draw_frame()


class BeeMonitorGUI(QMainWindow):
    """Main GUI application with video player controls."""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("BeeMonitor - Video Analysis Tool")
        self.setGeometry(100, 100, 1400, 900)
        
        # Video state
        self.video_path = None
        self.video_cap = None
        self.current_frame = None
        self.current_frame_idx = 0
        self.total_frames = 0
        self.fps = 0
        
        # Playback state
        self.playing = False
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self.play_next_frame)
        
        # Analysis results
        self.tracking_results = None
        self.results_loaded = False
        
        # Config and detectors
        self.config = Config.default()
        self.blob_detector = None
        self.output_folder = None  # Will be set when video loads
        
        self.analysis_thread = None
        
        # Setup UI
        self._create_menu_bar()
        self._create_main_widget()
        
        # Sync initial checkbox states with VideoCanvas
        self.video_canvas.show_detections = self.show_detections_cb.isChecked()
        self.video_canvas.show_tracks = self.show_tracks_cb.isChecked()
        print(f"Initial state: show_detections={self.video_canvas.show_detections}, show_tracks={self.video_canvas.show_tracks}")
        
        self.statusBar().showMessage("Ready - Load a video to begin")
        
        print("✓ BeeMonitor GUI v3.0 initialized (Video Player Edition)")
    
    def _create_menu_bar(self):
        """Create menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        load_action = QAction("&Load Video...", self)
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self.load_video)
        file_menu.addAction(load_action)
        
        load_results_action = QAction("Load &Results...", self)
        load_results_action.setShortcut("Ctrl+L")
        load_results_action.triggered.connect(self.load_results)
        file_menu.addAction(load_results_action)
        
        load_output_video_action = QAction("Load Output &Video...", self)
        load_output_video_action.triggered.connect(self.load_output_video)
        file_menu.addAction(load_output_video_action)
        
        file_menu.addSeparator()
        
        save_visualization_action = QAction("Save &Visualization Video...", self)
        save_visualization_action.triggered.connect(self.save_visualization_video)
        file_menu.addAction(save_visualization_action)
        
        file_menu.addSeparator()
        
        output_action = QAction("Set &Output Folder...", self)
        output_action.triggered.connect(self.set_output_folder)
        file_menu.addAction(output_action)
        
        file_menu.addSeparator()
        
        load_config_action = QAction("Load Config...", self)
        load_config_action.triggered.connect(self.load_config)
        file_menu.addAction(load_config_action)
        
        save_config_action = QAction("&Save Config...", self)
        save_config_action.setShortcut("Ctrl+S")
        save_config_action.triggered.connect(self.save_config)
        file_menu.addAction(save_config_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        
        self.show_detections_action = QAction("Show &Detections", self, checkable=True)
        self.show_detections_action.triggered.connect(self.toggle_detections)
        view_menu.addAction(self.show_detections_action)
        
        self.show_tracks_action = QAction("Show &Tracks", self, checkable=True)
        self.show_tracks_action.triggered.connect(self.toggle_tracks)
        view_menu.addAction(self.show_tracks_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        guide_action = QAction("Parameter &Guide", self)
        guide_action.triggered.connect(self.show_parameter_guide)
        help_menu.addAction(guide_action)
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def _create_main_widget(self):
        """Create main widget layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel
        left_widget = self._create_control_panel()
        splitter.addWidget(left_widget)
        
        # Right panel
        right_widget = self._create_video_panel()
        splitter.addWidget(right_widget)
        
        splitter.setSizes([400, 1000])
    
    def _create_control_panel(self):
        """Create left control panel."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(380)
        
        container = QWidget()
        layout = QVBoxLayout()
        container.setLayout(layout)
        
        # Video info
        video_group = QGroupBox("Video")
        video_layout = QVBoxLayout()
        
        load_btn = QPushButton("📁 Load Video")
        load_btn.clicked.connect(self.load_video)
        video_layout.addWidget(load_btn)
        
        self.video_info_label = QLabel("No video loaded")
        self.video_info_label.setWordWrap(True)
        video_layout.addWidget(self.video_info_label)
        
        self.output_folder_label = QLabel("<i>Output: (load video first)</i>")
        self.output_folder_label.setWordWrap(True)
        self.output_folder_label.setStyleSheet("color: #666; font-size: 9pt;")
        video_layout.addWidget(self.output_folder_label)
        
        video_group.setLayout(video_layout)
        layout.addWidget(video_group)
        
        # Detection parameters
        params_group = QGroupBox("Detection Parameters")
        params_layout = QVBoxLayout()
        
        params_layout.addWidget(QLabel("Min Area (px²):"))
        self.min_area_slider = QSlider(Qt.Orientation.Horizontal)
        self.min_area_slider.setMinimum(30)
        self.min_area_slider.setMaximum(300)
        self.min_area_slider.setValue(120)
        self.min_area_slider.valueChanged.connect(self.on_param_change)
        params_layout.addWidget(self.min_area_slider)
        self.min_area_label = QLabel("120")
        self.min_area_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        params_layout.addWidget(self.min_area_label)
        
        params_layout.addWidget(QLabel("Min Solidity (0-1):"))
        self.min_solidity_slider = QSlider(Qt.Orientation.Horizontal)
        self.min_solidity_slider.setMinimum(30)
        self.min_solidity_slider.setMaximum(95)
        self.min_solidity_slider.setValue(70)
        self.min_solidity_slider.valueChanged.connect(self.on_param_change)
        params_layout.addWidget(self.min_solidity_slider)
        self.min_solidity_label = QLabel("0.70")
        self.min_solidity_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        params_layout.addWidget(self.min_solidity_label)
        
        params_layout.addWidget(QLabel("Max Area (px²):"))
        self.max_area_slider = QSlider(Qt.Orientation.Horizontal)
        self.max_area_slider.setMinimum(1000)
        self.max_area_slider.setMaximum(10000)
        self.max_area_slider.setValue(4000)
        self.max_area_slider.valueChanged.connect(self.on_param_change)
        params_layout.addWidget(self.max_area_slider)
        self.max_area_label = QLabel("4000")
        self.max_area_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        params_layout.addWidget(self.max_area_label)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Presets
        presets_group = QGroupBox("Presets")
        presets_layout = QVBoxLayout()
        
        for name, preset_id in [("Sensitive", "sensitive"), ("Default", "default"),
                                ("Conservative ⭐", "conservative"), ("Very Conservative", "very_conservative")]:
            btn = QPushButton(name)
            btn.clicked.connect(lambda checked, p=preset_id: self.load_preset(p))
            presets_layout.addWidget(btn)
        
        presets_group.setLayout(presets_layout)
        layout.addWidget(presets_group)
        
        # Actions
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout()
        
        init_bg_btn = QPushButton("Initialize Background")
        init_bg_btn.clicked.connect(self.initialize_background)
        actions_layout.addWidget(init_bg_btn)
        
        test_btn = QPushButton("Test Detection (Space)")
        test_btn.clicked.connect(self.test_detection)
        actions_layout.addWidget(test_btn)
        
        analyze_btn = QPushButton("▶ Run Full Analysis")
        analyze_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        analyze_btn.clicked.connect(self.run_analysis)
        actions_layout.addWidget(analyze_btn)
        
        self.detection_count_label = QLabel("Detections: 0")
        self.detection_count_label.setStyleSheet("font-size: 12pt; font-weight: bold;")
        self.detection_count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        actions_layout.addWidget(self.detection_count_label)
        
        actions_group.setLayout(actions_layout)
        layout.addWidget(actions_group)
        
        layout.addStretch()
        
        scroll.setWidget(container)
        return scroll
    
    def _create_video_panel(self):
        """Create right video panel."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        
        # Video canvas
        self.video_canvas = VideoCanvas()
        layout.addWidget(self.video_canvas)
        
        # Video controls
        controls_layout = QHBoxLayout()
        
        self.play_pause_btn = QPushButton("▶ Play")
        self.play_pause_btn.clicked.connect(self.toggle_play_pause)
        self.play_pause_btn.setEnabled(False)
        controls_layout.addWidget(self.play_pause_btn)
        
        prev_btn = QPushButton("◀")
        prev_btn.clicked.connect(lambda: self.jump_frame(-1))
        prev_btn.setMaximumWidth(40)
        controls_layout.addWidget(prev_btn)
        
        next_btn = QPushButton("▶")
        next_btn.clicked.connect(lambda: self.jump_frame(1))
        next_btn.setMaximumWidth(40)
        controls_layout.addWidget(next_btn)
        
        controls_layout.addWidget(QLabel("Speed:"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(10)
        self.speed_slider.setValue(5)
        self.speed_slider.setMaximumWidth(100)
        self.speed_slider.valueChanged.connect(self.on_speed_change)
        controls_layout.addWidget(self.speed_slider)
        
        layout.addLayout(controls_layout)
        
        # Frame slider
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(100)
        self.frame_slider.valueChanged.connect(self.on_frame_slider_change)
        layout.addWidget(self.frame_slider)
        
        # Info bar
        info_layout = QHBoxLayout()
        
        self.frame_label = QLabel("Frame: 0 / 0")
        info_layout.addWidget(self.frame_label)
        
        # Add data status
        self.data_status_label = QLabel("No data")
        self.data_status_label.setStyleSheet("color: #999;")
        info_layout.addWidget(self.data_status_label)
        
        info_layout.addStretch()
        
        self.show_detections_cb = QCheckBox("Show Detections")
        self.show_detections_cb.stateChanged.connect(self.on_show_detections_changed)
        info_layout.addWidget(self.show_detections_cb)
        
        self.show_tracks_cb = QCheckBox("Show Tracks")
        self.show_tracks_cb.stateChanged.connect(self.on_show_tracks_changed)
        info_layout.addWidget(self.show_tracks_cb)
        
        layout.addLayout(info_layout)
        
        return widget
    
    # ========================================================================
    # Video Management
    # ========================================================================
    
    def load_video(self):
        """Load video file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        
        if not filepath:
            return
        
        self.video_path = filepath
        self.video_cap = cv2.VideoCapture(self.video_path)
        
        if not self.video_cap.isOpened():
            QMessageBox.critical(self, "Error", "Cannot open video file")
            return
        
        self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.video_info_label.setText(
            f"<b>{Path(filepath).name}</b><br>"
            f"{width}x{height} @ {self.fps:.1f} FPS<br>"
            f"{self.total_frames} frames ({self.total_frames/self.fps:.1f}s)"
        )
        
        # Set output folder to video directory
        video_dir = Path(filepath).parent
        video_name = Path(filepath).stem  # filename without extension
        self.output_folder = str(video_dir / f"{video_name}_output")
        
        # Create output folder if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Update output folder label
        self.output_folder_label.setText(
            f"<b>Output:</b> {Path(self.output_folder).name}/"
        )
        self.output_folder_label.setStyleSheet("color: #0a0; font-size: 9pt;")
        
        self.frame_slider.setMaximum(self.total_frames - 1)
        self.current_frame_idx = 0
        
        self.load_frame(0)
        self.play_pause_btn.setEnabled(True)
        self.statusBar().showMessage(
            f"Loaded: {Path(filepath).name} | Output: {self.output_folder}"
        )
        
        print(f"✓ Video loaded: {filepath}")
    
    def load_frame(self, frame_idx):
        """Load specific frame."""
        if self.video_cap is None:
            return
        
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.video_cap.read()
        
        if ret:
            self.current_frame = frame
            self.current_frame_idx = frame_idx
            self.frame_label.setText(f"Frame: {frame_idx} / {self.total_frames-1}")
            self.frame_slider.setValue(frame_idx)
            
            # Load tracks for this frame if results loaded
            tracks_for_frame = self.get_tracks_for_frame(frame_idx)
            
            # Update data status
            self._update_data_status(tracks_for_frame)
            
            self.video_canvas.set_frame(frame, tracks=tracks_for_frame)
    
    def _update_data_status(self, tracks=None):
        """Update data status label."""
        status_parts = []
        
        if self.video_canvas.detections:
            status_parts.append(f"{len(self.video_canvas.detections)} detections")
        
        if tracks:
            total_points = sum(len(traj) for traj in tracks.values())
            status_parts.append(f"{len(tracks)} tracks ({total_points} points)")
        elif self.results_loaded:
            status_parts.append("Results loaded")
        
        if status_parts:
            self.data_status_label.setText(" | ".join(status_parts))
            self.data_status_label.setStyleSheet("color: #0a0; font-weight: bold;")
        else:
            self.data_status_label.setText("No data")
            self.data_status_label.setStyleSheet("color: #999;")
    
    def jump_frame(self, delta):
        """Jump forward/backward by delta frames."""
        if self.video_cap is None:
            return
        
        new_idx = max(0, min(self.total_frames-1, self.current_frame_idx + delta))
        self.load_frame(new_idx)
    
    def on_frame_slider_change(self, value):
        """Handle frame slider change."""
        if value != self.current_frame_idx:
            self.load_frame(value)
    
    def toggle_play_pause(self):
        """Toggle video playback."""
        if self.playing:
            self.playing = False
            self.playback_timer.stop()
            self.play_pause_btn.setText("▶ Play")
            self.statusBar().showMessage("Paused")
        else:
            self.playing = True
            speed = self.speed_slider.value()
            interval = int(1000 / (self.fps * speed / 5))  # Adjust for speed
            self.playback_timer.start(interval)
            self.play_pause_btn.setText("⏸ Pause")
            self.statusBar().showMessage("Playing...")
    
    def play_next_frame(self):
        """Play next frame during playback."""
        if self.current_frame_idx < self.total_frames - 1:
            self.jump_frame(1)
        else:
            self.toggle_play_pause()  # Stop at end
    
    def on_speed_change(self, value):
        """Handle playback speed change."""
        if self.playing:
            interval = int(1000 / (self.fps * value / 5))
            self.playback_timer.setInterval(interval)
    
    # ========================================================================
    # Results Loading and Visualization
    # ========================================================================
    
    def load_results(self):
        """Load tracking results CSV."""
        # Default to output folder if available, otherwise user's home
        default_dir = self.output_folder if self.output_folder else str(Path.home())
        
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Load Tracking Results",
            default_dir,
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if not filepath:
            return
        
        try:
            df = pd.read_csv(filepath)
            
            # Check if this is the events file instead of tracking file
            if 'action' in df.columns and 'nest' in df.columns:
                # This is the events CSV, not tracking results
                QMessageBox.warning(
                    self,
                    "Wrong File",
                    f"This appears to be an <b>events CSV</b> file.\n\n"
                    f"For visualization, you need the <b>tracking results CSV</b>.\n\n"
                    f"Look for a file named:\n"
                    f"  • tracking_results.csv\n"
                    f"  • tracks.csv\n"
                    f"  • <video_name>_tracks.csv\n\n"
                    f"The events file is for entry/exit analysis,\n"
                    f"not for visualizing trajectories."
                )
                
                # Try to find the tracking file automatically
                tracking_file = self._find_tracking_file(filepath)
                if tracking_file:
                    reply = QMessageBox.question(
                        self,
                        "Found Tracking File",
                        f"I found a tracking results file:\n\n"
                        f"{Path(tracking_file).name}\n\n"
                        f"Would you like to load it instead?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                    )
                    
                    if reply == QMessageBox.StandardButton.Yes:
                        filepath = tracking_file
                        df = pd.read_csv(filepath)
                    else:
                        return
                else:
                    return
            
            # Validate required columns for tracking
            required = ['track_id']
            missing = [col for col in required if col not in df.columns]
            
            # Handle different frame column names
            frame_col = None
            for possible in ['frame', 'frame_number', 'frame_num', 'frame_id']:
                if possible in df.columns:
                    frame_col = possible
                    break
            
            if frame_col is None:
                missing.append('frame (or frame_number)')
            else:
                # Rename to standard 'frame' if different
                if frame_col != 'frame':
                    df = df.rename(columns={frame_col: 'frame'})
            
            if missing:
                QMessageBox.warning(
                    self,
                    "Invalid Tracking File",
                    f"This CSV is missing required columns for visualization:\n\n"
                    f"<b>Missing:</b> {', '.join(missing)}\n\n"
                    f"<b>Available:</b> {', '.join(df.columns)}\n\n"
                    f"Required format:\n"
                    f"  • frame (or frame_number)\n"
                    f"  • track_id\n"
                    f"  • Position columns (x1,y1,x2,y2 or x,y)\n\n"
                    f"Make sure you're loading the tracking results file,\n"
                    f"not the events file."
                )
                return
            
            # Check for position columns
            has_bbox = all(col in df.columns for col in ['x1', 'y1', 'x2', 'y2'])
            has_xy = all(col in df.columns for col in ['x', 'y'])
            has_centroid = all(col in df.columns for col in ['centroid_x', 'centroid_y'])
            
            if not (has_bbox or has_xy or has_centroid):
                QMessageBox.warning(
                    self,
                    "Missing Position Data",
                    f"CSV has track_id and frame, but no position data.\n\n"
                    f"Need one of:\n"
                    f"  • x1, y1, x2, y2 (bounding boxes)\n"
                    f"  • x, y (positions)\n"
                    f"  • centroid_x, centroid_y\n\n"
                    f"Available columns: {', '.join(df.columns)}"
                )
                return
            
            # Success! Store the dataframe
            self.tracking_results = df
            self.results_loaded = True
            
            total_tracks = df['track_id'].nunique()
            total_frames = df['frame'].nunique()
            
            msg = (
                f"✓ Tracking results loaded!\n\n"
                f"File: {Path(filepath).name}\n\n"
                f"Total tracks: {total_tracks}\n"
                f"Total frames: {total_frames}\n"
                f"Total detections: {len(df)}\n\n"
                f"Enable 'Show Tracks' checkbox to visualize."
            )
            
            QMessageBox.information(self, "Results Loaded", msg)
            
            # Auto-enable track display
            print(f"Auto-enabling track display")
            self.show_tracks_cb.setChecked(True)
            # Checkbox stateChanged will trigger VideoCanvas update
            
            # Refresh current frame with tracks
            if self.current_frame is not None:
                self.load_frame(self.current_frame_idx)
            
            self.statusBar().showMessage(f"✓ Results loaded: {total_tracks} tracks across {total_frames} frames")
            
            print(f"Results loaded: {filepath}")
            print(f"  Tracks: {total_tracks}")
            print(f"  Frames: {total_frames}")
            print(f"  Columns: {list(df.columns)}")
            
        except Exception as e:
            import traceback
            error_msg = f"Failed to load results:\n{e}\n\n{traceback.format_exc()}"
            QMessageBox.critical(self, "Error", error_msg)
            print(error_msg)
    
    def _find_tracking_file(self, events_filepath):
        """Try to find the tracking results file in the same directory."""
        directory = Path(events_filepath).parent
        
        # Common tracking file names
        possible_names = [
            'tracking_results.csv',
            'tracks.csv',
            'tracking.csv',
            'trajectories.csv'
        ]
        
        for name in possible_names:
            path = directory / name
            if path.exists():
                return str(path)
        
        # Try with video name prefix
        for file in directory.glob('*_tracks.csv'):
            return str(file)
        for file in directory.glob('*_tracking.csv'):
            return str(file)
        
        return None
    
    def get_tracks_for_frame(self, frame_idx):
        """Get all tracks visible in this frame."""
        if not self.results_loaded or self.tracking_results is None:
            return {}
        
        if 'frame' not in self.tracking_results.columns:
            print(f"Warning: 'frame' column not found in results")
            return {}
        
        if 'track_id' not in self.tracking_results.columns:
            print(f"Warning: 'track_id' column not found in results")
            return {}
        
        # Get all detections for this frame
        frame_data = self.tracking_results[self.tracking_results['frame'] == frame_idx]
        
        if len(frame_data) == 0:
            return {}
        
        # Build track trajectories (last 30 frames for each visible track)
        tracks = {}
        window = 30
        
        for track_id in frame_data['track_id'].unique():
            track_data = self.tracking_results[
                (self.tracking_results['track_id'] == track_id) &
                (self.tracking_results['frame'] <= frame_idx) &
                (self.tracking_results['frame'] > frame_idx - window)
            ]
            
            if len(track_data) > 0:
                # Try different column name formats
                centroids = []
                
                # Format 1: x1, y1, x2, y2 columns
                if all(col in track_data.columns for col in ['x1', 'y1', 'x2', 'y2']):
                    for _, row in track_data.iterrows():
                        cx = (row['x1'] + row['x2']) / 2
                        cy = (row['y1'] + row['y2']) / 2
                        centroids.append((int(cx), int(cy)))
                
                # Format 2: x, y columns
                elif all(col in track_data.columns for col in ['x', 'y']):
                    for _, row in track_data.iterrows():
                        centroids.append((int(row['x']), int(row['y'])))
                
                # Format 3: centroid_x, centroid_y columns
                elif all(col in track_data.columns for col in ['centroid_x', 'centroid_y']):
                    for _, row in track_data.iterrows():
                        centroids.append((int(row['centroid_x']), int(row['centroid_y'])))
                
                else:
                    # Only print column error once
                    if not hasattr(self, '_column_error_shown'):
                        print(f"Warning: Could not find position columns in results")
                        print(f"Available columns: {list(track_data.columns)}")
                        self._column_error_shown = True
                    return {}
                
                if centroids:
                    tracks[track_id] = centroids
        
        return tracks
    
    # ========================================================================
    # Detection
    # ========================================================================
    
    def on_param_change(self):
        """Handle parameter change."""
        min_area = self.min_area_slider.value()
        min_solidity = self.min_solidity_slider.value() / 100.0
        max_area = self.max_area_slider.value()
        
        self.min_area_label.setText(str(min_area))
        self.min_solidity_label.setText(f"{min_solidity:.2f}")
        self.max_area_label.setText(str(max_area))
        
        if self.blob_detector is not None:
            QTimer.singleShot(500, self.test_detection)
    
    def load_preset(self, preset_name):
        """Load parameter preset."""
        presets = {
            "sensitive": {"min_area": 30, "min_solidity": 30, "max_area": 10000},
            "default": {"min_area": 50, "min_solidity": 50, "max_area": 8000},
            "conservative": {"min_area": 120, "min_solidity": 70, "max_area": 4000},
            "very_conservative": {"min_area": 200, "min_solidity": 80, "max_area": 3000}
        }
        
        if preset_name not in presets:
            return
        
        preset = presets[preset_name]
        
        self.min_area_slider.setValue(preset["min_area"])
        self.min_solidity_slider.setValue(preset["min_solidity"])
        self.max_area_slider.setValue(preset["max_area"])
        
        self.on_param_change()
        self.statusBar().showMessage(f"Loaded preset: {preset_name}")
    
    def initialize_background(self):
        """Initialize background model."""
        if self.video_path is None:
            QMessageBox.warning(self, "Warning", "Load a video first")
            return
        
        self.statusBar().showMessage("Initializing background...")
        QApplication.processEvents()
        
        try:
            self.blob_detector = BlobDetector(
                min_area=self.min_area_slider.value(),
                min_solidity=self.min_solidity_slider.value() / 100.0
            )
            
            self.blob_detector.initialize_from_video(
                video_path=self.video_path,
                num_frames=100,
                start_frame=0
            )
            
            self.statusBar().showMessage("✓ Background initialized")
            QMessageBox.information(self, "Success", "Background model initialized")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to initialize background:\n{e}")
            self.statusBar().showMessage("✗ Background initialization failed")
    
    def test_detection(self):
        """Test detection on current frame."""
        if self.current_frame is None:
            QMessageBox.warning(self, "Warning", "Load a video first")
            return
        
        if self.blob_detector is None:
            reply = QMessageBox.question(
                self,
                "Background Not Initialized",
                "Initialize background model now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.initialize_background()
            else:
                return
        
        try:
            self.blob_detector.min_area = self.min_area_slider.value()
            self.blob_detector.min_solidity = self.min_solidity_slider.value() / 100.0
            
            detections = self.blob_detector.detect(self.current_frame)
            
            max_area = self.max_area_slider.value()
            detections = [d for d in detections 
                         if (d.bbox[2]-d.bbox[0])*(d.bbox[3]-d.bbox[1]) <= max_area]
            
            self.detection_count_label.setText(f"Detections: {len(detections)}")
            
            # Get tracks if results loaded
            tracks_for_frame = self.get_tracks_for_frame(self.current_frame_idx)
            
            # Update data status
            self._update_data_status(tracks_for_frame)
            
            # Always pass detections and auto-enable visualization
            self.video_canvas.set_frame(
                self.current_frame,
                detections=detections,
                tracks=tracks_for_frame
            )
            
            # Auto-enable detection display if we have detections
            if len(detections) > 0 and not self.show_detections_cb.isChecked():
                print(f"Auto-enabling detection display")
                self.show_detections_cb.setChecked(True)
                # Checkbox stateChanged will trigger VideoCanvas update
            elif self.show_detections_cb.isChecked():
                # Checkbox is already checked, but force sync
                print(f"Syncing detection display state")
                self.video_canvas.show_detections = True
                self.video_canvas._draw_frame()
            
            self.statusBar().showMessage(f"✓ Detected {len(detections)} objects (detections shown in green)")
            
        except Exception as e:
            import traceback
            QMessageBox.critical(self, "Error", f"Detection failed:\n{e}\n\n{traceback.format_exc()}")
    
    def run_analysis(self):
        """Run full video analysis."""
        if self.video_path is None:
            QMessageBox.warning(self, "Warning", "Load a video first")
            return
        
        # Ensure output folder is set (should be auto-set when video loads)
        if not self.output_folder:
            video_dir = Path(self.video_path).parent
            video_name = Path(self.video_path).stem
            self.output_folder = str(video_dir / f"{video_name}_output")
            os.makedirs(self.output_folder, exist_ok=True)
        
        reply = QMessageBox.question(
            self,
            "Run Analysis",
            f"Run full analysis on:\n{Path(self.video_path).name}\n\n"
            f"Output folder:\n{self.output_folder}\n\n"
            f"This may take several minutes. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        self.config.detection.min_area = self.min_area_slider.value()
        self.config.detection.min_solidity = self.min_solidity_slider.value() / 100.0
        self.config.detection.max_area = self.max_area_slider.value()
        
        self.config.detection.sync_to_tracking(self.config.tracking)
        
        os.makedirs(self.output_folder, exist_ok=True)
        
        monitor = BeeMonitor(config=self.config)
        
        self.analysis_thread = AnalysisThread(
            monitor,
            self.video_path,
            self.output_folder
        )
        
        self.analysis_thread.progress.connect(lambda msg: self.statusBar().showMessage(msg))
        self.analysis_thread.finished.connect(self.on_analysis_finished)
        self.analysis_thread.error.connect(lambda err: QMessageBox.critical(self, "Analysis Error", err))
        
        self.analysis_thread.start()
        self.statusBar().showMessage("Running analysis...")
    
    def on_analysis_finished(self, result, csv_path):
        """Handle analysis completion."""
        self.statusBar().showMessage("✓ Analysis complete")
        
        # Check if visualization video was created
        video_output = None
        if self.output_folder:
            # Common output video names
            possible_videos = [
                os.path.join(self.output_folder, 'tracking_visualization.mp4'),
                os.path.join(self.output_folder, 'output.mp4'),
                os.path.join(self.output_folder, 'result.mp4'),
            ]
            for path in possible_videos:
                if os.path.exists(path):
                    video_output = path
                    break
        
        msg = (
            f"✓ Analysis complete!\n\n"
            f"Output folder: {self.output_folder}\n\n"
            f"Files saved:\n"
            f"  • tracking_results.csv (tracking data)\n"
        )
        
        if video_output:
            msg += f"  • {os.path.basename(video_output)} (visualization video)\n"
        
        msg += (
            f"\nNext steps:\n"
            f"1. File → Load Results to visualize tracks on original video\n"
        )
        
        if video_output:
            msg += f"2. File → Load Output Video to play saved visualization\n"
        
        reply = QMessageBox.information(
            self,
            "Analysis Complete",
            msg,
            QMessageBox.StandardButton.Ok
        )
        
        # Offer to load results automatically
        if os.path.exists(csv_path):
            auto_load = QMessageBox.question(
                self,
                "Load Results?",
                "Would you like to load the results now to visualize tracks?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if auto_load == QMessageBox.StandardButton.Yes:
                # Load the CSV results
                try:
                    self.tracking_results = pd.read_csv(csv_path)
                    self.results_loaded = True
                    
                    # Auto-enable track display
                    self.show_tracks_cb.setChecked(True)
                    
                    # Refresh current frame
                    if self.current_frame is not None:
                        self.load_frame(self.current_frame_idx)
                    
                    self.statusBar().showMessage("✓ Results loaded - tracks visible on video")
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Could not load results:\n{e}")
    
    def load_output_video(self):
        """Load output visualization video (if created by analysis)."""
        # Default to output folder if available, otherwise user's home
        default_dir = self.output_folder if self.output_folder else str(Path.home())
        
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Load Output Video",
            default_dir,
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        
        if not filepath:
            return
        
        # Load as regular video
        self.video_path = filepath
        self.video_cap = cv2.VideoCapture(self.video_path)
        
        if not self.video_cap.isOpened():
            QMessageBox.critical(self, "Error", "Cannot open video file")
            return
        
        self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.video_info_label.setText(
            f"<b>OUTPUT: {Path(filepath).name}</b><br>"
            f"{width}x{height} @ {self.fps:.1f} FPS<br>"
            f"{self.total_frames} frames ({self.total_frames/self.fps:.1f}s)<br>"
            f"<i>This is pre-rendered visualization</i>"
        )
        
        self.frame_slider.setMaximum(self.total_frames - 1)
        self.current_frame_idx = 0
        
        self.load_frame(0)
        self.play_pause_btn.setEnabled(True)
        self.statusBar().showMessage(f"Loaded output video: {Path(filepath).name}")
        
        QMessageBox.information(
            self,
            "Output Video Loaded",
            "This is a pre-rendered visualization video.\n\n"
            "Tracks/detections are already drawn on the video.\n"
            "You don't need to enable overlays."
        )
    
    def save_visualization_video(self):
        """Save current video with tracks/detections as new video file."""
        if self.video_path is None:
            QMessageBox.warning(self, "Warning", "Load a video first")
            return
        
        if not self.results_loaded and not self.blob_detector:
            QMessageBox.warning(
                self,
                "Warning",
                "No data to visualize!\n\n"
                "Either:\n"
                "  • Load tracking results (File → Load Results), or\n"
                "  • Initialize background and test detection"
            )
            return
        
        # Ask for output file
        default_dir = self.output_folder if self.output_folder else str(Path.home())
        
        # Smart default filename based on input video
        if self.video_path:
            video_name = Path(self.video_path).stem
            default_filename = f"{video_name}_visualization.mp4"
        else:
            default_filename = "visualization.mp4"
        
        default_path = os.path.join(default_dir, default_filename)
        
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Visualization Video",
            default_path,
            "MP4 Video (*.mp4);;AVI Video (*.avi);;All Files (*)"
        )
        
        if not output_path:
            return
        
        # Ask what to include
        msg = QMessageBox()
        msg.setWindowTitle("Visualization Options")
        msg.setText("What should be included in the video?")
        
        include_detections = self.show_detections_cb.isChecked()
        include_tracks = self.show_tracks_cb.isChecked()
        
        msg.setInformativeText(
            f"Current settings:\n"
            f"  • Detections: {'Yes' if include_detections else 'No'}\n"
            f"  • Tracks: {'Yes' if include_tracks else 'No'}\n\n"
            f"This will process the entire video.\n"
            f"Continue?"
        )
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        reply = msg.exec()
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # Create progress dialog
        progress = QProgressDialog("Saving visualization video...", "Cancel", 0, self.total_frames, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        
        try:
            # Open video for writing
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.video_cap.get(cv2.CAP_PROP_FPS)
            
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Reset to beginning
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Process each frame
            for frame_idx in range(self.total_frames):
                if progress.wasCanceled():
                    break
                
                ret, frame = self.video_cap.read()
                if not ret:
                    break
                
                # Draw overlays
                vis_frame = frame.copy()
                
                # Draw detections if enabled and blob detector available
                if include_detections and self.blob_detector:
                    try:
                        detections = self.blob_detector.detect(frame)
                        for det in detections:
                            x1, y1, x2, y2 = [int(c) for c in det.bbox]
                            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            area = (x2-x1) * (y2-y1)
                            cv2.putText(vis_frame, f"{area:.0f}", (x1, y1-5),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    except:
                        pass  # Skip frames that fail
                
                # Draw tracks if enabled and results loaded
                if include_tracks and self.results_loaded:
                    tracks = self.get_tracks_for_frame(frame_idx)
                    colors = [(255, 0, 0), (0, 255, 255), (255, 0, 255), 
                             (255, 255, 0), (128, 0, 255), (255, 128, 0)]
                    
                    for i, (track_id, trajectory) in enumerate(tracks.items()):
                        color = colors[i % len(colors)]
                        
                        # Draw trajectory
                        if len(trajectory) > 1:
                            points = np.array(trajectory, dtype=np.int32)
                            cv2.polylines(vis_frame, [points], False, color, 2)
                        
                        # Draw current position
                        if trajectory:
                            x, y = trajectory[-1]
                            cv2.circle(vis_frame, (int(x), int(y)), 5, color, -1)
                            cv2.putText(vis_frame, f"ID:{track_id}", (int(x)+10, int(y)),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Add frame number
                cv2.putText(vis_frame, f"Frame: {frame_idx}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                out.write(vis_frame)
                progress.setValue(frame_idx)
                QApplication.processEvents()
            
            out.release()
            
            # Reset video to original position
            self.load_frame(self.current_frame_idx)
            
            progress.close()
            
            QMessageBox.information(
                self,
                "Success",
                f"Visualization video saved!\n\n{output_path}\n\n"
                f"You can now play this video in any media player."
            )
            
            self.statusBar().showMessage(f"✓ Visualization saved: {Path(output_path).name}")
            
        except Exception as e:
            import traceback
            progress.close()
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to save video:\n{e}\n\n{traceback.format_exc()}"
            )
    
    # ========================================================================
    # Utility
    # ========================================================================
    
    def set_output_folder(self):
        """Set output folder (override automatic location)."""
        current = self.output_folder if self.output_folder else str(Path.home())
        
        folder = QFileDialog.getExistingDirectory(
            self, 
            "Select Output Folder",
            current
        )
        
        if folder:
            self.output_folder = folder
            os.makedirs(self.output_folder, exist_ok=True)
            self.statusBar().showMessage(f"Output folder: {folder}")
            
            QMessageBox.information(
                self,
                "Output Folder Set",
                f"Results will be saved to:\n{folder}\n\n"
                f"Note: By default, output goes to:\n"
                f"<video_location>/<video_name>_output/\n\n"
                f"This override will be used for the current session."
            )
    
    def save_config(self):
        """Save configuration."""
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Configuration",
            "",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if not filepath:
            return
        
        config_data = {
            "detection": {
                "min_area": self.min_area_slider.value(),
                "min_solidity": self.min_solidity_slider.value() / 100.0,
                "max_area": self.max_area_slider.value()
            },
            "video_path": self.video_path,
            "output_folder": self.output_folder,
            "saved_at": datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        QMessageBox.information(self, "Success", f"Configuration saved")
    
    def load_config(self):
        """Load configuration."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Load Configuration",
            "",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if not filepath:
            return
        
        try:
            with open(filepath, 'r') as f:
                config_data = json.load(f)
            
            det = config_data.get("detection", {})
            self.min_area_slider.setValue(int(det.get("min_area", 120)))
            self.min_solidity_slider.setValue(int(det.get("min_solidity", 0.7) * 100))
            self.max_area_slider.setValue(int(det.get("max_area", 4000)))
            
            if "output_folder" in config_data:
                self.output_folder = config_data["output_folder"]
            
            self.on_param_change()
            
            QMessageBox.information(self, "Success", "Configuration loaded")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load configuration:\n{e}")
    
    def toggle_detections(self, checked):
        """Toggle detection display."""
        self.show_detections_cb.setChecked(checked)
    
    def toggle_tracks(self, checked):
        """Toggle track display."""
        self.show_tracks_cb.setChecked(checked)
    
    def on_show_detections_changed(self, state):
        """Handle Show Detections checkbox change."""
        enabled = (state == Qt.CheckState.Checked)
        print(f"Checkbox: Show Detections = {enabled}")
        self.video_canvas.toggle_detections(enabled)
    
    def on_show_tracks_changed(self, state):
        """Handle Show Tracks checkbox change."""
        enabled = (state == Qt.CheckState.Checked)
        print(f"Checkbox: Show Tracks = {enabled}")
        self.video_canvas.toggle_tracks(enabled)
    
    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About BeeMonitor",
            "<h3>BeeMonitor v3.0 - Video Player Edition</h3>"
            "<p>Interactive video analysis tool for bee tracking</p>"
            "<p><b>Features:</b></p>"
            "<ul>"
            "<li>Play/Pause video controls</li>"
            "<li>Real-time detection preview</li>"
            "<li>Track visualization overlay</li>"
            "<li>Parameter tuning</li>"
            "</ul>"
        )
    
    def show_parameter_guide(self):
        """Show parameter guide."""
        guide = """
<h2>QUICK GUIDE</h2>

<h3>Workflow:</h3>
<ol>
<li>Load video (Ctrl+O)</li>
<li>Initialize background (uses first 100 frames)</li>
<li>Load "Conservative" preset</li>
<li>Test detection (Space) - navigate frames to check</li>
<li>Adjust sliders if needed</li>
<li>Run Full Analysis</li>
<li>Load Results to visualize tracks</li>
</ol>

<h3>Video Controls:</h3>
<ul>
<li><b>Play/Pause:</b> Watch video with current settings</li>
<li><b>◀ ▶:</b> Step through frames</li>
<li><b>Speed slider:</b> Adjust playback speed</li>
<li><b>Frame slider:</b> Jump to specific frame</li>
</ul>

<h3>Visualization:</h3>
<ul>
<li><b>Show Detections:</b> Green boxes = detected blobs</li>
<li><b>Show Tracks:</b> Colored trails = bee trajectories</li>
</ul>

<h3>Parameters:</h3>
<ul>
<li><b>Min Area:</b> 120 (increase to reduce noise)</li>
<li><b>Min Solidity:</b> 0.7 (shape filtering)</li>
<li><b>Max Area:</b> 4000 (filter large objects)</li>
</ul>
"""
        
        msg = QMessageBox(self)
        msg.setWindowTitle("Quick Guide")
        msg.setTextFormat(Qt.TextFormat.RichText)
        msg.setText(guide)
        msg.exec()
    
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts."""
        if event.key() == Qt.Key.Key_Space:
            if self.playing:
                self.toggle_play_pause()
            else:
                self.test_detection()
        elif event.key() == Qt.Key.Key_Left:
            self.jump_frame(-1)
        elif event.key() == Qt.Key.Key_Right:
            self.jump_frame(1)


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = BeeMonitorGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()