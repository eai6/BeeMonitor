"""
Main Window
===========

Main application window orchestrating all components.
"""

import os
import json
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QSplitter,
    QMessageBox, QFileDialog, QApplication, QProgressDialog
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction

try:
    from beemonitor import BeeMonitor
    from beemonitor.core.config import Config
    from beemonitor.detection import BlobDetector
    from beemonitor.detection.base_detector import Detection
except ImportError:
    raise ImportError("Cannot import beemonitor. Please install the package.")

from .constants import VERSION, TITLE, DEFAULT_WINDOW_SIZE, TRAJECTORY_WINDOW
from .control_panel import ControlPanel
from .video_panel import VideoPanel
from .analysis_thread import AnalysisThread
from .dialogs import show_about_dialog, show_parameter_guide
from .utils import find_tracking_file, get_position_from_row


class BeeMonitorGUI(QMainWindow):
    """Main GUI application with video player controls."""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle(TITLE)
        self.setGeometry(100, 100, *DEFAULT_WINDOW_SIZE)
        
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
        self.sift_detector = None
        self.yolo_detector = None
        self.output_folder = None
        
        self.analysis_thread = None
        
        # Create panels
        self.control_panel = ControlPanel()
        self.video_panel = VideoPanel()
        self.video_canvas = self.video_panel.get_canvas()
        
        # Connect signals
        self._connect_signals()
        
        # Setup UI
        self._create_menu_bar()
        self._create_main_widget()
        
        # Sync initial checkbox states with VideoCanvas
        self.video_canvas.show_detections = self.video_panel.show_detections_cb.isChecked()
        self.video_canvas.show_tracks = self.video_panel.show_tracks_cb.isChecked()
        self.video_canvas.show_detection_sources = self.video_panel.show_sources_cb.isChecked()
        
        self.statusBar().showMessage("Ready - Load a video to begin")
        
        print(f"✓ BeeMonitor GUI v{VERSION} initialized (Modular with Detection Sources)")
    
    def _connect_signals(self):
        """Connect all signals from panels to methods."""
        # Control panel signals
        self.control_panel.load_video_requested.connect(self.load_video)
        self.control_panel.test_detection_requested.connect(self.test_detection)
        self.control_panel.initialize_background_requested.connect(self.initialize_background)
        self.control_panel.run_analysis_requested.connect(self.run_analysis)
        self.control_panel.parameters_changed.connect(self.on_parameters_changed)
        
        # Video panel signals
        self.video_panel.play_pause_toggled.connect(self.toggle_play_pause)
        self.video_panel.frame_changed.connect(self.on_frame_slider_change)
        self.video_panel.frame_step_requested.connect(self.jump_frame)
        self.video_panel.speed_changed.connect(self.on_speed_change)
        self.video_panel.show_detections_changed.connect(self.on_show_detections_changed)
        self.video_panel.show_tracks_changed.connect(self.on_show_tracks_changed)
        self.video_panel.show_sources_changed.connect(self.on_show_sources_changed)  # NEW!
    
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
        self.show_detections_action.triggered.connect(
            lambda checked: self.video_panel.show_detections_cb.setChecked(checked))
        view_menu.addAction(self.show_detections_action)
        
        self.show_tracks_action = QAction("Show &Tracks", self, checkable=True)
        self.show_tracks_action.triggered.connect(
            lambda checked: self.video_panel.show_tracks_cb.setChecked(checked))
        view_menu.addAction(self.show_tracks_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        guide_action = QAction("Parameter &Guide", self)
        guide_action.triggered.connect(lambda: show_parameter_guide(self))
        help_menu.addAction(guide_action)
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(lambda: show_about_dialog(self))
        help_menu.addAction(about_action)
    
    def _create_main_widget(self):
        """Create main widget layout with panels."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Add panels
        splitter.addWidget(self.control_panel)
        splitter.addWidget(self.video_panel)
        
        splitter.setSizes([400, 1000])
    
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
        
        # Update control panel video info
        self.control_panel.set_video_info(
            f"<b>{Path(filepath).name}</b><br>"
            f"{width}x{height} @ {self.fps:.1f} FPS<br>"
            f"{self.total_frames} frames ({self.total_frames/self.fps:.1f}s)"
        )
        
        # Set output folder to video directory
        video_dir = Path(filepath).parent
        video_name = Path(filepath).stem
        self.output_folder = str(video_dir / f"{video_name}_output")
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Update output folder label
        self.control_panel.set_output_folder_info(
            f"<b>Output:</b> {Path(self.output_folder).name}/"
        )
        
        # Update video panel
        self.video_panel.set_frame_range(self.total_frames - 1)
        self.current_frame_idx = 0
        
        self.load_frame(0)
        self.video_panel.enable_play_button(True)
        
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
            
            self.video_panel.set_frame_info(frame_idx, self.total_frames - 1)
            self.video_panel.set_frame_slider_value(frame_idx)
            
            # Load tracks and detections for this frame if results loaded
            tracks_for_frame = self.get_tracks_for_frame(frame_idx)
            detections_for_frame = self.get_detections_for_frame(frame_idx)
            
            # Update data status
            self._update_data_status(tracks_for_frame)
            
            # Pass both tracks AND detections to canvas
            self.video_canvas.set_frame(
                frame, 
                detections=detections_for_frame,
                tracks=tracks_for_frame
            )
    
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
            self.video_panel.set_data_status(" | ".join(status_parts), is_active=True)
        else:
            self.video_panel.set_data_status("No data", is_active=False)
    
    def jump_frame(self, delta):
        """Jump forward/backward by delta frames."""
        if self.video_cap is None:
            return
        
        new_idx = max(0, min(self.total_frames - 1, self.current_frame_idx + delta))
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
            self.video_panel.set_playing(False)
            self.statusBar().showMessage("Paused")
        else:
            self.playing = True
            speed = self.video_panel.speed_slider.value()
            interval = int(1000 / (self.fps * speed / 5))
            self.playback_timer.start(interval)
            self.video_panel.set_playing(True)
            self.statusBar().showMessage("Playing...")
    
    def play_next_frame(self):
        """Play next frame during playback."""
        if self.current_frame_idx < self.total_frames - 1:
            self.jump_frame(1)
        else:
            self.toggle_play_pause()
    
    def on_speed_change(self, value):
        """Handle playback speed change."""
        if self.playing:
            interval = int(1000 / (self.fps * value / 5))
            self.playback_timer.setInterval(interval)
    
    # ========================================================================
    # Detection
    # ========================================================================
    
    def on_parameters_changed(self, params):
        """Handle parameter changes from control panel."""
        if self.blob_detector is not None:
            # Auto-test if detector is ready
            QTimer.singleShot(500, self.test_detection)
    
    # def initialize_background(self):
    #     """Initialize background model."""
    #     if self.video_path is None:
    #         QMessageBox.warning(self, "Warning", "Load a video first")
    #         return
        
    #     self.statusBar().showMessage("Initializing background...")
    #     QApplication.processEvents()
        
    #     try:
    #         params = self.control_panel.get_parameters()
        
    #         # Get detection mode from control panel
    #         detection_mode = params.get("detection_mode", "yolo_only")
            
    #         # Use defaults - these will be learned in Phase 1b anyway
    #         self.blob_detector = BlobDetector(
    #             min_area=120.0,  # Default, will be overridden by learned values
    #             min_solidity=0.7  # Default, will be overridden by learned values
    #         )
            
    #         self.blob_detector.initialize_from_video(
    #             video_path=self.video_path,
    #             num_frames=100,
    #             start_frame=0
    #         )
            
    #         self.statusBar().showMessage("✓ Background initialized")
    #         QMessageBox.information(self, "Success", "Background model initialized")
            
    #     except Exception as e:
    #         QMessageBox.critical(self, "Error", f"Failed to initialize background:\n{e}")
    #         self.statusBar().showMessage("✗ Background initialization failed")


    def initialize_background(self):
        """Initialize background model."""
        if self.video_path is None:
            QMessageBox.warning(self, "Warning", "Load a video first")
            return
        
        self.statusBar().showMessage("Initializing background...")
        QApplication.processEvents()
        
        try:
            from beemonitor.detection import BlobDetector
            
            # Use defaults - these will be learned in Phase 1b during analysis
            self.blob_detector = BlobDetector(
                min_area=120.0,  # Default, will be overridden by learned values
                min_solidity=0.7  # Default, will be overridden by learned values
            )
            
            self.blob_detector.initialize_from_video(
                video_path=self.video_path,
                num_frames=30,  # Quick init for testing
                start_frame=0
            )
            
            self.statusBar().showMessage("✓ Background initialized")
            QMessageBox.information(
                self, 
                "Success", 
                "Background model initialized!\n\n"
                "Note: During full analysis:\n"
                "• Blob morphology will be learned (Phase 1b)\n"
                "• CNN filter will be applied automatically\n"
                "• Learned solidity filter will be applied"
            )
            
        except Exception as e:
            import traceback
            QMessageBox.critical(self, "Error", 
                f"Failed to initialize background:\n{e}\n\n{traceback.format_exc()}")
            self.statusBar().showMessage("✗ Background initialization failed")


    # ============================================================================
    # Change 4: Replace test_detection() method
    # ============================================================================
    def test_detection(self):
        """Test detection on current frame using selected mode."""
        if self.current_frame is None:
            QMessageBox.warning(self, "Warning", "Load a video first")
            return
        
        try:
            params = self.control_panel.get_parameters()
            detection_mode = params.get("detection_mode", "fgbg")
            
            detections = []
            
            # Handle mode
            if detection_mode in ['fgbg', 'fgbg_yolo']:
                # Blob-based modes: need background
                if self.blob_detector is None:
                    reply = QMessageBox.question(
                        self,
                        "Background Not Initialized",
                        "Blob detection requires initialized background.\n\n"
                        "During full analysis:\n"
                        "• CNN noise filter will be applied automatically\n"
                        "• Solidity threshold will be learned automatically\n\n"
                        "Initialize background now for testing?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                    )
                    if reply == QMessageBox.StandardButton.Yes:
                        self.initialize_background()
                    else:
                        return
                
                # Use blob detector (shows raw blobs in test)
                detections = self.blob_detector.detect(self.current_frame)
            
            elif detection_mode == 'yolo_only':
                # Pure YOLO (no blob/CNN/solidity)
                from beemonitor.detection import YOLODetector
                from ultralytics import YOLO
                model = YOLO('yolo11n.pt')
                yolo = YOLODetector(model, tracking_classes=['bee'], conf_threshold=0.25)
                detections = yolo.detect(self.current_frame)
            
            # Update display
            self.control_panel.set_detection_count(len(detections))
            tracks_for_frame = self.get_tracks_for_frame(self.current_frame_idx)
            self._update_data_status(tracks_for_frame)
            
            self.video_canvas.set_frame(
                self.current_frame,
                detections=detections,
                tracks=tracks_for_frame
            )
            
            # Auto-enable detection display
            if len(detections) > 0 and not self.video_panel.show_detections_cb.isChecked():
                self.video_panel.show_detections_cb.blockSignals(True)
                self.video_panel.show_detections_cb.setChecked(True)
                self.video_panel.show_detections_cb.blockSignals(False)
                self.video_canvas.show_detections = True
                self.video_canvas._draw_frame()
            
            # Status message
            mode_names = {
                'fgbg': 'Motion',
                'fgbg_yolo': 'Motion+YOLO', 
                'yolo_only': 'YOLO Only'
            }
            mode_name = mode_names.get(detection_mode, detection_mode)
            
            # Note about filtering
            note = " (raw blobs - CNN+solidity in full analysis)" if detection_mode in ['fgbg', 'fgbg_yolo'] else ""
            self.statusBar().showMessage(f"✓ {mode_name}: {len(detections)} detections{note}")
            
        except Exception as e:
            import traceback
            QMessageBox.critical(self, "Error", 
                f"Detection failed:\n{e}\n\n{traceback.format_exc()}")


    # ============================================================================
    # Change 5: Replace run_analysis() method
    # ============================================================================
    def run_analysis(self):
        """Run full video analysis with automatic CNN + solidity filtering."""
        from pathlib import Path
        import os
        from beemonitor.core.video_analyzer import BeeMonitor
        from .analysis_thread import AnalysisThread
        
        if self.video_path is None:
            QMessageBox.warning(self, "Warning", "Load a video first")
            return
        
        # Get detection mode
        params = self.control_panel.get_parameters()
        detection_mode = params.get("detection_mode", "fgbg")
        
        # Set output folder if not set
        if not self.output_folder:
            video_dir = Path(self.video_path).parent
            video_name = Path(self.video_path).stem
            self.output_folder = str(video_dir / f"{video_name}_output")
            os.makedirs(self.output_folder, exist_ok=True)
        
        # Check background init if needed
        if detection_mode in ['fgbg', 'fgbg_yolo']:
            if self.blob_detector is None:
                reply = QMessageBox.question(
                    self,
                    "Background Not Initialized",
                    "Blob detection requires background initialization.\n\n"
                    "During analysis:\n"
                    "• Background will be initialized (Phase 1)\n"
                    "• Blob morphology will be learned (Phase 1b)\n"
                    "• CNN noise filter will be applied automatically\n"
                    "• Learned solidity filter will be applied automatically\n\n"
                    "Initialize background now (recommended for testing)?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                
                if reply == QMessageBox.StandardButton.Yes:
                    self.initialize_background()
                else:
                    QMessageBox.information(
                        self,
                        "Automatic Initialization",
                        "No problem! Background will be initialized automatically\n"
                        "during the full analysis."
                    )
        
        # Build mode description
        mode_desc = {
            'fgbg': (
                '<b>Motion Detection (Recommended)</b><br>'
                '• Background initialization<br>'
                '• Morphology learning (solidity threshold)<br>'
                '• CNN noise filter (66% reduction)<br>'
                '• Learned solidity filter (safety net)'
            ),
            'fgbg_yolo': (
                '<b>Motion + YOLO (High Accuracy)</b><br>'
                '• Background initialization<br>'
                '• Morphology learning<br>'
                '• CNN noise filter<br>'
                '• Learned solidity filter<br>'
                '• YOLO confirmation + species ID'
            ),
            'yolo_only': (
                '<b>YOLO Only (Highest Accuracy)</b><br>'
                '• Deep learning every frame<br>'
                '• No background/CNN/solidity needed<br>'
                '• Slowest but most accurate'
            )
        }
        
        # Confirm analysis
        reply = QMessageBox.question(
            self,
            "Run Analysis",
            f"<b>Video:</b> {Path(self.video_path).name}<br><br>"
            f"{mode_desc.get(detection_mode, '')}<br><br>"
            f"<b>Output:</b> {self.output_folder}<br><br>"
            f"⚠️ <i>This may take several minutes.</i><br><br>"
            f"Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # Create monitor with config
        monitor = BeeMonitor(config=self.config)
        
        # Create analysis thread with detection_mode parameter
        self.analysis_thread = AnalysisThread(
            monitor,
            self.video_path,
            self.output_folder,
            detection_mode=detection_mode  # Pass mode to BeeMonitor
        )
        
        self.analysis_thread.progress.connect(
            lambda msg: self.statusBar().showMessage(msg))
        self.analysis_thread.finished.connect(self.on_analysis_finished)
        self.analysis_thread.error.connect(
            lambda err: QMessageBox.critical(self, "Analysis Error", err))
        
        self.analysis_thread.start()
        self.statusBar().showMessage(f"Running analysis ({detection_mode})...")
    
    def test_detection(self):
        """Test detection on current frame using selected mode."""
        if self.current_frame is None:
            QMessageBox.warning(self, "Warning", "Load a video first")
            return
        
        try:
            params = self.control_panel.get_parameters()
            detection_mode = params.get("detection_mode", "fgbg_sift_yolo")
            
            detections = []
            
            # Dispatch to correct detector(s) based on mode
            if detection_mode in ['fgbg', 'fgbg_only', 'fgbg_sift', 'fgbg_yolo', 'fgbg_sift_yolo']:
                # Need blob detector
                if self.blob_detector is None:
                    reply = QMessageBox.question(
                        self,
                        "Background Not Initialized",
                        "Blob detection requires initialized background.\nInitialize now?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                    )
                    if reply == QMessageBox.StandardButton.Yes:
                        self.initialize_background()
                    else:
                        return
                
                # Use blob detector
                blob_dets = self.blob_detector.detect(self.current_frame)
                detections.extend(blob_dets)
            
            if detection_mode in ['sift', 'sift_only', 'fgbg_sift', 'sift_yolo', 'fgbg_sift_yolo']:
                # Need SIFT detector
                if self.sift_detector is None:
                    QMessageBox.warning(
                        self,
                        "SIFT Not Initialized",
                        "SIFT detection requires template initialization.\n"
                        "Please run full analysis first, or select a different mode."
                    )
                    # Don't return - try other modes if available
                else:
                    sift_dets = self.sift_detector.detect(self.current_frame, use_templates=True)
                    detections.extend(sift_dets)
            
            if detection_mode in ['yolo', 'yolo_only', 'fgbg_yolo', 'sift_yolo', 'fgbg_sift_yolo']:
                # Need YOLO detector
                if self.yolo_detector is None:
                    from beemonitor.detection import YOLODetector
                    from ultralytics import YOLO
                    model = YOLO('yolo11n.pt')
                    self.yolo_detector = YOLODetector(
                        model=model,
                        tracking_classes=['bee'],
                        conf_threshold=0.25
                    )
                
                yolo_dets = self.yolo_detector.detect(self.current_frame)
                detections.extend(yolo_dets)
            
            # Update detection count
            self.control_panel.set_detection_count(len(detections))
            
            # Get tracks if results loaded
            tracks_for_frame = self.get_tracks_for_frame(self.current_frame_idx)
            
            # Update data status
            self._update_data_status(tracks_for_frame)
            
            # Pass detections to canvas
            self.video_canvas.set_frame(
                self.current_frame,
                detections=detections,
                tracks=tracks_for_frame
            )
            
            # Auto-enable detection display if we have detections
            if len(detections) > 0:
                if not self.video_panel.show_detections_cb.isChecked():
                    print("Auto-enabling detection display - setting checkbox to True")
                    # Block signals to prevent loops
                    self.video_panel.show_detections_cb.blockSignals(True)
                    self.video_panel.show_detections_cb.setChecked(True)
                    self.video_panel.show_detections_cb.blockSignals(False)
                    # Manually trigger the display
                    self.video_canvas.show_detections = True
                    self.video_canvas._draw_frame()
                    print("Detection display enabled and redrawn")
                else:
                    # Checkbox already checked - force redraw to ensure it displays
                    print(f"Detections already enabled, forcing sync")
                    self.video_canvas.show_detections = True
                    self.video_canvas._draw_frame()
            
            # Display mode-specific message
            mode_names = {
                'fgbg': 'Motion',
                'fgbg_only': 'Motion',
                'sift': 'SIFT',
                'sift_only': 'SIFT',
                'yolo': 'YOLO',
                'yolo_only': 'YOLO',
                'fgbg_sift': 'Motion+SIFT',
                'fgbg_yolo': 'Motion+YOLO',
                'sift_yolo': 'SIFT+YOLO',
                'fgbg_sift_yolo': 'Motion+SIFT+YOLO'
            }
            mode_name = mode_names.get(detection_mode, detection_mode)
            
            self.statusBar().showMessage(
                f"✓ {mode_name}: {len(detections)} detections "
                f"(shown in {'color-coded' if self.video_canvas.show_detection_sources else 'green'})"
            )
            
        except Exception as e:
            import traceback
            QMessageBox.critical(self, "Error", 
                f"Detection failed:\n{e}\n\n{traceback.format_exc()}")
    
    # ========================================================================
    # NEW: Detection Source Visualization
    # ========================================================================
    
    def on_show_detections_changed(self, enabled):
        """Handle Show Detections checkbox change."""
        print(f"on_show_detections_changed called: enabled={enabled}")
        self.video_canvas.toggle_detections(enabled)
        if enabled:
            self.statusBar().showMessage("Detection overlay: ENABLED")
            # Force redraw if we have detections
            if self.video_canvas.detections:
                print(f"  -> Have {len(self.video_canvas.detections)} detections, forcing redraw")
                self.video_canvas._draw_frame()
    
    def on_show_tracks_changed(self, enabled):
        """Handle Show Tracks checkbox change."""
        print(f"on_show_tracks_changed called: enabled={enabled}")
        self.video_canvas.toggle_tracks(enabled)
        if enabled:
            self.statusBar().showMessage("Track overlay: ENABLED")
            # Force redraw if we have tracks
            if self.video_canvas.tracks:
                print(f"  -> Have {len(self.video_canvas.tracks)} tracks, forcing redraw")
                self.video_canvas._draw_frame()
    
    def on_show_sources_changed(self, enabled):
        """Handle detection source visualization toggle."""
        print(f"on_show_sources_changed called: enabled={enabled}")
        self.video_canvas.toggle_detection_sources(enabled)
        if enabled:
            self.statusBar().showMessage(
                "Detection sources: ENABLED (RED=Blob, GREEN=SIFT, BLUE=YOLO)"
            )
            # Force redraw if we have detections
            if self.video_canvas.detections:
                print(f"  -> Have {len(self.video_canvas.detections)} detections, forcing redraw")
                self.video_canvas._draw_frame()
        else:
            self.statusBar().showMessage("Detection sources: DISABLED")
    
    # ========================================================================
    # Results Loading and Visualization
    # ========================================================================
    
    def load_results(self):
        """Load tracking results CSV."""
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
            # Check if file is empty
            import os
            if os.path.getsize(filepath) == 0:
                QMessageBox.critical(
                    self,
                    "Empty File",
                    f"The tracking results file is empty!\n\n"
                    f"File: {Path(filepath).name}\n\n"
                    f"This means the analysis didn't produce any results.\n\n"
                    f"Possible reasons:\n"
                    f"  • No detections found in video\n"
                    f"  • Analysis failed silently\n"
                    f"  • Detection parameters too strict\n\n"
                    f"Try:\n"
                    f"  1. Test detection on a frame first\n"
                    f"  2. Lower detection thresholds\n"
                    f"  3. Check analysis error messages"
                )
                return
            
            df = pd.read_csv(filepath)
            
            # Debug: Print CSV info
            print(f"\n{'='*60}")
            print(f"Loading CSV: {Path(filepath).name}")
            print(f"Shape: {df.shape} (rows={len(df)}, cols={len(df.columns)})")
            print(f"Columns: {list(df.columns)}")
            if len(df) > 0:
                print(f"First row: {df.iloc[0].to_dict()}")
            print(f"{'='*60}\n")
            
            # Check if this is the events file instead of tracking file
            if 'action' in df.columns and 'nest' in df.columns:
                QMessageBox.warning(
                    self,
                    "Wrong File",
                    "This appears to be an <b>events CSV</b> file.\n\n"
                    "For visualization, you need the <b>tracking results CSV</b>.\n\n"
                    "Look for a file named:\n"
                    "  • tracking_results.csv\n"
                    "  • tracks.csv\n"
                    "  • <video_name>_tracks.csv"
                )
                
                # Try to find the tracking file automatically
                tracking_file = find_tracking_file(filepath)
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
            
            # Validate required columns
            if 'track_id' not in df.columns:
                QMessageBox.warning(self, "Invalid File", "Missing 'track_id' column")
                return
            
            # Handle different frame column names
            frame_col = None
            for possible in ['frame', 'frame_number', 'frame_num', 'frame_id', 'frame_idx']:
                if possible in df.columns:
                    frame_col = possible
                    break
            
            if frame_col is None:
                # Show available columns for debugging
                QMessageBox.critical(
                    self,
                    "Invalid CSV Format",
                    f"Missing frame column in CSV!\n\n"
                    f"Available columns:\n{', '.join(df.columns)}\n\n"
                    f"Expected one of:\n"
                    f"  • frame\n"
                    f"  • frame_number\n"
                    f"  • frame_num\n"
                    f"  • frame_id\n\n"
                    f"This CSV may not be a tracking results file."
                )
                return
            
            # Rename to standard 'frame'
            if frame_col != 'frame':
                try:
                    df = df.rename(columns={frame_col: 'frame'})
                    print(f"Renamed column '{frame_col}' to 'frame'")
                except Exception as e:
                    QMessageBox.critical(
                        self,
                        "Column Rename Failed",
                        f"Failed to rename '{frame_col}' to 'frame':\n{e}\n\n"
                        f"Columns: {list(df.columns)}"
                    )
                    return
            
            # Verify frame column exists after rename
            if 'frame' not in df.columns:
                QMessageBox.critical(
                    self,
                    "Frame Column Missing",
                    f"Frame column disappeared after processing!\n\n"
                    f"Original column: {frame_col}\n"
                    f"Current columns: {list(df.columns)}"
                )
                return
            
            # Check for position columns
            has_bbox = all(col in df.columns for col in ['x1', 'y1', 'x2', 'y2'])
            has_xy = all(col in df.columns for col in ['x', 'y'])
            has_centroid = all(col in df.columns for col in ['centroid_x', 'centroid_y'])
            
            if not (has_bbox or has_xy or has_centroid):
                QMessageBox.warning(self, "Invalid File", "Missing position data")
                return
            
            # Success!
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
            
            # ============================================================
            # CRITICAL FIX: Enable visualization with signal blocking
            # ============================================================
            
            # Block signals to prevent unwanted state changes
            self.video_panel.show_tracks_cb.blockSignals(True)
            self.video_panel.show_detections_cb.blockSignals(True)
            self.video_panel.show_sources_cb.blockSignals(True)
            
            # Set checkbox visual state
            self.video_panel.show_tracks_cb.setChecked(True)
            self.video_panel.show_detections_cb.setChecked(True)
            self.video_panel.show_sources_cb.setChecked(True)
            
            # Unblock signals
            self.video_panel.show_tracks_cb.blockSignals(False)
            self.video_panel.show_detections_cb.blockSignals(False)
            self.video_panel.show_sources_cb.blockSignals(False)
            
            # CRITICAL: Directly set canvas flags (don't rely on signals)
            self.video_canvas.show_detections = True
            self.video_canvas.show_tracks = True
            self.video_canvas.show_detection_sources = True
            
            print(f"✓ Visualization enabled:")
            print(f"  Checkboxes: Detections={self.video_panel.show_detections_cb.isChecked()}, "
                  f"Tracks={self.video_panel.show_tracks_cb.isChecked()}, "
                  f"Sources={self.video_panel.show_sources_cb.isChecked()}")
            print(f"  Canvas flags: show_detections={self.video_canvas.show_detections}, "
                  f"show_tracks={self.video_canvas.show_tracks}, "
                  f"show_detection_sources={self.video_canvas.show_detection_sources}")
            
            # Refresh current frame to display overlays
            if self.current_frame is not None:
                self.load_frame(self.current_frame_idx)
            
            self.statusBar().showMessage(
                f"✓ Results loaded: {total_tracks} tracks across {total_frames} frames"
            )
            
            print(f"Results loaded: {filepath}")
            print(f"  Tracks: {total_tracks}")
            print(f"  Frames: {total_frames}")
            
        except Exception as e:
            import traceback
            error_msg = f"Failed to load results:\n{e}\n\n{traceback.format_exc()}"
            QMessageBox.critical(self, "Error", error_msg)
    
    def get_tracks_for_frame(self, frame_idx):
        """Get all tracks visible in this frame."""
        if not self.results_loaded or self.tracking_results is None:
            return {}
        
        # Check if frame column exists
        if 'frame' not in self.tracking_results.columns:
            print(f"WARNING: 'frame' column not found in tracking results")
            print(f"Available columns: {list(self.tracking_results.columns)}")
            return {}
        
        # Get all detections for this frame
        frame_data = self.tracking_results[
            self.tracking_results['frame'] == frame_idx
        ]
        
        if len(frame_data) == 0:
            return {}
        
        # Build track trajectories (last N frames for each visible track)
        tracks = {}
        
        for track_id in frame_data['track_id'].unique():
            track_data = self.tracking_results[
                (self.tracking_results['track_id'] == track_id) &
                (self.tracking_results['frame'] <= frame_idx) &
                (self.tracking_results['frame'] > frame_idx - TRAJECTORY_WINDOW)
            ]
            
            if len(track_data) > 0:
                centroids = []
                for _, row in track_data.iterrows():
                    pos = get_position_from_row(row)
                    if pos:
                        centroids.append(pos)
                
                if centroids:
                    tracks[track_id] = centroids
        
        return tracks
    
    def get_detections_for_frame(self, frame_idx):
        """Get all detections for this frame as Detection objects."""
        if not self.results_loaded or self.tracking_results is None:
            return []
        
        # Check if frame column exists
        if 'frame' not in self.tracking_results.columns:
            return []
        
        # Get all detections for this frame
        frame_data = self.tracking_results[
            self.tracking_results['frame'] == frame_idx
        ]
        
        if len(frame_data) == 0:
            return []
        
        # Convert CSV rows to Detection objects
        detections = []
        
        for _, row in frame_data.iterrows():
            # Check if we have bounding box columns
            if all(col in row.index for col in ['x1', 'y1', 'x2', 'y2']):
                bbox = (row['x1'], row['y1'], row['x2'], row['y2'])
                centroid = ((row['x1'] + row['x2']) / 2, (row['y1'] + row['y2']) / 2)
                
                # Create Detection object
                det = Detection(
                    bbox=bbox,
                    centroid=centroid,
                    confidence=row.get('confidence', 1.0),
                    label=row.get('species', 'bee'),
                    source=row.get('source', 'blob')  # Use source from CSV, default to blob
                )
                
                detections.append(det)
        
        return detections
    
    # ========================================================================
    # Analysis
    # ========================================================================
    
    def run_analysis(self):
        """Run full video analysis."""
        if self.video_path is None:
            QMessageBox.warning(self, "Warning", "Load a video first")
            return
        
        # Check if detection has been tested
        if self.blob_detector is None:
            reply = QMessageBox.question(
                self,
                "Background Not Initialized",
                "Background model not initialized.\n\n"
                "Initialize background now before running analysis?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.initialize_background()
                
                # Suggest testing
                test_reply = QMessageBox.question(
                    self,
                    "Test Detection First?",
                    "It's recommended to test detection on a frame first\n"
                    "to verify parameters work correctly.\n\n"
                    "Test detection now?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                
                if test_reply == QMessageBox.StandardButton.Yes:
                    return  # Let them test first
            else:
                return
        
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
            f"⚠️  Make sure test detection works before running full analysis!\n"
            f"If test detection shows 0 detections, the tracking file will be empty.\n\n"
            f"This may take several minutes. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        params = self.control_panel.get_parameters()
        
        # Get detection mode from control panel
        detection_mode = params.get("detection_mode", "yolo_only")
        
        # Use defaults - these will be learned automatically in Phase 1b
        self.config.detection.min_area = 120.0
        self.config.detection.min_solidity = 0.7
        self.config.detection.max_area = 4000.0
        
        self.config.detection.sync_to_tracking(self.config.tracking)
        
        os.makedirs(self.output_folder, exist_ok=True)
        
        monitor = BeeMonitor(config=self.config)
        
        self.analysis_thread = AnalysisThread(
            monitor,
            self.video_path,
            self.output_folder,
            detection_mode=detection_mode
        )
        
        self.analysis_thread.progress.connect(
            lambda msg: self.statusBar().showMessage(msg))
        self.analysis_thread.finished.connect(self.on_analysis_finished)
        self.analysis_thread.error.connect(
            lambda err: QMessageBox.critical(self, "Analysis Error", err))
        
        self.analysis_thread.start()
        self.statusBar().showMessage("Running analysis...")
    
    def on_analysis_finished(self, result, csv_path):
        """Handle analysis completion."""
        self.statusBar().showMessage("✓ Analysis complete")
        
        msg = (
            f"✓ Analysis complete!\n\n"
            f"Output folder: {self.output_folder}\n\n"
            f"Files saved:\n"
            f"  • tracking_results.csv (tracking data)\n\n"
            f"Next steps:\n"
            f"1. File → Load Results to visualize tracks on original video"
        )
        
        QMessageBox.information(self, "Analysis Complete", msg)
        
        # Offer to load results automatically
        if os.path.exists(csv_path):
            auto_load = QMessageBox.question(
                self,
                "Load Results?",
                "Would you like to load the results now to visualize tracks?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if auto_load == QMessageBox.StandardButton.Yes:
                try:
                    self.tracking_results = pd.read_csv(csv_path)
                    self.results_loaded = True
                    
                    self.video_panel.show_tracks_cb.setChecked(True)
                    
                    if self.current_frame is not None:
                        self.load_frame(self.current_frame_idx)
                    
                    self.statusBar().showMessage("✓ Results loaded - tracks visible on video")
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Could not load results:\n{e}")
    
    def load_output_video(self):
        """Load output visualization video."""
        default_dir = self.output_folder if self.output_folder else str(Path.home())
        
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Load Output Video",
            default_dir,
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        
        if filepath:
            # Reuse load_video logic
            self.load_video.__wrapped__(self, filepath)
            
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
        
        default_dir = self.output_folder if self.output_folder else str(Path.home())
        video_name = Path(self.video_path).stem
        default_path = os.path.join(default_dir, f"{video_name}_visualization.mp4")
        
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Visualization Video",
            default_path,
            "MP4 Video (*.mp4);;AVI Video (*.avi);;All Files (*)"
        )
        
        if not output_path:
            return
        
        include_detections = self.video_panel.show_detections_cb.isChecked()
        include_tracks = self.video_panel.show_tracks_cb.isChecked()
        
        reply = QMessageBox.question(
            self,
            "Visualization Options",
            f"Include in video:\n"
            f"  • Detections: {'Yes' if include_detections else 'No'}\n"
            f"  • Tracks: {'Yes' if include_tracks else 'No'}\n\n"
            f"This will process the entire video.\n"
            f"Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        progress = QProgressDialog(
            "Saving visualization video...", "Cancel", 0, self.total_frames, self
        )
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.video_cap.get(cv2.CAP_PROP_FPS)
            
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            for frame_idx in range(self.total_frames):
                if progress.wasCanceled():
                    break
                
                ret, frame = self.video_cap.read()
                if not ret:
                    break
                
                vis_frame = frame.copy()
                
                # Draw detections
                if include_detections and self.blob_detector:
                    try:
                        detections = self.blob_detector.detect(frame)
                        for det in detections:
                            x1, y1, x2, y2 = [int(c) for c in det.bbox]
                            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    except:
                        pass
                
                # Draw tracks
                if include_tracks and self.results_loaded:
                    tracks = self.get_tracks_for_frame(frame_idx)
                    colors = [(255, 0, 0), (0, 255, 255), (255, 0, 255), 
                             (255, 255, 0), (128, 0, 255), (255, 128, 0)]
                    
                    for i, (track_id, trajectory) in enumerate(tracks.items()):
                        color = colors[i % len(colors)]
                        
                        if len(trajectory) > 1:
                            points = np.array(trajectory, dtype=np.int32)
                            cv2.polylines(vis_frame, [points], False, color, 2)
                        
                        if trajectory:
                            x, y = trajectory[-1]
                            cv2.circle(vis_frame, (int(x), int(y)), 5, color, -1)
                            cv2.putText(vis_frame, f"ID:{track_id}", 
                                       (int(x)+10, int(y)),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                cv2.putText(vis_frame, f"Frame: {frame_idx}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                out.write(vis_frame)
                progress.setValue(frame_idx)
                QApplication.processEvents()
            
            out.release()
            self.load_frame(self.current_frame_idx)
            progress.close()
            
            QMessageBox.information(
                self, "Success",
                f"Visualization video saved!\n\n{output_path}"
            )
            
            self.statusBar().showMessage(f"✓ Visualization saved: {Path(output_path).name}")
            
        except Exception as e:
            import traceback
            progress.close()
            QMessageBox.critical(self, "Error", 
                f"Failed to save video:\n{e}\n\n{traceback.format_exc()}")
    
    # ========================================================================
    # Utility
    # ========================================================================
    
    def set_output_folder(self):
        """Set output folder."""
        current = self.output_folder if self.output_folder else str(Path.home())
        
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder", current)
        
        if folder:
            self.output_folder = folder
            os.makedirs(self.output_folder, exist_ok=True)
            self.statusBar().showMessage(f"Output folder: {folder}")
    
    def save_config(self):
        """Save configuration."""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Configuration", "", "JSON Files (*.json);;All Files (*)"
        )
        
        if not filepath:
            return
        
        params = self.control_panel.get_parameters()
        
        # Get detection mode from control panel
        detection_mode = params.get("detection_mode", "yolo_only")
        
        config_data = {
            "detection": params,
            "video_path": self.video_path,
            "output_folder": self.output_folder,
            "saved_at": datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        QMessageBox.information(self, "Success", "Configuration saved")
    
    def load_config(self):
        """Load configuration."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Configuration", "", "JSON Files (*.json);;All Files (*)"
        )
        
        if not filepath:
            return
        
        try:
            with open(filepath, 'r') as f:
                config_data = json.load(f)
            
            if "detection" in config_data:
                self.control_panel.set_parameters(config_data["detection"])
            
            if "output_folder" in config_data:
                self.output_folder = config_data["output_folder"]
            
            QMessageBox.information(self, "Success", "Configuration loaded")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load configuration:\n{e}")
    
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