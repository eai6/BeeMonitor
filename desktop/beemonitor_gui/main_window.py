"""Main Window - BeeMonitor Desktop App with Batch Folder Analysis"""

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
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
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


class FolderAnalysisThread(QThread):
    """Thread for analyzing multiple videos in a folder."""
    
    progress = pyqtSignal(str)
    video_completed = pyqtSignal(str, object)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, monitor, folder_path, output_folder, params):
        super().__init__()
        self.monitor = monitor
        self.folder_path = folder_path
        self.output_folder = output_folder
        self.params = params
    
    def run(self):
        """Run folder analysis."""
        try:
            self.progress.emit(f"Starting batch analysis of folder: {Path(self.folder_path).name}")
            
            results = self.monitor.analyze_videos_in_folder(
                video_folder=self.folder_path,
                output_folder=self.output_folder,
                visualize=self.params.get('visualize', True),
                detection_mode=self.params.get('detection_mode', 'fgbg_yolo'),
                use_fallback=self.params.get('use_fallback', True),
                max_workers=self.params.get('max_workers', 4)
            )
            
            self.finished.emit(results)
            
        except Exception as e:
            import traceback
            error_msg = f"Folder analysis failed: {e}\n\n{traceback.format_exc()}"
            self.error.emit(error_msg)


class BeeMonitorGUI(QMainWindow):
    """Main GUI application with video player controls and batch folder analysis."""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle(TITLE)
        self.setGeometry(100, 100, *DEFAULT_WINDOW_SIZE)
        
        self.video_path = None
        self.video_cap = None
        self.current_frame = None
        self.current_frame_idx = 0
        self.total_frames = 0
        self.fps = 0
        
        self.playing = False
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self.play_next_frame)
        
        self.tracking_results = None
        self.results_loaded = False
        
        self.config = Config.default()
        self.blob_detector = None
        self.yolo_detector_test = None
        self.output_folder = None
        
        self.analysis_thread = None
        self.folder_path = None
        self.folder_analysis_thread = None
        
        # Pre-initialize nest detection models (for fast reuse)
        self.nest_yolo_model = None
        self.nest_config = None
        self._init_nest_detection_models()
        
        self.control_panel = ControlPanel()
        self.video_panel = VideoPanel()
        self.video_canvas = self.video_panel.get_canvas()
        
        self._connect_signals()
        self._create_menu_bar()
        self._create_main_widget()
        
        self.video_canvas.show_detections = True
        self.video_canvas.show_tracks = True
        self.video_canvas.show_detection_sources = True
        
        self.statusBar().showMessage("Ready - Load a video to begin")
        
        print(f"✓ BeeMonitor GUI v{VERSION} initialized")
    
    def _init_nest_detection_models(self):
        """Initialize models for nest detection once at startup.
        
        Pre-loads:
        - YOLO model for nest detection (from config.model.nest_detection)
        - Config with GUI-specific settings (15 attempts vs 5 default)
        
        This makes subsequent nest detections much faster (reuses models).
        """
        try:
            from ultralytics import YOLO
            from beemonitor.core.config import Config
            
            print("Initializing nest detection models...")
            
            # Load config
            self.nest_config = Config.default()
            
            # Get nest detection model path from config
            nest_model_path = self.nest_config.models.nest_detection
            print(f"  Loading nest model: {nest_model_path}")
            
            # Check if model file exists
            import os
            if not os.path.exists(nest_model_path):
                raise FileNotFoundError(f"Nest detection model not found at: {nest_model_path}")
            
            # Load YOLO model
            self.nest_yolo_model = YOLO(nest_model_path)
            
            # Set GUI-specific config (more attempts than batch processing)
            self.nest_config.nest.max_detection_attempts = 15
            
            print("✓ Nest detection models ready")
            print(f"  Model: {nest_model_path}")
            print(f"  Max attempts: {self.nest_config.nest.max_detection_attempts}")
            
        except FileNotFoundError as e:
            print(f"⚠️  Nest detection model not found: {e}")
            print(f"   Please ensure model exists at configured path")
            print(f"   Nest detection will be unavailable")
            self.nest_yolo_model = None
            self.nest_config = None
        except Exception as e:
            print(f"⚠️  Could not initialize nest detection models: {e}")
            print(f"   Nest detection will be unavailable")
            import traceback
            traceback.print_exc()
            self.nest_yolo_model = None
            self.nest_config = None
    
    def _connect_signals(self):
        """Connect all signals from panels to methods."""
        self.control_panel.load_video_requested.connect(self.load_video)
        self.control_panel.initialize_background_requested.connect(self.initialize_background)
        self.control_panel.run_analysis_requested.connect(self.run_analysis)
        self.control_panel.stop_analysis_requested.connect(self.stop_analysis)
        self.control_panel.parameters_changed.connect(self.on_parameters_changed)
        
        # Folder analysis signals
        self.control_panel.folder_selected.connect(self.select_folder)
        self.control_panel.analyze_folder_requested.connect(self.run_folder_analysis)
        
        self.video_panel.play_pause_toggled.connect(self.toggle_play_pause)
        self.video_panel.frame_changed.connect(self.on_frame_slider_change)
        self.video_panel.frame_step_requested.connect(self.jump_frame)
        self.video_panel.speed_changed.connect(self.on_speed_change)
    
    def _create_menu_bar(self):
        """Create menu bar."""
        menubar = self.menuBar()
        
        file_menu = menubar.addMenu("&File")
        
        load_action = QAction("&Load Video...", self)
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self.load_video)
        file_menu.addAction(load_action)
        
        analyze_folder_action = QAction("Analyze &Folder...", self)
        analyze_folder_action.triggered.connect(self.select_folder)
        file_menu.addAction(analyze_folder_action)
        
        file_menu.addSeparator()
        
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
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        central_widget.setLayout(main_layout)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        splitter.addWidget(self.control_panel)
        splitter.addWidget(self.video_panel)
        
        splitter.setSizes([400, 1000])
    
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
        
        self.control_panel.set_video_info(
            f"<b>{Path(filepath).name}</b><br>"
            f"{width}x{height} @ {self.fps:.1f} FPS<br>"
            f"{self.total_frames} frames ({self.total_frames/self.fps:.1f}s)"
        )
        
        video_dir = Path(filepath).parent
        video_name = Path(filepath).stem
        self.output_folder = str(video_dir / f"{video_name}_output")
        os.makedirs(self.output_folder, exist_ok=True)
        
        self.control_panel.set_output_folder_info(
            f"<b>Output:</b> {Path(self.output_folder).name}/"
        )
        
        self.video_panel.set_frame_range(self.total_frames - 1)
        self.current_frame_idx = 0
        
        # Read first frame for nest detection
        ret, first_frame = self.video_cap.read()
        if ret:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start
            
            # Auto-detect nests on first frame
            self._auto_detect_nests(first_frame)
        
        self.load_frame(0)
        self.video_panel.enable_play_button(True)
        
        self.statusBar().showMessage(
            f"Loaded: {Path(filepath).name} | Output: {self.output_folder}"
        )
        
        print(f"✓ Video loaded: {filepath}")
        
        self._initialize_background()
        
        self.control_panel.set_video_loaded(True)
        self.control_panel.append_log(f"✓ Loaded: {Path(filepath).name}")
    
    def initialize_background(self):
        """Initialize background model."""
        if not self.video_path:
            QMessageBox.warning(self, "Warning", "Load video first")
            return
        
        self.control_panel.set_background_initialized(True)
        self.control_panel.append_log("✓ Background initialized")
        self.statusBar().showMessage("✓ Background ready")
    
    def stop_analysis(self):
        """Stop running analysis."""
        if self.analysis_thread and self.analysis_thread.isRunning():
            self.analysis_thread.terminate()
            self.control_panel.set_analysis_running(False)
            self.control_panel.append_log("Analysis stopped by user")
            self.statusBar().showMessage("Analysis stopped")
    
    def select_folder(self):
        """Select folder containing videos for batch analysis."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Video Folder",
            str(Path.home()),
            QFileDialog.Option.ShowDirsOnly
        )
        
        if folder:
            self.folder_path = folder
            
            video_files = [f for f in os.listdir(folder) 
                          if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
            
            if not video_files:
                QMessageBox.warning(
                    self,
                    "No Videos Found",
                    f"No video files found in:\n{folder}\n\n"
                    f"Supported formats: .mp4, .avi, .mov, .mkv"
                )
                return
            
            self.control_panel.set_folder_path(folder)
            self.control_panel.append_log(f"✓ Selected folder: {Path(folder).name}")
            self.control_panel.append_log(f"  Found {len(video_files)} video files")
            
            self.statusBar().showMessage(
                f"Folder selected: {len(video_files)} videos found"
            )
    
    def run_folder_analysis(self, params):
        """Run batch video analysis on folder."""
        if not self.folder_path:
            QMessageBox.warning(self, "Warning", "Select a video folder first")
            return
        
        video_files = [f for f in os.listdir(self.folder_path) 
                      if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        if not video_files:
            QMessageBox.warning(self, "Warning", "No video files in selected folder")
            return
        
        folder_name = Path(self.folder_path).name
        
        # Include detection mode in output folder name to avoid overwrites
        mode_suffix = params['detection_mode'].replace('_', '-')
        output_folder = str(Path(self.folder_path).parent / f"{folder_name}_output_{mode_suffix}")
        os.makedirs(output_folder, exist_ok=True)
        
        mode_names = {
            'fgbg_yolo': 'Motion + YOLO',
            'yolo_only': 'YOLO Only'
        }
        mode_name = mode_names.get(params['detection_mode'], params['detection_mode'])
        
        reply = QMessageBox.question(
            self,
            "Batch Video Analysis",
            f"Analyze {len(video_files)} videos?\n\n"
            f"Folder: {Path(self.folder_path).name}\n"
            f"Output: {Path(output_folder).name}\n"
            f"Detection mode: {mode_name}\n"
            f"Nest fallback: {'Enabled' if params['use_fallback'] else 'Disabled'}\n"
            f"Parallel workers: {params['max_workers']}\n\n"
            f"This may take a while...",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        self.control_panel.set_folder_analyzing(True)
        self.control_panel.set_folder_progress(0, len(video_files))
        self.control_panel.append_log(f"\n{'='*50}")
        self.control_panel.append_log(f"BATCH ANALYSIS STARTED")
        self.control_panel.append_log(f"{'='*50}")
        self.control_panel.append_log(f"Videos: {len(video_files)}")
        self.control_panel.append_log(f"Mode: {mode_name}")
        self.control_panel.append_log(f"Fallback: {'ON' if params['use_fallback'] else 'OFF'}")
        self.control_panel.append_log(f"Workers: {params['max_workers']}")
        
        self.statusBar().showMessage(f"Analyzing {len(video_files)} videos...")
        
        monitor = BeeMonitor(config=self.config)
        
        self.folder_analysis_thread = FolderAnalysisThread(
            monitor,
            self.folder_path,
            output_folder,
            params
        )
        
        self.folder_analysis_thread.progress.connect(
            lambda msg: self.control_panel.append_log(msg)
        )
        self.folder_analysis_thread.finished.connect(self.on_folder_analysis_finished)
        self.folder_analysis_thread.error.connect(
            lambda err: QMessageBox.critical(self, "Batch Analysis Error", err)
        )
        
        self.folder_analysis_thread.start()
    
    def on_folder_analysis_finished(self, results):
        """Handle folder analysis completion."""
        self.control_panel.set_folder_analyzing(False)
        
        total_videos = len(results)
        successful = sum(1 for r in results.values() if r is not None)
        failed = total_videos - successful
        total_events = sum(len(r.events) for r in results.values() if r is not None)
        
        self.control_panel.append_log(f"\n{'='*50}")
        self.control_panel.append_log(f"BATCH ANALYSIS COMPLETE")
        self.control_panel.append_log(f"{'='*50}")
        self.control_panel.append_log(f"Total videos: {total_videos}")
        self.control_panel.append_log(f"Successful: {successful}")
        self.control_panel.append_log(f"Failed: {failed}")
        self.control_panel.append_log(f"Total events: {total_events}")
        self.control_panel.append_log(f"{'='*50}")
        
        summary_text = (
            f"Batch analysis complete!\n\n"
            f"Total videos: {total_videos}\n"
            f"Successful: {successful}\n"
            f"Failed: {failed}\n"
            f"Total events detected: {total_events}\n\n"
        )
        
        if failed > 0:
            failed_videos = [Path(p).name for p, r in results.items() if r is None]
            summary_text += f"Failed videos:\n"
            for v in failed_videos[:5]:
                summary_text += f"  • {v}\n"
            if len(failed_videos) > 5:
                summary_text += f"  ... and {len(failed_videos) - 5} more\n"
        
        # Use actual output folder from thread (includes detection mode)
        output_path = self.folder_analysis_thread.output_folder if self.folder_analysis_thread else "output"
        summary_text += f"\nOutput folder:\n{output_path}"
        
        QMessageBox.information(
            self,
            "Batch Analysis Complete",
            summary_text
        )
        
        self.statusBar().showMessage(
            f"✓ Batch analysis complete: {successful}/{total_videos} successful"
        )
    
    def _auto_detect_nests(self, first_frame):
        """Auto-detect nest tubes on first frame using simple YOLO detection.
        
        Args:
            first_frame: First frame of video (BGR image)
        """
        try:
            from ultralytics import YOLO
            import cv2
            
            print("🔍 Auto-detecting nest tubes...")
            self.statusBar().showMessage("Detecting nest tubes...")
            QApplication.processEvents()
            
            # Use YOLO directly to detect objects that look like nest tubes
            yolo_model = YOLO('yolo11n.pt')
            
            # Get video dimensions
            height, width = first_frame.shape[:2]
            
            # Run YOLO detection on first frame
            results = yolo_model(first_frame, verbose=False)
            
            # Extract detections and filter for potential nest tubes
            # We'll look for small rectangular objects arranged in a grid pattern
            detections = []
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    
                    # Filter: small objects with decent confidence
                    w, h = x2 - x1, y2 - y1
                    if 20 < w < 100 and 20 < h < 100 and conf > 0.3:
                        # Create a simple Detection-like object
                        from collections import namedtuple
                        Detection = namedtuple('Detection', ['bbox', 'confidence'])
                        det = Detection(
                            bbox=(int(x1), int(y1), int(x2), int(y2)),
                            confidence=conf
                        )
                        detections.append(det)
            
            # If YOLO didn't find good candidates, try grid-based approach
            if len(detections) < 10:
                print("   YOLO found few candidates, trying grid-based detection...")
                detections = self._detect_nests_by_grid(first_frame)
            
            if detections and len(detections) > 0:
                # Sort by position (top to bottom, left to right)
                detections.sort(key=lambda d: (d.bbox[1], d.bbox[0]))
                
                # Store nests
                self.video_canvas.detected_nests = detections
                self.video_canvas.show_nests = True
                
                # Compute hotel ROI
                if len(detections) > 1:
                    xs = [d.bbox[0] for d in detections] + [d.bbox[2] for d in detections]
                    ys = [d.bbox[1] for d in detections] + [d.bbox[3] for d in detections]
                    padding = 20
                    hotel_roi = (
                        max(0, int(min(xs)) - padding),
                        max(0, int(min(ys)) - padding),
                        int(max(xs)) + padding,
                        int(max(ys)) + padding
                    )
                    self.video_canvas.hotel_roi = hotel_roi
                
                print(f"✓ Detected {len(detections)} nest tubes")
                self.control_panel.append_log(f"✓ Detected {len(detections)} nest tubes")
                self.statusBar().showMessage(f"✓ Detected {len(detections)} nest tubes")
            else:
                print("⚠️  No nest tubes detected - you can load nests from CSV instead")
                self.control_panel.append_log("⚠️  No nest tubes detected automatically")
                self.statusBar().showMessage("No nest tubes detected")
        
        except Exception as e:
            print(f"⚠️  Nest detection failed: {e}")
            print("   You can load nest positions from CSV instead")
            self.control_panel.append_log(f"⚠️  Automatic nest detection unavailable")
            self.statusBar().showMessage("Video loaded (nest detection unavailable)")
            import traceback
            traceback.print_exc()
    
    def _auto_detect_nests(self, first_frame):
        """Auto-detect nest tubes using comprehensive multi-frame detection.
        
        Uses pre-initialized models (loaded at startup) for fast detection:
        - YOLO model from config.model.nest_detection
        - Config with 15 max attempts (GUI-specific)
        
        Quality checks:
        - Correct nest count, grid alignment, spacing
        - Automatic retries with frame skipping
        
        Args:
            first_frame: First frame of video (used only to get dimensions)
        """
        try:
            from beemonitor.detection import NestDetector
            import logging
            
            # Check if models are available
            if self.nest_yolo_model is None or self.nest_config is None:
                print("⚠️  Nest detection unavailable (models not loaded at startup)")
                print("   Check that nest detection model exists at configured path")
                self.control_panel.append_log("⚠️  Nest detection unavailable")
                return
            
            # Enable detailed logging for nest detection
            logging.basicConfig(level=logging.INFO)
            nest_logger = logging.getLogger('beemonitor.detection.nest_detector')
            nest_logger.setLevel(logging.INFO)
            
            print("🔍 Auto-detecting nest tubes (comprehensive method)...")
            print("   This will try multiple frames with quality checks...")
            self.statusBar().showMessage("Detecting nest tubes (trying multiple frames)...")
            QApplication.processEvents()
            
            # Use pre-initialized models (much faster!)
            nest_detector = NestDetector(
                model=self.nest_yolo_model,  # ← Reused from startup
                config=self.nest_config       # ← Reused from startup
            )
            
            # Show config settings
            print(f"   Config: max_attempts={self.nest_config.nest.max_detection_attempts}, "
                  f"frame_skip={self.nest_config.nest.frame_skip}")
            print(f"   Expected: {self.nest_config.nest.expected_total_nests} nests "
                  f"({self.nest_config.nest.expected_rows} rows × "
                  f"{self.nest_config.nest.expected_nests_per_row} per row)")
            print(f"   Will try frames: 0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360, 390, 420")
            
            # Use comprehensive detection method (same as video analyzer)
            nests_dict = nest_detector.get_nests_and_hotel_detections(
                video_path=self.video_path
            )
            
            if nests_dict and 'nests' in nests_dict and len(nests_dict['nests']) > 0:
                # Convert to our format for display
                from collections import namedtuple
                Detection = namedtuple('Detection', ['bbox', 'confidence', 'nest_id'])
                
                nests = []
                for nest_id, bbox in nests_dict['nests'].items():
                    det = Detection(
                        bbox=bbox,
                        confidence=1.0,
                        nest_id=int(nest_id)
                    )
                    nests.append(det)
                
                # Store nests for display
                self.video_canvas.detected_nests = nests
                self.video_canvas.show_nests = True
                self.video_canvas.hotel_roi = nests_dict.get('hotel')
                
                # REFRESH FRAME to show nests immediately
                if self.current_frame_idx is not None:
                    self.load_frame(self.current_frame_idx)
                
                print(f"✓ Detected {len(nests)} nest tubes (quality verified)")
                print(f"  → Green boxes now displayed on video")
                self.control_panel.append_log(f"✓ Detected {len(nests)} nest tubes (quality verified)")
                self.control_panel.append_log(f"  → Nest boxes now visible on video (green)")
                self.statusBar().showMessage(f"✓ Detected {len(nests)} nest tubes")
            else:
                print("⚠️  No nest tubes detected after 15 attempts")
                print("   ")
                print("   Possible reasons:")
                print("   • Video doesn't show bee hotel clearly in first 7+ minutes")
                print("   • Hotel has different grid size (not 6×10)")
                print("   • Quality checks too strict for this video setup")
                print("   • Model not detecting nest tubes reliably")
                print("   ")
                print("   Workarounds:")
                print("   1. Check video shows hotel clearly")
                print("   2. Try different video or later timestamp")
                print("   3. Load nest positions from CSV")
                print("   4. Continue without nest visualization (analysis still works!)")
                print("   5. Check logs above to see which quality check failed")
                print("   ")
                self.control_panel.append_log("⚠️  No nest tubes detected (tried 15 frames)")
                self.control_panel.append_log("   You can load nest positions from CSV or continue without them")
                self.statusBar().showMessage("No nest tubes detected (see console for details)")
        
        except Exception as e:
            print(f"⚠️  Nest detection failed with error: {e}")
            print("   You can continue without nest visualization")
            self.control_panel.append_log(f"⚠️  Nest detection error: {e}")
            self.statusBar().showMessage("Video loaded (nest detection unavailable)")
            import traceback
            traceback.print_exc()
    
    def load_frame(self, frame_idx):
        """Load specific frame and display with nests."""
        if self.video_cap is None:
            return
        
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.video_cap.read()
        
        if ret:
            self.current_frame = frame
            self.current_frame_idx = frame_idx
            
            self.video_panel.set_frame_info(frame_idx, self.total_frames - 1)
            self.video_panel.set_frame_slider_value(frame_idx)
            
            tracks_for_frame = self.get_tracks_for_frame(frame_idx)
            detections_for_frame = self.get_detections_for_frame(frame_idx)
            
            self._update_data_status(tracks_for_frame)
            
            # Draw nests on frame if detected
            display_frame = frame.copy()
            if hasattr(self.video_canvas, 'show_nests') and self.video_canvas.show_nests:
                display_frame = self._draw_nests_on_frame(display_frame)
            
            self.video_canvas.set_frame(
                display_frame, 
                detections=detections_for_frame,
                tracks=tracks_for_frame
            )
    
    def _draw_nests_on_frame(self, frame):
        """Draw detected nests on frame.
        
        Args:
            frame: BGR image
            
        Returns:
            Frame with nest boxes drawn
        """
        if not hasattr(self.video_canvas, 'detected_nests'):
            return frame
        
        annotated = frame.copy()
        
        # Draw hotel ROI
        if hasattr(self.video_canvas, 'hotel_roi') and self.video_canvas.hotel_roi:
            x1, y1, x2, y2 = self.video_canvas.hotel_roi
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, "Hotel", (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw nest boxes
        for idx, nest in enumerate(self.video_canvas.detected_nests):
            x1, y1, x2, y2 = nest.bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Blue boxes for nests
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 1)
            
            # Get nest ID
            nest_id = idx
            if hasattr(nest, 'nest_id'):
                nest_id = nest.nest_id
            elif hasattr(nest, 'metadata') and 'nest_id' in nest.metadata:
                nest_id = nest.metadata['nest_id']
            
            # Put nest ID ABOVE the tube for visibility
            cx = (x1 + x2) // 2
            label = str(nest_id)
            
            # Get text size for centering
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
            )
            
            # Center the text horizontally above the box
            text_x = cx - text_width // 2
            text_y = y1 - 5  # 5 pixels above the box
            
            cv2.putText(annotated, label, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        return annotated
    
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
            self.current_frame_idx = 0
            self.load_frame(0)
            self.video_panel.set_frame_slider_value(0)
    
    def on_speed_change(self, value):
        """Handle playback speed change."""
        if self.playing:
            interval = int(1000 / (self.fps * value / 5))
            self.playback_timer.setInterval(interval)
    
    def on_parameters_changed(self, params):
        """Handle parameter changes from control panel."""
        pass
    
    def _initialize_background(self):
        """Initialize background with researched optimal thresholds."""
        self.statusBar().showMessage("Initializing background...")
        QApplication.processEvents()
        
        try:
            RESEARCHED_MIN_AREA = 30.0
            RESEARCHED_MIN_SOLIDITY = 0.56
            
            print("\n" + "="*70)
            print("BACKGROUND INITIALIZATION (Researched Optimal Defaults)")
            print("="*70)
            print(f"\nUsing researched optimal thresholds:")
            print(f"  min_area: {RESEARCHED_MIN_AREA}")
            print(f"  min_solidity: {RESEARCHED_MIN_SOLIDITY}")
            
            self.blob_detector = BlobDetector(
                min_area=RESEARCHED_MIN_AREA,
                min_solidity=RESEARCHED_MIN_SOLIDITY
            )
            
            self.blob_detector.initialize_from_video(
                video_path=self.video_path,
                num_frames=100,
                start_frame=0
            )
            
            self.statusBar().showMessage("✓ Background initialized")
            print("✓ Background initialized (100 frames)")
            print("="*70 + "\n")
            
        except Exception as e:
            import traceback
            print(f"✗ Background initialization failed: {e}")
            print(traceback.format_exc())
            self.statusBar().showMessage("✗ Background initialization failed")
    
    def test_detection(self):
        """Test detection on current frame."""
        if self.current_frame is None:
            QMessageBox.warning(self, "Warning", "Load a video first")
            return
        
        try:
            params = self.control_panel.get_parameters()
            detection_mode = params.get("detection_mode", "fgbg_yolo")
            
            detections = []
            
            if detection_mode == 'fgbg_yolo':
                if self.blob_detector is None:
                    QMessageBox.warning(
                        self,
                        "Background Not Ready",
                        "Background model is not initialized."
                    )
                    return
                
                detections = self.blob_detector.detect(self.current_frame)
            
            elif detection_mode == 'yolo_only':
                if not hasattr(self, 'yolo_detector_test') or self.yolo_detector_test is None:
                    self.statusBar().showMessage("⏳ Loading YOLO model...")
                    QApplication.processEvents()
                    
                    try:
                        from beemonitor.detection import YOLODetector
                        from ultralytics import YOLO
                        
                        model = YOLO('yolo11n.pt')
                        self.yolo_detector_test = YOLODetector(
                            model, 
                            tracking_classes=['bee'], 
                            conf_threshold=0.25
                        )
                        self.statusBar().showMessage("✓ YOLO model loaded")
                    except Exception as e:
                        QMessageBox.critical(
                            self, 
                            "YOLO Error",
                            f"Failed to load YOLO model:\n{e}"
                        )
                        return
                
                QApplication.processEvents()
                detections = self.yolo_detector_test.detect(self.current_frame)
            
            tracks_for_frame = self.get_tracks_for_frame(self.current_frame_idx)
            self._update_data_status(tracks_for_frame)
            
            self.video_canvas.set_frame(
                self.current_frame,
                detections=detections,
                tracks=tracks_for_frame
            )
            
            mode_names = {
                'fgbg_yolo': 'Motion+YOLO', 
                'yolo_only': 'YOLO Only'
            }
            mode_name = mode_names.get(detection_mode, detection_mode)
            
            self.statusBar().showMessage(f"✓ {mode_name}: {len(detections)} detections")
            
        except Exception as e:
            import traceback
            QMessageBox.critical(self, "Error", 
                f"Detection failed:\n{e}\n\n{traceback.format_exc()}")
    
    def run_analysis(self):
        """Run full video analysis."""
        if self.video_path is None:
            QMessageBox.warning(self, "Warning", "Load a video first")
            return
        
        params = self.control_panel.get_parameters()
        detection_mode = params.get("detection_mode", "fgbg_yolo")
        
        if not self.output_folder:
            video_dir = Path(self.video_path).parent
            video_name = Path(self.video_path).stem
            self.output_folder = str(video_dir / f"{video_name}_output")
            os.makedirs(self.output_folder, exist_ok=True)
        
        mode_names = {
            'fgbg_yolo': 'Motion + YOLO',
            'yolo_only': 'YOLO Only'
        }
        mode_name = mode_names.get(detection_mode, detection_mode)
        
        self.statusBar().showMessage(f"Starting analysis ({mode_name})...")
        
        self.control_panel.set_analysis_running(True)
        self.control_panel.append_log(f"Starting analysis ({mode_name})...")
        
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
            if os.path.getsize(filepath) == 0:
                QMessageBox.critical(
                    self,
                    "Empty File",
                    "The tracking results file is empty!"
                )
                return
            
            df = pd.read_csv(filepath)
            
            if 'action' in df.columns and 'nest' in df.columns:
                QMessageBox.warning(
                    self,
                    "Wrong File",
                    "This appears to be an events CSV file."
                )
                
                tracking_file = find_tracking_file(filepath)
                if tracking_file:
                    reply = QMessageBox.question(
                        self,
                        "Found Tracking File",
                        f"Load {Path(tracking_file).name} instead?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                    )
                    
                    if reply == QMessageBox.StandardButton.Yes:
                        filepath = tracking_file
                        df = pd.read_csv(filepath)
                    else:
                        return
                else:
                    return
            
            if 'track_id' not in df.columns:
                QMessageBox.warning(self, "Invalid File", "Missing 'track_id' column")
                return
            
            frame_col = None
            for possible in ['frame', 'frame_number', 'frame_num', 'frame_id', 'frame_idx']:
                if possible in df.columns:
                    frame_col = possible
                    break
            
            if frame_col is None:
                QMessageBox.critical(
                    self,
                    "Invalid CSV Format",
                    f"Missing frame column in CSV!"
                )
                return
            
            if frame_col != 'frame':
                df = df.rename(columns={frame_col: 'frame'})
            
            has_bbox = all(col in df.columns for col in ['x1', 'y1', 'x2', 'y2'])
            has_xy = all(col in df.columns for col in ['x', 'y'])
            has_centroid = all(col in df.columns for col in ['centroid_x', 'centroid_y'])
            
            if not (has_bbox or has_xy or has_centroid):
                QMessageBox.warning(self, "Invalid File", "Missing position data")
                return
            
            self.tracking_results = df
            self.results_loaded = True
            
            total_tracks = df['track_id'].nunique()
            total_frames = df['frame'].nunique()
            
            msg = (
                f"✓ Tracking results loaded!\n\n"
                f"File: {Path(filepath).name}\n\n"
                f"Total tracks: {total_tracks}\n"
                f"Total frames: {total_frames}\n"
                f"Total detections: {len(df)}"
            )
            
            QMessageBox.information(self, "Results Loaded", msg)
            
            if self.current_frame is not None:
                self.load_frame(self.current_frame_idx)
            
            self.statusBar().showMessage(
                f"✓ Results loaded: {total_tracks} tracks across {total_frames} frames"
            )
            
        except Exception as e:
            import traceback
            error_msg = f"Failed to load results:\n{e}\n\n{traceback.format_exc()}"
            QMessageBox.critical(self, "Error", error_msg)
    
    def get_tracks_for_frame(self, frame_idx):
        """Get all tracks visible in this frame."""
        if not self.results_loaded or self.tracking_results is None:
            return {}
        
        if 'frame' not in self.tracking_results.columns:
            return {}
        
        frame_data = self.tracking_results[
            self.tracking_results['frame'] == frame_idx
        ]
        
        if len(frame_data) == 0:
            return {}
        
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
        """Get all detections for this frame."""
        if not self.results_loaded or self.tracking_results is None:
            return []
        
        if 'frame' not in self.tracking_results.columns:
            return []
        
        frame_data = self.tracking_results[
            self.tracking_results['frame'] == frame_idx
        ]
        
        if len(frame_data) == 0:
            return []
        
        detections = []
        
        for _, row in frame_data.iterrows():
            if all(col in row.index for col in ['x1', 'y1', 'x2', 'y2']):
                bbox = (row['x1'], row['y1'], row['x2'], row['y2'])
                centroid = ((row['x1'] + row['x2']) / 2, (row['y1'] + row['y2']) / 2)
                
                det = Detection(
                    bbox=bbox,
                    centroid=centroid,
                    confidence=row.get('confidence', 1.0),
                    label=row.get('species', 'bee'),
                    source=row.get('source', 'blob')
                )
                
                detections.append(det)
        
        return detections
    
    def on_analysis_finished(self, result, csv_path):
        """Handle analysis completion and auto-load results."""
        self.control_panel.set_analysis_running(False)
        self.control_panel.append_log("✓ Analysis complete")
        self.statusBar().showMessage("✓ Analysis complete - Loading results...")
        
        # Auto-load and display results
        success = self._auto_load_and_display_results(csv_path)
        
        if success:
            msg = (
                f"✓ Analysis complete!\n\n"
                f"Output folder: {self.output_folder}\n\n"
                f"Results automatically loaded and displayed on video!\n\n"
                f"Files saved:\n"
                f"  • tracking_results.csv\n"
                f"  • events.csv (if applicable)\n\n"
                f"Tracks are now shown as blue boxes on the video.\n"
                f"Use video controls to step through frames."
            )
        else:
            msg = (
                f"✓ Analysis complete!\n\n"
                f"Output folder: {self.output_folder}\n\n"
                f"Files saved:\n"
                f"  • tracking_results.csv\n\n"
                f"Could not auto-load results for display.\n"
                f"You can try: File → Load Results"
            )
        
        QMessageBox.information(self, "Analysis Complete", msg)
    
    def _auto_load_and_display_results(self, csv_path):
        """Auto-load analysis results and display on video.
        
        Args:
            csv_path: Path to tracking_results.csv
            
        Returns:
            bool: True if successfully loaded, False otherwise
        """
        try:
            import pandas as pd
            
            if not os.path.exists(csv_path):
                print(f"⚠️  Results file not found: {csv_path}")
                self.control_panel.append_log("⚠️  Results file not found")
                return False
            
            print(f"📊 Auto-loading tracking results from {csv_path}")
            self.control_panel.append_log("📊 Loading tracking results for display...")
            
            # Load CSV
            df = pd.read_csv(csv_path)
            
            if df.empty:
                print("⚠️  Results file is empty")
                self.control_panel.append_log("⚠️  No tracking data in results")
                return False
            
            # Store for general use
            self.tracking_results = df
            self.results_loaded = True
            
            # Also create frame-indexed lookup for video display
            results_by_frame = {}
            for frame_num in df['frame'].unique():
                frame_data = df[df['frame'] == frame_num]
                
                tracks = []
                for _, row in frame_data.iterrows():
                    track = {
                        'track_id': int(row['track_id']),
                        'bbox': (float(row['x1']), float(row['y1']), 
                                float(row['x2']), float(row['y2'])),
                        'species': row.get('species', 'bee'),
                        'confidence': float(row.get('confidence', 1.0))
                    }
                    tracks.append(track)
                
                results_by_frame[int(frame_num)] = tracks
            
            # Store frame-indexed results for video display
            self.video_canvas.analysis_results = results_by_frame
            self.video_canvas.show_analysis_results = True
            
            # Refresh current frame to show results
            if self.current_frame_idx is not None:
                self.load_frame(self.current_frame_idx)
            
            num_frames = len(results_by_frame)
            total_tracks = df['track_id'].nunique()
            total_detections = len(df)
            
            print(f"✓ Auto-loaded tracking results:")
            print(f"  Frames with data: {num_frames}")
            print(f"  Total unique tracks: {total_tracks}")
            print(f"  Total detections: {total_detections}")
            
            self.control_panel.append_log(
                f"✓ Results loaded: {total_tracks} tracks, {total_detections} detections"
            )
            self.control_panel.append_log("  → Blue boxes on video show tracked bees")
            self.statusBar().showMessage(
                f"✓ Results displayed ({total_tracks} tracks across {num_frames} frames)"
            )
            
            return True
            
        except Exception as e:
            print(f"⚠️  Error auto-loading results: {e}")
            self.control_panel.append_log(f"⚠️  Could not auto-load results: {e}")
            import traceback
            traceback.print_exc()
            return False
    
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
            self.load_video()
            
            QMessageBox.information(
                self,
                "Output Video Loaded",
                "This is a pre-rendered visualization video."
            )
    
    def save_visualization_video(self):
        """Save current video with tracks/detections."""
        if self.video_path is None:
            QMessageBox.warning(self, "Warning", "Load a video first")
            return
        
        if not self.results_loaded and not self.blob_detector:
            QMessageBox.warning(
                self,
                "Warning",
                "No data to visualize!"
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
        
        reply = QMessageBox.question(
            self,
            "Save Visualization Video",
            f"This will save the video with overlays.\n"
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
                
                if self.blob_detector:
                    try:
                        detections = self.blob_detector.detect(frame)
                        for det in detections:
                            x1, y1, x2, y2 = [int(c) for c in det.bbox]
                            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    except:
                        pass
                
                if self.results_loaded:
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
            
            self.statusBar().showMessage(f"✓ Visualization saved")
            
        except Exception as e:
            import traceback
            progress.close()
            QMessageBox.critical(self, "Error", 
                f"Failed to save video:\n{e}\n\n{traceback.format_exc()}")
    
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
                detection_mode = config_data["detection"].get("detection_mode", "fgbg_yolo")
                self.control_panel.set_detection_mode(detection_mode)
            
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