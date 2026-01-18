"""Main Window - BeeMonitor Desktop App v2.3

v2.3 Features:
- Reference Configuration (nest rows/cols)
- Interaction Metrics Analysis
- Manual Nest Editing
- Crop Saving for ID Training
"""

import os
import json
import threading
import time
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
    """Thread for analyzing multiple videos in parallel with progress tracking."""
    
    progress = pyqtSignal(str)
    progress_update = pyqtSignal(int, int)
    video_completed = pyqtSignal(str, object)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, monitor, folder_path, output_folder, params):
        super().__init__()
        self.monitor = monitor
        self.folder_path = folder_path
        self.output_folder = output_folder
        self.params = params
        self._stop_flag = False
        self._lock = threading.Lock()
        self._completed_count = 0
        self._active_videos = []
        self._video_times = {}
    
    def run(self):
        """Run folder analysis with parallel processing."""
        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            start_time = time.time()
            self.progress.emit(f"Starting batch analysis of folder: {Path(self.folder_path).name}")
            
            video_files = [
                f for f in os.listdir(self.folder_path)
                if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))
            ]
            
            total_videos = len(video_files)
            
            if total_videos == 0:
                self.error.emit("No video files found in folder")
                return
            
            self.progress_update.emit(0, total_videos)
            
            max_workers = self.params.get('max_workers', 4)
            self.progress.emit(f"Processing {total_videos} videos with {max_workers} parallel workers")
            self.progress.emit(f"Started at: {time.strftime('%H:%M:%S')}")
            
            # Log advanced options if enabled
            if self.params.get('enable_interaction_metrics'):
                self.progress.emit(f"✓ Interaction metrics enabled (proximity={self.params.get('proximity_threshold', 50)}px)")
            if self.params.get('save_crops'):
                self.progress.emit(f"✓ Crop saving enabled ({self.params.get('crops_per_track', 5)} per track)")
            
            results = {}
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_video = {}
                for video_file in video_files:
                    if self._stop_flag:
                        break
                    
                    video_path = str(Path(self.folder_path) / video_file)
                    future = executor.submit(self._process_single_video, video_path, video_file)
                    future_to_video[future] = video_file
                
                for future in as_completed(future_to_video):
                    if self._stop_flag:
                        for f in future_to_video:
                            f.cancel()
                        break
                    
                    video_file = future_to_video[future]
                    
                    try:
                        result = future.result()
                        results[video_file] = result
                        
                        with self._lock:
                            self._completed_count += 1
                            completed = self._completed_count
                        
                        self.progress_update.emit(completed, total_videos)
                        
                        video_time = self._video_times.get(video_file, 0)
                        time_str = self._format_time(video_time)
                        
                        if result is not None:
                            self.progress.emit(f"✓ Completed {video_file} in {time_str} ({completed}/{total_videos})")
                        else:
                            self.progress.emit(f"✗ Failed {video_file} after {time_str} ({completed}/{total_videos})")
                            
                    except Exception as e:
                        with self._lock:
                            self._completed_count += 1
                            completed = self._completed_count
                        
                        self.progress.emit(f"✗ Error {video_file}: {e} ({completed}/{total_videos})")
                        results[video_file] = None
                        self.progress_update.emit(completed, total_videos)
            
            total_elapsed = time.time() - start_time
            
            if self._stop_flag:
                self.progress.emit(f"")
                self.progress.emit(f"Analysis stopped by user. Completed {self._completed_count}/{total_videos} videos.")
                self.progress.emit(f"Total time: {self._format_time(total_elapsed)}")
            else:
                avg_time = total_elapsed / len(results) if results else 0
                
                self.progress.emit(f"")
                self.progress.emit(f"{'='*50}")
                self.progress.emit(f"✓ BATCH ANALYSIS COMPLETE!")
                self.progress.emit(f"{'='*50}")
                self.progress.emit(f"Videos processed: {len(results)}/{total_videos}")
                self.progress.emit(f"Total time: {self._format_time(total_elapsed)}")
                self.progress.emit(f"Average per video: {self._format_time(avg_time)}")
                self.progress.emit(f"Finished at: {time.strftime('%H:%M:%S')}")
            
            self.finished.emit(results)
            
        except Exception as e:
            import traceback
            error_msg = f"Folder analysis failed: {e}\n\n{traceback.format_exc()}"
            self.error.emit(error_msg)
    
    def _process_single_video(self, video_path: str, video_file: str):
        """Process a single video (runs in worker thread)."""
        video_start_time = time.time()
        
        try:
            with self._lock:
                self._active_videos.append(video_file)
            
            active_list = ", ".join(self._active_videos[:3])
            if len(self._active_videos) > 3:
                active_list += f" (+{len(self._active_videos)-3} more)"
            self.progress.emit(f"⚙️  Processing: {active_list}")
            
            # Pass output_folder so crops save to correct location
            result = self.monitor.analyze_video(
                video_path=video_path,
                output_folder=self.output_folder,
                visualize=self.params.get('visualize', False),
                detection_mode=self.params.get('detection_mode', 'yolo')
            )
            
            # Run interaction analysis if enabled
            if result and self.params.get('enable_interaction_metrics'):
                self._run_interaction_analysis(result, video_path)
            
            video_elapsed = time.time() - video_start_time
            
            with self._lock:
                self._video_times[video_file] = video_elapsed
                if video_file in self._active_videos:
                    self._active_videos.remove(video_file)
            
            return result
            
        except Exception as e:
            video_elapsed = time.time() - video_start_time
            with self._lock:
                self._video_times[video_file] = video_elapsed
                if video_file in self._active_videos:
                    self._active_videos.remove(video_file)
            raise e
    
    def _run_interaction_analysis(self, result, video_path):
        """Run interaction analysis on completed result."""
        try:
            from beemonitor.processing.interaction_analyzer import (
                InteractionAnalyzer, nests_to_reference_objects
            )
            
            tracking_df = result.tracks
            if tracking_df is None or tracking_df.empty:
                return
            
            analyzer = InteractionAnalyzer(
                proximity_threshold=self.params.get('proximity_threshold', 50),
                min_interaction_frames=3,
                fps=30.0
            )
            
            # Track-to-track interactions
            track_interactions, track_summary = analyzer.analyze_track_interactions(tracking_df)
            
            # Track-to-nest interactions
            if result.nests and 'nests' in result.nests:
                ref_objects = nests_to_reference_objects(
                    [{'id': k, 'bbox': v} for k, v in result.nests['nests'].items()]
                )
                nest_interactions, nest_summary = analyzer.analyze_reference_interactions(
                    tracking_df, ref_objects
                )
            else:
                nest_interactions, nest_summary = [], pd.DataFrame()
            
            # Save interaction CSVs
            video_name = Path(video_path).stem
            
            if track_interactions:
                track_csv = os.path.join(self.output_folder, f"{video_name}_track_interactions.csv")
                analyzer.to_csv(track_interactions, track_csv, 'track')
            
            if nest_interactions:
                nest_csv = os.path.join(self.output_folder, f"{video_name}_nest_interactions.csv")
                analyzer.to_csv(nest_interactions, nest_csv, 'reference')
            
        except Exception as e:
            self.progress.emit(f"  ⚠️ Interaction analysis failed: {e}")
    
    def _format_time(self, seconds):
        """Format seconds into human-readable time string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours}h {mins}m {secs}s"
    
    def stop(self):
        """Request thread to stop processing."""
        self._stop_flag = True
        self.progress.emit("Stopping analysis... (waiting for active videos to finish)")


class BeeMonitorGUI(QMainWindow):
    """Main GUI application v2.3 with reference configuration and interaction metrics."""
    
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
        
        # Pre-initialize nest detection models
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
        """Initialize models for nest detection once at startup."""
        try:
            from ultralytics import YOLO
            from beemonitor.core.config import Config
            
            print("Initializing nest detection models...")
            
            self.nest_config = Config.default()
            nest_model_path = self.nest_config.models.nest_detection
            print(f"  Loading nest model: {nest_model_path}")
            
            if not os.path.exists(nest_model_path):
                raise FileNotFoundError(f"Nest detection model not found at: {nest_model_path}")
            
            self.nest_yolo_model = YOLO(nest_model_path)
            self.nest_config.nest.max_detection_attempts = 15
            
            print("✓ Nest detection models ready")
            
        except FileNotFoundError as e:
            print(f"⚠️  Nest detection model not found: {e}")
            self.nest_yolo_model = None
            self.nest_config = None
        except Exception as e:
            print(f"⚠️  Could not initialize nest detection models: {e}")
            self.nest_yolo_model = None
            self.nest_config = None
    
    def _connect_signals(self):
        """Connect all signals from panels to methods."""
        self.control_panel.load_video_requested.connect(self.load_video)
        self.control_panel.run_analysis_requested.connect(self.run_analysis)
        self.control_panel.stop_analysis_requested.connect(self.stop_analysis)
        self.control_panel.parameters_changed.connect(self.on_parameters_changed)
        
        # Folder analysis signals
        self.control_panel.folder_selected.connect(self.on_folder_selected)
        self.control_panel.analyze_folder_requested.connect(self.run_folder_analysis)
        
        # NEW v2.3: Reference configuration signals
        self.control_panel.reference_config_changed.connect(self.on_reference_config_changed)
        self.control_panel.edit_nests_requested.connect(self.show_nest_editor)
        
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
        
        # NEW v2.3: Edit menu
        edit_menu = menubar.addMenu("&Edit")
        
        edit_nests_action = QAction("Edit &Nests...", self)
        edit_nests_action.triggered.connect(self.show_nest_editor)
        edit_menu.addAction(edit_nests_action)
        
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
    
    # =========================================================================
    # NEW v2.3: Reference Configuration Methods
    # =========================================================================
    
    def on_reference_config_changed(self, config):
        """Handle reference configuration change.
        
        Args:
            config: Dict with 'rows', 'cols', 'total'
        """
        # Update config
        self.config.nest.expected_rows = config['rows']
        self.config.nest.expected_nests_per_row = config['cols']
        self.config.nest.expected_total_nests = config['total']
        
        # Also update nest_config if initialized
        if self.nest_config:
            self.nest_config.nest.expected_rows = config['rows']
            self.nest_config.nest.expected_nests_per_row = config['cols']
            self.nest_config.nest.expected_total_nests = config['total']
        
        self.statusBar().showMessage(
            f"Reference config updated: {config['rows']}×{config['cols']} = {config['total']} nests"
        )
        self.control_panel.append_log(
            f"✓ Reference config: {config['rows']} rows × {config['cols']} cols = {config['total']} nests"
        )
    
    def show_nest_editor(self):
        """Open visual nest editor dialog for manual nest editing."""
        try:
            from .nest_editor_dialog import show_visual_nest_editor
        except ImportError:
            QMessageBox.warning(
                self,
                "Module Not Found",
                "Nest editor dialog module not found.\n"
                "Please ensure nest_editor_dialog.py is in the GUI package."
            )
            return
        
        # Need a video frame to edit on
        if self.current_frame is None:
            QMessageBox.warning(
                self,
                "No Video",
                "Please load a video first to edit nests."
            )
            return
        
        # Get current nests - convert Detection namedtuples to dicts
        nests = []
        if hasattr(self.video_canvas, 'detected_nests') and self.video_canvas.detected_nests:
            for i, nest in enumerate(self.video_canvas.detected_nests):
                bbox = nest.bbox
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                
                nest_dict = {
                    'id': getattr(nest, 'nest_id', i + 1),
                    'x': cx,
                    'y': cy,
                    'w': int(w),
                    'h': int(h)
                }
                nests.append(nest_dict)
        
        # Get current hotel ROI
        hotel_roi = getattr(self.video_canvas, 'hotel_roi', None)
        
        # Get reference config
        ref_config = self.control_panel.get_reference_config()
        
        # Show visual editor with current frame
        result = show_visual_nest_editor(
            self,
            frame=self.current_frame,
            nests=nests,
            hotel_roi=hotel_roi,
            grid_rows=ref_config['rows'],
            grid_cols=ref_config['cols']
        )
        
        if result:
            updated_nests, updated_hotel = result
            
            # Convert back to Detection format
            from collections import namedtuple
            Detection = namedtuple('Detection', ['bbox', 'confidence', 'nest_id'])
            
            new_nests = []
            for nest in updated_nests:
                # Convert center + size to bbox
                x, y = nest['x'], nest['y']
                w = nest.get('w', 24)
                h = nest.get('h', 14)
                
                bbox = (x - w/2, y - h/2, x + w/2, y + h/2)
                
                det = Detection(
                    bbox=bbox,
                    confidence=1.0,
                    nest_id=nest['id']
                )
                new_nests.append(det)
            
            # Update video canvas
            self.video_canvas.detected_nests = new_nests
            self.video_canvas.show_nests = True
            
            # Update hotel ROI
            if updated_hotel:
                self.video_canvas.hotel_roi = tuple(int(v) for v in updated_hotel)
            else:
                self.video_canvas.hotel_roi = None
            
            # Refresh display
            if self.current_frame_idx is not None:
                self.load_frame(self.current_frame_idx)
            
            # Update control panel
            self.control_panel.set_detected_nests_count(len(new_nests))
            
            self.statusBar().showMessage(f"✓ Updated {len(new_nests)} nests")
            log_msg = f"✓ Manually edited {len(new_nests)} nests"
            if updated_hotel:
                log_msg += " + hotel ROI"
            self.control_panel.append_log(log_msg)
    
    # =========================================================================
    # Video Loading
    # =========================================================================
    
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
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._auto_detect_nests(first_frame)
        
        self.load_frame(0)
        self.video_panel.enable_play_button(True)
        
        self.statusBar().showMessage(
            f"Loaded: {Path(filepath).name} | Output: {self.output_folder}"
        )
        
        print(f"✓ Video loaded: {filepath}")
        
        self.control_panel.set_video_loaded(True)
        self.control_panel.append_log(f"✓ Loaded: {Path(filepath).name}")
    
    def _auto_detect_nests(self, first_frame):
        """Auto-detect nest tubes using comprehensive multi-frame detection."""
        try:
            from beemonitor.detection import NestDetector
            
            if self.nest_yolo_model is None or self.nest_config is None:
                print("⚠️  Nest detection unavailable (models not loaded at startup)")
                self.control_panel.append_log("⚠️  Nest detection unavailable")
                self._offer_grid_generation(first_frame, "Nest detection models not loaded")
                return
            
            print("🔍 Auto-detecting nest tubes (comprehensive method)...")
            self.statusBar().showMessage("Detecting nest tubes (trying multiple frames)...")
            QApplication.processEvents()
            
            nest_detector = NestDetector(
                model=self.nest_yolo_model,
                config=self.nest_config
            )
            
            nests_dict = nest_detector.get_nests_and_hotel_detections(
                video_path=self.video_path
            )
            
            if nests_dict and 'nests' in nests_dict and len(nests_dict['nests']) > 0:
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
                
                self.video_canvas.detected_nests = nests
                self.video_canvas.show_nests = True
                self.video_canvas.hotel_roi = nests_dict.get('hotel')
                
                if self.current_frame_idx is not None:
                    self.load_frame(self.current_frame_idx)
                
                print(f"✓ Detected {len(nests)} nest tubes (quality verified)")
                self.control_panel.append_log(f"✓ Detected {len(nests)} nest tubes (quality verified)")
                self.control_panel.set_detected_nests_count(len(nests))
                self.statusBar().showMessage(f"✓ Detected {len(nests)} nest tubes")
                
                # Update video info
                self.control_panel.set_video_info(
                    f"<b>{Path(self.video_path).name}</b><br>"
                    f"{int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                    f"{int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
                    f"@ {self.fps:.1f} FPS<br>"
                    f"{self.total_frames} frames ({self.total_frames/self.fps:.1f}s)<br>"
                    f"<span style='color: #4CAF50;'><b>🎯 {len(nests)} nests detected</b></span>"
                )
            else:
                print("⚠️  No nest tubes detected after multiple attempts")
                self.control_panel.append_log("⚠️  No nest tubes detected")
                self.control_panel.set_detected_nests_count(0)
                self.statusBar().showMessage("⚠️  Nest detection failed")
                
                # Offer to generate grid from reference config
                self._offer_grid_generation(first_frame, "Auto-detection found no nests")
        
        except Exception as e:
            print(f"⚠️  Nest detection failed: {e}")
            self.control_panel.append_log(f"⚠️  Nest detection error: {e}")
            self.statusBar().showMessage("Video loaded (nest detection error)")
            
            # Offer to generate grid from reference config
            self._offer_grid_generation(first_frame, f"Detection error: {e}")
    
    def _offer_grid_generation(self, frame, reason: str):
        """Offer to generate nest grid from reference config when auto-detection fails.
        
        Args:
            frame: Video frame for dimensions
            reason: Why auto-detection failed
        """
        ref_config = self.control_panel.get_reference_config()
        rows = ref_config['rows']
        cols = ref_config['cols']
        total = ref_config['total']
        
        reply = QMessageBox.question(
            self,
            "Generate Nest Grid?",
            f"{reason}.\n\n"
            f"Would you like to generate a {rows}×{cols} grid ({total} nests) "
            f"based on your reference configuration?\n\n"
            f"You can adjust positions in the Visual Nest Editor afterwards.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self._generate_grid_from_config(frame, rows, cols)
    
    def _generate_grid_from_config(self, frame, rows: int, cols: int):
        """Generate evenly spaced nest grid from reference config.
        
        Args:
            frame: Video frame for dimensions
            rows: Number of rows
            cols: Number of columns
        """
        from collections import namedtuple
        Detection = namedtuple('Detection', ['bbox', 'confidence', 'nest_id'])
        
        h, w = frame.shape[:2]
        
        # Default nest size and padding
        nest_w, nest_h = 24, 14
        padding = 50
        
        # Calculate spacing
        avail_w = w - 2 * padding
        avail_h = h - 2 * padding
        spacing_x = avail_w / cols
        spacing_y = avail_h / rows
        
        nests = []
        nest_id = 1
        
        for row in range(rows):
            for col in range(cols):
                cx = padding + (col + 0.5) * spacing_x
                cy = padding + (row + 0.5) * spacing_y
                
                bbox = (cx - nest_w/2, cy - nest_h/2, cx + nest_w/2, cy + nest_h/2)
                
                det = Detection(
                    bbox=bbox,
                    confidence=1.0,
                    nest_id=nest_id
                )
                nests.append(det)
                nest_id += 1
        
        # Update video canvas
        self.video_canvas.detected_nests = nests
        self.video_canvas.show_nests = True
        
        # Generate hotel ROI around all nests
        self.video_canvas.hotel_roi = (
            int(padding - 10),
            int(padding - 10),
            int(w - padding + 10),
            int(h - padding + 10)
        )
        
        # Refresh display
        if self.current_frame_idx is not None:
            self.load_frame(self.current_frame_idx)
        
        # Update UI
        self.control_panel.set_detected_nests_count(len(nests))
        
        self.control_panel.set_video_info(
            f"<b>{Path(self.video_path).name}</b><br>"
            f"{w}x{h} @ {self.fps:.1f} FPS<br>"
            f"{self.total_frames} frames ({self.total_frames/self.fps:.1f}s)<br>"
            f"<span style='color: #FF9800;'><b>🔲 {len(nests)} nests (generated grid)</b></span>"
        )
        
        self.statusBar().showMessage(f"✓ Generated {rows}×{cols} nest grid - use Edit Nests to adjust")
        self.control_panel.append_log(f"✓ Generated {rows}×{cols} = {len(nests)} nest grid")
        self.control_panel.append_log(f"  → Click 'Edit Nests' to adjust positions")
    
    # =========================================================================
    # Analysis Methods
    # =========================================================================
    
    def run_analysis(self):
        """Run full video analysis with v2.3 options."""
        if self.video_path is None:
            QMessageBox.warning(self, "Warning", "Load a video first")
            return
        
        params = self.control_panel.get_parameters()
        advanced = self.control_panel.get_advanced_options()
        
        if not self.output_folder:
            video_dir = Path(self.video_path).parent
            video_name = Path(self.video_path).stem
            self.output_folder = str(video_dir / f"{video_name}_output")
            os.makedirs(self.output_folder, exist_ok=True)
        
        # Update config with advanced options
        if advanced.get('save_crops'):
            self.config.tracking.save_crops = True
            self.config.tracking.crops_per_track = advanced.get('crops_per_track', 5)
            # Set crop output folder to match CSV output folder
            self.config.tracking.crop_output_folder = self.output_folder
        
        # Get manually edited nests to pass to analysis
        edited_nests = None
        if hasattr(self.video_canvas, 'detected_nests') and self.video_canvas.detected_nests:
            # Convert Detection namedtuples to dict format for EventProcessor
            edited_nests = {
                'nests': {},
                'hotel': getattr(self.video_canvas, 'hotel_roi', None)
            }
            for nest in self.video_canvas.detected_nests:
                nest_id = getattr(nest, 'nest_id', None)
                if nest_id is not None:
                    edited_nests['nests'][nest_id] = nest.bbox
        
        self.analysis_start_time = time.time()
        
        self.statusBar().showMessage("Starting analysis...")
        
        self.control_panel.set_analysis_running(True)
        self.control_panel.append_log(f"Starting analysis...")
        self.control_panel.append_log(f"Started at: {time.strftime('%H:%M:%S')}")
        
        if edited_nests and edited_nests['nests']:
            self.control_panel.append_log(f"✓ Using {len(edited_nests['nests'])} manually edited nests")
        
        if advanced.get('enable_interaction_metrics'):
            self.control_panel.append_log(f"✓ Interaction metrics enabled")
        if advanced.get('save_crops'):
            self.control_panel.append_log(f"✓ Crop saving enabled ({advanced.get('crops_per_track', 5)} per track)")
            self.control_panel.append_log(f"  → Crops will be saved to: {self.output_folder}")
        
        monitor = BeeMonitor(config=self.config)
        
        self.analysis_thread = AnalysisThread(
            monitor,
            self.video_path,
            self.output_folder,
            detection_mode='yolo',
            nests=edited_nests  # Pass edited nests to analysis
        )
        
        # Store advanced options for post-processing
        self._analysis_advanced_options = advanced
        
        self.analysis_thread.progress.connect(
            lambda msg: self.statusBar().showMessage(msg))
        self.analysis_thread.finished.connect(self.on_analysis_finished)
        self.analysis_thread.error.connect(
            lambda err: QMessageBox.critical(self, "Analysis Error", err))
        
        self.analysis_thread.start()
    
    def on_analysis_finished(self, result, csv_path):
        """Handle analysis completion with v2.3 post-processing."""
        if hasattr(self, 'analysis_start_time'):
            elapsed_time = time.time() - self.analysis_start_time
            elapsed_str = self._format_time(elapsed_time)
        else:
            elapsed_str = "Unknown"
        
        self.control_panel.set_analysis_running(False)
        self.control_panel.append_log(f"✓ Analysis complete in {elapsed_str}")
        self.control_panel.append_log(f"Finished at: {time.strftime('%H:%M:%S')}")
        
        # Run interaction analysis if enabled
        if hasattr(self, '_analysis_advanced_options'):
            advanced = self._analysis_advanced_options
            if advanced.get('enable_interaction_metrics') and result:
                self._run_interaction_analysis(result, advanced)
        
        self.statusBar().showMessage(f"✓ Analysis complete ({elapsed_str}) - Loading results...")
        
        success = self._auto_load_and_display_results(csv_path)
        
        msg = (
            f"✓ Analysis complete!\n\n"
            f"Execution time: {elapsed_str}\n"
            f"Output folder: {self.output_folder}\n\n"
        )
        
        if success:
            msg += "Results automatically loaded and displayed on video!"
        
        QMessageBox.information(self, "Analysis Complete", msg)
    
    def _run_interaction_analysis(self, result, advanced):
        """Run interaction analysis post-processing."""
        try:
            from beemonitor.processing.interaction_analyzer import (
                InteractionAnalyzer, nests_to_reference_objects
            )
            
            self.control_panel.append_log("Running interaction analysis...")
            
            tracking_df = result.tracks
            if tracking_df is None or tracking_df.empty:
                self.control_panel.append_log("  ⚠️ No tracking data for interactions")
                return
            
            analyzer = InteractionAnalyzer(
                proximity_threshold=advanced.get('proximity_threshold', 50),
                min_interaction_frames=3,
                fps=self.fps
            )
            
            # Track-to-track interactions
            track_interactions, _ = analyzer.analyze_track_interactions(tracking_df)
            
            # Track-to-nest interactions - ALWAYS use GUI edited nests first
            nest_interactions = []
            nests_used = 0
            nests_source = "none"
            
            # Priority 1: GUI edited nests (from visual editor)
            if hasattr(self.video_canvas, 'detected_nests') and self.video_canvas.detected_nests:
                gui_nests = []
                for nest in self.video_canvas.detected_nests:
                    nest_id = getattr(nest, 'nest_id', None)
                    if nest_id is not None:
                        gui_nests.append({'id': nest_id, 'bbox': nest.bbox})
                
                if gui_nests:
                    ref_objects = nests_to_reference_objects(gui_nests)
                    nest_interactions, _ = analyzer.analyze_reference_interactions(
                        tracking_df, ref_objects
                    )
                    nests_used = len(gui_nests)
                    nests_source = "GUI-edited"
                    self.control_panel.append_log(f"  ✓ Using {nests_used} {nests_source} nests for interactions")
            
            # Priority 2: Fallback to result.nests from auto-detection (only if no GUI nests)
            if nests_used == 0 and hasattr(result, 'nests') and result.nests and 'nests' in result.nests:
                ref_objects = nests_to_reference_objects(
                    [{'id': k, 'bbox': v} for k, v in result.nests['nests'].items()]
                )
                nest_interactions, _ = analyzer.analyze_reference_interactions(
                    tracking_df, ref_objects
                )
                nests_used = len(result.nests['nests'])
                nests_source = "auto-detected"
                self.control_panel.append_log(f"  ⚠️ Using {nests_used} {nests_source} nests (no GUI edits found)")
            
            if nests_used == 0:
                self.control_panel.append_log("  ⚠️ No nests available - skipping nest interactions")
            
            # Save CSVs
            video_name = Path(self.video_path).stem
            
            if track_interactions:
                track_csv = os.path.join(self.output_folder, f"{video_name}_track_interactions.csv")
                analyzer.to_csv(track_interactions, track_csv, 'track')
                self.control_panel.append_log(f"  ✓ Saved {len(track_interactions)} track interactions")
            
            if nest_interactions:
                nest_csv = os.path.join(self.output_folder, f"{video_name}_nest_interactions.csv")
                analyzer.to_csv(nest_interactions, nest_csv, 'reference')
                self.control_panel.append_log(f"  ✓ Saved {len(nest_interactions)} nest interactions ({nests_source} nests)")
            else:
                self.control_panel.append_log(f"  No nest interactions detected")
            
        except ImportError:
            self.control_panel.append_log("  ⚠️ InteractionAnalyzer module not found")
        except Exception as e:
            import traceback
            self.control_panel.append_log(f"  ⚠️ Interaction analysis failed: {e}")
            print(f"Interaction analysis error: {traceback.format_exc()}")
    
    def _auto_load_and_display_results(self, csv_path):
        """Auto-load and display results after analysis."""
        try:
            if not os.path.exists(csv_path):
                return False
            
            df = pd.read_csv(csv_path)
            
            if 'track_id' not in df.columns:
                return False
            
            # Find frame column
            frame_col = None
            for possible in ['frame', 'frame_number', 'frame_num']:
                if possible in df.columns:
                    frame_col = possible
                    break
            
            if frame_col is None:
                return False
            
            if frame_col != 'frame':
                df = df.rename(columns={frame_col: 'frame'})
            
            self.tracking_results = df
            self.results_loaded = True
            
            total_tracks = df['track_id'].nunique()
            total_detections = len(df)
            
            self.control_panel.append_log(f"✓ Loading tracking results for display...")
            self.control_panel.append_log(f"✓ Results loaded: {total_tracks} tracks, {total_detections} detections")
            self.control_panel.append_log(f"  → Blue boxes on video show tracked bees")
            
            if self.current_frame is not None:
                self.load_frame(self.current_frame_idx)
            
            return True
            
        except Exception as e:
            print(f"Auto-load failed: {e}")
            return False
    
    # =========================================================================
    # Folder Analysis
    # =========================================================================
    
    def stop_analysis(self):
        """Stop running analysis."""
        if self.analysis_thread and self.analysis_thread.isRunning():
            self.analysis_thread.terminate()
            self.control_panel.set_analysis_running(False)
            self.control_panel.append_log("Analysis stopped by user")
            self.statusBar().showMessage("Analysis stopped")
        
        if self.folder_analysis_thread and self.folder_analysis_thread.isRunning():
            self.folder_analysis_thread.stop()
            self.folder_analysis_thread.wait(3000)
            if self.folder_analysis_thread.isRunning():
                self.folder_analysis_thread.terminate()
            self.control_panel.set_folder_analyzing(False)
            self.control_panel.append_log("Batch analysis stopped by user")
            self.statusBar().showMessage("Batch analysis stopped")
    
    def on_folder_selected(self, folder_path):
        """Handle folder selection."""
        self.folder_path = folder_path
        
        video_files = [f for f in os.listdir(folder_path) 
                      if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        self.control_panel.append_log(f"✓ Selected folder: {Path(folder_path).name}")
        self.control_panel.append_log(f"  Found {len(video_files)} video files")
        
        self.statusBar().showMessage(f"Folder selected: {len(video_files)} videos found")
    
    def run_folder_analysis(self):
        """Run batch video analysis on folder."""
        if not self.folder_path:
            QMessageBox.warning(self, "Warning", "Select a video folder first")
            return
        
        params = self.control_panel.get_parameters()
        advanced = self.control_panel.get_advanced_options()
        
        # Merge advanced options into params for thread
        params.update(advanced)
        
        video_files = [f for f in os.listdir(self.folder_path) 
                      if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        if not video_files:
            QMessageBox.warning(self, "Warning", "No video files in selected folder")
            return
        
        folder_name = Path(self.folder_path).name
        output_folder = str(Path(self.folder_path).parent / f"{folder_name}_output")
        os.makedirs(output_folder, exist_ok=True)
        
        # Update config with crop settings
        if advanced.get('save_crops'):
            self.config.tracking.save_crops = True
            self.config.tracking.crops_per_track = advanced.get('crops_per_track', 5)
            self.config.tracking.crop_output_folder = output_folder
        
        # Build confirmation message
        confirm_msg = (
            f"Analyze {len(video_files)} videos?\n\n"
            f"Folder: {folder_name}\n"
            f"Output: {Path(output_folder).name}\n"
            f"Parallel workers: {params['max_workers']}\n"
        )
        
        if params.get('enable_interaction_metrics'):
            confirm_msg += f"Interaction metrics: Enabled (proximity={params.get('proximity_threshold', 50)}px)\n"
        if params.get('save_crops'):
            confirm_msg += f"Crop saving: Enabled ({params.get('crops_per_track', 5)} per track)\n"
        
        confirm_msg += "\nThis may take a while..."
        
        reply = QMessageBox.question(
            self,
            "Batch Video Analysis",
            confirm_msg,
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
        self.control_panel.append_log(f"Workers: {params['max_workers']}")
        if params.get('save_crops'):
            self.control_panel.append_log(f"Crops: Saving to {output_folder}")
        
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
        self.folder_analysis_thread.progress_update.connect(
            self.control_panel.set_folder_progress
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
        
        QMessageBox.information(
            self,
            "Batch Analysis Complete",
            f"Batch analysis complete!\n\n"
            f"Videos: {successful}/{total_videos} successful\n"
            f"Total events: {total_events}"
        )
        
        self.statusBar().showMessage(
            f"✓ Batch complete: {successful}/{total_videos} successful"
        )
    
    # =========================================================================
    # Video Playback and Display
    # =========================================================================
    
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
            
            display_frame = frame.copy()
            if hasattr(self.video_canvas, 'show_nests') and self.video_canvas.show_nests:
                display_frame = self._draw_nests_on_frame(display_frame)
            
            self.video_canvas.set_frame(
                display_frame, 
                detections=detections_for_frame,
                tracks=tracks_for_frame
            )
    
    def _draw_nests_on_frame(self, frame):
        """Draw detected nests on frame."""
        if not hasattr(self.video_canvas, 'detected_nests'):
            return frame
        
        annotated = frame.copy()
        
        if hasattr(self.video_canvas, 'hotel_roi') and self.video_canvas.hotel_roi:
            x1, y1, x2, y2 = self.video_canvas.hotel_roi
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, "Hotel", (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        for idx, nest in enumerate(self.video_canvas.detected_nests):
            x1, y1, x2, y2 = nest.bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 1)
            
            nest_id = getattr(nest, 'nest_id', idx + 1)
            
            cx = (x1 + x2) // 2
            label = str(nest_id)
            
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
            )
            
            text_x = cx - text_width // 2
            text_y = y1 - 5
            
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
    
    def on_speed_change(self, value):
        """Handle playback speed change."""
        if self.playing:
            interval = int(1000 / (self.fps * value / 5))
            self.playback_timer.setInterval(interval)
    
    def on_parameters_changed(self, params):
        """Handle parameter changes from control panel."""
        pass
    
    # =========================================================================
    # Track Data Methods
    # =========================================================================
    
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
                    source=row.get('source', 'yolo')
                )
                
                detections.append(det)
        
        return detections
    
    # =========================================================================
    # File Operations
    # =========================================================================
    
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
            df = pd.read_csv(filepath)
            
            if 'track_id' not in df.columns:
                QMessageBox.warning(self, "Invalid File", "Missing 'track_id' column")
                return
            
            frame_col = None
            for possible in ['frame', 'frame_number', 'frame_num']:
                if possible in df.columns:
                    frame_col = possible
                    break
            
            if frame_col is None:
                QMessageBox.critical(self, "Invalid CSV", "Missing frame column!")
                return
            
            if frame_col != 'frame':
                df = df.rename(columns={frame_col: 'frame'})
            
            self.tracking_results = df
            self.results_loaded = True
            
            total_tracks = df['track_id'].nunique()
            total_frames = df['frame'].nunique()
            
            QMessageBox.information(
                self,
                "Results Loaded",
                f"✓ Tracking results loaded!\n\n"
                f"Total tracks: {total_tracks}\n"
                f"Total frames: {total_frames}\n"
                f"Total detections: {len(df)}"
            )
            
            if self.current_frame is not None:
                self.load_frame(self.current_frame_idx)
            
            self.statusBar().showMessage(
                f"✓ Results loaded: {total_tracks} tracks"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load results:\n{e}")
    
    def load_output_video(self):
        """Load output video file."""
        default_dir = self.output_folder if self.output_folder else str(Path.home())
        
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Load Output Video",
            default_dir,
            "Video Files (*.mp4 *.avi);;All Files (*)"
        )
        
        if filepath:
            self.video_path = filepath
            self.video_cap = cv2.VideoCapture(filepath)
            
            if self.video_cap.isOpened():
                self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
                
                self.video_panel.set_frame_range(self.total_frames - 1)
                self.load_frame(0)
                self.video_panel.enable_play_button(True)
                
                self.statusBar().showMessage(f"Loaded output video: {Path(filepath).name}")
    
    def save_visualization_video(self):
        """Save video with visualization overlays."""
        if self.video_cap is None:
            QMessageBox.warning(self, "Warning", "No video loaded")
            return
        
        default_dir = self.output_folder if self.output_folder else str(Path.home())
        video_name = Path(self.video_path).stem
        default_path = os.path.join(default_dir, f"{video_name}_visualization.mp4")
        
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Visualization Video",
            default_path,
            "MP4 Video (*.mp4);;AVI Video (*.avi)"
        )
        
        if not output_path:
            return
        
        progress = QProgressDialog(
            "Saving visualization video...", "Cancel", 0, self.total_frames, self
        )
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            out = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
            
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            for frame_idx in range(self.total_frames):
                if progress.wasCanceled():
                    break
                
                ret, frame = self.video_cap.read()
                if not ret:
                    break
                
                # Draw nests
                if hasattr(self.video_canvas, 'detected_nests'):
                    frame = self._draw_nests_on_frame(frame)
                
                # Draw tracks
                if self.results_loaded:
                    tracks = self.get_tracks_for_frame(frame_idx)
                    colors = [(255, 0, 0), (0, 255, 255), (255, 0, 255)]
                    
                    for i, (track_id, trajectory) in enumerate(tracks.items()):
                        color = colors[i % len(colors)]
                        
                        if len(trajectory) > 1:
                            points = np.array(trajectory, dtype=np.int32)
                            cv2.polylines(frame, [points], False, color, 2)
                        
                        if trajectory:
                            x, y = trajectory[-1]
                            cv2.circle(frame, (int(x), int(y)), 5, color, -1)
                
                out.write(frame)
                progress.setValue(frame_idx)
                QApplication.processEvents()
            
            out.release()
            self.load_frame(self.current_frame_idx)
            progress.close()
            
            QMessageBox.information(self, "Success", f"Saved: {output_path}")
            
        except Exception as e:
            progress.close()
            QMessageBox.critical(self, "Error", f"Failed to save video:\n{e}")
    
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
            self, "Save Configuration", "", "JSON Files (*.json)"
        )
        
        if not filepath:
            return
        
        params = self.control_panel.get_parameters()
        ref_config = self.control_panel.get_reference_config()
        advanced = self.control_panel.get_advanced_options()
        
        config_data = {
            "detection": params,
            "reference": ref_config,
            "advanced": advanced,
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
            self, "Load Configuration", "", "JSON Files (*.json)"
        )
        
        if not filepath:
            return
        
        try:
            with open(filepath, 'r') as f:
                config_data = json.load(f)
            
            # Load reference config
            if "reference" in config_data:
                ref = config_data["reference"]
                self.control_panel.set_reference_config(
                    ref.get('rows', 6),
                    ref.get('cols', 10)
                )
            
            if "output_folder" in config_data:
                self.output_folder = config_data["output_folder"]
            
            QMessageBox.information(self, "Success", "Configuration loaded")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load configuration:\n{e}")
    
    def _format_time(self, seconds):
        """Format seconds into human-readable time string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours}h {mins}m {secs}s"
    
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts."""
        if event.key() == Qt.Key.Key_Space:
            self.toggle_play_pause()
        elif event.key() == Qt.Key.Key_Left:
            self.jump_frame(-1)
        elif event.key() == Qt.Key.Key_Right:
            self.jump_frame(1)