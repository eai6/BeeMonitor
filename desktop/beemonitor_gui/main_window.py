"""Main Window - BeeMonitor Desktop App"""

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
        self.yolo_detector_test = None  # For test detection (reusable)
        self.output_folder = None
        
        self.analysis_thread = None
        
        self.control_panel = ControlPanel()
        self.video_panel = VideoPanel()
        self.video_canvas = self.video_panel.get_canvas()
        
        self._connect_signals()
        self._create_menu_bar()
        self._create_main_widget()
        
        # Always show detections, tracks, and sources
        self.video_canvas.show_detections = True
        self.video_canvas.show_tracks = True
        self.video_canvas.show_detection_sources = True
        
        self.statusBar().showMessage("Ready - Load a video to begin")
        
        print(f"✓ BeeMonitor GUI v{VERSION} initialized")
    
    def _connect_signals(self):
        """Connect all signals from panels to methods."""
        self.control_panel.load_video_requested.connect(self.load_video)
        self.control_panel.test_detection_requested.connect(self.test_detection)
        self.control_panel.run_analysis_requested.connect(self.run_analysis)
        self.control_panel.parameters_changed.connect(self.on_parameters_changed)
        
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
        main_layout.setContentsMargins(0, 0, 0, 0)  # No margins
        main_layout.setSpacing(0)  # No spacing between panels
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
        
        self.load_frame(0)
        self.video_panel.enable_play_button(True)
        
        self.statusBar().showMessage(
            f"Loaded: {Path(filepath).name} | Output: {self.output_folder}"
        )
        
        print(f"✓ Video loaded: {filepath}")
        
        # Auto-initialize background for blob detection
        self._initialize_background()
    
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
            
            tracks_for_frame = self.get_tracks_for_frame(frame_idx)
            detections_for_frame = self.get_detections_for_frame(frame_idx)
            
            self._update_data_status(tracks_for_frame)
            
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
        """Play next frame during playback. Auto-loops when reaching end."""
        if self.current_frame_idx < self.total_frames - 1:
            self.jump_frame(1)
        else:
            # Loop back to start (auto-replay)
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
        # No auto-testing - let user manually test when ready
        pass
    
    def _initialize_background(self):
        """Initialize background with researched optimal thresholds.
        
        Uses proven values from ablation study:
        - min_area = 30.0 (conservative, catches most bees)
        - min_solidity = 0.56 (80% of typical 0.7, proven F1=53.0%)
        
        These adapt during full analysis if YOLO confirms different characteristics.
        """
        self.statusBar().showMessage("Initializing background...")
        QApplication.processEvents()
        
        try:
            # RESEARCHED OPTIMAL values from ablation study
            RESEARCHED_MIN_AREA = 30.0
            RESEARCHED_MIN_SOLIDITY = 0.56
            
            print("\n" + "="*70)
            print("BACKGROUND INITIALIZATION (Researched Optimal Defaults)")
            print("="*70)
            print(f"\nUsing researched optimal thresholds:")
            print(f"  min_area: {RESEARCHED_MIN_AREA} (from ablation study)")
            print(f"  min_solidity: {RESEARCHED_MIN_SOLIDITY} (80% scaling, proven F1=53.0%)")
            
            self.blob_detector = BlobDetector(
                min_area=RESEARCHED_MIN_AREA,
                min_solidity=RESEARCHED_MIN_SOLIDITY
            )
            
            self.blob_detector.initialize_from_video(
                video_path=self.video_path,
                num_frames=100,
                start_frame=0
            )
            
            self.statusBar().showMessage("✓ Background initialized (researched optimal thresholds)")
            print("✓ Background initialized (100 frames)")
            print("✓ Ready for detection with optimal thresholds")
            print("="*70 + "\n")
            
        except Exception as e:
            import traceback
            print(f"✗ Background initialization failed: {e}")
            print(traceback.format_exc())
            self.statusBar().showMessage("✗ Background initialization failed")
    
    def test_detection(self):
        """Test detection on current frame using selected mode."""
        if self.current_frame is None:
            QMessageBox.warning(self, "Warning", "Load a video first")
            return
        
        try:
            params = self.control_panel.get_parameters()
            detection_mode = params.get("detection_mode", "fgbg")
            
            detections = []
            
            if detection_mode in ['fgbg', 'fgbg_yolo']:
                if self.blob_detector is None:
                    QMessageBox.warning(
                        self,
                        "Background Not Ready",
                        "Background model is not initialized.\n\n"
                        "This shouldn't happen - background is initialized automatically when you load a video.\n\n"
                        "Try reloading the video or check the terminal for errors."
                    )
                    return
                
                detections = self.blob_detector.detect(self.current_frame)
            
            elif detection_mode == 'yolo_only':
                # Create YOLO detector if needed (reuse if exists)
                if not hasattr(self, 'yolo_detector_test') or self.yolo_detector_test is None:
                    self.statusBar().showMessage("⏳ Loading YOLO model (first time, takes ~5-10 seconds)...")
                    QApplication.processEvents()
                    
                    try:
                        from beemonitor.detection import YOLODetector
                        from ultralytics import YOLO
                        
                        print("Loading YOLO model for test detection...")
                        model = YOLO('yolo11n.pt')
                        self.yolo_detector_test = YOLODetector(
                            model, 
                            tracking_classes=['bee'], 
                            conf_threshold=0.25
                        )
                        print("✓ YOLO model loaded")
                        self.statusBar().showMessage("✓ YOLO model loaded, running detection...")
                    except Exception as e:
                        QMessageBox.critical(
                            self, 
                            "YOLO Error",
                            f"Failed to load YOLO model:\n{e}\n\n"
                            f"Make sure yolo11n.pt is available."
                        )
                        return
                
                QApplication.processEvents()
                detections = self.yolo_detector_test.detect(self.current_frame)
            
            self.control_panel.set_detection_count(len(detections))
            tracks_for_frame = self.get_tracks_for_frame(self.current_frame_idx)
            self._update_data_status(tracks_for_frame)
            
            self.video_canvas.set_frame(
                self.current_frame,
                detections=detections,
                tracks=tracks_for_frame
            )
            
            mode_names = {
                'fgbg': 'Motion',
                'fgbg_yolo': 'Motion+YOLO', 
                'yolo_only': 'YOLO Only'
            }
            mode_name = mode_names.get(detection_mode, detection_mode)
            
            if detection_mode in ['fgbg', 'fgbg_yolo']:
                note = " (raw blobs - CNN+solidity filters in full analysis)"
                print(f"\n⚠ Test Detection: {len(detections)} raw blobs (no CNN filter)")
                print("   Full analysis will apply CNN + learned solidity filters")
                print("   Expected: ~66% reduction in final results\n")
            else:
                note = ""
            
            self.statusBar().showMessage(f"✓ {mode_name}: {len(detections)} detections{note}")
            
        except Exception as e:
            import traceback
            QMessageBox.critical(self, "Error", 
                f"Detection failed:\n{e}\n\n{traceback.format_exc()}")
    
    def run_analysis(self):
        """Run full video analysis - auto-initializes everything in background."""
        if self.video_path is None:
            QMessageBox.warning(self, "Warning", "Load a video first")
            return
        
        params = self.control_panel.get_parameters()
        detection_mode = params.get("detection_mode", "fgbg")
        
        if not self.output_folder:
            video_dir = Path(self.video_path).parent
            video_name = Path(self.video_path).stem
            self.output_folder = str(video_dir / f"{video_name}_output")
            os.makedirs(self.output_folder, exist_ok=True)
        
        mode_names = {
            'fgbg': 'Motion Detection',
            'fgbg_yolo': 'Motion + YOLO',
            'yolo_only': 'YOLO Only'
        }
        mode_name = mode_names.get(detection_mode, detection_mode)
        
        self.statusBar().showMessage(
            f"Starting analysis ({mode_name})... "
            f"Background initialization, CNN filtering, and tracking will run automatically."
        )
        
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
        
        print(f"\n{'='*70}")
        print(f"STARTING ANALYSIS - Watch for CNN filter logs below")
        print(f"{'='*70}\n")
    
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
                    "The tracking results file is empty!\n\n"
                    "This means the analysis didn't produce any results."
                )
                return
            
            df = pd.read_csv(filepath)
            
            if 'action' in df.columns and 'nest' in df.columns:
                QMessageBox.warning(
                    self,
                    "Wrong File",
                    "This appears to be an events CSV file.\n\n"
                    "For visualization, you need the tracking results CSV."
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
                    f"Missing frame column in CSV!\n\n"
                    f"Available columns:\n{', '.join(df.columns)}"
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
            
            # Detections and tracks are always shown (no toggles)
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
        """Get all detections for this frame as Detection objects."""
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
            self.load_video()
            
            QMessageBox.information(
                self,
                "Output Video Loaded",
                "This is a pre-rendered visualization video.\n\n"
                "Tracks/detections are already drawn on the video."
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
        
        # Always include detections and tracks
        include_detections = True
        include_tracks = True
        
        reply = QMessageBox.question(
            self,
            "Save Visualization Video",
            f"This will save the video with:\n"
            f"  • Detections overlay\n"
            f"  • Track trajectories\n\n"
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
                
                if include_detections and self.blob_detector:
                    try:
                        detections = self.blob_detector.detect(frame)
                        for det in detections:
                            x1, y1, x2, y2 = [int(c) for c in det.bbox]
                            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    except:
                        pass
                
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