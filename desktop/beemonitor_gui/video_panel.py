"""
Video Panel
===========

Right panel with video display and playback controls.
Simplified - detections and tracks always show (no toggle controls).
Auto-detects hotel ROI and nest boxes on video load.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QSlider, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal
import cv2
import numpy as np

from .video_canvas import VideoCanvas

# Import nest detector for auto-detection
try:
    from beemonitor.detection import NestDetector
    from beemonitor.detection.base_detector import Detection
    NEST_DETECTOR_AVAILABLE = True
except ImportError:
    NEST_DETECTOR_AVAILABLE = False
    print("⚠️  NestDetector not available - auto-detection disabled")


class VideoPanel(QWidget):
    """Video panel with canvas, playback controls, and auto nest detection."""
    
    # Signals
    play_pause_toggled = pyqtSignal()
    frame_step_requested = pyqtSignal(int)  # delta
    frame_changed = pyqtSignal(int)  # frame index
    speed_changed = pyqtSignal(int)  # speed value
    hotel_nests_detected = pyqtSignal(dict)  # {'hotel': ROI, 'nests': List[Detection]}
    
    def __init__(self):
        """Initialize video panel."""
        super().__init__()
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.setLayout(layout)
        
        # Video canvas - takes all available space
        self.video_canvas = VideoCanvas()
        layout.addWidget(self.video_canvas, stretch=1)
        
        # Compact controls section at bottom
        controls_container = QWidget()
        controls_container.setMaximumHeight(80)
        controls_layout = QVBoxLayout()
        controls_layout.setContentsMargins(5, 2, 5, 2)
        controls_layout.setSpacing(2)
        controls_container.setLayout(controls_layout)
        
        # Video controls
        controls_layout.addLayout(self._create_controls())
        
        # Frame slider
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(100)
        self.frame_slider.setMaximumHeight(20)
        self.frame_slider.valueChanged.connect(self.frame_changed.emit)
        controls_layout.addWidget(self.frame_slider)
        
        # Info bar
        controls_layout.addLayout(self._create_info_bar())
        
        layout.addWidget(controls_container, stretch=0)
        
        # Initialize nest detector
        self.nest_detector = None
        self.nest_detector_available = NEST_DETECTOR_AVAILABLE
        self.detected_nests = []
        self.hotel_roi = None
        self.auto_detect_on_load = True
        self.show_nests = True
        
        # Tracking results
        self.tracking_df = None
        self.tracks_by_frame = {}
        self.trajectories = {}
        
        # Display options
        self.show_tracks = True
        self.show_trajectories = True
        self.show_track_ids = True
        self.trajectory_length = 30
        
        # Track colors
        self.track_colors = {}
        self.color_palette = [
            (255, 0, 0),    # Blue
            (0, 255, 0),    # Green
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 255),  # Purple
            (255, 128, 0),  # Orange
            (0, 128, 255),  # Light Blue
            (128, 255, 0),  # Lime
        ]
        
        # Video properties
        self.video_path = None
        self.cap = None
        self.width = 0
        self.height = 0
        self.fps = 0
        self.total_frames = 0
    
    def _create_controls(self):
        """Create compact playback controls."""
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(5)
        
        self.play_pause_btn = QPushButton("▶ Play")
        self.play_pause_btn.clicked.connect(self.play_pause_toggled.emit)
        self.play_pause_btn.setEnabled(False)
        self.play_pause_btn.setMaximumHeight(28)
        controls_layout.addWidget(self.play_pause_btn)
        
        prev_btn = QPushButton("◀")
        prev_btn.clicked.connect(lambda: self.frame_step_requested.emit(-1))
        prev_btn.setMaximumWidth(35)
        prev_btn.setMaximumHeight(28)
        controls_layout.addWidget(prev_btn)
        
        next_btn = QPushButton("▶")
        next_btn.clicked.connect(lambda: self.frame_step_requested.emit(1))
        next_btn.setMaximumWidth(35)
        next_btn.setMaximumHeight(28)
        controls_layout.addWidget(next_btn)
        
        speed_label = QLabel("Speed:")
        speed_label.setMaximumWidth(45)
        controls_layout.addWidget(speed_label)
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(10)
        self.speed_slider.setValue(5)
        self.speed_slider.setMaximumWidth(100)
        self.speed_slider.setMaximumHeight(20)
        self.speed_slider.valueChanged.connect(self.speed_changed.emit)
        controls_layout.addWidget(self.speed_slider)
        
        controls_layout.addStretch()
        
        return controls_layout
    
    def _create_info_bar(self):
        """Create compact info bar with frame info and data status."""
        info_layout = QHBoxLayout()
        info_layout.setSpacing(10)
        
        self.frame_label = QLabel("Frame: 0 / 0")
        self.frame_label.setStyleSheet("font-size: 10pt;")
        info_layout.addWidget(self.frame_label)
        
        self.data_status_label = QLabel("No data")
        self.data_status_label.setStyleSheet("color: #999; font-size: 10pt;")
        info_layout.addWidget(self.data_status_label)
        
        self.nest_count_label = QLabel("Nests: -")
        self.nest_count_label.setStyleSheet("color: #4CAF50; font-size: 10pt; font-weight: bold;")
        info_layout.addWidget(self.nest_count_label)
        
        info_layout.addStretch()
        
        return info_layout
    
    def get_canvas(self):
        """Get VideoCanvas widget."""
        return self.video_canvas
    
    def set_playing(self, playing):
        """Update play/pause button state."""
        if playing:
            self.play_pause_btn.setText("⏸ Pause")
        else:
            self.play_pause_btn.setText("▶ Play")
    
    def enable_play_button(self, enabled):
        """Enable/disable play button."""
        self.play_pause_btn.setEnabled(enabled)
    
    def set_frame_range(self, max_frame):
        """Set frame slider range."""
        self.frame_slider.setMaximum(max_frame)
    
    def set_frame_info(self, current, total):
        """Update frame label."""
        self.frame_label.setText(f"Frame: {current} / {total}")
    
    def set_frame_slider_value(self, value):
        """Set frame slider value without triggering signal."""
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(value)
        self.frame_slider.blockSignals(False)
    
    def set_data_status(self, text, is_active=False):
        """Update data status label."""
        self.data_status_label.setText(text)
        if is_active:
            self.data_status_label.setStyleSheet("color: #0a0; font-weight: bold;")
        else:
            self.data_status_label.setStyleSheet("color: #999;")
    
    def set_nest_count(self, count):
        """Update nest count label."""
        if count > 0:
            self.nest_count_label.setText(f"Nests: {count}")
            self.nest_count_label.setStyleSheet("color: #4CAF50; font-size: 10pt; font-weight: bold;")
        else:
            self.nest_count_label.setText("Nests: -")
            self.nest_count_label.setStyleSheet("color: #666; font-size: 10pt;")
    
    def get_hotel_and_nests(self):
        """Get detected hotel ROI and nests for analysis."""
        return {
            'hotel': self.hotel_roi,
            'nests': self.detected_nests,
            'nest_count': len(self.detected_nests)
        }
    
    def get_video_info(self):
        """Get video properties."""
        return {
            'path': self.video_path,
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'total_frames': self.total_frames
        }