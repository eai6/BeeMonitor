"""
Video Panel
===========

Right panel with video display and playback controls.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QSlider, QCheckBox
)
from PyQt6.QtCore import Qt, pyqtSignal

from .video_canvas import VideoCanvas


class VideoPanel(QWidget):
    """Video panel with canvas and playback controls."""
    
    # Signals
    play_pause_toggled = pyqtSignal()
    frame_step_requested = pyqtSignal(int)  # delta
    frame_changed = pyqtSignal(int)  # frame index
    speed_changed = pyqtSignal(int)  # speed value
    show_detections_changed = pyqtSignal(bool)
    show_tracks_changed = pyqtSignal(bool)
    show_sources_changed = pyqtSignal(bool)  # NEW!
    
    def __init__(self):
        """Initialize video panel."""
        super().__init__()
        
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Video canvas
        self.video_canvas = VideoCanvas()
        layout.addWidget(self.video_canvas)
        
        # Video controls
        layout.addLayout(self._create_controls())
        
        # Frame slider
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(100)
        self.frame_slider.valueChanged.connect(self.frame_changed.emit)
        layout.addWidget(self.frame_slider)
        
        # Info bar
        layout.addLayout(self._create_info_bar())
    
    def _create_controls(self):
        """Create playback controls."""
        controls_layout = QHBoxLayout()
        
        self.play_pause_btn = QPushButton("▶ Play")
        self.play_pause_btn.clicked.connect(self.play_pause_toggled.emit)
        self.play_pause_btn.setEnabled(False)
        controls_layout.addWidget(self.play_pause_btn)
        
        prev_btn = QPushButton("◀")
        prev_btn.clicked.connect(lambda: self.frame_step_requested.emit(-1))
        prev_btn.setMaximumWidth(40)
        controls_layout.addWidget(prev_btn)
        
        next_btn = QPushButton("▶")
        next_btn.clicked.connect(lambda: self.frame_step_requested.emit(1))
        next_btn.setMaximumWidth(40)
        controls_layout.addWidget(next_btn)
        
        controls_layout.addWidget(QLabel("Speed:"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(10)
        self.speed_slider.setValue(5)
        self.speed_slider.setMaximumWidth(100)
        self.speed_slider.valueChanged.connect(self.speed_changed.emit)
        controls_layout.addWidget(self.speed_slider)
        
        return controls_layout
    
    def _create_info_bar(self):
        """Create info bar with frame info and checkboxes."""
        info_layout = QHBoxLayout()
        
        self.frame_label = QLabel("Frame: 0 / 0")
        info_layout.addWidget(self.frame_label)
        
        self.data_status_label = QLabel("No data")
        self.data_status_label.setStyleSheet("color: #999;")
        info_layout.addWidget(self.data_status_label)
        
        info_layout.addStretch()
        
        # Show Detections checkbox
        self.show_detections_cb = QCheckBox("Show Detections")
        self.show_detections_cb.stateChanged.connect(
            lambda state: self.show_detections_changed.emit(
                state == Qt.CheckState.Checked
            )
        )
        info_layout.addWidget(self.show_detections_cb)
        
        # Show Tracks checkbox
        self.show_tracks_cb = QCheckBox("Show Tracks")
        self.show_tracks_cb.stateChanged.connect(
            lambda state: self.show_tracks_changed.emit(
                state == Qt.CheckState.Checked
            )
        )
        info_layout.addWidget(self.show_tracks_cb)
        
        # NEW: Show Detection Sources checkbox
        self.show_sources_cb = QCheckBox("Show Sources")
        self.show_sources_cb.setToolTip(
            "Color-code detections by source:\n"
            "🔴 RED = Blob/FG-BG (motion)\n"
            "🟢 GREEN = SIFT (stationary)\n"
            "🔵 BLUE = YOLO (deep learning)"
        )
        self.show_sources_cb.stateChanged.connect(
            lambda state: self.show_sources_changed.emit(
                state == Qt.CheckState.Checked
            )
        )
        info_layout.addWidget(self.show_sources_cb)
        
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
