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
        layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
        layout.setSpacing(5)  # Tight spacing
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
        
        # ===================================================================
        # AUTO NEST DETECTION
        # ===================================================================
        
        # Initialize nest detector (lazy initialization - will be created when needed)
        self.nest_detector = None
        self.nest_detector_available = NEST_DETECTOR_AVAILABLE
        self.detected_nests = []  # List[Detection]
        self.hotel_roi = None  # (x1, y1, x2, y2)
        self.auto_detect_on_load = True  # Enable by default
        self.show_nests = True  # Show nest boxes by default
        
        # Video properties
        self.video_path = None
        self.cap = None
        self.width = 0
        self.height = 0
        self.fps = 0
        self.total_frames = 0
    
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
        """Create info bar with frame info and data status."""
        info_layout = QHBoxLayout()
        
        self.frame_label = QLabel("Frame: 0 / 0")
        info_layout.addWidget(self.frame_label)
        
        self.data_status_label = QLabel("No data")
        self.data_status_label.setStyleSheet("color: #999;")
        info_layout.addWidget(self.data_status_label)
        
        # Nest count label (NEW)
        self.nest_count_label = QLabel("Nests: -")
        self.nest_count_label.setStyleSheet("color: #666;")
        info_layout.addWidget(self.nest_count_label)
        
        info_layout.addStretch()
        
        # Note: Detections and tracks always show (no toggles)
        note_label = QLabel("💡 Detections & tracks always visible")
        note_label.setStyleSheet("color: #666; font-size: 9pt; font-style: italic;")
        info_layout.addWidget(note_label)
        
        return info_layout
    
    # =========================================================================
    # VIDEO LOADING WITH AUTO NEST DETECTION
    # =========================================================================
    
    def load_video(self, video_path: str):
        """Load video and automatically detect hotel/nests.
        
        Args:
            video_path: Path to video file
        """
        print(f"Loading video: {video_path}")
        
        # Store path
        self.video_path = video_path
        
        # Open video
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", f"Cannot open video: {video_path}")
            return False
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video properties: {self.width}×{self.height}, {self.fps:.1f} fps, {self.total_frames} frames")
        
        # Read first frame
        ret, first_frame = self.cap.read()
        if not ret:
            QMessageBox.critical(self, "Error", "Cannot read first frame")
            return False
        
        # Reset to start
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Auto-detect hotel and nests
        if self.auto_detect_on_load and self.nest_detector_available:
            self._auto_detect_hotel_and_nests(first_frame)
        
        # Update UI
        self.set_frame_range(self.total_frames - 1)
        self.set_frame_info(0, self.total_frames)
        self.enable_play_button(True)
        
        print(f"✓ Video loaded successfully")
        return True
    
    def _auto_detect_hotel_and_nests(self, frame):
        """Automatically detect hotel ROI and nest boxes.
        
        Args:
            frame: First frame of video (BGR image)
        """
        try:
            # Lazy initialization of nest detector
            if self.nest_detector is None and self.nest_detector_available:
                print("🔧 Initializing NestDetector...")
                try:
                    from ultralytics import YOLO
                    # Use lightweight YOLO model for nest detection
                    yolo_model = YOLO('yolo11n.pt')
                    self.nest_detector = NestDetector(model=yolo_model)
                    print("✓ NestDetector initialized")
                except Exception as e:
                    print(f"⚠️  Failed to initialize NestDetector: {e}")
                    self.nest_detector_available = False
                    return
            
            if self.nest_detector is None:
                print("⚠️  NestDetector not available")
                return
            
            print("🔍 Auto-detecting hotel and nests...")
            
            # Detect nests using NestDetector
            detections = self.nest_detector.detect_nests(frame)
            
            if not detections or len(detections) == 0:
                print("⚠️  No nests detected automatically")
                self.nest_count_label.setText("Nests: 0")
                return
            
            # Store nest detections
            self.detected_nests = detections
            
            # Compute hotel ROI as bounding box of all nests
            self.hotel_roi = self._compute_hotel_roi(detections)
            
            # Log results
            nest_count = len(detections)
            print(f"✓ Detected {nest_count} nests")
            
            if self.hotel_roi:
                x1, y1, x2, y2 = self.hotel_roi
                w, h = x2 - x1, y2 - y1
                print(f"✓ Hotel ROI: ({x1}, {y1}) - ({x2}, {y2}) [{w}×{h} px]")
            
            # Update UI
            self.nest_count_label.setText(f"Nests: {nest_count}")
            self.nest_count_label.setStyleSheet("color: #0a0; font-weight: bold;")
            
            # Emit signal with detected data
            self.hotel_nests_detected.emit({
                'hotel': self.hotel_roi,
                'nests': self.detected_nests,
                'nest_count': nest_count
            })
            
            print(f"✓ Hotel and nest detection complete")
            
        except Exception as e:
            print(f"⚠️  Auto-detection failed: {e}")
            import traceback
            traceback.print_exc()
            self.nest_count_label.setText("Nests: Error")
            self.nest_count_label.setStyleSheet("color: #f00;")
    
    def _compute_hotel_roi(self, nest_detections):
        """Compute hotel ROI from nest detections.
        
        Args:
            nest_detections: List[Detection] with nest bboxes
            
        Returns:
            (x1, y1, x2, y2) hotel bounding box with padding
        """
        if not nest_detections or len(nest_detections) == 0:
            return None
        
        # Get all nest bboxes
        all_x1 = []
        all_y1 = []
        all_x2 = []
        all_y2 = []
        
        for detection in nest_detections:
            x1, y1, x2, y2 = detection.bbox
            all_x1.append(x1)
            all_y1.append(y1)
            all_x2.append(x2)
            all_y2.append(y2)
        
        # Compute bounding box of all nests
        min_x = min(all_x1)
        min_y = min(all_y1)
        max_x = max(all_x2)
        max_y = max(all_y2)
        
        # Add padding (20 pixels on each side)
        padding = 20
        hotel_x1 = max(0, min_x - padding)
        hotel_y1 = max(0, min_y - padding)
        hotel_x2 = min(self.width, max_x + padding)
        hotel_y2 = min(self.height, max_y + padding)
        
        return (int(hotel_x1), int(hotel_y1), int(hotel_x2), int(hotel_y2))
    
    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    
    def draw_hotel_and_nests(self, frame):
        """Draw hotel ROI and nest boxes on frame.
        
        Args:
            frame: BGR image to draw on (modified in-place)
            
        Returns:
            Modified frame with annotations
        """
        if not self.show_nests:
            return frame
        
        # Make a copy to avoid modifying original
        annotated = frame.copy()
        
        # Draw hotel ROI (green box)
        if self.hotel_roi:
            x1, y1, x2, y2 = self.hotel_roi
            cv2.rectangle(annotated, (x1, y1), (x2, y2), 
                         color=(0, 255, 0), thickness=2)
            cv2.putText(annotated, "Hotel", (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw nest boxes (blue boxes with IDs)
        for idx, nest in enumerate(self.detected_nests):
            x1, y1, x2, y2 = nest.bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Draw nest box
            cv2.rectangle(annotated, (x1, y1), (x2, y2),
                         color=(255, 0, 0), thickness=1)
            
            # Draw nest ID at center
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            
            # Get nest ID (from detection or use index)
            if hasattr(nest, 'nest_id'):
                nest_id = nest.nest_id
            elif hasattr(nest, 'metadata') and 'nest_id' in nest.metadata:
                nest_id = nest.metadata['nest_id']
            else:
                nest_id = idx
            
            cv2.putText(annotated, str(nest_id), (cx - 8, cy + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        return annotated
    
    def toggle_nest_display(self, show: bool):
        """Toggle nest box display on/off.
        
        Args:
            show: True to show nests, False to hide
        """
        self.show_nests = show
        print(f"Nest display: {'ON' if show else 'OFF'}")
    
    def display_frame_with_nests(self, frame):
        """Display frame with hotel and nest annotations.
        
        This is a convenience method that draws nests and displays on canvas.
        Use this instead of video_canvas.display_frame() directly.
        
        Args:
            frame: BGR image to display
        """
        # Draw annotations
        annotated = self.draw_hotel_and_nests(frame)
        
        # Display on canvas
        self.video_canvas.display_frame(annotated)
    
    def set_manual_hotel_roi(self, roi, nests=None):
        """Set hotel ROI manually (bypass auto-detection).
        
        Args:
            roi: (x1, y1, x2, y2) hotel bounding box
            nests: List[Detection] or None to auto-generate grid
        """
        self.hotel_roi = roi
        
        if nests is None:
            # Generate simple 6x10 grid
            print("Generating 6×10 nest grid...")
            nests = self._generate_nest_grid(roi, rows=6, cols=10)
        
        self.detected_nests = nests
        count = len(nests)
        self.nest_count_label.setText(f"Nests: {count}")
        self.nest_count_label.setStyleSheet("color: #0a0; font-weight: bold;")
        
        print(f"✓ Manual hotel ROI set: {roi}")
        print(f"✓ {count} nests configured")
        
        # Emit signal
        self.hotel_nests_detected.emit({
            'hotel': self.hotel_roi,
            'nests': self.detected_nests,
            'nest_count': count
        })
    
    def _generate_nest_grid(self, roi, rows, cols):
        """Generate regular grid of nest positions.
        
        Args:
            roi: (x1, y1, x2, y2) hotel bounding box
            rows: Number of rows
            cols: Number of columns
            
        Returns:
            List[Detection] of generated nests
        """
        from beemonitor.detection.base_detector import Detection
        
        x1, y1, x2, y2 = roi
        w = (x2 - x1) / cols
        h = (y2 - y1) / rows
        
        nests = []
        nest_id = 0
        
        for row in range(rows):
            for col in range(cols):
                nx1 = x1 + col * w
                ny1 = y1 + row * h
                nx2 = nx1 + w
                ny2 = ny1 + h
                
                # Create Detection for nest
                nest = Detection(
                    bbox=(nx1, ny1, nx2, ny2),
                    centroid=((nx1 + nx2) / 2, (ny1 + ny2) / 2),
                    confidence=1.0,
                    label='nest',
                    source='manual_grid'
                )
                nest.nest_id = nest_id
                nests.append(nest)
                nest_id += 1
        
        return nests
    
    # =========================================================================
    # DATA ACCESS
    # =========================================================================
    
    def get_hotel_and_nests(self):
        """Get detected hotel ROI and nests for analysis.
        
        Returns:
            Dictionary with:
                'hotel': (x1, y1, x2, y2) or None
                'nests': List[Detection] or []
                'nest_count': int
        """
        return {
            'hotel': self.hotel_roi,
            'nests': self.detected_nests,
            'nest_count': len(self.detected_nests)
        }
    
    def get_video_info(self):
        """Get video properties.
        
        Returns:
            Dictionary with video properties
        """
        return {
            'path': self.video_path,
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'total_frames': self.total_frames
        }
    
    def has_hotel_roi(self):
        """Check if hotel ROI was detected.
        
        Returns:
            True if hotel ROI exists, False otherwise
        """
        return self.hotel_roi is not None
    
    def has_nests(self):
        """Check if nests were detected.
        
        Returns:
            True if nests exist, False otherwise
        """
        return len(self.detected_nests) > 0
    
    # =========================================================================
    # MANUAL DETECTION CONTROLS
    # =========================================================================
    
    def manual_detect_nests(self):
        """Manually trigger nest detection on current frame."""
        if self.cap is None or not self.cap.isOpened():
            QMessageBox.warning(self, "No Video", "Please load a video first")
            return
        
        if not self.nest_detector_available:
            QMessageBox.warning(self, "Detector Unavailable", 
                              "NestDetector is not available")
            return
        
        # Get current frame
        current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
        ret, frame = self.cap.read()
        
        if ret:
            self._auto_detect_hotel_and_nests(frame)
        else:
            QMessageBox.warning(self, "Error", "Cannot read current frame")
    
    def clear_detections(self):
        """Clear detected hotel and nests."""
        self.detected_nests = []
        self.hotel_roi = None
        self.nest_count_label.setText("Nests: -")
        self.nest_count_label.setStyleSheet("color: #666;")
        print("Cleared hotel and nest detections")
    
    def set_auto_detect(self, enabled: bool):
        """Enable/disable auto-detection on video load.
        
        Args:
            enabled: True to enable, False to disable
        """
        self.auto_detect_on_load = enabled
        print(f"Auto-detect on load: {'ENABLED' if enabled else 'DISABLED'}")
    
    # =========================================================================
    # EXISTING VIDEO PANEL METHODS
    # =========================================================================
    
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