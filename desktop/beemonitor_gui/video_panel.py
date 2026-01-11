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
        layout.setSpacing(2)  # Minimal spacing
        self.setLayout(layout)
        
        # Video canvas - takes all available space
        self.video_canvas = VideoCanvas()
        layout.addWidget(self.video_canvas, stretch=1)  # Stretch factor 1 = takes all space
        
        # Compact controls section at bottom
        controls_container = QWidget()
        controls_container.setMaximumHeight(80)  # Limit controls to 80px
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
        self.frame_slider.setMaximumHeight(20)  # Compact slider
        self.frame_slider.valueChanged.connect(self.frame_changed.emit)
        controls_layout.addWidget(self.frame_slider)
        
        # Info bar
        controls_layout.addLayout(self._create_info_bar())
        
        layout.addWidget(controls_container, stretch=0)  # No stretch = fixed size
        
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
        
        # ===================================================================
        # TRACKING DATA OVERLAY
        # ===================================================================
        
        # Tracking results
        self.tracking_df = None  # DataFrame with tracking results
        self.tracks_by_frame = {}  # Dict[frame_num, List[track_data]]
        self.trajectories = {}  # Dict[track_id, List[(frame, (x, y))]]
        
        # Display options
        self.show_tracks = True  # Show track boxes
        self.show_trajectories = True  # Show trajectory lines
        self.show_track_ids = True  # Show track ID labels
        self.trajectory_length = 30  # Frames to show in trajectory
        
        # Track colors (consistent per track ID)
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
        self.play_pause_btn.setMaximumHeight(28)  # Compact height
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
        
        # Nest count label (NEW)
        self.nest_count_label = QLabel("Nests: -")
        self.nest_count_label.setStyleSheet("color: #4CAF50; font-size: 10pt; font-weight: bold;")
        info_layout.addWidget(self.nest_count_label)
        
        info_layout.addStretch()
        
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
    
    def display_frame_with_nests(self, frame, frame_num=None):
        """Display frame with hotel and nest annotations, optionally with tracks.
        
        This is a convenience method that draws nests and displays on canvas.
        Use display_frame_with_overlays() for nests + tracks together.
        
        Args:
            frame: BGR image to display
            frame_num: Optional frame number (for track overlay)
        """
        # Draw annotations
        annotated = self.draw_hotel_and_nests(frame)
        
        # Optionally draw tracks
        if frame_num is not None and self.tracking_df is not None and self.show_tracks:
            annotated = self.draw_tracks(annotated, frame_num)
        
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
    
    def load_nest_positions(self, nest_df_or_path):
        """Load nest positions from CSV or DataFrame (alternative to NestDetector).
        
        Simpler and faster than auto-detection - just load positions from file!
        
        Args:
            nest_df_or_path: DataFrame or path to CSV file
                Expected columns: nest_id, x1, y1, x2, y2
                Optional: row, col, label
        
        Example CSV:
            nest_id,x1,y1,x2,y2,row,col
            0,100,50,150,100,0,0
            1,150,50,200,100,0,1
            ...
        """
        import pandas as pd
        
        # Load from CSV if path
        if isinstance(nest_df_or_path, str):
            print(f"📂 Loading nest positions from: {nest_df_or_path}")
            nest_df = pd.read_csv(nest_df_or_path)
        else:
            nest_df = nest_df_or_path
        
        # Validate columns
        required = ['nest_id', 'x1', 'y1', 'x2', 'y2']
        missing = [col for col in required if col not in nest_df.columns]
        if missing:
            raise ValueError(f"Nest data missing columns: {missing}")
        
        # Convert to Detection objects (for compatibility)
        from beemonitor.detection.base_detector import Detection
        self.detected_nests = []
        
        for _, row in nest_df.iterrows():
            nest = Detection(
                bbox=(int(row['x1']), int(row['y1']), 
                     int(row['x2']), int(row['y2'])),
                centroid=(int((row['x1'] + row['x2']) / 2),
                         int((row['y1'] + row['y2']) / 2)),
                confidence=1.0,
                label='nest',
                source='csv'
            )
            nest.nest_id = int(row['nest_id'])
            nest.row = int(row['row']) if 'row' in row else None
            nest.col = int(row['col']) if 'col' in row else None
            self.detected_nests.append(nest)
        
        # Compute hotel ROI
        self.hotel_roi = self._compute_hotel_roi(self.detected_nests)
        
        print(f"✓ Loaded {len(self.detected_nests)} nests from CSV")
        if self.hotel_roi:
            x1, y1, x2, y2 = self.hotel_roi
            print(f"✓ Hotel ROI: ({x1}, {y1}) - ({x2}, {y2})")
        
        # Update UI
        self.nest_count_label.setText(f"Nests: {len(self.detected_nests)}")
        self.nest_count_label.setStyleSheet("color: #0a0; font-weight: bold;")
        
        # Emit signal
        self.hotel_nests_detected.emit({
            'hotel': self.hotel_roi,
            'nests': self.detected_nests,
            'nest_count': len(self.detected_nests)
        })
    
    def generate_nest_positions_csv(self, roi, rows, cols, output_path='nest_positions.csv'):
        """Generate CSV with nest positions in a regular grid.
        
        Args:
            roi: (x1, y1, x2, y2) hotel bounding box
            rows: Number of rows
            cols: Number of columns
            output_path: Path to save CSV
            
        Returns:
            DataFrame with nest positions
        """
        import pandas as pd
        
        x1, y1, x2, y2 = roi
        nest_width = (x2 - x1) / cols
        nest_height = (y2 - y1) / rows
        
        data = []
        nest_id = 0
        
        for row in range(rows):
            for col in range(cols):
                nx1 = x1 + col * nest_width
                ny1 = y1 + row * nest_height
                nx2 = nx1 + nest_width
                ny2 = ny1 + nest_height
                
                data.append({
                    'nest_id': nest_id,
                    'x1': int(nx1),
                    'y1': int(ny1),
                    'x2': int(nx2),
                    'y2': int(ny2),
                    'row': row,
                    'col': col,
                    'label': f'R{row}C{col}'
                })
                nest_id += 1
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        
        print(f"✓ Generated {len(data)} nest positions ({rows}×{cols})")
        print(f"✓ Saved to: {output_path}")
        
        return df
    
    # =========================================================================
    # TRACKING DATA MANAGEMENT
    # =========================================================================
    
    def load_tracking_results(self, tracking_df_or_path):
        """Load tracking results for visualization.
        
        Args:
            tracking_df_or_path: DataFrame or path to CSV file
                Expected columns: frame, track_id, x1, y1, x2, y2
                Optional: species, confidence
        """
        import pandas as pd
        from collections import defaultdict
        
        # Load from CSV if path
        if isinstance(tracking_df_or_path, str):
            print(f"Loading tracking results from: {tracking_df_or_path}")
            self.tracking_df = pd.read_csv(tracking_df_or_path)
        else:
            self.tracking_df = tracking_df_or_path
        
        # Validate columns
        required = ['frame', 'track_id', 'x1', 'y1', 'x2', 'y2']
        missing = [col for col in required if col not in self.tracking_df.columns]
        if missing:
            raise ValueError(f"Tracking data missing columns: {missing}")
        
        # Build frame index for fast lookup
        self._build_frame_index()
        
        # Build trajectories
        self._build_trajectories()
        
        # Assign colors to tracks
        unique_tracks = self.tracking_df['track_id'].unique()
        for i, track_id in enumerate(unique_tracks):
            self.track_colors[track_id] = self.color_palette[i % len(self.color_palette)]
        
        print(f"✓ Loaded {len(self.tracking_df)} detections")
        print(f"✓ {len(unique_tracks)} unique tracks")
        print(f"✓ Frames: {self.tracking_df['frame'].min()} - {self.tracking_df['frame'].max()}")
        
        # Update UI
        self.set_data_status(f"{len(unique_tracks)} tracks loaded", is_active=True)
    
    def _build_frame_index(self):
        """Build index for fast frame-based lookup."""
        self.tracks_by_frame = {}
        
        for _, row in self.tracking_df.iterrows():
            frame_num = int(row['frame'])
            
            if frame_num not in self.tracks_by_frame:
                self.tracks_by_frame[frame_num] = []
            
            track_data = {
                'track_id': int(row['track_id']),
                'bbox': (int(row['x1']), int(row['y1']), 
                        int(row['x2']), int(row['y2'])),
                'centroid': (int((row['x1'] + row['x2']) / 2), 
                           int((row['y1'] + row['y2']) / 2)),
                'species': row.get('species', 'bee'),
                'confidence': row.get('confidence', 1.0)
            }
            
            self.tracks_by_frame[frame_num].append(track_data)
    
    def _build_trajectories(self):
        """Build trajectory history for each track."""
        from collections import defaultdict
        self.trajectories = defaultdict(list)
        
        # Sort by frame
        df_sorted = self.tracking_df.sort_values('frame')
        
        for _, row in df_sorted.iterrows():
            track_id = int(row['track_id'])
            centroid = (int((row['x1'] + row['x2']) / 2),
                       int((row['y1'] + row['y2']) / 2))
            frame_num = int(row['frame'])
            
            self.trajectories[track_id].append((frame_num, centroid))
    
    # =========================================================================
    # DRAW TRACKS ON FRAME
    # =========================================================================
    
    def draw_tracks(self, frame, frame_num):
        """Draw tracking results on frame.
        
        Args:
            frame: BGR image
            frame_num: Current frame number
            
        Returns:
            Annotated frame with tracks drawn
        """
        if self.tracking_df is None or not self.show_tracks:
            return frame
        
        # Make copy
        annotated = frame.copy()
        
        # Get tracks for this frame
        tracks = self.tracks_by_frame.get(frame_num, [])
        
        if not tracks:
            return annotated
        
        # Draw trajectories first (behind boxes)
        if self.show_trajectories:
            for track in tracks:
                track_id = track['track_id']
                self._draw_trajectory(annotated, track_id, frame_num)
        
        # Draw bounding boxes and IDs
        for track in tracks:
            track_id = track['track_id']
            bbox = track['bbox']
            centroid = track['centroid']
            confidence = track['confidence']
            
            # Get color for this track
            color = self.track_colors.get(track_id, (0, 255, 0))
            
            # Draw bounding box
            x1, y1, x2, y2 = bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw track ID
            if self.show_track_ids:
                label = f"ID:{track_id}"
                if confidence < 1.0:
                    label += f" {confidence:.2f}"
                
                # Background for text
                (label_w, label_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(annotated, (x1, y1 - label_h - 4), 
                            (x1 + label_w, y1), color, -1)
                
                # Text
                cv2.putText(annotated, label, (x1, y1 - 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw centroid dot
            cv2.circle(annotated, centroid, 3, color, -1)
        
        return annotated
    
    def _draw_trajectory(self, frame, track_id, current_frame):
        """Draw trajectory line for a track.
        
        Args:
            frame: Image to draw on (modified in-place)
            track_id: Track ID
            current_frame: Current frame number
        """
        if track_id not in self.trajectories:
            return
        
        # Get trajectory points
        trajectory = self.trajectories[track_id]
        
        # Filter to recent frames only
        recent_points = [
            (f, pt) for f, pt in trajectory
            if current_frame - self.trajectory_length <= f <= current_frame
        ]
        
        if len(recent_points) < 2:
            return
        
        # Get color
        color = self.track_colors.get(track_id, (0, 255, 0))
        
        # Draw lines connecting points
        points = [pt for _, pt in recent_points]
        for i in range(len(points) - 1):
            cv2.line(frame, points[i], points[i + 1], color, 1)
    
    # =========================================================================
    # COMBINED DISPLAY WITH ALL OVERLAYS
    # =========================================================================
    
    def display_frame_with_overlays(self, frame, frame_num):
        """Display frame with all overlays (nests + tracks).
        
        Args:
            frame: BGR image
            frame_num: Current frame number
        """
        # Draw nests
        if self.show_nests:
            frame = self.draw_hotel_and_nests(frame)
        
        # Draw tracks
        if self.show_tracks and self.tracking_df is not None:
            frame = self.draw_tracks(frame, frame_num)
        
        # Display on canvas
        self.video_canvas.display_frame(frame)
    
    # =========================================================================
    # TRACK DISPLAY TOGGLES
    # =========================================================================
    
    def toggle_tracks(self, show: bool):
        """Toggle track display on/off."""
        self.show_tracks = show
        print(f"Track display: {'ON' if show else 'OFF'}")
    
    def toggle_trajectories(self, show: bool):
        """Toggle trajectory display on/off."""
        self.show_trajectories = show
        print(f"Trajectory display: {'ON' if show else 'OFF'}")
    
    def toggle_track_ids(self, show: bool):
        """Toggle track ID labels on/off."""
        self.show_track_ids = show
        print(f"Track ID labels: {'ON' if show else 'OFF'}")
    
    def set_trajectory_length(self, frames: int):
        """Set number of frames to show in trajectory.
        
        Args:
            frames: Number of past frames (1-100)
        """
        self.trajectory_length = max(1, min(100, frames))
        print(f"Trajectory length: {self.trajectory_length} frames")
    
    def clear_tracking_data(self):
        """Clear loaded tracking data."""
        self.tracking_df = None
        self.tracks_by_frame = {}
        self.trajectories = {}
        self.track_colors = {}
        self.set_data_status("No data", is_active=False)
        print("Tracking data cleared")
    
    def get_track_statistics(self, frame_num=None):
        """Get tracking statistics.
        
        Args:
            frame_num: If provided, stats for this frame only
            
        Returns:
            Dictionary with statistics
        """
        if self.tracking_df is None:
            return {}
        
        if frame_num is not None:
            # Stats for specific frame
            tracks = self.tracks_by_frame.get(frame_num, [])
            return {
                'frame': frame_num,
                'active_tracks': len(tracks),
                'track_ids': [t['track_id'] for t in tracks]
            }
        else:
            # Overall stats
            return {
                'total_detections': len(self.tracking_df),
                'total_tracks': self.tracking_df['track_id'].nunique(),
                'frame_range': (int(self.tracking_df['frame'].min()), 
                               int(self.tracking_df['frame'].max())),
                'avg_detections_per_frame': len(self.tracking_df) / 
                    (self.tracking_df['frame'].max() - self.tracking_df['frame'].min() + 1)
            }
    
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