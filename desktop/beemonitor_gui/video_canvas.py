"""
Video Canvas Widget
===================

Custom QLabel widget for displaying video with overlays:
- Detection boxes (with source color-coding)
- Track trajectories
- ROI drawing
"""

import cv2
import numpy as np
from PyQt6.QtWidgets import QLabel
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap

from .constants import TRACK_COLORS, DETECTION_BOX_THICKNESS, TRACK_CIRCLE_RADIUS
from .detection_visualizer import DetectionSourceVisualizer


class VideoCanvas(QLabel):
    """Video display widget with overlay support."""
    
    roi_changed = pyqtSignal(tuple)
    
    def __init__(self):
        """Initialize video canvas."""
        super().__init__()
        self.setMinimumSize(640, 480)
        self.setStyleSheet("background-color: black;")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.current_frame = None
        self.original_pixmap = None
        
        # Visualization overlays
        self.show_detections = False
        self.show_tracks = False
        self.show_detection_sources = False
        self.detections = []
        self.tracks = {}
        self.roi = None
        
        # Nest overlays
        self.detected_nests = []
        self.hotel_roi = None
        self.show_nests = True
        
        # Detection source visualizer
        self.source_visualizer = DetectionSourceVisualizer()
        
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
        if tracks is not None:
            self.tracks = tracks
        if roi is not None:
            self.roi = roi
        
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
        if self.show_detections and self.detections:
            if self.show_detection_sources:
                frame = self.source_visualizer.draw_detections_with_sources(
                    frame,
                    self.detections,
                    show_labels=True,
                    show_confidence=False,
                    thickness=DETECTION_BOX_THICKNESS
                )
                
                counts = self.source_visualizer.get_detection_counts(self.detections)
                frame = self.source_visualizer.draw_source_legend(
                    frame,
                    position='top_right',
                    counts=counts
                )
            else:
                for det in self.detections:
                    x1, y1, x2, y2 = [int(c) for c in det.bbox]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    area = (x2-x1) * (y2-y1)
                    cv2.putText(frame, f"{area:.0f}", (x1, y1-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Draw tracks
        if self.show_tracks and self.tracks:
            for i, (track_id, trajectory) in enumerate(self.tracks.items()):
                color = TRACK_COLORS[i % len(TRACK_COLORS)]
                
                if len(trajectory) > 1:
                    points = np.array(trajectory, dtype=np.int32)
                    cv2.polylines(frame, [points], False, color, 2)
                
                if trajectory:
                    x, y = trajectory[-1]
                    cv2.circle(frame, (int(x), int(y)), TRACK_CIRCLE_RADIUS, color, -1)
                    cv2.putText(frame, f"ID:{track_id}", (int(x)+10, int(y)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add status overlay
        if not self.show_detection_sources:
            status_lines = []
            if self.show_detections:
                status_lines.append(f"Detections: {len(self.detections)}")
            if self.show_tracks:
                status_lines.append(f"Tracks: {len(self.tracks)}")
            
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
        self._draw_frame()
    
    def toggle_tracks(self, enabled):
        """Toggle track overlay."""
        self.show_tracks = enabled
        self._draw_frame()
    
    def toggle_detection_sources(self, enabled):
        """Toggle detection source color-coding."""
        self.show_detection_sources = enabled
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