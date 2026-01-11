"""
Detection Source Visualizer
============================

Color-coded visualization of detection sources:
- RED: Blob/FG-BG (motion detection)
- GREEN: SIFT (stationary detection)
- BLUE: YOLO (deep learning)
"""

import cv2
import numpy as np
from typing import List, Dict, Optional

from .constants import DETECTION_SOURCE_COLORS


class DetectionSourceVisualizer:
    """Visualize detections with color-coded sources."""
    
    def __init__(self):
        """Initialize visualizer with color scheme."""
        self.colors = DETECTION_SOURCE_COLORS
    
    def draw_detections_with_sources(
        self,
        frame: np.ndarray,
        detections: List,
        show_labels: bool = True,
        show_confidence: bool = False,
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw detections with color-coded sources.
        
        Args:
            frame: Input frame (BGR)
            detections: List of Detection objects with .source attribute
            show_labels: Show source label text
            show_confidence: Show confidence scores
            thickness: Line thickness
        
        Returns:
            Frame with visualizations
        """
        vis_frame = frame.copy()
        
        for det in detections:
            # Get detection info
            x1, y1, x2, y2 = map(int, det.bbox)
            source = getattr(det, 'source', 'unknown').lower()
            confidence = getattr(det, 'confidence', 0.0)
            
            # Get color for this source
            color = self.colors.get(source, self.colors['unknown'])
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Build label text
            label_parts = []
            if show_labels:
                label_parts.append(source.upper())
            if show_confidence and confidence > 0:
                label_parts.append(f"{confidence:.2f}")
            
            if label_parts:
                label = " ".join(label_parts)
                
                # Get label size for background
                (label_w, label_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                
                # Draw label background
                cv2.rectangle(
                    vis_frame,
                    (x1, y1 - label_h - baseline - 5),
                    (x1 + label_w + 5, y1),
                    color,
                    -1  # Filled
                )
                
                # Draw label text
                cv2.putText(
                    vis_frame,
                    label,
                    (x1 + 2, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),  # White text
                    1
                )
            
            # Draw centroid
            if hasattr(det, 'centroid'):
                cx, cy = map(int, det.centroid)
                cv2.circle(vis_frame, (cx, cy), 3, color, -1)
        
        return vis_frame
    
    def draw_source_legend(
        self,
        frame: np.ndarray,
        position: str = 'top_right',
        counts: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Draw legend showing detection source colors.
        
        Args:
            frame: Input frame
            position: 'top_right', 'top_left', 'bottom_right', 'bottom_left'
            counts: Optional dict with detection counts per source
        
        Returns:
            Frame with legend
        """
        vis_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Legend config
        legend_width = 180
        legend_height = 120 if counts else 80
        padding = 10
        line_height = 20
        
        # Position
        if position == 'top_right':
            x = w - legend_width - padding
            y = padding
        elif position == 'top_left':
            x = padding
            y = padding
        elif position == 'bottom_right':
            x = w - legend_width - padding
            y = h - legend_height - padding
        else:  # bottom_left
            x = padding
            y = h - legend_height - padding
        
        # Draw legend background
        cv2.rectangle(
            vis_frame,
            (x, y),
            (x + legend_width, y + legend_height),
            (0, 0, 0),
            -1
        )
        cv2.rectangle(
            vis_frame,
            (x, y),
            (x + legend_width, y + legend_height),
            (255, 255, 255),
            1
        )
        
        # Title
        cv2.putText(
            vis_frame,
            "Detection Sources",
            (x + 5, y + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1
        )
        
        # Draw each source
        sources = [
            ('BLOB/FG-BG', 'blob'),
            ('SIFT', 'sift'),
            ('YOLO', 'yolo')
        ]
        
        for i, (label, source_key) in enumerate(sources):
            color = self.colors[source_key]
            y_pos = y + 30 + (i * line_height)
            
            # Color box
            cv2.rectangle(
                vis_frame,
                (x + 5, y_pos),
                (x + 20, y_pos + 12),
                color,
                -1
            )
            
            # Label
            text = label
            if counts and source_key in counts:
                text += f" ({counts[source_key]})"
            
            cv2.putText(
                vis_frame,
                text,
                (x + 25, y_pos + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1
            )
        
        # Total count
        if counts:
            total = sum(counts.values())
            cv2.putText(
                vis_frame,
                f"Total: {total}",
                (x + 5, y + legend_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1
            )
        
        return vis_frame
    
    @staticmethod
    def get_detection_counts(detections: List) -> Dict[str, int]:
        """
        Count detections by source.
        
        Args:
            detections: List of Detection objects
        
        Returns:
            Dict with counts: {'blob': N, 'sift': M, 'yolo': K}
        """
        counts = {'blob': 0, 'sift': 0, 'yolo': 0}
        
        for det in detections:
            source = getattr(det, 'source', 'unknown').lower()
            if source in counts:
                counts[source] += 1
        
        return counts
