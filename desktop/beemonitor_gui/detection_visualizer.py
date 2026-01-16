"""
Detection Source Visualizer - v2.3
===================================

Color-coded visualization of detection sources:
- RED: Blob motion detection (two-mode optimization)
- BLUE: YOLO tracking (100% accuracy)
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
        """Draw detections with color-coded sources."""
        vis_frame = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            source = getattr(det, 'source', 'unknown').lower()
            confidence = getattr(det, 'confidence', 0.0)
            
            color = self.colors.get(source, self.colors['unknown'])
            
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, thickness)
            
            label_parts = []
            if show_labels:
                label_parts.append(source.upper())
            if show_confidence and confidence > 0:
                label_parts.append(f"{confidence:.2f}")
            
            if label_parts:
                label = " ".join(label_parts)
                
                (label_w, label_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                
                cv2.rectangle(
                    vis_frame,
                    (x1, y1 - label_h - baseline - 5),
                    (x1 + label_w + 5, y1),
                    color,
                    -1
                )
                
                cv2.putText(
                    vis_frame,
                    label,
                    (x1 + 2, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
            
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
        """Draw legend showing detection source colors."""
        vis_frame = frame.copy()
        h, w = frame.shape[:2]
        
        legend_width = 180
        legend_height = 90 if counts else 70
        padding = 10
        line_height = 20
        
        if position == 'top_right':
            x = w - legend_width - padding
            y = padding
        elif position == 'top_left':
            x = padding
            y = padding
        elif position == 'bottom_right':
            x = w - legend_width - padding
            y = h - legend_height - padding
        else:
            x = padding
            y = h - legend_height - padding
        
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
        
        cv2.putText(
            vis_frame,
            "Detection Sources (v2.3)",
            (x + 5, y + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1
        )
        
        sources = [
            ('Motion (Blob)', 'blob'),
            ('Tracking (YOLO)', 'yolo')
        ]
        
        for i, (label, source_key) in enumerate(sources):
            color = self.colors[source_key]
            y_pos = y + 30 + (i * line_height)
            
            cv2.rectangle(
                vis_frame,
                (x + 5, y_pos),
                (x + 20, y_pos + 12),
                color,
                -1
            )
            
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
        """Count detections by source."""
        counts = {'blob': 0, 'yolo': 0}
        
        for det in detections:
            source = getattr(det, 'source', 'unknown').lower()
            if source in counts:
                counts[source] += 1
        
        return counts