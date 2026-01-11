"""
Enhanced Detection Visualizer - Show Detection Sources

Add this to your GUI to visualize which detector found each detection:
- Blob (FG/BG): RED
- SIFT: GREEN  
- YOLO: BLUE

This helps diagnose detection pipeline issues.
"""

import cv2
import numpy as np
from typing import List, Optional


class DetectionSourceVisualizer:
    """Visualize detections with color-coded sources."""
    
    # Color scheme for detection sources
    COLORS = {
        'blob': (0, 0, 255),      # RED - Blob/FG/BG
        'fgbg': (0, 0, 255),      # RED - Alias for blob
        'sift': (0, 255, 0),      # GREEN - SIFT
        'yolo': (255, 0, 0),      # BLUE - YOLO
        'unknown': (128, 128, 128) # GRAY - Unknown source
    }
    
    @staticmethod
    def draw_detections_with_sources(
        frame: np.ndarray,
        detections: List,
        show_labels: bool = True,
        show_confidence: bool = True,
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
            color = DetectionSourceVisualizer.COLORS.get(source, DetectionSourceVisualizer.COLORS['unknown'])
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Build label text
            label_parts = []
            if show_labels:
                label_parts.append(source.upper())
            if show_confidence:
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
    
    @staticmethod
    def draw_source_legend(
        frame: np.ndarray,
        position: str = 'top_right',
        counts: Optional[dict] = None
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
            color = DetectionSourceVisualizer.COLORS[source_key]
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
    def get_detection_counts(detections: List) -> dict:
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


# ============================================================================
# Integration with GUI
# ============================================================================

def add_detection_source_visualization_to_gui(gui_class):
    """
    Add detection source visualization to your GUI.
    
    Usage:
        # In your GUI class
        add_detection_source_visualization_to_gui(YourGUIClass)
    """
    
    # Add visualizer instance
    gui_class.detection_visualizer = DetectionSourceVisualizer()
    
    # Override or enhance draw_detections method
    original_draw = gui_class.draw_overlays if hasattr(gui_class, 'draw_overlays') else None
    
    def enhanced_draw_overlays(self, frame, results, frame_num):
        """Enhanced overlay drawing with detection sources."""
        
        # Get detections if available
        detections = results.get('detections', [])
        
        # Draw detection sources if checkbox enabled
        if getattr(self, 'show_detection_sources', False) and detections:
            # Draw detections with color-coded sources
            frame = self.detection_visualizer.draw_detections_with_sources(
                frame,
                detections,
                show_labels=True,
                show_confidence=True
            )
            
            # Draw legend
            counts = self.detection_visualizer.get_detection_counts(detections)
            frame = self.detection_visualizer.draw_legend(
                frame,
                position='top_right',
                counts=counts
            )
        
        # Call original draw method if exists
        if original_draw:
            frame = original_draw(frame, results, frame_num)
        
        return frame
    
    gui_class.enhanced_draw_overlays = enhanced_draw_overlays


# ============================================================================
# PyQt6 GUI Integration Example
# ============================================================================

def integrate_with_pyqt6_gui():
    """
    Example integration with PyQt6 GUI.
    
    Add this to your beemonitor_gui_player.py:
    """
    
    example_code = '''
# In your GUI class __init__():
self.detection_visualizer = DetectionSourceVisualizer()

# Add checkbox to show detection sources
self.show_detection_sources_checkbox = QCheckBox("Show Detection Sources")
self.show_detection_sources_checkbox.setChecked(False)
self.show_detection_sources_checkbox.stateChanged.connect(self.update_display)

# Add to control panel layout
control_layout.addWidget(self.show_detection_sources_checkbox)

# In your draw_overlays() or display_frame() method:
def draw_overlays(self, frame, results, frame_num):
    """Draw overlays on frame."""
    
    # ... existing overlay code ...
    
    # NEW: Detection source visualization
    if self.show_detection_sources_checkbox.isChecked():
        detections = results.get('detections', [])
        
        if detections:
            # Draw color-coded detections
            frame = self.detection_visualizer.draw_detections_with_sources(
                frame,
                detections,
                show_labels=True,
                show_confidence=True,
                thickness=2
            )
            
            # Draw legend
            counts = self.detection_visualizer.get_detection_counts(detections)
            frame = self.detection_visualizer.draw_source_legend(
                frame,
                position='top_right',
                counts=counts
            )
    
    return frame
'''
    
    return example_code


# ============================================================================
# Standalone Usage Example
# ============================================================================

if __name__ == '__main__':
    """
    Example usage for testing.
    """
    import cv2
    
    # Create dummy detections
    class DummyDetection:
        def __init__(self, bbox, source, confidence):
            self.bbox = bbox
            self.centroid = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            self.source = source
            self.confidence = confidence
    
    # Test frame
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    
    # Create test detections
    detections = [
        DummyDetection((100, 100, 150, 150), 'blob', 0.95),
        DummyDetection((200, 100, 250, 150), 'sift', 0.85),
        DummyDetection((300, 100, 350, 150), 'yolo', 0.92),
        DummyDetection((100, 200, 150, 250), 'blob', 0.88),
        DummyDetection((200, 200, 250, 250), 'blob', 0.90),
    ]
    
    # Visualize
    visualizer = DetectionSourceVisualizer()
    
    # Draw detections
    vis_frame = visualizer.draw_detections_with_sources(
        frame,
        detections,
        show_labels=True,
        show_confidence=True
    )
    
    # Draw legend
    counts = visualizer.get_detection_counts(detections)
    vis_frame = visualizer.draw_source_legend(
        vis_frame,
        position='top_right',
        counts=counts
    )
    
    # Display
    cv2.imshow('Detection Sources', vis_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("Visualizer test complete!")
    print(f"Detection counts: {counts}")