"""Base detector interface for all detection methods.

All detectors (nest, blob, SIFT, YOLO) inherit from this abstract class.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np

# Type aliases
BBox = Tuple[float, float, float, float]
Point = Tuple[float, float]


@dataclass
class Detection:
    """Single detection result from any detector.
    
    Attributes:
        bbox: Bounding box (x1, y1, x2, y2)
        centroid: Center point (x, y)
        confidence: Detection confidence (0-1)
        label: Class label (e.g., 'bee', 'wasp', 'nest')
        source: Detector name ('yolo', 'sift', 'blob', etc.)
        metadata: Additional detector-specific data
    """
    bbox: BBox
    centroid: Point
    confidence: float
    label: str
    source: str
    metadata: Optional[Dict[str, Any]] = None


class BaseDetector(ABC):
    """Abstract base class for all detectors.
    
    All detection methods (YOLO, SIFT, FG/BG blobs) implement this interface.
    This provides a unified API for detecting objects in images.
    """
    
    @abstractmethod
    def detect(self, frame: np.ndarray, **kwargs) -> List[Detection]:
        """Detect objects in a single frame.
        
        Args:
            frame: Input image (BGR format)
            **kwargs: Detector-specific parameters
            
        Returns:
            List of Detection objects
            
        Example:
            >>> detector = YOLODetector(model)
            >>> detections = detector.detect(frame, conf=0.5)
            >>> for det in detections:
            ...     print(f"{det.label}: {det.confidence:.2f}")
        """
        pass
    
    @abstractmethod
    def configure(self, **kwargs) -> None:
        """Configure detector parameters.
        
        Args:
            **kwargs: Detector-specific configuration
            
        Example:
            >>> detector.configure(conf_threshold=0.5, iou_threshold=0.7)
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset detector state (if stateful).
        
        For stateful detectors like background subtractors.
        """
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get detector information.
        
        Returns:
            Dictionary with detector name, type, configuration
        """
        return {
            'detector_type': self.__class__.__name__,
            'source': self.get_source_name()
        }
    
    @abstractmethod
    def get_source_name(self) -> str:
        """Get detector source identifier.
        
        Returns:
            Source name (e.g., 'yolo', 'sift', 'blob')
        """
        pass
    
    # Utility methods (implemented for all detectors)
    
    def filter_by_confidence(
        self,
        detections: List[Detection],
        min_confidence: float
    ) -> List[Detection]:
        """Filter detections by confidence threshold.
        
        Args:
            detections: List of detections
            min_confidence: Minimum confidence (0-1)
            
        Returns:
            Filtered detections
        """
        return [d for d in detections if d.confidence >= min_confidence]
    
    def filter_by_labels(
        self,
        detections: List[Detection],
        labels: List[str]
    ) -> List[Detection]:
        """Filter detections by class labels.
        
        Args:
            detections: List of detections
            labels: List of allowed labels
            
        Returns:
            Filtered detections
        """
        label_set = set(labels)
        return [d for d in detections if d.label in label_set]
    
    def filter_by_area(
        self,
        detections: List[Detection],
        min_area: Optional[float] = None,
        max_area: Optional[float] = None
    ) -> List[Detection]:
        """Filter detections by bounding box area.
        
        Args:
            detections: List of detections
            min_area: Minimum area (pixels)
            max_area: Maximum area (pixels)
            
        Returns:
            Filtered detections
        """
        filtered = []
        for d in detections:
            area = self.bbox_area(d.bbox)
            if min_area is not None and area < min_area:
                continue
            if max_area is not None and area > max_area:
                continue
            filtered.append(d)
        return filtered
    
    def validate_detections(
        self,
        detections: List[Detection],
        frame_width: int,
        frame_height: int
    ) -> List[Detection]:
        """Validate detections are within frame bounds.
        
        Args:
            detections: List of detections
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            
        Returns:
            Valid detections only
        """
        valid = []
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            
            # Check bounds
            if x1 < 0 or y1 < 0 or x2 > frame_width or y2 > frame_height:
                continue
            
            # Check validity
            if x2 <= x1 or y2 <= y1:
                continue
            
            valid.append(d)
        
        return valid
    
    def get_detection_stats(self, detections: List[Detection]) -> Dict[str, Any]:
        """Get statistics about detections.
        
        Args:
            detections: List of detections
            
        Returns:
            Dictionary with statistics
        """
        if not detections:
            return {
                'count': 0,
                'labels': {},
                'avg_confidence': 0.0,
                'avg_area': 0.0
            }
        
        # Count by label
        label_counts = {}
        for d in detections:
            label_counts[d.label] = label_counts.get(d.label, 0) + 1
        
        # Average confidence
        avg_conf = sum(d.confidence for d in detections) / len(detections)
        
        # Average area
        avg_area = sum(self.bbox_area(d.bbox) for d in detections) / len(detections)
        
        return {
            'count': len(detections),
            'labels': label_counts,
            'avg_confidence': avg_conf,
            'avg_area': avg_area
        }
    
    @staticmethod
    def bbox_area(bbox: BBox) -> float:
        """Calculate bounding box area.
        
        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            Area in pixels
        """
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)
    
    @staticmethod
    def bbox_iou(bbox1: BBox, bbox2: BBox) -> float:
        """Calculate Intersection over Union between two boxes.
        
        Args:
            bbox1: First bounding box (x1, y1, x2, y2)
            bbox2: Second bounding box (x1, y1, x2, y2)
            
        Returns:
            IoU value (0-1)
        """
        # Get intersection coordinates
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        # Check if boxes intersect
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        # Calculate intersection area
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union area
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        # Avoid division by zero
        if union <= 0:
            return 0.0
        
        return intersection / union
    
    @staticmethod
    def bbox_to_xywh(bbox: BBox) -> Tuple[float, float, float, float]:
        """Convert bbox from (x1, y1, x2, y2) to (x, y, w, h).
        
        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            Tuple of (x, y, width, height)
        """
        x1, y1, x2, y2 = bbox
        return (x1, y1, x2 - x1, y2 - y1)
    
    @staticmethod
    def xywh_to_bbox(xywh: Tuple[float, float, float, float]) -> BBox:
        """Convert from (x, y, w, h) to (x1, y1, x2, y2).
        
        Args:
            xywh: Tuple of (x, y, width, height)
            
        Returns:
            Bounding box (x1, y1, x2, y2)
        """
        x, y, w, h = xywh
        return (x, y, x + w, y + h)
    
    def nms(
        self,
        detections: List[Detection],
        iou_threshold: float = 0.5
    ) -> List[Detection]:
        """Non-Maximum Suppression to remove overlapping detections.
        
        Args:
            detections: List of detections
            iou_threshold: IoU threshold for suppression
            
        Returns:
            Filtered detections after NMS
        """
        if not detections:
            return []
        
        # Sort by confidence (descending)
        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        
        keep = []
        
        while sorted_dets:
            # Keep highest confidence detection
            current = sorted_dets.pop(0)
            keep.append(current)
            
            # Remove overlapping detections
            sorted_dets = [
                d for d in sorted_dets
                if self.bbox_iou(current.bbox, d.bbox) < iou_threshold
            ]
        
        return keep
    
    def merge_detections(
        self,
        detections_list: List[List[Detection]],
        iou_threshold: float = 0.5
    ) -> List[Detection]:
        """Merge detections from multiple sources/frames.
        
        Args:
            detections_list: List of detection lists to merge
            iou_threshold: IoU threshold for NMS
            
        Returns:
            Merged and deduplicated detections
        """
        # Combine all detections
        all_detections = []
        for dets in detections_list:
            all_detections.extend(dets)
        
        # Apply NMS to remove duplicates
        return self.nms(all_detections, iou_threshold)