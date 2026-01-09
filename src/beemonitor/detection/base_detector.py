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
