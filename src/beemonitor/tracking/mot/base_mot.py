"""Base interface for Multiple Object Tracking (MOT) algorithms.

Allows easy swapping between different MOT implementations (Kalman, ByteTrack, etc.)
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Type aliases
BBox = Tuple[float, float, float, float]
Point = Tuple[float, float]


@dataclass
class Detection:
    """Single detection from motion or YOLO."""
    bbox: BBox
    centroid: Point
    label: str
    confidence: float = 1.0
    source: str = 'unknown'  # 'motion', 'yolo', etc.
    cost_weight: float = 1.0 # Weight for Hungarian cost matrix


@dataclass
class Track:
    """Active track state."""
    track_id: int
    bbox: BBox
    centroid: Point
    label: str
    age: int
    frames_without_detection: int
    last_confirmation_frame: int
    trajectory: List[Tuple[int, Point]]  # [(frame, centroid), ...]
    velocity: Optional[Tuple[float, float]] = None
    source: str = 'unknown'  # Detection source: 'blob', 'sift', 'yolo'
    
    
class BaseMOT(ABC):
    """Abstract base class for Multiple Object Tracking algorithms."""
    
    @abstractmethod
    def predict(self, frame_num: int) -> Dict[int, Track]:
        """Predict track positions for current frame.
        
        Args:
            frame_num: Current frame number
            
        Returns:
            Dictionary of predicted tracks {track_id: Track}
        """
        pass
    
    @abstractmethod
    def update(self, detections: List[Detection], frame_num: int) -> Dict[int, Track]:
        """Update tracks with new detections.
        
        Args:
            detections: List of detections for current frame
            frame_num: Current frame number
            
        Returns:
            Dictionary of updated tracks {track_id: Track}
        """
        pass
    
    @abstractmethod
    def get_tracks(self) -> Dict[int, Track]:
        """Get all active tracks.
        
        Returns:
            Dictionary of active tracks {track_id: Track}
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Reset tracker state (clear all tracks)."""
        pass
    
    @abstractmethod
    def get_search_regions(self, frame_num: int) -> Dict[int, Dict]:
        """Get search regions for each track (for visualization/debugging).
        
        Args:
            frame_num: Current frame number
            
        Returns:
            Dictionary {track_id: search_region_dict}
        """
        pass