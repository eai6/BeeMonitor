"""Base tracking system interface.

Abstract class for high-level tracking systems (BeeTracking, etc.)
Combines detection + MOT + domain-specific logic.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np

from beemonitor.detection.base_detector import Detection


class BaseTracking(ABC):
    """Abstract base class for tracking systems.
    
    High-level tracking systems (like BeeTracking) inherit from this.
    Orchestrates: detection pipeline + MOT algorithm + domain logic.
    
    This is different from BaseMOT which is just the tracking algorithm.
    """
    
    @abstractmethod
    def process_video(
        self,
        video_path: str,
        roi: Optional[tuple] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Process entire video and return tracking results.
        
        Args:
            video_path: Path to video file
            roi: Region of interest (x1, y1, x2, y2)
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with tracking results
        """
        pass
    
    @abstractmethod
    def process_frame(
        self,
        frame: np.ndarray,
        frame_num: int
    ) -> Dict[str, Any]:
        """Process single frame.
        
        Args:
            frame: Input frame
            frame_num: Frame number
            
        Returns:
            Dictionary with frame results (detections, tracks, etc.)
        """
        pass
    
    @abstractmethod
    def configure_detection(self, **kwargs) -> None:
        """Configure detection pipeline.
        
        Args:
            **kwargs: Detector-specific parameters
        """
        pass
    
    @abstractmethod
    def configure_tracking(self, **kwargs) -> None:
        """Configure tracking algorithm.
        
        Args:
            **kwargs: MOT-specific parameters
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset tracking system state."""
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """Get tracking statistics.
        
        Returns:
            Dictionary with stats (total_tracks, avg_length, etc.)
        """
        pass
