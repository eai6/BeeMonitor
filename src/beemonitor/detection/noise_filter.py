"""CNN-based noise filter for blob detections.

Filters false positive blobs using a trained CNN classifier.
"""

import logging
from typing import List
import cv2
import numpy as np

from .base_detector import Detection

logger = logging.getLogger(__name__)


class NoiseFilter:
    """CNN-based noise filter for detections.
    
    Filters out noise blobs using a trained classifier.
    Not a detector itself, but filters detector outputs.
    
    Attributes:
        classifier: Bee noise classifier model
        threshold: Confidence threshold for accepting detections
    """
    
    def __init__(
        self,
        classifier,
        threshold: float = 0.7
    ):
        """Initialize noise filter.
        
        Args:
            classifier: BeeNoiseFilter instance
            threshold: Confidence threshold (0-1)
        """
        self.classifier = classifier
        self.threshold = threshold
        
        logger.info(f"NoiseFilter initialized: threshold={threshold}")
    
    def filter_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection]
    ) -> List[Detection]:
        """Filter detections using CNN classifier.
        
        Args:
            frame: Original frame
            detections: List of detections to filter
            
        Returns:
            Filtered list of detections
        """
        filtered = []
        
        for det in detections:
            # Extract crop from frame
            x1, y1, x2, y2 = [int(c) for c in det.bbox]
            
            # Validate bbox
            if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            
            crop = frame[y1:y2, x1:x2]
            
            if crop.size == 0:
                continue
            
            # Classify
            is_bee = self.classifier.is_bee(crop)
            
            if is_bee:
                # Update confidence with classifier score
                # Could combine with original confidence
                filtered.append(det)
        
        logger.debug(f"NoiseFilter: {len(detections)} → {len(filtered)} detections")
        
        return filtered
    
    def configure(self, threshold: float) -> None:
        """Configure noise filter threshold.
        
        Args:
            threshold: New threshold (0-1)
        """
        self.threshold = threshold
        logger.debug(f"NoiseFilter threshold set to {threshold}")
