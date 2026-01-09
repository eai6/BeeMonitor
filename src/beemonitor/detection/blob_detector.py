"""Blob detector using foreground/background subtraction.

Detects motion blobs using background subtraction and morphological operations.
"""

import logging
from typing import List, Optional, Tuple
import cv2
import numpy as np

from .base_detector import BaseDetector, Detection, BBox, Point

logger = logging.getLogger(__name__)


class BlobDetector(BaseDetector):
    """Foreground/background subtraction-based blob detector.
    
    Uses MOG2 background subtraction to detect moving objects.
    Applies morphological operations and filtering for noise reduction.
    
    Attributes:
        bg_subtractor: Background subtractor (MOG2)
        min_area: Minimum blob area (pixels)
        min_solidity: Minimum solidity (blob_area / convex_hull_area)
        morph_kernel_size: Morphology kernel size
        morph_iterations: Number of morphology iterations
    """
    
    def __init__(
        self,
        min_area: float = 50.0,
        min_solidity: float = 0.5,
        morph_kernel_size: int = 5,
        morph_iterations: int = 2,
        history: int = 500,
        var_threshold: int = 16
    ):
        """Initialize blob detector.
        
        Args:
            min_area: Minimum blob area in pixels
            min_solidity: Minimum solidity ratio (0-1)
            morph_kernel_size: Morphology kernel size (odd number)
            morph_iterations: Morphology iterations
            history: Background subtractor history frames
            var_threshold: Background subtractor variance threshold
        """
        self.min_area = min_area
        self.min_solidity = min_solidity
        self.morph_kernel_size = morph_kernel_size
        self.morph_iterations = morph_iterations
        
        # Initialize background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=False
        )
        
        logger.info(f"BlobDetector initialized: min_area={min_area}, min_solidity={min_solidity}")
    
    def detect(self, frame: np.ndarray, **kwargs) -> List[Detection]:
        """Detect motion blobs in frame.
        
        Args:
            frame: Input frame (BGR)
            **kwargs: Optional overrides (min_area, min_solidity)
            
        Returns:
            List of blob detections
        """
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Morphological operations to reduce noise
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.morph_kernel_size, self.morph_kernel_size)
        )
        fg_mask = cv2.morphologyEx(
            fg_mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=self.morph_iterations
        )
        
        # Find contours
        contours, _ = cv2.findContours(
            fg_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter and convert to detections
        detections = []
        min_area = kwargs.get('min_area', self.min_area)
        min_solidity = kwargs.get('min_solidity', self.min_solidity)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Area filter
            if area < min_area:
                continue
            
            # Solidity filter
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = area / hull_area
                if solidity < min_solidity:
                    continue
            
            # Create detection
            x, y, w, h = cv2.boundingRect(contour)
            bbox = (float(x), float(y), float(x + w), float(y + h))
            centroid = (float(x + w / 2), float(y + h / 2))
            
            detections.append(Detection(
                bbox=bbox,
                centroid=centroid,
                confidence=1.0,  # Blob detection doesn't have confidence
                label='blob',
                source='fgbg',
                metadata={
                    'area': float(area),
                    'solidity': float(solidity) if hull_area > 0 else 0.0,
                    'contour': contour
                }
            ))
        
        return detections
    
    def configure(self, **kwargs) -> None:
        """Configure blob detector parameters.
        
        Args:
            **kwargs: min_area, min_solidity, morph_kernel_size, morph_iterations
        """
        if 'min_area' in kwargs:
            self.min_area = kwargs['min_area']
        if 'min_solidity' in kwargs:
            self.min_solidity = kwargs['min_solidity']
        if 'morph_kernel_size' in kwargs:
            self.morph_kernel_size = kwargs['morph_kernel_size']
        if 'morph_iterations' in kwargs:
            self.morph_iterations = kwargs['morph_iterations']
        
        logger.debug(f"BlobDetector configured: {kwargs}")
    
    def reset(self) -> None:
        """Reset background subtractor."""
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=False
        )
        logger.debug("BlobDetector reset")
    
    def get_source_name(self) -> str:
        """Get detector source name."""
        return 'fgbg'
