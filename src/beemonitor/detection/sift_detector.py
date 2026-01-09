"""SIFT-based detector for detecting bees without motion.

Uses SIFT (Scale-Invariant Feature Transform) to detect bees even when stationary.
Works by detecting keypoints and clustering them into bee candidates.
"""

import logging
from typing import List, Optional, Tuple
import cv2
import numpy as np
from sklearn.cluster import DBSCAN

from .base_detector import BaseDetector, Detection, BBox, Point

logger = logging.getLogger(__name__)


class SIFTDetector(BaseDetector):
    """SIFT-based detector for stationary object detection.
    
    Detects objects using SIFT keypoints and spatial clustering.
    Good for detecting bees that aren't moving (FG/BG misses these).
    
    Attributes:
        sift: SIFT detector
        min_keypoints: Minimum keypoints per detection
        cluster_eps: DBSCAN clustering epsilon (pixels)
        min_cluster_size: Minimum cluster size
    """
    
    def __init__(
        self,
        min_keypoints: int = 3,
        cluster_eps: float = 30.0,
        min_cluster_size: int = 3,
        contrast_threshold: float = 0.04,
        edge_threshold: float = 10
    ):
        """Initialize SIFT detector.
        
        Args:
            min_keypoints: Minimum keypoints to form detection
            cluster_eps: DBSCAN epsilon (max distance for clustering)
            min_cluster_size: Minimum keypoints in cluster
            contrast_threshold: SIFT contrast threshold (lower = more features)
            edge_threshold: SIFT edge threshold
        """
        self.min_keypoints = min_keypoints
        self.cluster_eps = cluster_eps
        self.min_cluster_size = min_cluster_size
        
        # Initialize SIFT
        try:
            self.sift = cv2.SIFT_create(
                contrastThreshold=contrast_threshold,
                edgeThreshold=edge_threshold
            )
        except AttributeError:
            # Older OpenCV versions
            self.sift = cv2.xfeatures2d.SIFT_create(
                contrastThreshold=contrast_threshold,
                edgeThreshold=edge_threshold
            )
        
        logger.info(f"SIFTDetector initialized: min_keypoints={min_keypoints}")
    
    def detect(self, frame: np.ndarray, **kwargs) -> List[Detection]:
        """Detect objects using SIFT keypoints.
        
        Args:
            frame: Input frame (BGR)
            **kwargs: Optional overrides
            
        Returns:
            List of SIFT-based detections
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect keypoints
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        
        if len(keypoints) < self.min_keypoints:
            return []
        
        # Extract keypoint coordinates
        coords = np.array([kp.pt for kp in keypoints])
        
        # Cluster keypoints spatially using DBSCAN
        clustering = DBSCAN(
            eps=self.cluster_eps,
            min_samples=self.min_cluster_size
        ).fit(coords)
        
        # Convert clusters to detections
        detections = []
        unique_labels = set(clustering.labels_)
        unique_labels.discard(-1)  # Remove noise label
        
        for label in unique_labels:
            # Get keypoints in this cluster
            cluster_mask = clustering.labels_ == label
            cluster_coords = coords[cluster_mask]
            cluster_kps = [kp for kp, mask in zip(keypoints, cluster_mask) if mask]
            
            if len(cluster_coords) < self.min_keypoints:
                continue
            
            # Compute bounding box around cluster
            x_coords = cluster_coords[:, 0]
            y_coords = cluster_coords[:, 1]
            
            x1, y1 = float(x_coords.min()), float(y_coords.min())
            x2, y2 = float(x_coords.max()), float(y_coords.max())
            
            # Add padding
            padding = 10
            x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
            x2 = min(frame.shape[1], x2 + padding)
            y2 = min(frame.shape[0], y2 + padding)
            
            bbox = (x1, y1, x2, y2)
            centroid = (float(x_coords.mean()), float(y_coords.mean()))
            
            # Confidence based on keypoint strength
            avg_response = float(np.mean([kp.response for kp in cluster_kps]))
            confidence = min(1.0, avg_response)  # Normalize
            
            detections.append(Detection(
                bbox=bbox,
                centroid=centroid,
                confidence=confidence,
                label='sift_blob',
                source='sift',
                metadata={
                    'num_keypoints': len(cluster_kps),
                    'avg_response': avg_response,
                    'keypoints': cluster_kps
                }
            ))
        
        return detections
    
    def configure(self, **kwargs) -> None:
        """Configure SIFT detector parameters.
        
        Args:
            **kwargs: min_keypoints, cluster_eps, min_cluster_size
        """
        if 'min_keypoints' in kwargs:
            self.min_keypoints = kwargs['min_keypoints']
        if 'cluster_eps' in kwargs:
            self.cluster_eps = kwargs['cluster_eps']
        if 'min_cluster_size' in kwargs:
            self.min_cluster_size = kwargs['min_cluster_size']
        
        logger.debug(f"SIFTDetector configured: {kwargs}")
    
    def reset(self) -> None:
        """Reset SIFT detector (no state to reset)."""
        pass
    
    def get_source_name(self) -> str:
        """Get detector source name."""
        return 'sift'
