"""Detection module for bee monitoring.

This module provides various detection methods for finding objects in images.
All detectors implement the BaseDetector interface for consistent API.

Available Detectors:
- BlobDetector: FG/BG motion-based blob detection
- SIFTDetector: SIFT keypoint clustering (detects stationary bees!)
- YOLODetector: Deep learning object detection
- NestDetector: Nest tube detection
- NoiseFilter: CNN-based false positive filtering

Example:
    >>> from beemonitor.detection import BlobDetector, YOLODetector
    >>> blob_det = BlobDetector(min_area=50)
    >>> yolo_det = YOLODetector(model)
    >>> 
    >>> # Get detections
    >>> blob_dets = blob_det.detect(frame)
    >>> yolo_dets = yolo_det.detect(frame)
"""

from beemonitor.detection.base_detector import BaseDetector, Detection
from beemonitor.detection.blob_detector import BlobDetector
from beemonitor.detection.sift_detector import SIFTDetector
from beemonitor.detection.yolo_detector import YOLODetector
from beemonitor.detection.nest_detector import NestDetector
from beemonitor.detection.noise_filter import NoiseFilter, BeeNoiseFilter

__all__ = [
    'BaseDetector',
    'Detection',
    'BlobDetector',
    'SIFTDetector',
    'YOLODetector',
    'NestDetector',
    'NoiseFilter',
    'BeeNoiseFilter',
]