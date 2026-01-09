"""BeeTracking - High-level tracking system for bee hotels.

Combines detection methods (FG/BG, SIFT, YOLO) with MOT algorithms
to track bees in bee hotel videos.
"""

import logging
from typing import List, Dict, Optional, Any
from enum import Enum
import cv2
import numpy as np
import pandas as pd

from beemonitor.tracking.base_tracking import BaseTracking
from beemonitor.detection.base_detector import Detection
from beemonitor.detection.blob_detector import BlobDetector
from beemonitor.detection.sift_detector import SIFTDetector
from beemonitor.detection.yolo_detector import YOLODetector
from beemonitor.detection.noise_filter import NoiseFilter

logger = logging.getLogger(__name__)


class DetectionMode(Enum):
    """Detection modes for BeeTracking."""
    FGBG_ONLY = "fgbg"              # FG/BG blob detection only
    SIFT_ONLY = "sift"              # SIFT-based detection only
    FGBG_SIFT = "fgbg_sift"         # Both FG/BG and SIFT
    FGBG_YOLO = "fgbg_yolo"         # FG/BG with YOLO confirmation
    SIFT_YOLO = "sift_yolo"         # SIFT with YOLO confirmation
    FGBG_SIFT_YOLO = "fgbg_sift_yolo"  # All three methods
    YOLO_ONLY = "yolo"              # YOLO every frame (expensive)


class BeeTracking(BaseTracking):
    """High-level tracking system for bee hotels.
    
    Designed specifically for solitary bee hotels with:
    - Configurable detection pipeline (FG/BG, SIFT, YOLO)
    - Noise filtering (CNN)
    - Pluggable MOT algorithm
    - Adaptive mode switching (motion detection ↔ tracking)
    - Frame merging for efficiency
    
    Attributes:
        mot_algorithm: MOT algorithm (BeeTracker, ByteTrack, etc.)
        detection_mode: Which detectors to use
        blob_detector: FG/BG blob detector
        sift_detector: SIFT-based detector
        yolo_detector: YOLO detector
        noise_filter: CNN noise filter
    """
    
    def __init__(
        self,
        mot_algorithm,
        yolo_model = None,
        detection_mode: DetectionMode = DetectionMode.FGBG_YOLO,
        use_noise_filter: bool = True,
        noise_filter_model = None,
        config = None
    ):
        """Initialize BeeTracking system.
        
        Args:
            mot_algorithm: MOT algorithm (BeeTracker, ByteTrack, etc.)
            yolo_model: YOLO model for detection
            detection_mode: Which detection methods to use
            use_noise_filter: Whether to use CNN noise filter
            noise_filter_model: Noise filter classifier
            config: Configuration object
        """
        self.mot = mot_algorithm
        self.detection_mode = detection_mode
        self.config = config
        
        # Initialize detectors based on mode
        self._init_detectors(yolo_model, noise_filter_model, use_noise_filter)
        
        # Tracking state
        self.current_mode = 'motion_detection'  # or 'tracking'
        self.frames_without_tracks = 0
        self.motion_detection_threshold = 1
        self.tracking_to_detection_delay = 30
        
        # Frame merging for motion detection
        self.frame_merge_size = 10
        self.frame_buffer = []
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'total_detections': 0,
            'total_tracks': 0,
            'mode_switches': 0
        }
        
        logger.info(f"BeeTracking initialized")
        logger.info(f"  Detection mode: {detection_mode.value}")
        logger.info(f"  MOT: {type(mot_algorithm).__name__}")
        logger.info(f"  Noise filter: {use_noise_filter}")
    
    def _init_detectors(self, yolo_model, noise_filter_model, use_noise_filter):
        """Initialize detectors based on detection mode."""
        mode = self.detection_mode
        
        # FG/BG blob detector
        if mode in [DetectionMode.FGBG_ONLY, DetectionMode.FGBG_SIFT,
                    DetectionMode.FGBG_YOLO, DetectionMode.FGBG_SIFT_YOLO]:
            self.blob_detector = BlobDetector(
                min_area=50.0,
                min_solidity=0.5
            )
        else:
            self.blob_detector = None
        
        # SIFT detector
        if mode in [DetectionMode.SIFT_ONLY, DetectionMode.FGBG_SIFT,
                    DetectionMode.SIFT_YOLO, DetectionMode.FGBG_SIFT_YOLO]:
            self.sift_detector = SIFTDetector(
                min_keypoints=3,
                cluster_eps=30.0
            )
        else:
            self.sift_detector = None
        
        # YOLO detector
        if mode in [DetectionMode.FGBG_YOLO, DetectionMode.SIFT_YOLO,
                    DetectionMode.FGBG_SIFT_YOLO, DetectionMode.YOLO_ONLY]:
            if yolo_model is None:
                raise ValueError("YOLO model required for this detection mode")
            self.yolo_detector = YOLODetector(
                model=yolo_model,
                conf_threshold=0.25,
                tracking_classes=['bee', 'wasp'] if self.config is None 
                    else self.config.tracking.tracking_classes
            )
        else:
            self.yolo_detector = None
        
        # Noise filter
        if use_noise_filter and noise_filter_model is not None:
            self.noise_filter = NoiseFilter(
                classifier=noise_filter_model,
                threshold=0.7
            )
        else:
            self.noise_filter = None
    
    def process_video(
        self,
        video_path: str,
        roi: Optional[tuple] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Process entire video.
        
        Args:
            video_path: Path to video
            roi: Region of interest (x1, y1, x2, y2)
            **kwargs: visualize, progress_callback, etc.
            
        Returns:
            DataFrame with tracking results
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Processing {total_frames} frames from {video_path}")
        
        # Reset state
        self.reset()
        
        all_detections = []
        frame_num = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Apply ROI if specified
            if roi:
                frame = self._apply_roi_mask(frame, roi)
            
            # Process frame
            frame_result = self.process_frame(frame, frame_num)
            
            # Record tracks
            for track_id, track in frame_result['tracks'].items():
                all_detections.append({
                    'frame': frame_num,
                    'track_id': track_id,
                    'x1': track.bbox[0],
                    'y1': track.bbox[1],
                    'x2': track.bbox[2],
                    'y2': track.bbox[3],
                    'species': track.label,
                    'confidence': 1.0
                })
            
            frame_num += 1
            self.stats['total_frames'] = frame_num
        
        cap.release()
        
        # Convert to grouped format
        return self._convert_to_dataframe(all_detections)
    
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
            Dict with detections and tracks
        """
        # Detect objects in frame
        detections = self._detect_in_frame(frame, frame_num)
        
        # Update MOT
        from beemonitor.tracking.mot.base_mot import Detection as MOTDetection
        mot_detections = [
            MOTDetection(
                bbox=d.bbox,
                centroid=d.centroid,
                label=d.label,
                confidence=d.confidence,
                source=d.source
            )
            for d in detections
        ]
        
        # Check if MOT needs frame (Ultralytics trackers)
        try:
            from beemonitor.tracking.mot.ultralytics_tracker import UltralyticsTracker
            if isinstance(self.mot, UltralyticsTracker):
                tracks = self.mot.update(mot_detections, frame_num, frame=frame)
            else:
                tracks = self.mot.update(mot_detections, frame_num)
        except ImportError:
            tracks = self.mot.update(mot_detections, frame_num)
        
        self.stats['total_detections'] += len(detections)
        self.stats['total_tracks'] = max(self.stats['total_tracks'], len(tracks))
        
        return {
            'detections': detections,
            'tracks': tracks,
            'mode': self.current_mode
        }
    
    def _detect_in_frame(
        self,
        frame: np.ndarray,
        frame_num: int
    ) -> List[Detection]:
        """Run detection pipeline on frame.
        
        Args:
            frame: Input frame
            frame_num: Frame number
            
        Returns:
            List of detections
        """
        detections = []
        
        # FG/BG blob detection
        if self.blob_detector:
            blob_dets = self.blob_detector.detect(frame)
            detections.extend(blob_dets)
        
        # SIFT detection
        if self.sift_detector:
            sift_dets = self.sift_detector.detect(frame)
            detections.extend(sift_dets)
        
        # Apply noise filter if enabled
        if self.noise_filter and detections:
            detections = self.noise_filter.filter_detections(frame, detections)
        
        # YOLO confirmation/detection
        if self.yolo_detector:
            if self.detection_mode == DetectionMode.YOLO_ONLY:
                # Replace all with YOLO
                detections = self.yolo_detector.detect(frame)
            else:
                # Periodic YOLO confirmation
                if frame_num % 10 == 0:
                    yolo_dets = self.yolo_detector.detect(frame)
                    detections.extend(yolo_dets)
        
        return detections
    
    def configure_detection(self, **kwargs) -> None:
        """Configure detection pipeline."""
        if 'blob_min_area' in kwargs and self.blob_detector:
            self.blob_detector.configure(min_area=kwargs['blob_min_area'])
        
        if 'sift_min_keypoints' in kwargs and self.sift_detector:
            self.sift_detector.configure(min_keypoints=kwargs['sift_min_keypoints'])
        
        if 'yolo_conf' in kwargs and self.yolo_detector:
            self.yolo_detector.configure(conf_threshold=kwargs['yolo_conf'])
        
        logger.debug(f"Detection configured: {kwargs}")
    
    def configure_tracking(self, **kwargs) -> None:
        """Configure MOT algorithm."""
        # Delegate to MOT algorithm if it has configure method
        if hasattr(self.mot, 'configure'):
            self.mot.configure(**kwargs)
        
        logger.debug(f"Tracking configured: {kwargs}")
    
    def reset(self) -> None:
        """Reset tracking system."""
        if self.blob_detector:
            self.blob_detector.reset()
        if self.sift_detector:
            self.sift_detector.reset()
        if self.yolo_detector:
            self.yolo_detector.reset()
        
        self.mot.reset()
        
        self.frame_buffer.clear()
        self.current_mode = 'motion_detection'
        self.frames_without_tracks = 0
        
        self.stats = {
            'total_frames': 0,
            'total_detections': 0,
            'total_tracks': 0,
            'mode_switches': 0
        }
        
        logger.debug("BeeTracking reset")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get tracking statistics."""
        return self.stats.copy()
    
    def _apply_roi_mask(self, frame: np.ndarray, roi: tuple) -> np.ndarray:
        """Apply ROI mask to frame."""
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        x1, y1, x2, y2 = [int(c) for c in roi]
        mask[y1:y2, x1:x2] = 255
        return cv2.bitwise_and(frame, frame, mask=mask)
    
    def _convert_to_dataframe(self, detections: List[dict]) -> pd.DataFrame:
        """Convert detections to DataFrame."""
        if not detections:
            return pd.DataFrame(columns=['frame', 'track_id', 'x1', 'y1', 'x2', 'y2', 'species'])
        
        return pd.DataFrame(detections)
