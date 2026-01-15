"""BeeTracking - High-level tracking system for bee hotels.

Combines detection methods (FG/BG, YOLO) with MOT algorithms
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
    """Detection modes for BeeTracking.
    
    Modes:
        FGBG_ONLY: Motion detection only (fast, misses stationary bees)
        SIFT_ONLY: SIFT keypoint detection only (slower, finds stationary)
        YOLO_ONLY: Deep learning every frame (slowest, most accurate)
        FGBG_SIFT: Motion + SIFT (balanced speed + stationary detection)
        FGBG_YOLO: Motion + periodic YOLO (good balance)
        SIFT_YOLO: SIFT + periodic YOLO (stationary + species ID)
        FGBG_SIFT_YOLO: All three methods (comprehensive, RECOMMENDED)
    """
    FGBG_ONLY = "fgbg"                      # FG/BG blob detection only
    SIFT_ONLY = "sift"                      # SIFT keypoint detection only
    YOLO_ONLY = "yolo"                      # YOLO every frame
    FGBG_SIFT = "fgbg_sift"                 # FG/BG + SIFT
    FGBG_YOLO = "fgbg_yolo"                 # FG/BG + YOLO confirmation
    SIFT_YOLO = "sift_yolo"                 # SIFT + YOLO confirmation
    FGBG_SIFT_YOLO = "fgbg_sift_yolo"       # All three (RECOMMENDED)


class BeeTracking(BaseTracking):
    """High-level tracking system for bee hotels.
    
    Designed specifically for solitary bee hotels with:
    - Configurable detection pipeline (FG/BG, YOLO)
    - Noise filtering (CNN)
    - Pluggable MOT algorithm
    - Adaptive mode switching (motion detection ↔ tracking)
    - Frame merging for efficiency
    
    Attributes:
        mot_algorithm: MOT algorithm (BeeTracker, ByteTrack, etc.)
        detection_mode: Which detectors to use
        blob_detector: FG/BG blob detector
        yolo_detector: YOLO detector
        noise_filter: CNN noise filter
    """
    
    def __init__(
        self,
        mot_algorithm,
        yolo_model = None,
        detection_mode: DetectionMode = DetectionMode.FGBG_YOLO,
        use_noise_filter: bool = False,
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
        
        # Two-mode tracking optimization (configurable)
        if config and hasattr(config, 'tracking'):
            self.enable_two_mode = config.tracking.enable_two_mode_tracking
            self.motion_detection_threshold = config.tracking.motion_detection_threshold
            self.tracking_to_detection_delay = config.tracking.tracking_to_detection_delay
            self.frame_merge_size = config.tracking.motion_mode_frame_merge
        else:
            # Default values if no config
            self.enable_two_mode = True
            self.motion_detection_threshold = 1
            self.tracking_to_detection_delay = 30
            self.frame_merge_size = 10
        
        # Tracking state - ALWAYS start in tracking mode to catch initial activity
        self.current_mode = 'tracking'
        self.frames_without_tracks = 0
        self.frame_buffer = []
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'total_detections': 0,
            'total_tracks': 0,
            'mode_switches': 0,
            'frames_in_motion_mode': 0,
            'frames_in_tracking_mode': 0
        }
        
        logger.info(f"BeeTracking initialized")
        logger.info(f"  Detection mode: {detection_mode.value}")
        logger.info(f"  MOT: {type(mot_algorithm).__name__}")
        logger.info(f"  Noise filter: {use_noise_filter}")
        logger.info(f"  Two-mode tracking: {'ENABLED' if self.enable_two_mode else 'DISABLED'}")
        if self.enable_two_mode:
            logger.info(f"    Motion threshold: {self.motion_detection_threshold}")
            logger.info(f"    Switch delay: {self.tracking_to_detection_delay} frames")
    
    def _init_detectors(self, yolo_model, noise_filter_model, use_noise_filter):
        """Initialize detectors based on detection mode."""
        mode = self.detection_mode
        
        # FG/BG blob detector
        if mode in [DetectionMode.FGBG_ONLY, DetectionMode.FGBG_YOLO, 
                    DetectionMode.FGBG_SIFT, DetectionMode.FGBG_SIFT_YOLO]:
            self.blob_detector = BlobDetector(
                min_area=50.0,
                min_solidity=0.5
            )
        else:
            self.blob_detector = None
        
        # SIFT detector (for stationary bee detection)
        # NOTE: Must be initialized with learn_from_video() before use!
        if mode in [DetectionMode.SIFT_ONLY, DetectionMode.FGBG_SIFT,
                    DetectionMode.SIFT_YOLO, DetectionMode.FGBG_SIFT_YOLO]:
            self.sift_detector = SIFTDetector(
                min_keypoints=3,
                use_templates=True,
                require_movement=True  # Filter out static nest holes
            )
            logger.info("  SIFT detector created (needs initialization before use)")
        else:
            self.sift_detector = None
        
        # YOLO detector
        if mode in [DetectionMode.FGBG_YOLO, DetectionMode.YOLO_ONLY,
                    DetectionMode.SIFT_YOLO, DetectionMode.FGBG_SIFT_YOLO]:
            if yolo_model is None:
                raise ValueError("YOLO model required for this detection mode")
            
            # Convert class IDs to class names
            if self.config is None:
                tracking_classes = ['bee', 'wasp']
            else:
                tracking_classes = []
                if hasattr(self.config.tracking, 'label_map'):
                    for class_id in self.config.tracking.tracking_classes:
                        class_name = self.config.tracking.label_map.get(class_id, f'class_{class_id}')
                        tracking_classes.append(class_name)
                else:
                    tracking_classes = [str(cid) for cid in self.config.tracking.tracking_classes]
            
            self.yolo_detector = YOLODetector(
                model=yolo_model,
                conf_threshold=0.25,
                tracking_classes=tracking_classes
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
            roi: Region of interest (x1, y1, x2, y2) - NOT used for masking.
                 Detections and tracking happen in full frame to allow
                 tracking bees that move outside the hotel region.
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
            
            # Note: ROI is NOT applied to frame masking
            # This allows tracking to follow bees outside the hotel region
            # Detections happen in full frame, tracking persists everywhere
            
            # Process frame (full frame, no ROI masking)
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
                    'confidence': 1.0,
                    'source': track.source if hasattr(track, 'source') else 'unknown'
                })
            
            frame_num += 1
            self.stats['total_frames'] = frame_num
        
        cap.release()
        
        # Convert to DataFrame
        return self._convert_to_dataframe(all_detections)
    
    def process_frame(
        self,
        frame: np.ndarray,
        frame_num: int
    ) -> Dict[str, Any]:
        """Process single frame with optional two-mode optimization.
        
        Two-Mode Tracking:
        - Motion Detection Mode: Lightweight motion check only (fast)
        - Tracking Mode: Full detection + tracking (comprehensive)
        - Switches based on bee activity
        
        Args:
            frame: Input frame
            frame_num: Frame number
            
        Returns:
            Dict with detections, tracks, and current mode
        """
        # Check if two-mode tracking is enabled
        if not self.enable_two_mode:
            # Original behavior: full tracking every frame
            return self._process_full_tracking(frame, frame_num)
        
        # Two-mode optimization enabled
        if self.current_mode == 'motion_detection':
            # Lightweight mode: just check for motion
            return self._process_motion_detection_mode(frame, frame_num)
        else:
            # Full mode: complete detection + tracking
            return self._process_tracking_mode(frame, frame_num)
    
    def _process_full_tracking(
        self,
        frame: np.ndarray,
        frame_num: int
    ) -> Dict[str, Any]:
        """Process frame with full tracking (original behavior).
        
        Used when two-mode tracking is disabled.
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
            'mode': 'full_tracking'
        }
    
    def _process_motion_detection_mode(
        self,
        frame: np.ndarray,
        frame_num: int
    ) -> Dict[str, Any]:
        """Process frame in lightweight motion detection mode.
        
        Only checks for motion using blob detector with RELAXED thresholds.
        If motion detected, switches to tracking mode.
        """
        self.stats['frames_in_motion_mode'] += 1
        
        # Lightweight detection: blob only with MOTION MODE thresholds
        # More sensitive to catch ANY movement and trigger switch
        detections = []
        if self.blob_detector:
            blob_dets = self.blob_detector.detect(frame, mode='motion')
            detections.extend(blob_dets)
        
        # Check if we should switch to tracking mode
        if len(detections) >= self.motion_detection_threshold:
            # Motion detected! Switch to tracking mode
            self._switch_to_tracking_mode()
            # Process this frame in tracking mode
            return self._process_tracking_mode(frame, frame_num)
        
        # No motion - stay in motion detection mode
        # Return empty tracks
        return {
            'detections': detections,
            'tracks': {},
            'mode': 'motion_detection'
        }
    
    def _process_tracking_mode(
        self,
        frame: np.ndarray,
        frame_num: int
    ) -> Dict[str, Any]:
        """Process frame in full tracking mode.
        
        Runs complete detection pipeline + MOT tracking.
        If no tracks for N frames, switches back to motion detection mode.
        """
        self.stats['frames_in_tracking_mode'] += 1
        
        # Full detection pipeline
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
        
        # Check if we should switch to motion detection mode
        if len(tracks) == 0:
            self.frames_without_tracks += 1
            if self.frames_without_tracks >= self.tracking_to_detection_delay:
                # No tracks for N frames - switch to motion detection mode
                self._switch_to_motion_detection_mode()
        else:
            # Reset counter when tracks are present
            self.frames_without_tracks = 0
        
        return {
            'detections': detections,
            'tracks': tracks,
            'mode': 'tracking'
        }
    
    def _switch_to_tracking_mode(self):
        """Switch from motion detection mode to tracking mode."""
        if self.current_mode != 'tracking':
            logger.info(f"Mode switch: motion_detection → tracking (motion detected)")
            self.current_mode = 'tracking'
            self.frames_without_tracks = 0
            self.stats['mode_switches'] += 1
    
    def _switch_to_motion_detection_mode(self):
        """Switch from tracking mode to motion detection mode."""
        if self.current_mode != 'motion_detection':
            logger.info(f"Mode switch: tracking → motion_detection ({self.frames_without_tracks} frames without tracks)")
            self.current_mode = 'motion_detection'
            self.frames_without_tracks = 0
            self.stats['mode_switches'] += 1
    
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
        
        # FG/BG blob detection with TRACKING MODE thresholds (precise)
        if self.blob_detector:
            blob_dets = self.blob_detector.detect(frame, mode='tracking')
            detections.extend(blob_dets)
        
        # SIFT stationary detection
        if self.sift_detector:
            sift_dets = self.sift_detector.detect(frame, use_templates=True)
            detections.extend(sift_dets)
        
        # Apply noise filter if enabled (to blob and SIFT detections)
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
        if self.yolo_detector:
            self.yolo_detector.reset()
        
        self.mot.reset()
        
        self.frame_buffer.clear()
        # ALWAYS start in tracking mode to catch initial activity
        self.current_mode = 'tracking'
        self.frames_without_tracks = 0
        
        self.stats = {
            'total_frames': 0,
            'total_detections': 0,
            'total_tracks': 0,
            'mode_switches': 0,
            'frames_in_motion_mode': 0,
            'frames_in_tracking_mode': 0
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
            return pd.DataFrame(columns=['frame', 'track_id', 'x1', 'y1', 'x2', 'y2', 'species', 'confidence', 'source'])
        
        return pd.DataFrame(detections)