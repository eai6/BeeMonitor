"""BeeTracking - High-level tracking system for bee hotels with online learning.

Combines detection methods (FG/BG, YOLO) with MOT algorithms
to track bees in bee hotel videos. Includes adaptive threshold learning.
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
    """High-level tracking system for bee hotels with adaptive learning.
    
    Designed specifically for solitary bee hotels with:
    - Configurable detection pipeline (FG/BG, YOLO)
    - Noise filtering (CNN)
    - Pluggable MOT algorithm
    - Adaptive threshold learning (online learning from YOLO confirmations)
    - Adaptive mode switching (motion detection ↔ tracking)
    - Frame merging for efficiency
    
    Attributes:
        mot_algorithm: MOT algorithm (BeeTracker, ByteTrack, etc.)
        detection_mode: Which detectors to use
        blob_detector: FG/BG blob detector
        yolo_detector: YOLO detector
        noise_filter: CNN noise filter
        enable_online_learning: Whether to adapt thresholds during tracking
    """
    
    def __init__(
        self,
        mot_algorithm,
        yolo_model = None,
        detection_mode: DetectionMode = DetectionMode.FGBG_YOLO,
        use_noise_filter: bool = False,
        noise_filter_model = None,
        config = None,
        enable_online_learning: bool = True
    ):
        """Initialize BeeTracking system.
        
        Args:
            mot_algorithm: MOT algorithm (BeeTracker, ByteTrack, etc.)
            yolo_model: YOLO model for detection
            detection_mode: Which detection methods to use
            use_noise_filter: Whether to use CNN noise filter
            noise_filter_model: Noise filter classifier
            config: Configuration object
            enable_online_learning: Enable adaptive threshold learning (default: True)
        """
        self.mot = mot_algorithm
        self.detection_mode = detection_mode
        self.config = config
        self.enable_online_learning = enable_online_learning
        
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
        logger.info(f"  Online learning: {'ENABLED' if enable_online_learning else 'DISABLED'}")
    
    def _init_detectors(self, yolo_model, noise_filter_model, use_noise_filter):
        """Initialize detectors based on detection mode."""
        mode = self.detection_mode
        
        # FG/BG blob detector with RESEARCHED OPTIMAL defaults
        if mode in [DetectionMode.FGBG_ONLY, DetectionMode.FGBG_YOLO, 
                    DetectionMode.FGBG_SIFT, DetectionMode.FGBG_SIFT_YOLO]:
            # Use researched optimal defaults from ablation study
            self.blob_detector = BlobDetector(
                min_area=30.0,      # Researched optimal (conservative)
                min_solidity=0.56   # 80% of 0.7 (proven F1=53.0%)
            )
            logger.info(f"  Blob detector: RESEARCHED DEFAULTS (area=30.0, solidity=0.56)")
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
            roi: Region of interest (x1, y1, x2, y2)
            **kwargs: visualize, progress_callback, etc.
            
        Returns:
            DataFrame with tracking results
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames to process: {total_frames}")
        
        # Reset state
        self.reset()
        
        all_detections = []
        frame_num = 0
        last_print = 0
        print_interval = max(1, total_frames // 20)  # Print 20 progress updates
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Apply ROI if specified
            if roi:
                frame = self._apply_roi_mask(frame, roi)
            
            # Process frame
            frame_result = self.process_frame(frame, frame_num)
            
            # Progress logging
            if frame_num - last_print >= print_interval or frame_num == total_frames - 1:
                progress = (frame_num / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_num}/{total_frames} frames) - "
                      f"{len(frame_result['detections'])} detections, "
                      f"{len(frame_result['tracks'])} active tracks")
                last_print = frame_num
            
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
        
        print(f"\n✓ Video processing complete!")
        print(f"  Frames processed: {frame_num}")
        print(f"  Total track records: {len(all_detections)}")
        
        # Convert to DataFrame
        return self._convert_to_dataframe(all_detections)
    
    def process_frame(
        self,
        frame: np.ndarray,
        frame_num: int
    ) -> Dict[str, Any]:
        """Process single frame with online learning.
        
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
        
        # Online learning update (adapt thresholds from YOLO-confirmed bees)
        if self.enable_online_learning:
            self._update_online_learning(frame, frame_num, detections)
        
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
            
            # Diagnostic logging (only first time and when interesting)
            if frame_num == 0:
                print(f"\n🔍 FIRST FRAME DIAGNOSTICS:")
                print(f"  Blob detections: {len(blob_dets)}")
                print(f"  Thresholds: area≥{self.blob_detector.min_area:.1f}, "
                      f"solidity≥{self.blob_detector.min_solidity:.3f}")
                if len(blob_dets) == 0:
                    print(f"  ⚠ No blobs detected - may indicate:")
                    print(f"    • Background model too aggressive")
                    print(f"    • Thresholds too strict")
                    print(f"    • No motion in this frame")
            
            detections.extend(blob_dets)
        
        # SIFT stationary detection
        if self.sift_detector:
            sift_dets = self.sift_detector.detect(frame, use_templates=True)
            detections.extend(sift_dets)
        
        # Apply noise filter if enabled (to blob and SIFT detections)
        pre_filter_count = len(detections)
        if self.noise_filter and detections:
            detections = self.noise_filter.filter_detections(frame, detections)
            
            # Log filter effectiveness (only when interesting)
            if pre_filter_count > 0 and frame_num % 100 == 0:
                filtered = pre_filter_count - len(detections)
                print(f"  🧹 CNN Filter (frame {frame_num}): "
                      f"{pre_filter_count} → {len(detections)} "
                      f"(removed {filtered}, {filtered/pre_filter_count*100:.1f}%)")
        
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
    
    def _update_online_learning(
        self,
        frame: np.ndarray,
        frame_num: int,
        all_detections: List[Detection]
    ) -> None:
        """Update blob detector thresholds from YOLO-confirmed bees.
        
        This is called after each frame is processed. It matches blob
        detections to YOLO detections and updates the blob detector's
        thresholds based on confirmed bee characteristics.
        
        Args:
            frame: Current video frame
            frame_num: Frame number
            all_detections: All detections from current frame
        """
        # Only update if blob detector exists and has online learning
        if not hasattr(self, 'blob_detector') or self.blob_detector is None:
            return
        
        if not hasattr(self.blob_detector, 'online_learning_enabled'):
            return
        
        if not self.blob_detector.online_learning_enabled:
            return
        
        # Only update when YOLO runs (periodic confirmation)
        should_run_yolo = False
        
        if self.detection_mode in [DetectionMode.FGBG_YOLO, DetectionMode.YOLO_ONLY,
                                   DetectionMode.SIFT_YOLO, DetectionMode.FGBG_SIFT_YOLO]:
            should_run_yolo = (frame_num % 10 == 0)  # YOLO runs every 10 frames
        
        if not should_run_yolo:
            return
        
        # Get YOLO detections (ground truth)
        yolo_dets = [d for d in all_detections if d.source == 'yolo']
        
        if len(yolo_dets) == 0:
            return
        
        # Get blob detections
        blob_dets = [d for d in all_detections if d.source == 'fgbg']
        
        if len(blob_dets) == 0:
            return
        
        # Match blobs to YOLO (IoU > 0.3 = confirmed bee)
        for blob_det in blob_dets:
            for yolo_det in yolo_dets:
                iou = self.blob_detector.compute_iou(blob_det.bbox, yolo_det.bbox)
                if iou >= 0.3:
                    # This blob is YOLO-confirmed!
                    self.blob_detector.update_with_yolo_confirmation(blob_det, frame_num)
                    break  # Each blob only matched once
    
    def configure_detection(self, **kwargs) -> None:
        """Configure detection pipeline.
        
        Args:
            **kwargs: Configuration parameters
                blob_min_area: Minimum blob area
                blob_min_solidity: Minimum blob solidity
                yolo_conf: YOLO confidence threshold
                enable_online_learning: Enable/disable adaptive learning
        """
        if 'blob_min_area' in kwargs and self.blob_detector:
            self.blob_detector.configure(min_area=kwargs['blob_min_area'])
        
        if 'blob_min_solidity' in kwargs and self.blob_detector:
            self.blob_detector.configure(min_solidity=kwargs['blob_min_solidity'])
        
        if 'yolo_conf' in kwargs and self.yolo_detector:
            self.yolo_detector.configure(conf_threshold=kwargs['yolo_conf'])
        
        if 'enable_online_learning' in kwargs:
            self.enable_online_learning = kwargs['enable_online_learning']
            if hasattr(self, 'blob_detector') and self.blob_detector:
                if hasattr(self.blob_detector, 'enable_online_learning'):
                    self.blob_detector.enable_online_learning(self.enable_online_learning)
        
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
        """Get tracking statistics including online learning stats.
        
        Returns:
            Dictionary with tracking and learning statistics
        """
        stats = self.stats.copy()
        
        # Add online learning statistics if available
        if hasattr(self, 'blob_detector') and self.blob_detector:
            if hasattr(self.blob_detector, 'get_learning_stats'):
                learning_stats = self.blob_detector.get_learning_stats()
                stats['online_learning'] = learning_stats
        
        return stats
    
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