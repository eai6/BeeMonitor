"""
BeeTracking - v2.3 YOLO-Only with ADAPTIVE Tracker
===================================================

Simplified bee tracking using YOLO-only detection with adaptive, resolution and FPS-independent tracking.

v2.3 Changes:
- Pass frame_height to tracker for resolution-relative fallback
- Robust bee size with IQR outlier rejection (in BeeTracker)

v2.2 Changes:
- YOLO-only detection (no FGBG_YOLO mode)
- Two-mode optimization (motion detection + YOLO tracking)
- Adaptive tracker with FPS and bee-size relative thresholds
- Auto-calculates bee size from detections
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
from ultralytics import YOLO

from beemonitor.detection.yolo_detector import YOLODetector
from beemonitor.detection.blob_detector import BlobDetector
from beemonitor.tracking.mot.bee_tracker import BeeTracker

logger = logging.getLogger(__name__)


class BeeTracking:
    """
    YOLO-only bee tracking with two-mode optimization and adaptive parameters.
    
    Features:
    - Motion detection mode (fast blob detection)
    - Tracking mode (YOLO when motion detected)
    - Adaptive tracker (resolution & FPS independent)
    - Auto-calculates bee size
    - Configurable distance multipliers for tuning
    """
    
    def __init__(
        self,
        yolo_model_path: str,
        confidence_threshold: float = 0.25,
        roi: Optional[Tuple[int, int, int, int]] = None,
        # Adaptive tracker parameters
        max_age_seconds: float = 1.0,
        min_hits_seconds: float = 0.1,
        max_resurrection_seconds: float = 0.5,
        match_distance_multiplier: float = 5.0,
        resurrection_search_multiplier: float = 3.0,
        duplicate_distance_multiplier: float = 1.2,
        iou_threshold: float = 0.3
    ):
        """
        Initialize BeeTracking with YOLO-only detection and adaptive tracking.
        
        Args:
            yolo_model_path: Path to YOLO model weights
            confidence_threshold: YOLO confidence threshold
            roi: Region of interest (x1, y1, x2, y2)
            
            Adaptive tracker parameters (resolution & FPS independent):
                max_age_seconds: Max time without detection before track dies
                min_hits_seconds: Min time before track is confirmed
                max_resurrection_seconds: Max time to resurrect dead tracks
                match_distance_multiplier: Max distance for matching (multiplier of bee size)
                resurrection_search_multiplier: Search radius (multiplier of bee size)
                duplicate_distance_multiplier: Duplicate threshold (multiplier of bee size)
                iou_threshold: IoU threshold for matching
        """
        logger.info("Initializing BeeTracking (v2.3 YOLO-only with adaptive tracker)")
        
        # YOLO detector
        logger.info(f"Loading YOLO model from {yolo_model_path}")

        # self.yolo_detector = YOLODetector(
        #     model_path=yolo_model_path,
        #     confidence_threshold=confidence_threshold
        # )

        yolo_model = YOLO(yolo_model_path)
        self.yolo_detector = YOLODetector(
            model=yolo_model,
            conf_threshold=confidence_threshold,
            iou_threshold=iou_threshold
        )
        
        # Blob detector (for motion detection mode)
        logger.info("Initializing blob detector for motion detection")
        self.blob_detector = BlobDetector()
        
        # ROI
        self.roi = roi
        
        # Video properties (will be set when video is opened)
        self.fps = None
        self.video_width = None
        self.video_height = None
        
        # Store tracker parameters for initialization
        self.tracker_params = {
            'max_age_seconds': max_age_seconds,
            'min_hits_seconds': min_hits_seconds,
            'max_resurrection_seconds': max_resurrection_seconds,
            'match_distance_multiplier': match_distance_multiplier,
            'resurrection_search_multiplier': resurrection_search_multiplier,
            'duplicate_distance_multiplier': duplicate_distance_multiplier,
            'iou_threshold': iou_threshold
        }
        
        # Tracker (will be initialized when video properties are known)
        self.tracker = None
        
        # Two-mode system state
        self.mode = 'motion_detection'  # 'motion_detection' or 'tracking'
        self.frames_since_motion = 0
        self.motion_cooldown = 30  # Frames to stay in tracking mode after motion stops
        
        logger.info("BeeTracking initialized (v2.2 YOLO-only with adaptive tracker)")
    
    def _initialize_tracker(self, fps: float, frame_height: int = None):
        """
        Initialize adaptive tracker with video FPS and frame height.
        
        Args:
            fps: Video frame rate
            frame_height: Video frame height for resolution-relative fallback
        """
        logger.info(f"Initializing adaptive tracker with FPS={fps}, frame_height={frame_height}")
        
        self.tracker = BeeTracker(
            fps=fps,
            bee_size=None,  # Auto-calculate from detections
            frame_height=frame_height,  # For resolution-relative fallback
            **self.tracker_params
        )
        
        logger.info("Adaptive tracker initialized")
    
    def _get_video_properties(self, video_path: str) -> Tuple[float, int, int]:
        """
        Get video properties (FPS, width, height).
        
        Args:
            video_path: Path to video file
            
        Returns:
            (fps, width, height)
        """
        cap = cv2.VideoCapture(video_path)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cap.release()
        
        # Default to 30fps if unable to read
        if fps <= 0:
            logger.warning(f"Could not read FPS from video, defaulting to 30.0")
            fps = 30.0
        
        logger.info(f"Video properties: {width}x{height} @ {fps} fps")
        
        return fps, width, height
    
    def initialize_video(self, video_path: str, output_path: Optional[str] = None):
        """
        Initialize tracking for a specific video.
        
        Args:
            video_path: Path to video file
            output_path: Optional path to output folder
        """
        logger.info(f"Initializing video tracking for: {video_path}")
        if output_path:
            logger.info(f"Output path: {output_path}")
        
        # Get video properties
        self.fps, self.video_width, self.video_height = self._get_video_properties(video_path)
        logger.info(f"Video properties: {self.video_width}x{self.video_height} @ {self.fps} fps")
        
        # Initialize adaptive tracker with video FPS and frame height
        self._initialize_tracker(self.fps, self.video_height)
        
        # Initialize blob detector background
        # Note: BlobDetector builds background model internally from detect() calls
        # We don't need to manually feed it frames
        logger.info("Blob detector initialized (will build background on first detect() call)")
        
        # Optionally save initial background if output_path is provided
        # This will be saved after the first few frames are processed
        if output_path:
            import os
            os.makedirs(output_path, exist_ok=True)
            self.background_save_path = os.path.join(output_path, "background.png")
            logger.info(f"Background will be saved to: {self.background_save_path}")
        else:
            self.background_save_path = None
    
    def detect_motion(self, frame: np.ndarray) -> bool:
        """
        Detect if there is motion in the frame.
        
        Args:
            frame: Input frame
            
        Returns:
            True if motion detected, False otherwise
        """
        # Get blob detections (fast)
        blob_detections = self.blob_detector.detect(frame)
        
        # Motion detected if we have any blobs
        return len(blob_detections) > 0
    
    def process_frame(
        self,
        frame: np.ndarray,
        frame_num: int,
        visualize: bool = False
    ) -> Dict[str, Any]:
        """
        Process a single frame with two-mode optimization.
        
        Args:
            frame: Input frame
            frame_num: Frame number
            visualize: Whether to create visualization
            
        Returns:
            Dictionary with detections, tracks, and mode info
        """
        # Apply ROI for motion detection only
        if self.roi:
            x1, y1, x2, y2 = self.roi
            roi_frame = frame[y1:y2, x1:x2]
        else:
            roi_frame = frame
        
        detections = []
        
        # TWO-MODE SYSTEM
        if self.mode == 'motion_detection':
            # Fast motion detection mode (ROI only)
            has_motion = self.detect_motion(roi_frame)
            
            if has_motion:
                # Motion detected - switch to tracking mode
                logger.debug(f"Frame {frame_num}: Motion detected, switching to tracking mode")
                self.mode = 'tracking'
                self.frames_since_motion = 0
                
                # Run YOLO on FULL FRAME (allows tracking beyond ROI)
                yolo_detections = self.yolo_detector.detect(frame)
                
                # Convert to tracking format and add source
                for det in yolo_detections:
                    bbox = det.bbox
                    conf = det.confidence
                    det_with_source = list(bbox) + [conf, 'yolo']
                    detections.append(det_with_source)
        
        else:  # tracking mode
            # Run YOLO on FULL FRAME (allows tracking beyond ROI)
            yolo_detections = self.yolo_detector.detect(frame)
            
            # Convert to tracking format and add source
            for det in yolo_detections:
                bbox = det.bbox
                conf = det.confidence
                det_with_source = list(bbox) + [conf, 'yolo']
                detections.append(det_with_source)
            
            # Check if motion has stopped (ROI only)
            has_motion = self.detect_motion(roi_frame)
            
            if has_motion:
                self.frames_since_motion = 0
            else:
                self.frames_since_motion += 1
            
            # Switch back to motion detection mode if no motion for cooldown period
            if self.frames_since_motion > self.motion_cooldown:
                logger.debug(f"Frame {frame_num}: No motion for {self.motion_cooldown} frames, switching to motion detection mode")
                self.mode = 'motion_detection'
        
        # Update tracker (full frame coordinates, no adjustment needed)
        tracks = []
        if self.tracker is not None:
            tracks = self.tracker.update(detections, frame_num)
        
        result = {
            'frame_num': frame_num,
            'detections': detections,
            'tracks': tracks,
            'mode': self.mode,
            'num_detections': len(detections),
            'num_tracks': len(tracks)
        }
        
        if visualize:
            result['visualization'] = self._create_visualization(frame, detections, tracks)
        
        return result
    
    def _create_visualization(
        self,
        frame: np.ndarray,
        detections: List,
        tracks: List[Dict]
    ) -> np.ndarray:
        """
        Create visualization of detections and tracks.
        
        Args:
            frame: Input frame
            detections: List of detections
            tracks: List of tracks
            
        Returns:
            Annotated frame
        """
        vis_frame = frame.copy()
        
        # Draw ROI
        if self.roi:
            x1, y1, x2, y2 = self.roi
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw detections (BLUE for YOLO)
        for det in detections:
            x1, y1, x2, y2 = map(int, det[:4])
            source = det[5] if len(det) > 5 else 'unknown'
            
            color = (255, 0, 0) if source == 'yolo' else (128, 128, 128)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis_frame, source.upper(), (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw tracks
        colors = [(255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
        
        for i, track in enumerate(tracks):
            color = colors[i % len(colors)]
            track_id = track['track_id']
            cx, cy = map(int, track['centroid'])
            
            # Draw trajectory
            if len(track['history']) > 1:
                points = np.array(track['history'], dtype=np.int32)
                cv2.polylines(vis_frame, [points], False, color, 2)
            
            # Draw current position
            cv2.circle(vis_frame, (cx, cy), 5, color, -1)
            cv2.putText(vis_frame, f"ID:{track_id}", (cx + 10, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw mode indicator
        mode_text = f"Mode: {self.mode.upper()}"
        cv2.putText(vis_frame, mode_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return vis_frame
    
    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        visualize: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Process entire video.
        
        Args:
            video_path: Path to input video
            output_path: Path to output video (if visualize=True)
            visualize: Whether to create output video
            
        Returns:
            List of results for each frame
        """
        # Initialize for this video
        self.initialize_video(video_path, output_path)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        # Setup video writer if visualizing
        if visualize and output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                output_path,
                fourcc,
                self.fps,
                (self.video_width, self.video_height)
            )
        
        results = []
        frame_num = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            result = self.process_frame(frame, frame_num, visualize=visualize)
            results.append(result)
            
            # Write visualization if enabled
            if visualize and output_path and 'visualization' in result:
                out.write(result['visualization'])
            
            frame_num += 1
            
            if frame_num % 100 == 0:
                logger.info(f"Processed {frame_num} frames")
        
        cap.release()
        if visualize and output_path:
            out.release()
        
        logger.info(f"Processed {frame_num} frames total")
        
        # Convert results to DataFrame
        import pandas as pd
        
        if not results:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=['frame', 'frame_num', 'track_id', 
                                        'x1', 'y1', 'x2', 'y2', 'cx', 'cy',
                                        'bbox', 'centroid', 'confidence', 'mode'])
        
        # Flatten results - one row per track per frame
        flattened_rows = []
        
        for result in results:
            frame_num = result['frame_num']
            mode = result['mode']
            
            # Extract each track into a separate row
            for track in result['tracks']:
                # Get track ID - should be 'track_id' based on debug output
                track_id = track.get('track_id')
                
                if track_id is None:
                    logger.warning(f"Track missing track_id! Track keys: {track.keys()}")
                    continue  # Skip tracks without ID
                
                # Unpack bbox (x1, y1, x2, y2)
                bbox = track['bbox']
                x1, y1, x2, y2 = bbox
                
                # Unpack centroid (cx, cy)
                centroid = track['centroid']
                cx, cy = centroid
                
                row = {
                    'frame': frame_num,
                    'frame_num': frame_num,
                    'track_id': track_id,
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'cx': cx,
                    'cy': cy,
                    'bbox': bbox,  # Keep original bbox too
                    'centroid': centroid,  # Keep original centroid too
                    'confidence': track.get('confidence', 0.0),
                    'mode': mode
                }
                flattened_rows.append(row)
        
        # Create DataFrame from flattened rows
        if not flattened_rows:
            # No tracks detected
            logger.warning("No tracks detected in video")
            return pd.DataFrame(columns=['frame', 'frame_num', 'track_id', 
                                        'x1', 'y1', 'x2', 'y2', 'cx', 'cy',
                                        'bbox', 'centroid', 'confidence', 'mode'])
        
        df = pd.DataFrame(flattened_rows)
        
        logger.info(f"Created tracking DataFrame with {len(df)} rows ({len(results)} frames, "
                   f"{len(df['track_id'].unique())} unique tracks)")
        
        return df