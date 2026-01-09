"""Motion detection system for bee tracking.

Handles:
- Background subtraction (MOG2)
- Blob detection and filtering
- ROI masking
- YOLO inference with auto device detection (CUDA/MPS/CPU)
- Adaptive blob size filtering
- Frame merging for motion detection

Delegates tracking to pluggable MOT algorithms via BaseMOT interface.
"""

import logging
from typing import List, Tuple, Optional
import cv2
import numpy as np
import pandas as pd
import os

from beemonitor.core.config import Config
from multiple_object_tracking.base_mot import BaseMOT, Detection
from multiple_object_tracking.bee_tracker import BeeTracker

logger = logging.getLogger(__name__)

# Type aliases
BBox = Tuple[float, float, float, float]
Point = Tuple[float, float]


class MotionTracking:
    """Bee tracking system with motion detection and pluggable MOT backend."""
    
    def __init__(
        self,
        model,
        config: Optional[Config] = None,
        mot_algorithm: Optional[BaseMOT] = None,
        use_gpu: Optional[bool] = None,
        fast_mode: bool = True
    ):
        """Initialize motion tracking system.
        
        Args:
            model: YOLO model for detection
            config: Configuration object
            mot_algorithm: MOT algorithm (default: KalmanMOT)
            use_gpu: Use GPU if available (default: auto-detect)
            fast_mode: Enable performance optimizations (default: True)
        """
        self.model = model
        self.config = config if config is not None else Config.default()
        self.fast_mode = fast_mode
        
        # Auto-detect and configure device
        if use_gpu is None:
            self.device = self._detect_device()
        else:
            self.device = 'cuda' if use_gpu else 'cpu'
        
        self._configure_yolo_device()
        
        # Initialize MOT algorithm
        if mot_algorithm is None:
            self.mot = BeeTracker(
                config=self.config,
                tracking_classes=self.config.tracking.tracking_classes
            )
        else:
            self.mot = mot_algorithm
        
        # Initialize background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=False
        )
        
        # Initialize AI noise filter (optional)
        self.noise_filter = None
        self._init_noise_filter()
        
        # Mode state
        self.current_mode = 'motion_detection'
        self.frames_without_tracks = 0
        self.motion_detection_threshold = 1
        self.tracking_to_detection_delay = 30
        
        # Frame merging
        self.frame_merge_size = 10
        self.frame_buffer = []
        self.frame_buffer_start = 0
        
        # Adaptive blob filtering
        self.recorded_bee_areas = []
        self.min_blob_area_dynamic = config.tracking.min_bee_blob_area
        
        # Species mapping
        self.label_map = self.config.tracking.label_map
        self.tracking_classes = self.config.tracking.tracking_classes
        
        logger.info(f"Motion detection initialized")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  MOT: {type(self.mot).__name__}")
        logger.info(f"  Tracking classes: {self.tracking_classes}")
        
        if self.fast_mode:
            logger.info(f"  [FAST] YOLO resolution: 0.5x")
            logger.info(f"  [FAST] Morphology: 3x3 kernel, 1 iteration")
    
    def _detect_device(self) -> str:
        """Auto-detect best available device (MPS/CUDA/CPU)."""
        try:
            import torch
            
            # Check for Apple Silicon MPS
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                logger.info("Apple MPS detected")
                return 'mps'
            
            # Check for NVIDIA CUDA
            if torch.cuda.is_available():
                logger.info(f"CUDA detected: {torch.cuda.get_device_name(0)}")
                return 'cuda'
            
            logger.info("No GPU detected, using CPU")
            return 'cpu'
            
        except ImportError:
            logger.warning("PyTorch not available, using CPU")
            return 'cpu'
    
    def _configure_yolo_device(self):
        """Configure YOLO model to use detected device."""
        try:
            # Map device names to YOLO format
            if self.device == 'cuda':
                device_id = 0  # Use first GPU
            elif self.device == 'mps':
                device_id = 'mps'
            else:
                device_id = 'cpu'
            
            self.model.to(device_id)
            logger.info(f"YOLO model moved to: {device_id}")
            
        except Exception as e:
            logger.warning(f"Failed to move YOLO to {self.device}: {e}")
            logger.warning("Falling back to CPU")
            self.device = 'cpu'
    
    def _init_noise_filter(self):
        """Initialize optional AI noise filter."""
        try:
            from beemonitor.ml.bee_noise_filter import BeeNoiseFilter
            
            noise_filter_path = self.config.model.blob_noise_classifier
            
            if os.path.exists(noise_filter_path):
                self.noise_filter = BeeNoiseFilter(
                    model_path=noise_filter_path,
                    device='cpu',
                    noise_threshold=0.7,
                    image_size=64
                )
                logger.info(f"AI noise filter loaded: {noise_filter_path}")
            else:
                logger.warning(f"Noise filter model not found: {noise_filter_path}")
        except ImportError as e:
            logger.warning(f"BeeNoiseFilter not available: {e}")
    
    def process_video(
        self,
        video_path: str,
        roi_coords: Optional[BBox] = None,
        progress_callback=None,
        visualize: bool = False
    ) -> pd.DataFrame:
        """Process video and extract bee tracks.
        
        Args:
            video_path: Path to video file
            roi_coords: ROI as (x1, y1, x2, y2)
            progress_callback: Optional callback(frame_num, total_frames)
            visualize: Return visualization frames
            
        Returns:
            DataFrame with tracking results
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Processing {total_frames} frames from {video_path}")
        
        # Expand ROI with padding
        if roi_coords:
            roi_coords = self._expand_roi(roi_coords, padding=200)
            logger.info(f"ROI with padding: {roi_coords}")
        
        # Reset tracker
        self.mot.reset()
        self.frame_buffer.clear()
        self.current_mode = 'motion_detection'
        
        all_detections = []
        viz_frames = [] if visualize else None
        frame_num = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Apply ROI mask
            if roi_coords:
                frame = self._apply_roi_mask(frame, roi_coords)
            
            # Process frame
            detections = self._process_frame(frame, frame_num, roi_coords)
            
            # Convert to Detection objects
            det_objects = []
            for det in detections:
                det_objects.append(Detection(
                    bbox=det['bbox'],
                    centroid=det['centroid'],
                    label=det['label'],
                    confidence=det.get('confidence', 1.0),
                    source=det.get('source', 'unknown')
                ))
            
            # Update MOT
            # Check if tracker needs frame (Ultralytics trackers)
            from ultralytics_tracker import UltralyticsTracker
            if isinstance(self.mot, UltralyticsTracker):
                tracks = self.mot.update(det_objects, frame_num, frame=frame)
            else:
                tracks = self.mot.update(det_objects, frame_num)
            
            # Record detections
            for track_id, track in tracks.items():
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
            
            # Visualization
            if visualize:
                viz = self._visualize_frame(frame, tracks, frame_num, detections)
                viz_frames.append(viz)
            
            # Progress callback
            if progress_callback:
                progress_callback(frame_num, total_frames)
            
            frame_num += 1
        
        cap.release()
        
        # Convert to grouped format
        df = self._convert_to_grouped_format(all_detections)
        
        if visualize:
            return df, viz_frames
        return df
    
    def _process_frame(
        self,
        frame: np.ndarray,
        frame_num: int,
        roi_coords: Optional[BBox] = None
    ) -> List[dict]:
        """Process single frame for motion/tracking.
        
        Returns:
            List of detections with bbox, centroid, label, source
        """
        # Add to frame buffer for merging
        self.frame_buffer.append((frame_num, frame.copy()))
        if len(self.frame_buffer) > self.frame_merge_size:
            self.frame_buffer.pop(0)
        
        # Mode switching logic
        if self.current_mode == 'motion_detection':
            # Check if we should switch to tracking
            if len(self.frame_buffer) == self.frame_merge_size:
                blobs = self._detect_motion_blobs(self.frame_buffer)
                
                if len(blobs) >= self.motion_detection_threshold:
                    self.current_mode = 'tracking'
                    self.frames_without_tracks = 0
                    logger.info(f"Frame {frame_num}: Switched to tracking mode ({len(blobs)} blobs)")
                    
                    # Return blob detections to initialize tracks
                    return [self._blob_to_detection(blob, 'motion') for blob in blobs]
            
            return []
        
        else:  # tracking mode
            # Detect motion blobs
            blobs = self._detect_single_frame_blobs(frame)
            
            # Run YOLO periodically
            yolo_dets = []
            if frame_num % 10 == 0:  # Every 10 frames
                yolo_dets = self._run_yolo(frame, roi_coords)
            
            # Combine detections
            all_dets = [self._blob_to_detection(b, 'motion') for b in blobs]
            all_dets.extend([self._yolo_to_detection(d) for d in yolo_dets])
            
            # Check if we should switch back to motion detection
            if len(self.mot.get_tracks()) == 0:
                self.frames_without_tracks += 1
                if self.frames_without_tracks >= self.tracking_to_detection_delay:
                    self.current_mode = 'motion_detection'
                    self.frame_buffer.clear()
                    logger.info(f"Frame {frame_num}: Switched to motion detection mode")
            else:
                self.frames_without_tracks = 0
            
            return all_dets
    
    def _detect_motion_blobs(self, frame_buffer: List[Tuple[int, np.ndarray]]) -> List[dict]:
        """Detect motion blobs from merged frames."""
        # Merge frames
        merged = self._merge_frames([f for _, f in frame_buffer])
        
        # Background subtraction
        fg_mask = self.bg_subtractor.apply(merged)
        
        # Morphological operations
        if self.fast_mode:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and convert to blobs
        blobs = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_blob_area_dynamic:
                continue
            
            # Solidity filter
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = area / hull_area
                if solidity < 0.5:
                    continue
            
            # Bounding box
            x, y, w, h = cv2.boundingRect(contour)
            bbox = (x, y, x + w, y + h)
            centroid = (x + w / 2, y + h / 2)
            
            blobs.append({
                'bbox': bbox,
                'centroid': centroid,
                'area': area,
                'contour': contour
            })
        
        # Update dynamic area threshold
        if blobs:
            areas = [b['area'] for b in blobs]
            self.recorded_bee_areas.extend(areas)
            if len(self.recorded_bee_areas) > 100:
                self.min_blob_area_dynamic = float(np.percentile(self.recorded_bee_areas, 10))
        
        # Apply noise filter
        if self.noise_filter:
            blobs = self._filter_blobs_with_ai(frame_buffer[-1][1], blobs)
        
        return blobs
    
    def _detect_single_frame_blobs(self, frame: np.ndarray) -> List[dict]:
        """Detect blobs from single frame."""
        fg_mask = self.bg_subtractor.apply(frame)
        
        if self.fast_mode:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        blobs = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_blob_area_dynamic:
                continue
            
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = area / hull_area
                if solidity < 0.5:
                    continue
            
            x, y, w, h = cv2.boundingRect(contour)
            bbox = (x, y, x + w, y + h)
            centroid = (x + w / 2, y + h / 2)
            
            blobs.append({
                'bbox': bbox,
                'centroid': centroid,
                'area': area,
                'contour': contour
            })
        
        if self.noise_filter and blobs:
            blobs = self._filter_blobs_with_ai(frame, blobs)
        
        return blobs
    
    def _run_yolo(self, frame: np.ndarray, roi_coords: Optional[BBox] = None) -> List[dict]:
        """Run YOLO detection on frame."""
        try:
            # Resize for fast mode
            if self.fast_mode:
                h, w = frame.shape[:2]
                small_frame = cv2.resize(frame, (w // 2, h // 2))
                scale = 2.0
            else:
                small_frame = frame
                scale = 1.0
            
            # Run inference
            results = self.model(small_frame, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                for i in range(len(boxes)):
                    cls = int(boxes.cls[i])
                    label = result.names[cls]
                    
                    if label not in self.tracking_classes:
                        continue
                    
                    # Scale back bbox
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = x1 * scale, y1 * scale, x2 * scale, y2 * scale
                    
                    conf = float(boxes.conf[i])
                    
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'centroid': ((x1 + x2) / 2, (y1 + y2) / 2),
                        'label': self.label_map.get(label, label),
                        'confidence': conf
                    })
            
            return detections
            
        except Exception as e:
            logger.error(f"YOLO inference failed: {e}")
            return []
    
    def _filter_blobs_with_ai(self, frame: np.ndarray, blobs: List[dict]) -> List[dict]:
        """Filter blobs using AI noise classifier."""
        if not self.noise_filter or not blobs:
            return blobs
        
        filtered = []
        for blob in blobs:
            x1, y1, x2, y2 = [int(c) for c in blob['bbox']]
            crop = frame[y1:y2, x1:x2]
            
            if crop.size == 0:
                continue
            
            is_bee = self.noise_filter.is_bee(crop)
            if is_bee:
                filtered.append(blob)
        
        return filtered
    
    def _blob_to_detection(self, blob: dict, source: str) -> dict:
        """Convert blob dict to detection dict."""
        return {
            'bbox': blob['bbox'],
            'centroid': blob['centroid'],
            'label': 'bee',
            'confidence': 1.0,
            'source': source
        }
    
    def _yolo_to_detection(self, yolo_det: dict) -> dict:
        """Convert YOLO detection to standardized format."""
        return {
            'bbox': yolo_det['bbox'],
            'centroid': yolo_det['centroid'],
            'label': yolo_det['label'],
            'confidence': yolo_det['confidence'],
            'source': 'yolo'
        }
    
    def _merge_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        """Merge multiple frames using median."""
        if len(frames) == 1:
            return frames[0]
        
        stacked = np.stack(frames, axis=0)
        return np.median(stacked, axis=0).astype(np.uint8)
    
    def _expand_roi(self, roi: BBox, padding: int = 200) -> BBox:
        """Expand ROI with padding."""
        x1, y1, x2, y2 = roi
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = x2 + padding
        y2 = y2 + padding
        return (x1, y1, x2, y2)
    
    def _apply_roi_mask(self, frame: np.ndarray, roi: BBox) -> np.ndarray:
        """Apply ROI mask to frame."""
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        x1, y1, x2, y2 = [int(c) for c in roi]
        mask[y1:y2, x1:x2] = 255
        
        return cv2.bitwise_and(frame, frame, mask=mask)
    
    def _visualize_frame(
        self,
        frame: np.ndarray,
        tracks: dict,
        frame_num: int,
        detections: List[dict]
    ) -> np.ndarray:
        """Create visualization frame."""
        viz = frame.copy()
        
        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = [int(c) for c in det['bbox']]
            color = (255, 255, 0) if det['source'] == 'motion' else (0, 165, 255)
            cv2.rectangle(viz, (x1, y1), (x2, y2), color, 1)
        
        # Draw tracks
        search_regions = self.mot.get_search_regions(frame_num)
        
        for track_id, track in tracks.items():
            # Draw trajectory
            if len(track.trajectory) > 1:
                for i in range(len(track.trajectory) - 1):
                    pt1 = tuple(map(int, track.trajectory[i][1]))
                    pt2 = tuple(map(int, track.trajectory[i+1][1]))
                    cv2.line(viz, pt1, pt2, (255, 255, 0), 1)
            
            # Draw bbox
            frames_since_confirmation = frame_num - track.last_confirmation_frame
            if frames_since_confirmation == 0:
                color = (0, 255, 0)  # Green = confirmed
            elif frames_since_confirmation < 10:
                color = (0, 255, 255)  # Yellow = recent
            else:
                color = (255, 0, 0)  # Blue = old
            
            x1, y1, x2, y2 = [int(c) for c in track.bbox]
            cv2.rectangle(viz, (x1, y1), (x2, y2), color, 2)
            
            # Draw search region
            region = search_regions.get(track_id, {})
            if region.get('type') == 'circle':
                cx, cy = map(int, region['center'])
                radius = int(region['radius'])
                cv2.circle(viz, (cx, cy), radius, (128, 128, 128), 1)
            elif region.get('type') == 'ellipse':
                cx, cy = map(int, region['center'])
                axes = (int(region['major_axis']), int(region['minor_axis']))
                angle = region['angle']
                cv2.ellipse(viz, (cx, cy), axes, angle, 0, 360, (128, 128, 128), 1)
            
            # Label
            label = f"ID:{track_id} {track.label} (age={track.age})"
            cv2.putText(viz, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Status text
        mode_color = (0, 255, 0) if self.current_mode == 'tracking' else (255, 255, 0)
        cv2.putText(viz, f"Mode: {self.current_mode}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
        cv2.putText(viz, f"Tracks: {len(tracks)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(viz, f"Frame: {frame_num}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return viz
    
    def _convert_to_grouped_format(self, all_detections: List[dict]) -> pd.DataFrame:
        """Convert flat detections to grouped format."""
        if not all_detections:
            return pd.DataFrame(columns=['frame_number', 'tracks', 'detections'])
        
        df = pd.DataFrame(all_detections)
        
        # Group into periods
        periods = self._split_into_periods(df, gap_threshold=int(self.config.tracking.max_age * 1.1))
        
        result_rows = []
        for period_df in periods:
            track_groups = {}
            
            for track_id in period_df['track_id'].unique():
                track_df = period_df[period_df['track_id'] == track_id].sort_values('frame')
                segments = self._split_track_by_gaps(track_df, gap_threshold=self.config.tracking.max_age)
                
                for seg_idx, seg_df in enumerate(segments):
                    unique_id = f"{track_id}_{seg_idx}" if len(segments) > 1 else track_id
                    
                    centroids = [((row['x1'] + row['x2']) / 2, (row['y1'] + row['y2']) / 2)
                                for _, row in seg_df.iterrows()]
                    bboxes = [(row['x1'], row['y1'], row['x2'], row['y2'])
                             for _, row in seg_df.iterrows()]
                    frame_numbers = seg_df['frame'].tolist()
                    
                    if len(frame_numbers) >= self.config.tracking.min_track_length:
                        track_groups[unique_id] = (unique_id, centroids, bboxes, frame_numbers)
            
            if not track_groups:
                continue
            
            all_tracks = list(track_groups.values())
            min_frame = period_df['frame'].min()
            max_frame = period_df['frame'].max()
            
            frame_detections = {}
            for frame_num in period_df['frame'].unique():
                frame_df = period_df[period_df['frame'] == frame_num]
                frame_detections[int(frame_num)] = {
                    'boxes': [(row['x1'], row['y1'], row['x2'], row['y2'])
                             for _, row in frame_df.iterrows()],
                    'label': frame_df['species'].tolist()
                }
            
            result_rows.append({
                'frame_number': (int(min_frame), int(max_frame)),
                'tracks': all_tracks,
                'detections': frame_detections
            })
        
        return pd.DataFrame(result_rows) if result_rows else pd.DataFrame(columns=['frame_number', 'tracks', 'detections'])
    
    def _split_into_periods(self, df: pd.DataFrame, gap_threshold: int = 100) -> List[pd.DataFrame]:
        """Split detections into activity periods."""
        df = df.sort_values('frame')
        frames = df['frame'].tolist()
        
        periods = []
        current_start = 0
        
        for i in range(len(frames) - 1):
            gap = frames[i + 1] - frames[i]
            if gap > gap_threshold:
                periods.append(df.iloc[current_start:i+1].copy())
                current_start = i + 1
        
        if current_start < len(df):
            periods.append(df.iloc[current_start:].copy())
        
        return periods
    
    def _split_track_by_gaps(self, track_df: pd.DataFrame, gap_threshold: int = 30) -> List[pd.DataFrame]:
        """Split track into segments by gaps."""
        frames = track_df['frame'].tolist()
        
        segments = []
        current_start = 0
        
        for i in range(len(frames) - 1):
            gap = frames[i + 1] - frames[i]
            if gap > gap_threshold:
                segments.append(track_df.iloc[current_start:i+1].copy())
                current_start = i + 1
        
        if current_start < len(track_df):
            segments.append(track_df.iloc[current_start:].copy())
        
        return segments