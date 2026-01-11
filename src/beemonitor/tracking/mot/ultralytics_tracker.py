"""Wrapper for Ultralytics YOLO built-in trackers (ByteTrack, BoT-SORT, etc.).

Allows using Ultralytics' native tracking algorithms with the motion detection system.

Supported trackers:
- bytetrack.yaml
- botsort.yaml
- Custom tracker configs
"""

import logging
from typing import Dict, List, Optional
import numpy as np

from beemonitor.tracking.mot.base_mot import BaseMOT, Detection, Track, BBox, Point

logger = logging.getLogger(__name__)


class UltralyticsTracker(BaseMOT):
    """Wrapper for Ultralytics YOLO built-in trackers.
    
    Example trackers:
    - 'bytetrack.yaml' - ByteTrack (fast, good for occlusion)
    - 'botsort.yaml' - BoT-SORT (uses appearance features)
    """
    
    def __init__(
        self,
        model,
        tracker_config: str = 'bytetrack.yaml',
        tracking_classes: Optional[List[str]] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.7
    ):
        """Initialize Ultralytics tracker wrapper.
        
        Args:
            model: YOLO model instance
            tracker_config: Tracker config file ('bytetrack.yaml', 'botsort.yaml', or path)
            tracking_classes: List of class names to track (None = all)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IOU threshold for NMS
        """
        self.model = model
        self.tracker_config = tracker_config
        self.tracking_classes = tracking_classes or []
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Track state
        self._tracks: Dict[int, TrackInfo] = {}
        self._frame_counter = 0
        
        logger.info(f"UltralyticsTracker initialized")
        logger.info(f"  Tracker: {tracker_config}")
        logger.info(f"  Classes: {tracking_classes}")
    
    def predict(self, frame_num: int) -> Dict[int, Track]:
        """Predict is not used in Ultralytics tracking (returns current tracks)."""
        return self.get_tracks()
    
    def update(self, detections: List[Detection], frame_num: int, frame: np.ndarray = None) -> Dict[int, Track]:
        """Update tracks using Ultralytics tracker.
        
        Args:
            detections: List of detections (from motion or YOLO)
            frame_num: Current frame number
            frame: Frame image (required for Ultralytics tracking)
            
        Returns:
            Dictionary of updated tracks
        """
        if frame is None:
            logger.warning("Frame required for Ultralytics tracking, using empty frame")
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Run YOLO tracking (combines detection + tracking)
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_config,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        # Update internal tracks from YOLO results
        self._tracks.clear()
        
        for result in results:
            boxes = result.boxes
            
            if boxes is None or len(boxes) == 0:
                continue
            
            for i in range(len(boxes)):
                # Get class and check if we should track it
                cls = int(boxes.cls[i])
                label = result.names[cls]
                
                if self.tracking_classes and label not in self.tracking_classes:
                    continue
                
                # Get track ID (None if not tracked)
                track_id = boxes.id[i] if boxes.id is not None else None
                
                if track_id is None:
                    continue
                
                track_id = int(track_id)
                
                # Get bbox
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                bbox = (float(x1), float(y1), float(x2), float(y2))
                centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
                
                # Get confidence
                conf = float(boxes.conf[i])
                
                # Update track info
                if track_id not in self._tracks:
                    self._tracks[track_id] = TrackInfo(
                        track_id=track_id,
                        bbox=bbox,
                        centroid=centroid,
                        label=label,
                        age=0,
                        frames_without_detection=0,
                        last_confirmation_frame=frame_num,
                        trajectory=[(frame_num, centroid)],
                        confidence=conf,
                        source='yolo'  # UltralyticsTracker always uses YOLO
                    )
                else:
                    track_info = self._tracks[track_id]
                    track_info.bbox = bbox
                    track_info.centroid = centroid
                    track_info.age += 1
                    track_info.frames_without_detection = 0
                    track_info.last_confirmation_frame = frame_num
                    track_info.trajectory.append((frame_num, centroid))
                    track_info.confidence = conf
        
        self._frame_counter = frame_num
        return self.get_tracks()
    
    def get_tracks(self) -> Dict[int, Track]:
        """Get all active tracks."""
        tracks = {}
        
        for track_id, track_info in self._tracks.items():
            # Calculate velocity from last 2 points
            velocity = None
            if len(track_info.trajectory) >= 2:
                (f1, p1), (f2, p2) = track_info.trajectory[-2:]
                if f2 > f1:
                    vx = (p2[0] - p1[0]) / (f2 - f1)
                    vy = (p2[1] - p1[1]) / (f2 - f1)
                    velocity = (vx, vy)
            
            tracks[track_id] = Track(
                track_id=track_id,
                bbox=track_info.bbox,
                centroid=track_info.centroid,
                label=track_info.label,
                age=track_info.age,
                frames_without_detection=track_info.frames_without_detection,
                last_confirmation_frame=track_info.last_confirmation_frame,
                trajectory=track_info.trajectory.copy(),
                velocity=velocity,
                source=track_info.source  # Preserve source
            )
        
        return tracks
    
    def reset(self):
        """Reset tracker state."""
        self._tracks.clear()
        self._frame_counter = 0
        
        # Reset YOLO tracker
        # Note: Ultralytics resets tracker automatically when persist=False
        # or when starting a new video
        logger.info("UltralyticsTracker reset")
    
    def get_search_regions(self, frame_num: int) -> Dict[int, Dict]:
        """Get search regions (not applicable for Ultralytics tracking).
        
        Returns empty regions since Ultralytics handles this internally.
        """
        regions = {}
        
        for track_id, track_info in self._tracks.items():
            # Return simple circular region for visualization
            regions[track_id] = {
                'type': 'circle',
                'center': track_info.centroid,
                'radius': 50.0
            }
        
        return regions


class TrackInfo:
    """Internal track information for UltralyticsTracker."""
    
    def __init__(
        self,
        track_id: int,
        bbox: BBox,
        centroid: Point,
        label: str,
        age: int,
        frames_without_detection: int,
        last_confirmation_frame: int,
        trajectory: List,
        confidence: float,
        source: str = 'yolo'  # UltralyticsTracker always uses YOLO
    ):
        self.track_id = track_id
        self.bbox = bbox
        self.centroid = centroid
        self.label = label
        self.age = age
        self.frames_without_detection = frames_without_detection
        self.last_confirmation_frame = last_confirmation_frame
        self.trajectory = trajectory
        self.confidence = confidence
        self.source = source


# Convenience functions for common trackers
def create_bytetrack(model, tracking_classes=None, conf=0.25, iou=0.7) -> UltralyticsTracker:
    """Create ByteTrack tracker (fast, good for occlusion)."""
    return UltralyticsTracker(
        model=model,
        tracker_config='bytetrack.yaml',
        tracking_classes=tracking_classes,
        conf_threshold=conf,
        iou_threshold=iou
    )


def create_botsort(model, tracking_classes=None, conf=0.25, iou=0.7) -> UltralyticsTracker:
    """Create BoT-SORT tracker (uses appearance features)."""
    return UltralyticsTracker(
        model=model,
        tracker_config='botsort.yaml',
        tracking_classes=tracking_classes,
        conf_threshold=conf,
        iou_threshold=iou
    )