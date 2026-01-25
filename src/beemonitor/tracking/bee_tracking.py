# """
# BeeTracking - v2.4 YOLO-Only with ADAPTIVE Tracker + Bee Identification
# ========================================================================

# Simplified bee tracking using YOLO-only detection with adaptive, resolution and FPS-independent tracking.

# v2.4 Changes:
# - Added bee identification support (color, number, QR code)
# - taxon field from YOLO class label (always present)
# - bee_id field for individual identification (optional)
# - Configurable identifier via ColorIdentifier or BeeIdentifierManager

# v2.3 Changes:
# - Pass frame_height to tracker for resolution-relative fallback
# - Robust bee size with IQR outlier rejection (in BeeTracker)
# - Track IDs start from 1 for confirmed tracks only

# v2.2 Changes:
# - YOLO-only detection (no FGBG_YOLO mode)
# - Two-mode optimization (motion detection + YOLO tracking)
# - Adaptive tracker with FPS and bee-size relative thresholds
# - Auto-calculates bee size from detections
# """

# import cv2
# import numpy as np
# from typing import List, Tuple, Optional, Dict, Any
# import logging
# from ultralytics import YOLO

# from beemonitor.detection.yolo_detector import YOLODetector
# from beemonitor.detection.blob_detector import BlobDetector
# from beemonitor.tracking.mot.bee_tracker import BeeTracker

# logger = logging.getLogger(__name__)


# class BeeTracking:
#     """
#     YOLO-only bee tracking with two-mode optimization and adaptive parameters.
    
#     Features:
#     - Motion detection mode (fast blob detection)
#     - Tracking mode (YOLO when motion detected)
#     - Adaptive tracker (resolution & FPS independent)
#     - Auto-calculates bee size
#     - Configurable distance multipliers for tuning
#     """
    
#     def __init__(
#         self,
#         yolo_model_path: str,
#         confidence_threshold: float = 0.25,
#         roi: Optional[Tuple[int, int, int, int]] = None,
#         # Adaptive tracker parameters
#         max_age_seconds: float = 1.0,
#         min_hits_seconds: float = 0.1,
#         max_resurrection_seconds: float = 0.5,
#         match_distance_multiplier: float = 8.0,
#         resurrection_search_multiplier: float = 3.0,
#         duplicate_distance_multiplier: float = 1.2,
#         iou_threshold: float = 0.3,
#         # Bee identification
#         identifier=None,  # BeeIdentifierManager or ColorIdentifier instance
#         # Crop saving for identification training
#         save_crops: bool = False,
#         crop_output_dir: Optional[str] = None,
#         crops_per_track: int = 5
#     ):
#         """
#         Initialize BeeTracking with YOLO-only detection and adaptive tracking.
        
#         Args:
#             yolo_model_path: Path to YOLO model weights
#             confidence_threshold: YOLO confidence threshold
#             roi: Region of interest (x1, y1, x2, y2)
            
#             Adaptive tracker parameters (resolution & FPS independent):
#                 max_age_seconds: Max time without detection before track dies
#                 min_hits_seconds: Min time before track is confirmed
#                 max_resurrection_seconds: Max time to resurrect dead tracks
#                 match_distance_multiplier: Max distance for matching (multiplier of bee size)
#                 resurrection_search_multiplier: Search radius (multiplier of bee size)
#                 duplicate_distance_multiplier: Duplicate threshold (multiplier of bee size)
#                 iou_threshold: IoU threshold for matching
            
#             Bee identification:
#                 identifier: Identifier instance (ColorIdentifier or BeeIdentifierManager)
            
#             Crop saving (for identification training data):
#                 save_crops: Whether to save bbox crops of tracked bees
#                 crop_output_dir: Directory to save crops (default: ./crops)
#                 crops_per_track: Number of crops to save per track (default: 5)
#         """
#         logger.info("Initializing BeeTracking (v2.4 YOLO-only with adaptive tracker + identification)")
        
#         # YOLO detector
#         logger.info(f"Loading YOLO model from {yolo_model_path}")

#         # self.yolo_detector = YOLODetector(
#         #     model_path=yolo_model_path,
#         #     confidence_threshold=confidence_threshold
#         # )

#         yolo_model = YOLO(yolo_model_path)
#         self.yolo_detector = YOLODetector(
#             model=yolo_model,
#             conf_threshold=confidence_threshold,
#             iou_threshold=iou_threshold
#         )
        
#         # Blob detector (for motion detection mode)
#         logger.info("Initializing blob detector for motion detection")
#         self.blob_detector = BlobDetector()
        
#         # ROI
#         self.roi = roi
        
#         # Video properties (will be set when video is opened)
#         self.fps = None
#         self.video_width = None
#         self.video_height = None
        
#         # Store tracker parameters for initialization
#         self.tracker_params = {
#             'max_age_seconds': max_age_seconds,
#             'min_hits_seconds': min_hits_seconds,
#             'max_resurrection_seconds': max_resurrection_seconds,
#             'match_distance_multiplier': match_distance_multiplier,
#             'resurrection_search_multiplier': resurrection_search_multiplier,
#             'duplicate_distance_multiplier': duplicate_distance_multiplier,
#             'iou_threshold': iou_threshold
#         }
        
#         # Tracker (will be initialized when video properties are known)
#         self.tracker = None
        
#         # Bee identifier (optional)
#         self.identifier = identifier
        
#         # Crop saving for identification training
#         self.save_crops = save_crops
#         self.crop_output_dir = crop_output_dir or './crops'
#         self.crops_per_track = crops_per_track
#         self.track_crop_counts = {}  # track_id -> number of crops saved
        
#         # Two-mode system state
#         self.mode = 'motion_detection'  # 'motion_detection' or 'tracking'
#         self.frames_since_motion = 0
#         self.motion_cooldown = 30  # Frames to stay in tracking mode after motion stops
        
#         # Lookback buffer for catching motion before detection
#         self.lookback_seconds = 0.5  # How far back to look when motion detected
#         self.frame_buffer = []  # Ring buffer of (frame_num, frame) tuples
#         self.lookback_frames = 0  # Will be set based on FPS
#         self.processing_lookback = False  # Flag to prevent infinite loops
        
#         id_status = "enabled" if identifier else "disabled"
#         crop_status = f"enabled ({crops_per_track}/track)" if save_crops else "disabled"
#         logger.info(f"BeeTracking initialized (v2.4, identification={id_status}, crops={crop_status})")
    
#     def _initialize_tracker(self, fps: float, frame_height: int = None):
#         """
#         Initialize adaptive tracker with video FPS and frame height.
        
#         Args:
#             fps: Video frame rate
#             frame_height: Video frame height for resolution-relative fallback
#         """
#         logger.info(f"Initializing adaptive tracker with FPS={fps}, frame_height={frame_height}")
        
#         self.tracker = BeeTracker(
#             fps=fps,
#             bee_size=None,  # Auto-calculate from detections
#             frame_height=frame_height,  # For resolution-relative fallback
#             **self.tracker_params
#         )
        
#         # Set lookback buffer size based on FPS
#         self.lookback_frames = int(fps * self.lookback_seconds)
#         self.frame_buffer = []  # Reset buffer
#         logger.info(f"Lookback buffer: {self.lookback_frames} frames ({self.lookback_seconds}s)")
        
#         logger.info("Adaptive tracker initialized")
    
#     def _get_video_properties(self, video_path: str) -> Tuple[float, int, int]:
#         """
#         Get video properties (FPS, width, height).
        
#         Args:
#             video_path: Path to video file
            
#         Returns:
#             (fps, width, height)
#         """
#         cap = cv2.VideoCapture(video_path)
        
#         fps = cap.get(cv2.CAP_PROP_FPS)
#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
#         cap.release()
        
#         # Default to 30fps if unable to read
#         if fps <= 0:
#             logger.warning(f"Could not read FPS from video, defaulting to 30.0")
#             fps = 30.0
        
#         logger.info(f"Video properties: {width}x{height} @ {fps} fps")
        
#         return fps, width, height
    
#     def initialize_video(self, video_path: str, output_path: Optional[str] = None):
#         """
#         Initialize tracking for a specific video.
        
#         Args:
#             video_path: Path to video file
#             output_path: Optional path to output folder
#         """
#         import os
        
#         logger.info(f"Initializing video tracking for: {video_path}")
#         if output_path:
#             logger.info(f"Output path: {output_path}")
        
#         # Get video properties
#         self.fps, self.video_width, self.video_height = self._get_video_properties(video_path)
#         logger.info(f"Video properties: {self.video_width}x{self.video_height} @ {self.fps} fps")
        
#         # Initialize adaptive tracker with video FPS and frame height
#         self._initialize_tracker(self.fps, self.video_height)
        
#         # Reset crop counts for new video
#         self.track_crop_counts = {}
        
#         # Setup crop output directory (same location as other outputs)
#         if self.save_crops:
#             if output_path:
#                 # output_path might be a file (video) or directory
#                 # Use parent directory if it's a file
#                 if os.path.splitext(output_path)[1]:  # Has extension = file
#                     output_dir = os.path.dirname(output_path)
#                 else:
#                     output_dir = output_path
#                 self.crop_output_dir = os.path.join(output_dir, 'crops')
#             os.makedirs(self.crop_output_dir, exist_ok=True)
#             abs_crop_dir = os.path.abspath(self.crop_output_dir)
#             logger.info(f"Crop saving enabled: {abs_crop_dir} ({self.crops_per_track} per track)")
        
#         # Initialize blob detector background
#         # Note: BlobDetector builds background model internally from detect() calls
#         # We don't need to manually feed it frames
#         logger.info("Blob detector initialized (will build background on first detect() call)")
        
#         # Optionally save initial background if output_path is provided
#         # This will be saved after the first few frames are processed
#         if output_path:
#             os.makedirs(output_path, exist_ok=True)
#             self.background_save_path = os.path.join(output_path, "background.png")
#             logger.info(f"Background will be saved to: {self.background_save_path}")
#         else:
#             self.background_save_path = None
    
#     def detect_motion(self, frame: np.ndarray) -> bool:
#         """
#         Detect if there is motion in the frame.
        
#         Args:
#             frame: Input frame
            
#         Returns:
#             True if motion detected, False otherwise
#         """
#         # Get blob detections (fast)
#         blob_detections = self.blob_detector.detect(frame)
        
#         # Motion detected if we have any blobs
#         return len(blob_detections) > 0
    
#     def _save_track_crops(self, frame: np.ndarray, tracks: List[Dict], frame_num: int):
#         """
#         Save bbox crops for tracked bees (for identification training data).
        
#         Args:
#             frame: Full video frame
#             tracks: List of confirmed tracks
#             frame_num: Current frame number
#         """
#         import os
        
#         if not self.save_crops or not tracks:
#             return
        
#         h, w = frame.shape[:2]
        
#         for track in tracks:
#             track_id = track['track_id']
            
#             # Check if we've saved enough crops for this track
#             saved = self.track_crop_counts.get(track_id, 0)
#             if saved >= self.crops_per_track:
#                 continue
            
#             # Extract bbox
#             x1 = max(0, int(track['x1']))
#             y1 = max(0, int(track['y1']))
#             x2 = min(w, int(track['x2']))
#             y2 = min(h, int(track['y2']))
            
#             if x1 >= x2 or y1 >= y2:
#                 continue
            
#             # Crop bee region
#             crop = frame[y1:y2, x1:x2]
            
#             if crop.size == 0:
#                 continue
            
#             # Save crop: crops/track_{id}/frame_{num}.jpg
#             track_dir = os.path.join(self.crop_output_dir, f'track_{track_id:04d}')
#             os.makedirs(track_dir, exist_ok=True)
            
#             crop_path = os.path.join(track_dir, f'frame_{frame_num:06d}.jpg')
#             success = cv2.imwrite(crop_path, crop)
            
#             if success:
#                 self.track_crop_counts[track_id] = saved + 1
#                 logger.info(f"Saved crop {saved + 1}/{self.crops_per_track} for track {track_id}: {crop_path}")
#             else:
#                 logger.warning(f"Failed to save crop: {crop_path}")
    
#     def process_frame(
#         self,
#         frame: np.ndarray,
#         frame_num: int,
#         visualize: bool = False
#     ) -> Dict[str, Any]:
#         """
#         Process a single frame with two-mode optimization.
        
#         Args:
#             frame: Input frame
#             frame_num: Frame number
#             visualize: Whether to create visualization
            
#         Returns:
#             Dictionary with detections, tracks, and mode info
#         """
#         # Apply ROI for motion detection only
#         if self.roi:
#             x1, y1, x2, y2 = self.roi
#             roi_frame = frame[y1:y2, x1:x2]
#         else:
#             roi_frame = frame
        
#         detections = []
#         lookback_results = []  # Results from lookback frames
        
#         # DISABLE TWO-MODE FOR NOW
#         self.mode = "tracking" # allow two mode for now.

#         #print(self.mode)

#         # TWO-MODE SYSTEM
#         if self.mode == 'motion_detection':
#             # Add frame to lookback buffer (only in motion_detection mode)
#             if not self.processing_lookback:
#                 self.frame_buffer.append((frame_num, frame.copy()))
#                 # Keep buffer at max size
#                 if len(self.frame_buffer) > self.lookback_frames:
#                     self.frame_buffer.pop(0)
            
#             # Fast motion detection mode (ROI only)
#             has_motion = self.detect_motion(roi_frame)
            
#             if has_motion:
#                 # Motion detected - switch to tracking mode
#                 logger.debug(f"Frame {frame_num}: Motion detected, switching to tracking mode")
#                 self.mode = 'tracking'
#                 self.frames_since_motion = 0
                
#                 # Process lookback buffer FIRST (if we have buffered frames)
#                 if self.frame_buffer and not self.processing_lookback:
#                     self.processing_lookback = True  # Prevent infinite loops
#                     logger.debug(f"Processing {len(self.frame_buffer)} lookback frames")
                    
#                     for buf_frame_num, buf_frame in self.frame_buffer:
#                         # Run YOLO on buffered frames
#                         yolo_detections = self.yolo_detector.detect(buf_frame)
                        
#                         buf_detections = []
#                         for det in yolo_detections:
#                             bbox = det.bbox
#                             conf = det.confidence
#                             taxon = det.label
#                             det_with_source = list(bbox) + [conf, 'yolo', taxon]
#                             buf_detections.append(det_with_source)
                        
#                         # Update tracker with lookback detections
#                         if self.tracker is not None:
#                             buf_tracks = self.tracker.update(buf_detections, buf_frame_num)
#                             lookback_results.append({
#                                 'frame_num': buf_frame_num,
#                                 'detections': buf_detections,
#                                 'tracks': buf_tracks,
#                                 'mode': 'lookback'
#                             })
                    
#                     self.frame_buffer = []  # Clear buffer
#                     self.processing_lookback = False
                
#                 # Now run YOLO on current frame
#                 yolo_detections = self.yolo_detector.detect(frame)
                
#                 for det in yolo_detections:
#                     bbox = det.bbox
#                     conf = det.confidence
#                     taxon = det.label
#                     det_with_source = list(bbox) + [conf, 'yolo', taxon]
#                     detections.append(det_with_source)
        
#         else:  # tracking mode
#             # Run YOLO on FULL FRAME (allows tracking beyond ROI)
#             yolo_detections = self.yolo_detector.detect(frame)
            
#             # Convert to tracking format: [x1, y1, x2, y2, conf, source, taxon]
#             for det in yolo_detections:
#                 bbox = det.bbox
#                 conf = det.confidence
#                 taxon = det.label  # YOLO class label = taxonomic ID
#                 det_with_source = list(bbox) + [conf, 'yolo', taxon]
#                 detections.append(det_with_source)
            
#             # Check if motion has stopped (ROI only)
#             has_motion = self.detect_motion(roi_frame)
            
#             if has_motion:
#                 self.frames_since_motion = 0
#             else:
#                 self.frames_since_motion += 1
            
#             # Switch back to motion detection mode if no motion for cooldown period
#             if self.frames_since_motion > self.motion_cooldown:
#                 logger.debug(f"Frame {frame_num}: No motion for {self.motion_cooldown} frames, switching to motion detection mode")
#                 self.mode = 'motion_detection'
#                 self.frame_buffer = []  # Clear buffer when switching back
        
#         # Update tracker (full frame coordinates, no adjustment needed)
#         tracks = []
#         if self.tracker is not None:
#             tracks = self.tracker.update(detections, frame_num)
            
#             # Run identification on confirmed tracks
#             if self.identifier and tracks:
#                 for track_obj in self.tracker.tracks:
#                     if track_obj.is_confirmed and track_obj.bee_id is None:
#                         # Try to identify this bee
#                         result = self.identifier.identify(frame, track_obj.last_bbox)
#                         if result:
#                             bee_id, method, confidence = result
#                             track_obj.set_bee_id(bee_id, method, confidence)
                
#                 # Refresh tracks list with updated bee_ids
#                 tracks = self.tracker.get_active_tracks()
        
#         # Save crops for identification training
#         if self.save_crops and tracks:
#             self._save_track_crops(frame, tracks, frame_num)
        
#         result = {
#             'frame_num': frame_num,
#             'detections': detections,
#             'tracks': tracks,
#             'mode': self.mode,
#             'num_detections': len(detections),
#             'num_tracks': len(tracks),
#             'lookback_results': lookback_results  # Include lookback frames
#         }
        
#         if visualize:
#             result['visualization'] = self._create_visualization(frame, detections, tracks)
        
#         return result
    
#     def _create_visualization(
#         self,
#         frame: np.ndarray,
#         detections: List,
#         tracks: List[Dict]
#     ) -> np.ndarray:
#         """
#         Create visualization of detections and tracks.
        
#         Args:
#             frame: Input frame
#             detections: List of detections
#             tracks: List of tracks
            
#         Returns:
#             Annotated frame
#         """
#         vis_frame = frame.copy()
        
#         # Draw ROI
#         if self.roi:
#             x1, y1, x2, y2 = self.roi
#             cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
#         # Draw detections (BLUE for YOLO)
#         for det in detections:
#             x1, y1, x2, y2 = map(int, det[:4])
#             source = det[5] if len(det) > 5 else 'unknown'
            
#             color = (255, 0, 0) if source == 'yolo' else (128, 128, 128)
#             cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
#             cv2.putText(vis_frame, source.upper(), (x1, y1 - 5),
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
#         # Draw tracks
#         colors = [(255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
        
#         for i, track in enumerate(tracks):
#             color = colors[i % len(colors)]
#             track_id = track['track_id']
#             cx, cy = int(track['cx']), int(track['cy'])
            
#             # Draw trajectory
#             if len(track['history']) > 1:
#                 points = np.array(track['history'], dtype=np.int32)
#                 cv2.polylines(vis_frame, [points], False, color, 2)
            
#             # Draw current position
#             cv2.circle(vis_frame, (cx, cy), 5, color, -1)
            
#             # Show bee_id if available, otherwise track_id
#             label = f"B:{track['bee_id']}" if track.get('bee_id') else f"T:{track_id}"
#             cv2.putText(vis_frame, label, (cx + 10, cy),
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
#         # Draw mode indicator
#         mode_text = f"Mode: {self.mode.upper()}"
#         cv2.putText(vis_frame, mode_text, (10, 30),
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
#         return vis_frame
    
#     def process_video(
#         self,
#         video_path: str,
#         output_path: Optional[str] = None,
#         visualize: bool = False
#     ) -> List[Dict[str, Any]]:
#         """
#         Process entire video.
        
#         Args:
#             video_path: Path to input video
#             output_path: Path to output video (if visualize=True)
#             visualize: Whether to create output video
            
#         Returns:
#             List of results for each frame
#         """
#         # Initialize for this video
#         self.initialize_video(video_path, output_path)
        
#         # Open video
#         cap = cv2.VideoCapture(video_path)
        
#         # Setup video writer if visualizing
#         if visualize and output_path:
#             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#             out = cv2.VideoWriter(
#                 output_path,
#                 fourcc,
#                 self.fps,
#                 (self.video_width, self.video_height)
#             )
        
#         results = []
#         frame_num = 0
        
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             # Process frame
#             result = self.process_frame(frame, frame_num, visualize=visualize)
            
#             # Handle lookback results (frames processed before current frame)
#             lookback_results = result.pop('lookback_results', [])
#             if lookback_results:
#                 logger.debug(f"Adding {len(lookback_results)} lookback results")
#                 results.extend(lookback_results)
            
#             results.append(result)
            
#             # Write visualization if enabled
#             if visualize and output_path and 'visualization' in result:
#                 out.write(result['visualization'])
            
#             frame_num += 1
            
#             if frame_num % 100 == 0:
#                 logger.info(f"Processed {frame_num} frames")
        
#         cap.release()
#         if visualize and output_path:
#             out.release()
        
#         logger.info(f"Processed {frame_num} frames total")
        
#         # Log crop saving summary
#         if self.save_crops and self.track_crop_counts:
#             import os
#             total_crops = sum(self.track_crop_counts.values())
#             num_tracks = len(self.track_crop_counts)
#             abs_crop_dir = os.path.abspath(self.crop_output_dir)
#             logger.info(f"Saved {total_crops} crops for {num_tracks} tracks in: {abs_crop_dir}")
        
#         # Convert results to DataFrame
#         import pandas as pd
        
#         # Define clean column list
#         columns = ['frame', 'track_id', 'x1', 'y1', 'x2', 'y2', 'cx', 'cy',
#                    'confidence', 'source', 'taxon', 'bee_id', 'bee_id_method', 'bee_id_confidence', 'mode']
        
#         if not results:
#             # Return empty DataFrame with expected columns
#             return pd.DataFrame(columns=columns)
        
#         # Flatten results - one row per track per frame
#         flattened_rows = []
        
#         for result in results:
#             frame_num = result['frame_num']
#             mode = result['mode']
            
#             # Extract each track into a separate row
#             for track in result['tracks']:
#                 track_id = track.get('track_id')
                
#                 if track_id is None:
#                     logger.warning(f"Track missing track_id! Track keys: {track.keys()}")
#                     continue  # Skip tracks without ID
                
#                 row = {
#                     'frame': frame_num,
#                     'track_id': track_id,
#                     'x1': track['x1'],
#                     'y1': track['y1'],
#                     'x2': track['x2'],
#                     'y2': track['y2'],
#                     'cx': track['cx'],
#                     'cy': track['cy'],
#                     'confidence': track.get('confidence', 0.0),
#                     'source': track.get('source', 'unknown'),
#                     'taxon': track.get('taxon', 'bee'),
#                     'bee_id': track.get('bee_id'),
#                     'bee_id_method': track.get('bee_id_method'),
#                     'bee_id_confidence': track.get('bee_id_confidence', 0.0),
#                     'mode': mode
#                 }
#                 flattened_rows.append(row)
        
#         # Create DataFrame from flattened rows
#         if not flattened_rows:
#             logger.warning("No tracks detected in video")
#             return pd.DataFrame(columns=columns)
        
#         df = pd.DataFrame(flattened_rows)
        
#         # Sort by frame number to ensure correct temporal order (lookback frames may be added later)
#         df = df.sort_values(['frame', 'track_id']).reset_index(drop=True)
        
#         logger.info(f"Created tracking DataFrame with {len(df)} rows ({len(results)} frames, "
#                    f"{len(df['track_id'].unique())} unique tracks)")
        
#         return df





"""
BeeTracking - v2.4 YOLO-Only with ADAPTIVE Tracker + Bee Identification
========================================================================

Simplified bee tracking using YOLO-only detection with adaptive, resolution and FPS-independent tracking.

v2.4 Changes:
- Added bee identification support (color, number, QR code)
- taxon field from YOLO class label (always present)
- bee_id field for individual identification (optional)
- Configurable identifier via ColorIdentifier or BeeIdentifierManager

v2.3 Changes:
- Pass frame_height to tracker for resolution-relative fallback
- Robust bee size with IQR outlier rejection (in BeeTracker)
- Track IDs start from 1 for confirmed tracks only

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
        match_distance_multiplier: float = 8.0,
        resurrection_search_multiplier: float = 3.0,
        duplicate_distance_multiplier: float = 1.2,
        iou_threshold: float = 0.3,
        # Bee identification
        identifier=None,  # BeeIdentifierManager or ColorIdentifier instance
        # Crop saving for identification training
        save_crops: bool = False,
        crop_output_dir: Optional[str] = None,
        crops_per_track: int = 5
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
            
            Bee identification:
                identifier: Identifier instance (ColorIdentifier or BeeIdentifierManager)
            
            Crop saving (for identification training data):
                save_crops: Whether to save bbox crops of tracked bees
                crop_output_dir: Directory to save crops (default: ./crops)
                crops_per_track: Number of crops to save per track (default: 5)
        """
        logger.info("Initializing BeeTracking (v2.4 YOLO-only with adaptive tracker + identification)")
        
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
        
        # Bee identifier (optional)
        self.identifier = identifier
        
        # Crop saving for identification training
        self.save_crops = save_crops
        self.crop_output_dir = crop_output_dir or './crops'
        self.crops_per_track = crops_per_track
        self.track_crop_counts = {}  # track_id -> number of crops saved
        
        # Two-mode system state
        self.mode = 'motion_detection'  # 'motion_detection' or 'tracking'
        self.frames_since_motion = 0
        self.motion_cooldown = 30  # Frames to stay in tracking mode after motion stops
        
        # Lookback buffer for catching motion before detection
        self.lookback_seconds = 0.5  # How far back to look when motion detected
        self.frame_buffer = []  # Ring buffer of (frame_num, frame) tuples
        self.lookback_frames = 0  # Will be set based on FPS
        self.processing_lookback = False  # Flag to prevent infinite loops
        
        id_status = "enabled" if identifier else "disabled"
        crop_status = f"enabled ({crops_per_track}/track)" if save_crops else "disabled"
        logger.info(f"BeeTracking initialized (v2.4, identification={id_status}, crops={crop_status})")
    
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
        
        # Set lookback buffer size based on FPS
        self.lookback_frames = int(fps * self.lookback_seconds)
        self.frame_buffer = []  # Reset buffer
        logger.info(f"Lookback buffer: {self.lookback_frames} frames ({self.lookback_seconds}s)")
        
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
        import os
        
        logger.info(f"Initializing video tracking for: {video_path}")
        if output_path:
            logger.info(f"Output path: {output_path}")
        
        # Get video properties
        self.fps, self.video_width, self.video_height = self._get_video_properties(video_path)
        logger.info(f"Video properties: {self.video_width}x{self.video_height} @ {self.fps} fps")
        
        # Initialize adaptive tracker with video FPS and frame height
        self._initialize_tracker(self.fps, self.video_height)
        
        # Reset crop counts for new video
        self.track_crop_counts = {}
        
        # Extract video name for per-video crop folders
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Setup crop output directory (per-video to avoid track ID collisions)
        if self.save_crops:
            if output_path:
                # output_path might be a file (video) or directory
                # Use parent directory if it's a file
                if os.path.splitext(output_path)[1]:  # Has extension = file
                    output_dir = os.path.dirname(output_path)
                else:
                    output_dir = output_path
                # Include video name to separate crops from different videos
                self.crop_output_dir = os.path.join(output_dir, 'crops', video_name)
            else:
                # No output_path provided - use default with video name
                self.crop_output_dir = os.path.join('./crops', video_name)
            os.makedirs(self.crop_output_dir, exist_ok=True)
            abs_crop_dir = os.path.abspath(self.crop_output_dir)
            logger.info(f"Crop saving enabled: {abs_crop_dir} ({self.crops_per_track} per track)")
        
        # Initialize blob detector background
        # Note: BlobDetector builds background model internally from detect() calls
        # We don't need to manually feed it frames
        logger.info("Blob detector initialized (will build background on first detect() call)")
        
        # Optionally save initial background if output_path is provided
        # This will be saved after the first few frames are processed
        if output_path:
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
    
    def _save_track_crops(self, frame: np.ndarray, tracks: List[Dict], frame_num: int):
        """
        Save bbox crops for tracked bees (for identification training data).
        
        Args:
            frame: Full video frame
            tracks: List of confirmed tracks
            frame_num: Current frame number
        """
        import os
        
        if not self.save_crops or not tracks:
            return
        
        h, w = frame.shape[:2]
        
        for track in tracks:
            track_id = track['track_id']
            
            # Check if we've saved enough crops for this track
            saved = self.track_crop_counts.get(track_id, 0)
            if saved >= self.crops_per_track:
                continue
            
            # Extract bbox
            x1 = max(0, int(track['x1']))
            y1 = max(0, int(track['y1']))
            x2 = min(w, int(track['x2']))
            y2 = min(h, int(track['y2']))
            
            if x1 >= x2 or y1 >= y2:
                continue
            
            # Crop bee region
            crop = frame[y1:y2, x1:x2]
            
            if crop.size == 0:
                continue
            
            # Save crop: crops/track_{id}/frame_{num}.jpg
            track_dir = os.path.join(self.crop_output_dir, f'track_{track_id:04d}')
            os.makedirs(track_dir, exist_ok=True)
            
            crop_path = os.path.join(track_dir, f'frame_{frame_num:06d}.jpg')
            success = cv2.imwrite(crop_path, crop)
            
            if success:
                self.track_crop_counts[track_id] = saved + 1
                logger.info(f"Saved crop {saved + 1}/{self.crops_per_track} for track {track_id}: {crop_path}")
            else:
                logger.warning(f"Failed to save crop: {crop_path}")
    
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
        lookback_results = []  # Results from lookback frames
        
        # DISABLE TWO-MODE FOR NOW
        self.mode = "tracking" # allow two mode for now.

        #print(self.mode)

        # TWO-MODE SYSTEM
        if self.mode == 'motion_detection':
            # Add frame to lookback buffer (only in motion_detection mode)
            if not self.processing_lookback:
                self.frame_buffer.append((frame_num, frame.copy()))
                # Keep buffer at max size
                if len(self.frame_buffer) > self.lookback_frames:
                    self.frame_buffer.pop(0)
            
            # Fast motion detection mode (ROI only)
            has_motion = self.detect_motion(roi_frame)
            
            if has_motion:
                # Motion detected - switch to tracking mode
                logger.debug(f"Frame {frame_num}: Motion detected, switching to tracking mode")
                self.mode = 'tracking'
                self.frames_since_motion = 0
                
                # Process lookback buffer FIRST (if we have buffered frames)
                if self.frame_buffer and not self.processing_lookback:
                    self.processing_lookback = True  # Prevent infinite loops
                    logger.debug(f"Processing {len(self.frame_buffer)} lookback frames")
                    
                    for buf_frame_num, buf_frame in self.frame_buffer:
                        # Run YOLO on buffered frames
                        yolo_detections = self.yolo_detector.detect(buf_frame)
                        
                        buf_detections = []
                        for det in yolo_detections:
                            bbox = det.bbox
                            conf = det.confidence
                            taxon = det.label
                            det_with_source = list(bbox) + [conf, 'yolo', taxon]
                            buf_detections.append(det_with_source)
                        
                        # Update tracker with lookback detections
                        if self.tracker is not None:
                            buf_tracks = self.tracker.update(buf_detections, buf_frame_num)
                            lookback_results.append({
                                'frame_num': buf_frame_num,
                                'detections': buf_detections,
                                'tracks': buf_tracks,
                                'mode': 'lookback'
                            })
                    
                    self.frame_buffer = []  # Clear buffer
                    self.processing_lookback = False
                
                # Now run YOLO on current frame
                yolo_detections = self.yolo_detector.detect(frame)
                
                for det in yolo_detections:
                    bbox = det.bbox
                    conf = det.confidence
                    taxon = det.label
                    det_with_source = list(bbox) + [conf, 'yolo', taxon]
                    detections.append(det_with_source)
        
        else:  # tracking mode
            # Run YOLO on FULL FRAME (allows tracking beyond ROI)
            yolo_detections = self.yolo_detector.detect(frame)
            
            # Convert to tracking format: [x1, y1, x2, y2, conf, source, taxon]
            for det in yolo_detections:
                bbox = det.bbox
                conf = det.confidence
                taxon = det.label  # YOLO class label = taxonomic ID
                det_with_source = list(bbox) + [conf, 'yolo', taxon]
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
                self.frame_buffer = []  # Clear buffer when switching back
        
        # Update tracker (full frame coordinates, no adjustment needed)
        tracks = []
        if self.tracker is not None:
            tracks = self.tracker.update(detections, frame_num)
            
            # Run identification on confirmed tracks
            if self.identifier and tracks:
                for track_obj in self.tracker.tracks:
                    if track_obj.is_confirmed and track_obj.bee_id is None:
                        # Try to identify this bee
                        result = self.identifier.identify(frame, track_obj.last_bbox)
                        if result:
                            bee_id, method, confidence = result
                            track_obj.set_bee_id(bee_id, method, confidence)
                
                # Refresh tracks list with updated bee_ids
                tracks = self.tracker.get_active_tracks()
        
        # Save crops for identification training
        if self.save_crops and tracks:
            self._save_track_crops(frame, tracks, frame_num)
        
        result = {
            'frame_num': frame_num,
            'detections': detections,
            'tracks': tracks,
            'mode': self.mode,
            'num_detections': len(detections),
            'num_tracks': len(tracks),
            'lookback_results': lookback_results  # Include lookback frames
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
            cx, cy = int(track['cx']), int(track['cy'])
            
            # Draw trajectory
            if len(track['history']) > 1:
                points = np.array(track['history'], dtype=np.int32)
                cv2.polylines(vis_frame, [points], False, color, 2)
            
            # Draw current position
            cv2.circle(vis_frame, (cx, cy), 5, color, -1)
            
            # Show bee_id if available, otherwise track_id
            label = f"B:{track['bee_id']}" if track.get('bee_id') else f"T:{track_id}"
            cv2.putText(vis_frame, label, (cx + 10, cy),
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
            
            # Handle lookback results (frames processed before current frame)
            lookback_results = result.pop('lookback_results', [])
            if lookback_results:
                logger.debug(f"Adding {len(lookback_results)} lookback results")
                results.extend(lookback_results)
            
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
        
        # Log crop saving summary
        if self.save_crops and self.track_crop_counts:
            import os
            total_crops = sum(self.track_crop_counts.values())
            num_tracks = len(self.track_crop_counts)
            abs_crop_dir = os.path.abspath(self.crop_output_dir)
            logger.info(f"Saved {total_crops} crops for {num_tracks} tracks in: {abs_crop_dir}")
        
        # Convert results to DataFrame
        import pandas as pd
        
        # Define clean column list
        columns = ['frame', 'track_id', 'x1', 'y1', 'x2', 'y2', 'cx', 'cy',
                   'confidence', 'source', 'taxon', 'bee_id', 'bee_id_method', 'bee_id_confidence', 'mode']
        
        if not results:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=columns)
        
        # Flatten results - one row per track per frame
        flattened_rows = []
        
        for result in results:
            frame_num = result['frame_num']
            mode = result['mode']
            
            # Extract each track into a separate row
            for track in result['tracks']:
                track_id = track.get('track_id')
                
                if track_id is None:
                    logger.warning(f"Track missing track_id! Track keys: {track.keys()}")
                    continue  # Skip tracks without ID
                
                row = {
                    'frame': frame_num,
                    'track_id': track_id,
                    'x1': track['x1'],
                    'y1': track['y1'],
                    'x2': track['x2'],
                    'y2': track['y2'],
                    'cx': track['cx'],
                    'cy': track['cy'],
                    'confidence': track.get('confidence', 0.0),
                    'source': track.get('source', 'unknown'),
                    'taxon': track.get('taxon', 'bee'),
                    'bee_id': track.get('bee_id'),
                    'bee_id_method': track.get('bee_id_method'),
                    'bee_id_confidence': track.get('bee_id_confidence', 0.0),
                    'mode': mode
                }
                flattened_rows.append(row)
        
        # Create DataFrame from flattened rows
        if not flattened_rows:
            logger.warning("No tracks detected in video")
            return pd.DataFrame(columns=columns)
        
        df = pd.DataFrame(flattened_rows)
        
        # Sort by frame number to ensure correct temporal order (lookback frames may be added later)
        df = df.sort_values(['frame', 'track_id']).reset_index(drop=True)
        
        logger.info(f"Created tracking DataFrame with {len(df)} rows ({len(results)} frames, "
                   f"{len(df['track_id'].unique())} unique tracks)")
        
        return df