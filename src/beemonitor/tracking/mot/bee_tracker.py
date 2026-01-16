"""
BeeTracker - v2.3 ADAPTIVE (Resolution & FPS Independent)
==========================================================

Enhanced multi-object tracking with ADAPTIVE thresholds based on bee size and FPS.

v2.3 Changes:
- Robust bee size calculation with IQR outlier rejection
- Rolling window for continuous bee size adaptation
- Resolution-relative fallback for cold start

v2.2 Changes:
- All distance thresholds as MULTIPLIERS of bee size
- All time thresholds as RATIOS of FPS
- Resolution-independent tracking
- FPS-independent tracking
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import defaultdict, deque
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class KalmanFilter:
    """Simple Kalman filter for 2D position tracking."""
    
    def __init__(self, initial_position, process_noise=1.0, measurement_noise=1.0):
        """
        Initialize Kalman filter.
        
        Args:
            initial_position: (x, y) initial position
            process_noise: Process noise covariance
            measurement_noise: Measurement noise covariance
        """
        x, y = initial_position
        self.state = np.array([x, y, 0, 0], dtype=float)  # [x, y, vx, vy]
        
        # State transition matrix
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=float)
        
        # Measurement matrix
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=float)
        
        # Covariance matrices
        self.P = np.eye(4) * 500  # Initial uncertainty
        self.Q = np.eye(4) * process_noise
        self.R = np.eye(2) * measurement_noise
    
    def predict(self):
        """Predict next state."""
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state[:2]
    
    def update(self, measurement):
        """Update state with measurement."""
        measurement = np.array(measurement)
        y = measurement - (self.H @ self.state)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.state[:2]
    
    @property
    def position(self):
        """Get current position estimate."""
        return self.state[:2]
    
    @property
    def velocity(self):
        """Get current velocity estimate."""
        return self.state[2:]


class Track:
    """Represents a single tracked object (bee)."""
    
    _next_confirmed_id = 1
    
    @classmethod
    def reset_id_counter(cls):
        """Reset track ID counter (call at start of new video)."""
        cls._next_confirmed_id = 1
    
    def __init__(self, detection, frame_num, min_hits=3, max_age=30):
        """
        Initialize track.
        
        Args:
            detection: Initial detection (x1, y1, x2, y2, confidence, source, taxon)
            frame_num: Frame number where track started
            min_hits: Minimum hits before track is confirmed
            max_age: Maximum frames without detection before deletion
        """
        # ID assigned only when confirmed
        self._id = None
        
        bbox = detection[:4]
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        
        self.kf = KalmanFilter((cx, cy))
        self.history = deque(maxlen=30)
        self.history.append((cx, cy))
        
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.min_hits = min_hits
        self.max_age = max_age
        
        self.last_bbox = tuple(bbox)
        self.last_confidence = float(detection[4]) if len(detection) > 4 else 0.0
        self.last_source = detection[5] if len(detection) > 5 else 'unknown'
        self.last_detection_frame = frame_num
        self.start_frame = frame_num
        
        # Taxonomic identification (from YOLO class label - always present)
        self.taxon = detection[6] if len(detection) > 6 else 'bee'
        
        # Anti-duplicate tracking
        self.initial_position = (cx, cy)
        self.position_variance = 0.0
        self.is_stable = False
        
        # Unique individual identification (optional)
        # Determined by: color marking, number reading, QR code
        self.bee_id = None  # None = unidentified individual
        self.bee_id_method = None  # 'color', 'number', 'qrcode'
        self.bee_id_confidence = 0.0
    
    @property
    def id(self):
        """Get track ID (assigned when first confirmed)."""
        return self._id
    
    def _assign_id(self):
        """Assign ID when track becomes confirmed."""
        if self._id is None:
            self._id = Track._next_confirmed_id
            Track._next_confirmed_id += 1
    
    def predict(self):
        """Predict next position."""
        predicted_pos = self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return predicted_pos
    
    def update(self, detection, frame_num):
        """Update track with detection."""
        bbox = detection[:4]
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        
        self.kf.update((cx, cy))
        self.history.append((cx, cy))
        
        self.hits += 1
        self.time_since_update = 0
        self.last_bbox = tuple(bbox)
        self.last_confidence = float(detection[4]) if len(detection) > 4 else 0.0
        self.last_source = detection[5] if len(detection) > 5 else 'unknown'
        self.last_detection_frame = frame_num
        
        # Update taxon if provided (use most recent)
        if len(detection) > 6 and detection[6]:
            self.taxon = detection[6]
        
        # Update stability metrics
        if len(self.history) >= 5:
            recent_positions = np.array(list(self.history)[-5:])
            self.position_variance = np.var(recent_positions)
            self.is_stable = self.position_variance < 100
    
    def set_bee_id(self, bee_id: str, method: str, confidence: float = 1.0):
        """
        Set the unique individual identity of this bee.
        
        Args:
            bee_id: Unique identifier for this individual bee
            method: How the ID was determined ('color', 'number', 'qrcode')
            confidence: Confidence in the identification (0.0 - 1.0)
        """
        # Only update if higher confidence or no existing ID
        if self.bee_id is None or confidence > self.bee_id_confidence:
            self.bee_id = bee_id
            self.bee_id_method = method
            self.bee_id_confidence = confidence
    
    @property
    def is_confirmed(self):
        """Check if track is confirmed."""
        confirmed = self.hits >= self.min_hits
        if confirmed and self._id is None:
            self._assign_id()
        return confirmed
    
    @property
    def is_dead(self):
        """Check if track should be deleted."""
        return self.time_since_update > self.max_age
    
    @property
    def centroid(self):
        """Get current centroid."""
        return self.kf.position
    
    def get_distance_to(self, position):
        """Get Euclidean distance to a position."""
        return np.linalg.norm(self.centroid - np.array(position))


class BeeTracker:
    """
    Adaptive bee tracking with resolution and FPS-independent thresholds.
    
    v2.3 Features:
    - Robust bee size with IQR outlier rejection
    - Rolling window for continuous adaptation
    - Resolution-relative fallback
    
    v2.2 Features:
    - All distance thresholds as multipliers of bee size
    - All time thresholds as ratios of FPS
    - Works across different resolutions and frame rates
    """
    
    def __init__(
        self,
        fps: float = 30.0,
        bee_size: Optional[float] = None,
        frame_height: Optional[int] = None,
        # Time thresholds (as ratios of FPS)
        max_age_seconds: float = 1.0,
        min_hits_seconds: float = 0.1,
        max_resurrection_seconds: float = 0.5,
        # Distance thresholds (as multipliers of bee size)
        match_distance_multiplier: float = 8.0,
        resurrection_search_multiplier: float = 3.0,
        duplicate_distance_multiplier: float = 1.2,
        # Other
        iou_threshold: float = 0.3
    ):
        """
        Initialize BeeTracker with ADAPTIVE thresholds.
        
        Args:
            fps: Video frame rate (frames per second)
            bee_size: Average bee size in pixels (auto-calculated if None)
            frame_height: Video frame height for resolution-relative fallback
            
            Time thresholds (in seconds):
                max_age_seconds: Max time without detection before track dies
                min_hits_seconds: Min time before track is confirmed
                max_resurrection_seconds: Max time to keep dead tracks for resurrection
            
            Distance thresholds (as multipliers of bee size):
                match_distance_multiplier: Max distance for detection-to-track matching
                resurrection_search_multiplier: Search radius for resurrections
                duplicate_distance_multiplier: Distance for duplicate detection
            
            Other:
                iou_threshold: IoU threshold for matching
        """
        self.fps = fps
        self.bee_size = bee_size
        self.frame_height = frame_height
        
        # Convert time thresholds to frames
        self.max_age = int(max_age_seconds * fps)
        self.min_hits = max(1, int(min_hits_seconds * fps))
        self.max_resurrection_frames = int(max_resurrection_seconds * fps)
        
        # Distance multipliers (will be applied to bee_size)
        self.match_distance_multiplier = match_distance_multiplier
        self.resurrection_search_multiplier = resurrection_search_multiplier
        self.duplicate_distance_multiplier = duplicate_distance_multiplier
        
        self.iou_threshold = iou_threshold
        
        # Reset track ID counter for new video
        Track.reset_id_counter()
        
        self.tracks: List[Track] = []
        self.dead_tracks: List[Tuple[Track, int]] = []
        self.frame_count = 0
        
        # Anti-duplicate tracking
        self.track_positions = defaultdict(list)
        self.confirmed_track_ids = set()
        
        # Rolling window for adaptive bee size (v2.3)
        self.detection_sizes = deque(maxlen=100)
        self._calculated_bee_size = None
        
        logger.info(f"BeeTracker initialized (v2.3 ADAPTIVE):")
        logger.info(f"  FPS: {fps}")
        logger.info(f"  Frame height: {frame_height}")
        logger.info(f"  Max age: {max_age_seconds}s ({self.max_age} frames)")
        logger.info(f"  Min hits: {min_hits_seconds}s ({self.min_hits} frames)")
        logger.info(f"  Max resurrection: {max_resurrection_seconds}s ({self.max_resurrection_frames} frames)")
        logger.info(f"  Match distance multiplier: {match_distance_multiplier}x bee size")
        logger.info(f"  Resurrection multiplier: {resurrection_search_multiplier}x bee size")
        logger.info(f"  Duplicate multiplier: {duplicate_distance_multiplier}x bee size")
    
    def _update_bee_size(self, detections):
        """
        Continuously update bee size with IQR outlier rejection.
        
        Args:
            detections: List of detections (x1, y1, x2, y2, ...)
        """
        if len(detections) == 0:
            return
        
        for detection in detections:
            bbox = detection[:4]
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            size = (width + height) / 2
            self.detection_sizes.append(size)
        
        # Calculate robust bee size with IQR filtering
        if len(self.detection_sizes) >= 5:
            sizes = np.array(self.detection_sizes)
            q1, q3 = np.percentile(sizes, [25, 75])
            iqr = q3 - q1
            
            # Filter outliers (1.5x IQR rule)
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            valid_sizes = sizes[(sizes >= lower_bound) & (sizes <= upper_bound)]
            
            if len(valid_sizes) > 0:
                new_bee_size = np.median(valid_sizes)
            else:
                new_bee_size = np.median(sizes)
            
            # Log when bee size changes significantly
            if self._calculated_bee_size is None:
                logger.info(f"Initial bee size calculated: {new_bee_size:.1f} pixels")
            elif abs(new_bee_size - self._calculated_bee_size) > 5:
                logger.debug(f"Bee size updated: {self._calculated_bee_size:.1f} -> {new_bee_size:.1f} pixels")
            
            self._calculated_bee_size = new_bee_size
    
    def _get_bee_size(self):
        """
        Get bee size with intelligent fallback chain.
        
        Priority:
        1. User-provided bee_size
        2. Calculated from detections (IQR-filtered)
        3. Resolution-relative fallback (~2.5% of frame height)
        4. Hard fallback (40px)
        """
        # 1. User-provided
        if self.bee_size is not None:
            return self.bee_size
        
        # 2. Calculated from detections
        if self._calculated_bee_size is not None:
            return self._calculated_bee_size
        
        # 3. Resolution-relative fallback
        if self.frame_height is not None and self.frame_height > 0:
            fallback = self.frame_height * 0.025
            logger.debug(f"Using resolution-relative bee size fallback: {fallback:.1f} pixels")
            return fallback
        
        # 4. Hard fallback
        logger.debug("Using hard fallback bee size: 40.0 pixels")
        return 40.0
    
    def _calculate_iou(self, bbox1, bbox2):
        """Calculate IoU between two bounding boxes."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
        
        union_area = bbox1_area + bbox2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _get_cost_matrix(self, detections):
        """Calculate cost matrix between tracks and detections."""
        if len(self.tracks) == 0 or len(detections) == 0:
            return np.array([])
        
        cost_matrix = np.zeros((len(self.tracks), len(detections)))
        
        for i, track in enumerate(self.tracks):
            track_pos = track.centroid
            
            for j, detection in enumerate(detections):
                bbox = detection[:4]
                det_cx = (bbox[0] + bbox[2]) / 2
                det_cy = (bbox[1] + bbox[3]) / 2
                det_pos = np.array([det_cx, det_cy])
                
                # Euclidean distance
                distance = np.linalg.norm(track_pos - det_pos)
                
                # IoU
                iou = self._calculate_iou(track.last_bbox, bbox)
                
                # Combined cost (lower is better)
                cost = distance * (1 - iou)
                cost_matrix[i, j] = cost
        
        return cost_matrix
    
    def _is_duplicate_track(self, track: Track) -> bool:
        """
        Check if track is a duplicate using ADAPTIVE distance threshold.
        
        Args:
            track: Track to check
            
        Returns:
            True if track is likely a duplicate
        """
        if not track.is_confirmed:
            return False
        
        # Get adaptive threshold
        bee_size = self._get_bee_size()
        duplicate_threshold = bee_size * self.duplicate_distance_multiplier
        
        track_pos = np.array(track.centroid)
        
        # Check against all confirmed tracks
        for other_id in self.confirmed_track_ids:
            if other_id == track.id:
                continue
            
            # Find the other track
            other_track = None
            for t in self.tracks:
                if t.id == other_id:
                    other_track = t
                    break
            
            if other_track is None:
                continue
            
            # Check distance
            other_pos = np.array(other_track.centroid)
            distance = np.linalg.norm(track_pos - other_pos)
            
            if distance < duplicate_threshold:
                # Close tracks - check which is older/more stable
                if other_track.hits > track.hits:
                    logger.debug(f"Track {track.id} is duplicate of {other_track.id} (distance: {distance:.1f} < {duplicate_threshold:.1f})")
                    return True
                elif other_track.hits == track.hits and other_track.is_stable and not track.is_stable:
                    return True
        
        return False
    
    def _try_resurrect(self, detection, frame_num):
        """
        Try to resurrect a dead track using ADAPTIVE search radius.
        
        Args:
            detection: Detection to match
            frame_num: Current frame number
            
        Returns:
            Resurrected track if successful, None otherwise
        """
        if not self.dead_tracks:
            return None
        
        bbox = detection[:4]
        det_cx = (bbox[0] + bbox[2]) / 2
        det_cy = (bbox[1] + bbox[3]) / 2
        det_pos = np.array([det_cx, det_cy])
        
        best_track = None
        best_distance = float('inf')
        
        # Calculate ADAPTIVE search radius
        bee_size = self._get_bee_size()
        search_radius = bee_size * self.resurrection_search_multiplier
        
        logger.debug(f"Resurrection search radius: {search_radius:.1f} pixels ({self.resurrection_search_multiplier}x {bee_size:.1f})")
        
        for track, frame_died in self.dead_tracks[:]:
            # Remove tracks that have been dead too long
            if frame_num - frame_died > self.max_resurrection_frames:
                self.dead_tracks.remove((track, frame_died))
                continue
            
            # Check distance
            distance = track.get_distance_to(det_pos)
            
            # STRICTER resurrection criteria
            if distance < search_radius and distance < best_distance:
                # Additional checks for resurrection
                frames_dead = frame_num - frame_died
                
                # Prefer tracks that just died recently
                time_penalty = frames_dead / self.max_resurrection_frames
                adjusted_distance = distance * (1 + time_penalty)
                
                if adjusted_distance < best_distance:
                    best_distance = adjusted_distance
                    best_track = track
        
        if best_track is not None:
            # Resurrect track
            logger.debug(f"Resurrecting track {best_track.id} (distance: {best_distance:.1f}, radius: {search_radius:.1f})")
            self.dead_tracks = [(t, f) for t, f in self.dead_tracks if t.id != best_track.id]
            best_track.update(detection, frame_num)
            return best_track
        
        return None
    
    def update(self, detections, frame_num=None):
        """
        Update tracks with new detections.
        
        Args:
            detections: List of detections (x1, y1, x2, y2, ...)
            frame_num: Current frame number
            
        Returns:
            List of active track IDs and their centroids
        """
        if frame_num is None:
            frame_num = self.frame_count
        self.frame_count = frame_num
        
        # Continuously update bee size (v2.3 - rolling window)
        self._update_bee_size(detections)
        
        # Predict all tracks
        for track in self.tracks:
            track.predict()
        
        # Calculate cost matrix
        cost_matrix = self._get_cost_matrix(detections)
        
        # Hungarian assignment
        matched_tracks = set()
        matched_detections = set()
        
        if cost_matrix.size > 0:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            for row, col in zip(row_indices, col_indices):
                cost = cost_matrix[row, col]
                
                # Adaptive distance threshold
                bee_size = self._get_bee_size()
                max_distance = bee_size * self.match_distance_multiplier
                
                # IoU is already factored into cost, no need for hard gate
                if cost < max_distance:
                    track = self.tracks[row]
                    detection = detections[col]
                    track.update(detection, frame_num)
                    matched_tracks.add(row)
                    matched_detections.add(col)
        
        # Handle unmatched detections
        unmatched_detections = set(range(len(detections))) - matched_detections
        
        for det_idx in unmatched_detections:
            detection = detections[det_idx]
            
            # Try resurrection first (with ADAPTIVE radius)
            resurrected = self._try_resurrect(detection, frame_num)
            if resurrected is not None:
                self.tracks.append(resurrected)
            else:
                # Create new track
                new_track = Track(detection, frame_num, self.min_hits, self.max_age)
                self.tracks.append(new_track)
        
        # Remove dead/duplicate tracks
        active_tracks = []
        for track in self.tracks:
            if track.is_dead:
                # Move to dead tracks for potential resurrection
                self.dead_tracks.append((track, frame_num))
                logger.debug(f"Track {track.id} died at frame {frame_num}")
            elif self._is_duplicate_track(track):
                # Remove duplicate
                logger.debug(f"Removing duplicate track {track.id}")
            else:
                active_tracks.append(track)
                if track.is_confirmed:
                    self.confirmed_track_ids.add(track.id)
        
        self.tracks = active_tracks
        
        # Return confirmed tracks with clean format
        results = []
        for track in self.tracks:
            if track.is_confirmed:
                cx, cy = track.centroid
                x1, y1, x2, y2 = track.last_bbox
                results.append({
                    'track_id': track.id,
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'cx': float(cx),
                    'cy': float(cy),
                    'confidence': track.last_confidence,
                    'source': track.last_source,
                    'taxon': track.taxon,
                    'bee_id': track.bee_id,
                    'bee_id_method': track.bee_id_method,
                    'bee_id_confidence': track.bee_id_confidence,
                    'history': list(track.history)
                })
        
        return results
    
    def get_active_tracks(self):
        """Get all confirmed active tracks."""
        results = []
        for track in self.tracks:
            if track.is_confirmed:
                cx, cy = track.centroid
                x1, y1, x2, y2 = track.last_bbox
                results.append({
                    'track_id': track.id,
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'cx': float(cx),
                    'cy': float(cy),
                    'confidence': track.last_confidence,
                    'source': track.last_source,
                    'taxon': track.taxon,
                    'bee_id': track.bee_id,
                    'bee_id_method': track.bee_id_method,
                    'bee_id_confidence': track.bee_id_confidence,
                    'history': list(track.history)
                })
        return results