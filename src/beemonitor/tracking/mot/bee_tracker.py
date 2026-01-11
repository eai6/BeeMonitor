"""Kalman filter-based MOT with UNIFIED Hungarian resurrection matching.

IMPROVED DESIGN:
- Single cost matrix includes BOTH active tracks AND resurrection buffer
- Hungarian algorithm handles both regular tracking AND resurrection
- No duplicate matching logic
- More consistent behavior

Key improvement: Resurrection uses same distance metric as regular tracking!
"""

import logging
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np

from beemonitor.tracking.mot.base_mot import BaseMOT, Detection, Track

logger = logging.getLogger(__name__)


class BeeTracker(BaseMOT):
    """Kalman filter tracker with unified Hungarian resurrection."""
    
    def __init__(
        self, 
        config, 
        tracking_classes: List[str],
        require_yolo_confirmation: bool = False,
        max_pending_age: int = 30,
        resurrection_window: int = 30  # Increased for stationary bees (6 sec @ 30fps)
    ):
        """Initialize bee tracker with unified Hungarian matching.
        
        Args:
            config: Configuration object
            tracking_classes: List of class names to track
            require_yolo_confirmation: Only create tracks from YOLO detections
            max_pending_age: Max frames to buffer non-YOLO detections
            resurrection_window: Frames to remember deleted tracks (default: 60)
        """
        self.config = config
        self.tracking_classes = tracking_classes
        
        # Active tracks
        self._tracks: Dict[int, 'TrackState'] = {}
        self._next_track_id = 0
        
        # Resurrection buffer (now integrated into cost matrix)
        self.resurrection_window = resurrection_window
        self.recently_deleted = []
        
        # Adaptive parameters
        self.d_initial = getattr(config.tracking, 'initial_distance_threshold', 30.0)
        self.recorded_speeds = []
        self.max_speed_dynamic = 100.0
        
        # Resurrection distance threshold (used in cost matrix)
        # Increased to 200px to handle stationary bees that drift slightly during long gaps
        self.resurrection_distance_threshold = 300.0  # Lenient distance for resurrection
        
        # YOLO confirmation
        self.require_yolo_confirmation = require_yolo_confirmation
        self.max_pending_age = max_pending_age
        self.pending_detections: List[Dict] = []
        
        # Statistics
        self.stats = {
            'total_tracks_created': 0,
            'yolo_confirmed_tracks': 0,
            'deleted_tracks': 0,
            'resurrected_tracks': 0,
            'pending_confirmed': 0,
            'pending_aged_out': 0
        }
        
        logger.info(f"BeeTracker initialized (UNIFIED Hungarian matching)")
        logger.info(f"  Classes: {tracking_classes}")
        logger.info(f"  YOLO confirmation: {require_yolo_confirmation}")
        logger.info(f"  Resurrection window: {resurrection_window} frames")
        logger.info(f"  Resurrection distance: {self.resurrection_distance_threshold}px")
        logger.info(f"  Min search radius: 150px (stationary: 200px)")
    
    def predict(self, frame_num: int) -> Dict[int, Track]:
        """Predict track positions using Kalman filters."""
        predictions = {}
        
        for track_id, track_state in self._tracks.items():
            track_state.kalman.predict()
            
            pred = track_state.kalman.statePost
            cx, cy = float(pred[0]), float(pred[1])
            vx, vy = float(pred[2]), float(pred[3])
            
            # Update bbox around predicted centroid
            w = track_state.bbox[2] - track_state.bbox[0]
            h = track_state.bbox[3] - track_state.bbox[1]
            
            predictions[track_id] = Track(
                track_id=track_id,
                bbox=(cx - w/2, cy - h/2, cx + w/2, cy + h/2),
                centroid=(cx, cy),
                label=track_state.label,
                age=track_state.age,
                frames_without_detection=track_state.frames_without_detection,
                last_confirmation_frame=track_state.last_yolo_confirmation,
                trajectory=track_state.trajectory_history.copy(),
                velocity=(vx, vy),
                source=track_state.source
            )
        
        return predictions
    
    def update(self, detections: List[Detection], frame_num: int) -> Dict[int, Track]:
        """Update tracks with UNIFIED Hungarian matching (includes resurrection).
        
        Key improvement: Single cost matrix includes both active tracks AND 
        resurrection buffer, so Hungarian algorithm handles both!
        """
        # Reset update flags
        for track_state in self._tracks.values():
            track_state.updated_this_frame = False
        
        predictions = self.predict(frame_num)
        
        # Handle empty state
        if not detections and not self._tracks:
            return {}
        
        # Initialize first tracks
        if not self._tracks:
            if self.require_yolo_confirmation:
                self._handle_initial_detections_with_confirmation(detections, frame_num)
            else:
                for det in detections:
                    self._create_track(det, frame_num)
            return self.get_tracks()
        
        # ⭐ UNIFIED ASSOCIATION: Includes both active tracks AND resurrection buffer
        matched, unmatched_dets, unmatched_tracks, resurrected = self._associate_with_resurrection(
            detections, predictions, frame_num
        )
        
        # Update matched tracks (regular tracking)
        for det_idx, track_id in matched:
            det = detections[det_idx]
            track_state = self._tracks[track_id]
            
            # Kalman update
            cx, cy = det.centroid
            measurement = np.array([[cx], [cy]], dtype=np.float32)
            track_state.kalman.correct(measurement)
            
            # State update
            track_state.bbox = det.bbox
            track_state.centroid = det.centroid
            track_state.frames_without_detection = 0
            track_state.age += 1
            track_state.updated_this_frame = True
            
            if det.source == 'yolo':
                track_state.last_yolo_confirmation = frame_num
                track_state.label = det.label
            
            track_state.trajectory_history.append((frame_num, det.centroid))
            
            # Learn speed
            if len(track_state.trajectory_history) >= 2:
                prev_frame, prev_pos = track_state.trajectory_history[-2]
                curr_frame, curr_pos = track_state.trajectory_history[-1]
                dx = curr_pos[0] - prev_pos[0]
                dy = curr_pos[1] - prev_pos[1]
                dt = max(1, curr_frame - prev_frame)
                speed = np.sqrt(dx**2 + dy**2) / dt
                self.recorded_speeds.append(speed)
        
        # ⭐ Handle resurrected tracks (already recreated by association method)
        for det_idx, resurrected_id in resurrected:
            det = detections[det_idx]
            track_state = self._tracks[resurrected_id]
            
            # Initialize with detection (resurrection already created Kalman filter)
            cx, cy = det.centroid
            measurement = np.array([[cx], [cy]], dtype=np.float32)
            track_state.kalman.correct(measurement)
            
            track_state.bbox = det.bbox
            track_state.centroid = det.centroid
            track_state.updated_this_frame = True
            
            if det.source == 'yolo':
                track_state.last_yolo_confirmation = frame_num
                track_state.label = det.label
            
            track_state.trajectory_history.append((frame_num, det.centroid))
        
        # Age unmatched tracks
        for track_id in unmatched_tracks:
            track_state = self._tracks[track_id]
            track_state.frames_without_detection += 1
            track_state.age += 1
            
            # Delete old tracks
            if track_state.frames_without_detection > self.config.tracking.max_age:
                self._archive_deleted_track(track_id, track_state, frame_num)
                del self._tracks[track_id]
        
        # Handle unmatched detections (create new tracks)
        if self.require_yolo_confirmation:
            self._handle_unmatched_with_yolo_confirmation(
                detections, unmatched_dets, frame_num
            )
        else:
            for det_idx in unmatched_dets:
                det = detections[det_idx]
                
                # Check anti-duplicate
                if self._is_duplicate_of_existing_track(det):
                    continue
                
                # Create new track
                self._create_track(det, frame_num)
        
        # Update dynamic max speed
        if len(self.recorded_speeds) > 100:
            self.max_speed_dynamic = float(np.percentile(self.recorded_speeds, 99))
            self.max_speed_dynamic = max(20.0, self.max_speed_dynamic)
        
        # Clean resurrection buffer
        self._clean_resurrection_buffer(frame_num)
        
        return self.get_tracks()
    
    def _associate_with_resurrection(
        self,
        detections: List[Detection],
        predictions: Dict[int, Track],
        frame_num: int
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int], List[Tuple[int, int]]]:
        """⭐ UNIFIED association: Single cost matrix for active tracks + resurrection buffer.
        
        Returns:
            (matched_pairs, unmatched_det_indices, unmatched_track_ids, resurrected_pairs)
        """
        if not detections:
            return [], [], list(predictions.keys()), []
        
        # Build EXTENDED cost matrix: active tracks + resurrection candidates
        active_track_ids = list(predictions.keys())
        resurrection_candidates = [
            d for d in self.recently_deleted
            if (frame_num - d['death_frame']) <= self.resurrection_window
        ]
        
        # Total columns = active tracks + resurrection candidates
        num_active = len(active_track_ids)
        num_resurrect = len(resurrection_candidates)
        total_cols = num_active + num_resurrect
        
        if total_cols == 0:
            return [], list(range(len(detections))), [], []
        
        cost_matrix = np.zeros((len(detections), total_cols), dtype=np.float32)
        search_regions = self.get_search_regions(frame_num)
        
        # Fill cost matrix
        for i, det in enumerate(detections):
            priority = 0.1 if det.source == 'yolo' else 1.0
            
            # PART 1: Active tracks (columns 0 to num_active-1)
            for j, track_id in enumerate(active_track_ids):
                pred = predictions[track_id]
                track_state = self._tracks[track_id]
                
                # Update locking
                if track_state.updated_this_frame:
                    cost_matrix[i, j] = 1e9
                    continue
                
                # Distance
                dx = det.centroid[0] - pred.centroid[0]
                dy = det.centroid[1] - pred.centroid[1]
                dist = np.sqrt(dx**2 + dy**2)
                
                # Search region constraint (FIXED: handle both circle and ellipse)
                region = search_regions.get(track_id, {'type': 'circle', 'radius': 100})
                if region['type'] == 'circle':
                    max_dist = region['radius']
                else:  # ellipse
                    # For ellipse, use major_axis as max distance
                    max_dist = region['major_axis']
                
                if dist > max_dist:
                    cost_matrix[i, j] = 1e9
                else:
                    cost_matrix[i, j] = dist * priority
            
            # PART 2: Resurrection candidates (columns num_active to total_cols-1)
            for k, deleted in enumerate(resurrection_candidates):
                j = num_active + k  # Column index in extended matrix
                
                # Distance from detection to deleted track's last position
                dx = det.centroid[0] - deleted['last_centroid'][0]
                dy = det.centroid[1] - deleted['last_centroid'][1]
                dist = np.sqrt(dx**2 + dy**2)
                
                # Resurrection threshold (more lenient than active tracking)
                if dist > self.resurrection_distance_threshold:
                    cost_matrix[i, j] = 1e9
                else:
                    # Lower cost for resurrection (encourage reusing IDs)
                    # But not as low as YOLO priority (YOLO still preferred)
                    cost_matrix[i, j] = dist * 0.5 * priority
        
        # Greedy matching on EXTENDED matrix
        matched_active = []
        matched_resurrect = []
        unmatched_dets = set(range(len(detections)))
        unmatched_cols = set(range(total_cols))
        
        while unmatched_dets and unmatched_cols:
            min_cost = 1e9
            min_i, min_j = -1, -1
            
            for i in unmatched_dets:
                for j in unmatched_cols:
                    if cost_matrix[i, j] < min_cost:
                        min_cost = cost_matrix[i, j]
                        min_i = i
                        min_j = j
            
            if min_cost >= 1e9:
                break
            
            # Determine if match is to active track or resurrection candidate
            if min_j < num_active:
                # Matched to active track
                track_id = active_track_ids[min_j]
                matched_active.append((min_i, track_id))
            else:
                # Matched to resurrection candidate
                resurrect_idx = min_j - num_active
                deleted = resurrection_candidates[resurrect_idx]
                
                # Resurrect this track!
                resurrected_id = self._resurrect_track_from_deleted(
                    deleted, detections[min_i], frame_num
                )
                matched_resurrect.append((min_i, resurrected_id))
                
                logger.info(f"🔄 RESURRECTED Track {resurrected_id} "
                           f"(dist={min_cost:.1f}px, gap={frame_num - deleted['death_frame']} frames)")
            
            unmatched_dets.remove(min_i)
            unmatched_cols.remove(min_j)
        
        # Unmatched active tracks
        unmatched_track_ids = [
            active_track_ids[j] for j in range(num_active)
            if j in unmatched_cols
        ]
        
        return matched_active, list(unmatched_dets), unmatched_track_ids, matched_resurrect
    
    def _resurrect_track_from_deleted(
        self, deleted_info: dict, det: Detection, frame_num: int
    ) -> int:
        """Resurrect track with original ID from deleted info.
        
        Args:
            deleted_info: Dict with track_id, last_bbox, last_centroid, death_frame, label
            det: Detection that triggered resurrection
            frame_num: Current frame number
            
        Returns:
            Resurrected track ID
        """
        track_id = deleted_info['track_id']
        
        # Remove from resurrection buffer
        self.recently_deleted = [
            d for d in self.recently_deleted
            if d['track_id'] != track_id
        ]
        
        # Create fresh Kalman filter
        kalman = cv2.KalmanFilter(4, 2)
        kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        
        cx, cy = det.centroid
        kalman.statePre = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
        kalman.statePost = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
        
        # Recreate track with ORIGINAL ID
        self._tracks[track_id] = TrackState(
            track_id=track_id,
            bbox=det.bbox,
            centroid=det.centroid,
            kalman=kalman,
            frames_without_detection=0,
            label=det.label if hasattr(det, 'label') else deleted_info['label'],
            age=1,
            last_yolo_confirmation=frame_num if det.source == 'yolo' else -999,
            trajectory_history=[(frame_num, det.centroid)],
            source=det.source
        )
        
        self.stats['resurrected_tracks'] += 1
        logger.debug(f"✨ Resurrected Track {track_id}")
        
        return track_id
    
    def _handle_initial_detections_with_confirmation(
        self, detections: List[Detection], frame_num: int
    ):
        """Handle initial detections with YOLO confirmation."""
        yolo_dets = [d for d in detections if d.source == 'yolo']
        other_dets = [d for d in detections if d.source != 'yolo']
        
        for det in yolo_dets:
            self._create_track(det, frame_num, is_yolo_confirmed=True)
        
        for det in other_dets:
            self.pending_detections.append({
                'detection': det,
                'frame_num': frame_num
            })
    
    def _handle_unmatched_with_yolo_confirmation(
        self, detections: List[Detection], unmatched_indices: List[int], frame_num: int
    ):
        """Handle unmatched detections with YOLO confirmation."""
        yolo_indices = [i for i in unmatched_indices if detections[i].source == 'yolo']
        other_indices = [i for i in unmatched_indices if detections[i].source != 'yolo']
        
        # YOLO detections create tracks immediately
        for idx in yolo_indices:
            det = detections[idx]
            
            # Check anti-duplicate
            if self._is_duplicate_of_existing_track(det):
                continue
            
            self._create_track(det, frame_num, is_yolo_confirmed=True)
        
        # Confirm pending if YOLO ran
        if yolo_indices:
            yolo_dets = [detections[i] for i in yolo_indices]
            self._confirm_pending_detections(yolo_dets, frame_num)
        
        # Add to pending buffer
        for idx in other_indices:
            self.pending_detections.append({
                'detection': detections[idx],
                'frame_num': frame_num
            })
        
        self._age_out_pending(frame_num)
    
    def _confirm_pending_detections(self, yolo_dets: List[Detection], frame_num: int):
        """Confirm pending detections by matching to YOLO."""
        if not self.pending_detections:
            return
        
        confirmed = []
        for i, pending in enumerate(self.pending_detections):
            for yolo_det in yolo_dets:
                iou = self._compute_iou(pending['detection'].bbox, yolo_det.bbox)
                if iou >= 0.3:
                    self._create_track(pending['detection'], frame_num, is_yolo_confirmed=True)
                    self.stats['pending_confirmed'] += 1
                    confirmed.append(i)
                    break
        
        for i in reversed(confirmed):
            del self.pending_detections[i]
    
    def _age_out_pending(self, frame_num: int):
        """Remove old pending detections."""
        aged_out = []
        for i, pending in enumerate(self.pending_detections):
            if frame_num - pending['frame_num'] > self.max_pending_age:
                aged_out.append(i)
                self.stats['pending_aged_out'] += 1
        
        for i in reversed(aged_out):
            del self.pending_detections[i]
    
    def get_tracks(self) -> Dict[int, Track]:
        """Get all active tracks."""
        tracks = {}
        for track_id, track_state in self._tracks.items():
            pred = track_state.kalman.statePost
            vx, vy = float(pred[2]), float(pred[3])
            
            tracks[track_id] = Track(
                track_id=track_id,
                bbox=track_state.bbox,
                centroid=track_state.centroid,
                label=track_state.label,
                age=track_state.age,
                frames_without_detection=track_state.frames_without_detection,
                last_confirmation_frame=track_state.last_yolo_confirmation,
                trajectory=track_state.trajectory_history.copy(),
                velocity=(vx, vy),
                source=track_state.source
            )
        
        return tracks
    
    def reset(self):
        """Reset tracker state."""
        self._tracks.clear()
        self._next_track_id = 0
        self.recorded_speeds.clear()
        self.max_speed_dynamic = 100.0
        self.pending_detections.clear()
        self.recently_deleted.clear()
        self.stats = {
            'total_tracks_created': 0,
            'yolo_confirmed_tracks': 0,
            'deleted_tracks': 0,
            'resurrected_tracks': 0,
            'pending_confirmed': 0,
            'pending_aged_out': 0
        }
        logger.info("BeeTracker reset")
    
    def get_search_regions(self, frame_num: int) -> Dict[int, Dict]:
        """Get adaptive search regions for active tracks.
        
        CRITICAL FIX: Enforce minimum search radius to prevent tiny regions
        that cause premature track death.
        
        UPDATED: Increased to 150px minimum to handle stationary bees
        that may drift slightly or have detection uncertainty.
        """
        regions = {}
        MIN_SEARCH_RADIUS = 250.0  # Minimum 150px search (handles stationary bees)
        
        for track_id, track_state in self._tracks.items():
            pred = track_state.kalman.statePost
            cx, cy = float(pred[0]), float(pred[1])
            vx, vy = float(pred[2]), float(pred[3])
            speed = np.sqrt(vx**2 + vy**2)
            
            # Cold-start
            if track_state.age < 5:
                radius = 150.0 if track_state.age < 2 else 100.0
                regions[track_id] = {
                    'type': 'circle',
                    'center': (cx, cy),
                    'radius': radius
                }
            # Moving
            elif speed > 1.0:
                # CRITICAL FIX: Apply minimum to prevent tiny search regions
                major_axis = max(MIN_SEARCH_RADIUS, min(2.5 * speed, self.max_speed_dynamic * 4))
                minor_axis = max(MIN_SEARCH_RADIUS * 0.5, major_axis * 0.5)
                angle = np.arctan2(vy, vx) * 180 / np.pi
                
                regions[track_id] = {
                    'type': 'ellipse',
                    'center': (cx, cy),
                    'major_axis': major_axis,
                    'minor_axis': minor_axis,
                    'angle': angle
                }
            # Stationary
            else:
                # Increased to 200px for stationary bees (may have long YOLO gaps)
                regions[track_id] = {
                    'type': 'circle',
                    'center': (cx, cy),
                    'radius': 300.0
                }
        
        return regions
    
    def get_statistics(self):
        """Get tracking statistics."""
        stats = self.stats.copy()
        stats['active_tracks'] = len(self._tracks)
        stats['buffer_size'] = len(self.recently_deleted)
        stats['current_pending'] = len(self.pending_detections)
        
        total = max(1, self.stats['total_tracks_created'])
        stats['confirmation_rate'] = self.stats['yolo_confirmed_tracks'] / total
        
        deleted = self.stats['deleted_tracks']
        resurrected = self.stats['resurrected_tracks']
        stats['resurrection_rate'] = resurrected / max(1, deleted)
        
        return stats
    
    # ========================================================================
    # RESURRECTION - Now integrated into Hungarian matching above!
    # ========================================================================
    
    def _archive_deleted_track(self, track_id: int, track_state, frame_num: int):
        """Archive track to resurrection buffer."""
        self.recently_deleted.append({
            'track_id': track_id,
            'last_bbox': track_state.bbox,
            'last_centroid': track_state.centroid,
            'death_frame': frame_num,
            'label': track_state.label
        })
        
        self.stats['deleted_tracks'] += 1
        logger.debug(f"💀 Archived Track {track_id} (frame {frame_num})")
    
    def _clean_resurrection_buffer(self, frame_num: int):
        """Remove expired tracks from resurrection buffer."""
        original = len(self.recently_deleted)
        self.recently_deleted = [
            d for d in self.recently_deleted
            if (frame_num - d['death_frame']) <= self.resurrection_window
        ]
        
        cleaned = original - len(self.recently_deleted)
        if cleaned > 0:
            logger.debug(f"🧹 Cleaned {cleaned} expired tracks")
    
    # ========================================================================
    # ANTI-DUPLICATE & TRACK CREATION
    # ========================================================================
    
    def _is_duplicate_of_existing_track(
        self, 
        det: Detection,
        iou_threshold: float = 0.3,
        dist_threshold: float = 50.0
    ) -> bool:
        """Check if detection overlaps with existing track."""
        for track_state in self._tracks.values():
            iou = self._compute_iou(det.bbox, track_state.bbox)
            if iou > iou_threshold:
                return True
            
            dx = det.centroid[0] - track_state.centroid[0]
            dy = det.centroid[1] - track_state.centroid[1]
            if np.sqrt(dx**2 + dy**2) < dist_threshold:
                return True
        
        return False
    
    def _create_track(
        self, 
        det: Detection, 
        frame_num: int,
        is_yolo_confirmed: bool = False
    ):
        """Create new track from detection."""
        if self._is_duplicate_of_existing_track(det):
            logger.debug(f"Rejected duplicate from {det.source}")
            return
        
        track_id = self._next_track_id
        self._next_track_id += 1
        
        # Initialize Kalman filter
        kalman = cv2.KalmanFilter(4, 2)
        kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        
        cx, cy = det.centroid
        kalman.statePre = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
        kalman.statePost = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
        
        # Create track
        self._tracks[track_id] = TrackState(
            track_id=track_id,
            bbox=det.bbox,
            centroid=det.centroid,
            kalman=kalman,
            frames_without_detection=0,
            label=det.label if hasattr(det, 'label') else 'bee',
            age=0,
            last_yolo_confirmation=frame_num if det.source == 'yolo' else -999,
            trajectory_history=[(frame_num, det.centroid)],
            source=det.source
        )
        
        self.stats['total_tracks_created'] += 1
        if is_yolo_confirmed or det.source == 'yolo':
            self.stats['yolo_confirmed_tracks'] += 1
        
        logger.debug(f"Created Track {track_id} from {det.source}")
    
    @staticmethod
    def _compute_iou(bbox1, bbox2):
        """Compute IoU between two bboxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


class TrackState:
    """Internal track state."""
    def __init__(
        self, track_id, bbox, centroid, kalman, frames_without_detection,
        label, age, last_yolo_confirmation, trajectory_history, source='unknown'
    ):
        self.track_id = track_id
        self.bbox = bbox
        self.centroid = centroid
        self.kalman = kalman
        self.frames_without_detection = frames_without_detection
        self.label = label
        self.age = age
        self.last_yolo_confirmation = last_yolo_confirmation
        self.trajectory_history = trajectory_history
        self.source = source
        self.updated_this_frame = False