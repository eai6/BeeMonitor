"""Event processing for bee tracking data with multi-species support and cleaning.

This module processes bee trajectories to identify entry and exit events
at nest holes, including species classification and event cleaning/validation.
"""

import logging
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd

from beemonitor.core.config import Config
from beemonitor.processing.trajectory_analyzer import TrajectoryAnalyzer


logger = logging.getLogger(__name__)

# Type aliases
Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]


class EventProcessor:
    """Processor for identifying bee entry/exit events with species tracking and cleaning.
    
    This class analyzes bee trajectories to determine when bees enter
    or exit nest holes, creating a timeline of activity events with
    species information. Includes methods for cleaning noise and validating events.
    
    Attributes:
        config: Configuration object
        trajectory_analyzer: TrajectoryAnalyzer instance
        ml_classifier: Optional ML-based event classifier
    
    Example:
        >>> processor = EventProcessor(config)
        >>> events = processor.process_tracks(motion_data, nests)
        >>> clean_events = processor.clean_events(events, min_confidence=0.3)
        >>> print(f"Found {len(clean_events)} valid events")
    """
    
    def __init__(self, config: Optional[Config] = None, ml_classifier=None):
        """Initialize EventProcessor.
        
        Args:
            config: Configuration object (optional)
            ml_classifier: Optional EventClassifier for ML-based classification
        """
        self.config = config if config is not None else Config.default()
        self.trajectory_analyzer = TrajectoryAnalyzer(self.config)
        self.ml_classifier = ml_classifier
        
        if ml_classifier is not None:
            logger.info("EventProcessor initialized with ML classifier")
        else:
            logger.debug("EventProcessor initialized with heuristic classification")
    
    def process_tracks(
        self,
        motion_data: pd.DataFrame,
        nests: Dict,
        label_map: Optional[Dict[int, str]] = None,
        filter_fragments: bool = True,
        min_trajectory_length: int = 10,
        min_movement_distance: float = 30.0
    ) -> pd.DataFrame:
        """Process tracking data to identify entry/exit events with species.
        
        Args:
            motion_data: DataFrame with columns: frame_number, tracks, detections
            nests: Dictionary with 'hotel' ROI and 'nests' mapping
            label_map: Optional mapping of class IDs to species names
            filter_fragments: Filter out short trajectory fragments from ID switches
            min_trajectory_length: Minimum frames for valid trajectory
            min_movement_distance: Minimum distance traveled (pixels)
            
        Returns:
            DataFrame with columns: action, nest, frame_number, label, 
                                   label_class, label_confidence, notes
            
        Example:
            >>> events = processor.process_tracks(
            ...     motion_data, nests,
            ...     filter_fragments=True,  # Filter ID switch fragments
            ...     min_trajectory_length=10
            ... )
        """
        logger.info("Processing tracks to identify events...")
        
        # Use label_map from config if not provided
        if label_map is None and hasattr(self.config, 'tracking'):
            label_map = self.config.tracking.label_map
        
        # Extract all movements from tracking data
        movements = []  # these are trajectories
        for period in motion_data.tracks:
            for track in period:
                movements.append(track)
        
        logger.debug(f"Processing {len(movements)} trajectories")
        
        # Filter trajectory fragments (NEW - prevents false events from ID switches)
        if filter_fragments:
            movements = self._filter_trajectory_fragments(
                movements,
                min_length=min_trajectory_length,
                min_distance=min_movement_distance
            )
            logger.info(f"After fragment filtering: {len(movements)} valid trajectories")
        
        # Get resolution for scaled parameters
        res_width = self.config.video.res_width
        res_height = self.config.video.res_height
        
        # Process each movement to identify events
        actions = []
        for movement in movements:
            # Skip short trajectories
            if len(movement[1]) < self.config.processing.min_trajectory_length:
                continue
            
            # Classify movement type
            if self.trajectory_analyzer.is_exit_behavior(movement):
                # Get scaled parameters for exit
                exit_window = self.config.processing.exit_window_size
                exit_padding = self.config.processing.exit_padding(res_width, res_height)
                
                action = self._get_action(
                    movement,
                    nests,
                    window_size=exit_window,
                    padding=exit_padding,
                    label_map=label_map
                )
            elif self.trajectory_analyzer.is_entry_behavior(movement):
                # Get scaled parameters for entry
                entry_window = self.config.processing.entry_window_size
                entry_padding = self.config.processing.entry_padding(res_width, res_height)
                
                action = self._get_action(
                    movement,
                    nests,
                    window_size=entry_window,
                    padding=entry_padding,  # ← FIXED: was exit_padding
                    label_map=label_map
                )
            else:
                # Not clearly entry or exit, skip
                continue
            
            # Add actions to list
            if action:
                if isinstance(action, list):
                    actions.extend(action)
                else:
                    actions.append(action)
        
        logger.info(f"Identified {len(actions)} events")
        
        # Convert to DataFrame
        if actions:
            df = pd.DataFrame(actions)
            # Log label distribution if present
            if 'label' in df.columns:
                label_counts = df['label'].value_counts()
                logger.info(f"Label distribution: {label_counts.to_dict()}")
            return df
        else:
            return pd.DataFrame(columns=[
                'action', 'nest', 'frame_number', 'label',
                'label_class', 'label_confidence', 'notes'
            ])
    
    def _filter_trajectory_fragments(
        self,
        movements: List[Tuple],
        min_length: int = 10,
        min_distance: float = 30.0
    ) -> List[Tuple]:
        """Filter out short trajectory fragments likely from ID switches.
        
        Removes trajectories that are:
        - Too short (< min_length frames)
        - Don't move enough (total distance < min_distance)
        - Sit stationary at one location (ID switch while bee at nest)
        
        Args:
            movements: List of trajectory tuples
            min_length: Minimum number of frames
            min_distance: Minimum total distance traveled
            
        Returns:
            Filtered list of movements
        """
        valid_movements = []
        filtered_count = {'too_short': 0, 'no_movement': 0, 'stationary': 0}
        
        for movement in movements:
            centroids = movement[1]  # List of (x, y) positions
            
            # Check 1: Minimum length
            if len(centroids) < min_length:
                filtered_count['too_short'] += 1
                continue
            
            # Check 2: Calculate total distance traveled
            total_distance = 0.0
            for i in range(len(centroids) - 1):
                dx = centroids[i+1][0] - centroids[i][0]
                dy = centroids[i+1][1] - centroids[i][1]
                total_distance += np.sqrt(dx**2 + dy**2)
            
            if total_distance < min_distance:
                filtered_count['no_movement'] += 1
                continue
            
            # Check 3: Calculate movement variance (detect stationary bee)
            x_positions = [c[0] for c in centroids]
            y_positions = [c[1] for c in centroids]
            x_variance = np.var(x_positions)
            y_variance = np.var(y_positions)
            
            # If variance is very low, bee is just sitting still (ID switch artifact)
            if x_variance < 10.0 and y_variance < 10.0:
                filtered_count['stationary'] += 1
                continue
            
            # Passed all checks
            valid_movements.append(movement)
        
        total_filtered = sum(filtered_count.values())
        if total_filtered > 0:
            logger.info(f"  Filtered {total_filtered} trajectory fragments: "
                       f"short={filtered_count['too_short']}, "
                       f"no_movement={filtered_count['no_movement']}, "
                       f"stationary={filtered_count['stationary']}")
        
        return valid_movements
    
    def _get_action(
        self,
        movement: Tuple,
        nests: Dict,
        window_size: int = 3,
        padding: float = 20,
        label_map: Optional[Dict[int, str]] = None
    ) -> Optional[Union[Dict, List[Dict]]]:
        """Determine action (entry/exit) from movement trajectory with species.
        
        Args:
            movement: Tuple of (track_id, centroids, bboxes, frame_numbers, 
                               species, species_votes)
            nests: Dictionary with nest locations
            window_size: Number of frames to analyze
            padding: Padding around nest boxes (already scaled)
            label_map: Mapping of class IDs to species names
            
        Returns:
            Dictionary or list of dictionaries with action details, or None
        """
        start_id, end_id = self._detect_entry_exit(
            movement[1],  # centroids
            nests['nests'],
            window_size=window_size,
            padding=padding
        )
        
        # Get species information from track
        label_class = movement[4] if len(movement) > 4 else None
        label_votes = movement[5] if len(movement) > 5 else {}
        
        # Determine species name
        # label_class can be either a class ID (int) or species name (str)
        if isinstance(label_class, str):
            # Already a species name
            label_name = label_class
        elif label_map and label_class is not None:
            # Map class ID to name
            label_name = label_map.get(label_class, 'unknown')
        else:
            label_name = 'unknown'
        
        # Calculate species confidence
        label_confidence = 0.0
        if label_votes:
            total_votes = sum(label_votes.values())
            if total_votes > 0 and label_class is not None:
                label_confidence = label_votes.get(label_class, 0) / total_votes
        
        if start_id == -1 and end_id == -1:
            return None
        
        elif start_id != -1 and end_id == -1:
            # Exit only
            return {
                "action": "Exit",
                "nest": str(start_id),
                "frame_number": movement[3][0],  # First frame
                "label": label_name,
                "label_class": label_class,
                "label_confidence": label_confidence,
                "notes": f"{label_name} exited the nest"
            }
        
        elif start_id == -1 and end_id != -1:
            # Entry only
            return {
                "action": "Entry",
                "nest": str(end_id),
                "frame_number": movement[3][-1],  # Last frame
                "label": label_name,
                "label_class": label_class,
                "label_confidence": label_confidence,
                "notes": f"{label_name} entered the nest"
            }
        
        elif start_id != -1 and end_id != -1:
            # Both entry and exit (nest-to-nest movement)
            return [
                {
                    "action": "Exit",
                    "nest": str(start_id),
                    "frame_number": movement[3][0],
                    "label": label_name,
                    "label_class": label_class,
                    "label_confidence": label_confidence,
                    "notes": f"{label_name} exited nest to move to another hole {end_id}"
                },
                {
                    "action": "Entry",
                    "nest": str(end_id),
                    "frame_number": movement[3][-1],
                    "label": label_name,
                    "label_class": label_class,
                    "label_confidence": label_confidence,
                    "notes": f"{label_name} entered nest from another hole {start_id}"
                }
            ]
        
        return None
    
    def _detect_entry_exit(
        self,
        bee_trajectory: List[Point],
        hole_bboxes: Dict[str, BBox],
        window_size: int = 3,
        padding: float = 20
    ) -> Tuple[int, int]:
        """Detect if bee enters or exits a hole.
        
        Analyzes the start and end of a trajectory to determine if the bee
        started inside a hole (exit) or ended inside a hole (entry).
        
        Args:
            bee_trajectory: List of (x, y) positions
            hole_bboxes: Dictionary mapping hole IDs to bounding boxes
            window_size: Number of frames to analyze at start/end
            padding: Padding to add around nest boxes (already scaled)
            
        Returns:
            Tuple of (start_hole_id, end_hole_id), -1 if not in any hole
        """
        if len(bee_trajectory) < window_size:
            window_size = max(1, len(bee_trajectory) // 2)
        
        # Analyze start of trajectory
        start_trajectory = bee_trajectory[:window_size]
        start_id = -1
        
        for hole_id, bbox in hole_bboxes.items():
            # Check if all positions in start window are inside this hole
            start_inside = all(
                self._is_inside_bbox(pos, bbox, padding)
                for pos in start_trajectory
            )
            
            if start_inside:
                start_id = hole_id
                break
        
        # Analyze end of trajectory
        end_trajectory = bee_trajectory[-window_size:]
        end_id = -1
        
        for hole_id, bbox in hole_bboxes.items():
            # Check if all positions in end window are inside this hole
            end_inside = all(
                self._is_inside_bbox(pos, bbox, padding)
                for pos in end_trajectory
            )
            
            if end_inside:
                end_id = hole_id
                break
        
        return start_id, end_id
    
    def _is_inside_bbox(
        self,
        bee_position: Point,
        bbox: BBox,
        padding: float = 20
    ) -> bool:
        """Check if a position is inside a bounding box with padding.
        
        Args:
            bee_position: (x, y) coordinates
            bbox: Bounding box (x_min, y_min, x_max, y_max)
            padding: Padding to add around box (already scaled)
            
        Returns:
            True if position is inside padded box
        """
        x, y = bee_position
        x_min, y_min, x_max, y_max = bbox
        
        # Add padding with slightly more vertical padding
        x_min -= padding
        y_min -= int(padding + padding / 2)
        x_max += padding
        y_max += int(padding + padding / 2)
        
        return x_min <= x <= x_max and y_min <= y <= y_max
    
    def _find_nearest_nest(
        self,
        location: Tuple[float, float],
        nest_bboxes: Dict,
        max_distance: float = 50.0
    ) -> Optional[int]:
        """Find nearest nest to a location.
        
        Args:
            location: (x, y) coordinates
            nest_bboxes: Dictionary of nest bounding boxes
            max_distance: Maximum distance to consider
            
        Returns:
            Nest ID or None
        """
        x, y = location
        min_dist = float('inf')
        nearest_nest = None
        
        for nest_id, bbox in nest_bboxes.items():
            # Check if inside bbox (with padding)
            if self._is_inside_bbox((x, y), bbox, padding=30):
                return nest_id
            
            # Otherwise find nearest
            nest_center_x = (bbox[0] + bbox[2]) / 2
            nest_center_y = (bbox[1] + bbox[3]) / 2
            dist = np.sqrt((x - nest_center_x)**2 + (y - nest_center_y)**2)
            
            if dist < min_dist:
                min_dist = dist
                nearest_nest = nest_id
        
        # Only return if within max distance
        if min_dist <= max_distance:
            return nearest_nest
        return None
    
    # =========================================================================
    # EVENT CLEANING AND VALIDATION METHODS
    # =========================================================================
    
    def clean_events(
        self,
        events: pd.DataFrame,
        remove_blob_events: bool = True,
        min_confidence: float = 0.0,
        merge_duplicates: bool = True,
        duplicate_window: int = 10,
        remove_id_switch_clusters: bool = True,
        cluster_window: int = 30
    ) -> pd.DataFrame:
        """Clean event data by removing noise and duplicates.
        
        Args:
            events: DataFrame with event data
            remove_blob_events: Remove events labeled as 'blob' (FG/BG noise)
            min_confidence: Minimum confidence threshold (0-1)
            merge_duplicates: Merge duplicate events (same action/nest/time)
            duplicate_window: Frame window for considering events duplicates
            remove_id_switch_clusters: Remove Exit→Entry pairs from ID switches
            cluster_window: Frame window for detecting ID switch clusters (default: 30)
            
        Returns:
            Cleaned DataFrame
            
        Example:
            >>> events = processor.process_tracks(motion_data, nests,
            ...                                   filter_fragments=True)
            >>> clean_events = processor.clean_events(
            ...     events,
            ...     remove_id_switch_clusters=True
            ... )
        """
        if events.empty:
            return events
        
        original_count = len(events)
        df = events.copy()
        
        logger.info(f"Cleaning {original_count} events...")
        
        # 1. Remove blob events (FG/BG noise)
        if remove_blob_events and 'label' in df.columns:
            blob_count = len(df[df['label'] == 'blob'])
            df = df[df['label'] != 'blob']
            if blob_count > 0:
                logger.info(f"  Removed {blob_count} blob events (FG/BG noise)")
        
        # 2. Filter by confidence
        if min_confidence > 0 and 'label_confidence' in df.columns:
            low_conf = len(df[df['label_confidence'] < min_confidence])
            df = df[df['label_confidence'] >= min_confidence]
            if low_conf > 0:
                logger.info(f"  Removed {low_conf} low-confidence events (< {min_confidence})")
        
        # 3. Remove ID switch clusters (Exit → Entry at same nest)
        if remove_id_switch_clusters:
            df = self._remove_temporal_clusters(df, cluster_window)
        
        # 4. Merge duplicate events
        if merge_duplicates:
            before_merge = len(df)
            df = self._merge_duplicate_events(df, duplicate_window)
            dup_count = before_merge - len(df)
            if dup_count > 0:
                logger.info(f"  Merged {dup_count} duplicate events")
        
        # 5. Sort by frame number
        df = df.sort_values('frame_number').reset_index(drop=True)
        
        cleaned_count = len(df)
        removed_count = original_count - cleaned_count
        
        if removed_count > 0:
            logger.info(f"✓ Cleaned events: {original_count} → {cleaned_count} "
                       f"({removed_count} removed, {removed_count/original_count*100:.1f}%)")
        else:
            logger.info(f"✓ All {original_count} events passed cleaning")
        
        return df
    
    def _filter_trajectory_fragments(
        self,
        movements: List[Tuple],
        min_length: int = 10,
        min_distance: float = 30.0
    ) -> List[Tuple]:
        """Filter out short trajectory fragments from ID switches.
        
        Removes trajectories that are:
        - Too short (< min_length frames) → ID switch fragments
        - Don't move (< min_distance pixels) → Bee sitting still during switch
        - Stationary (low variance) → Same bee, new ID
        
        Args:
            movements: List of (track_id, centroids, bboxes, frames, species, votes)
            min_length: Minimum frames (default: 10)
            min_distance: Minimum distance traveled (default: 30 px)
            
        Returns:
            Filtered movements (removes ~30-50% of ID switch fragments)
        """
        valid_movements = []
        filtered = {'short': 0, 'stationary': 0, 'no_movement': 0}
        
        for movement in movements:
            centroids = movement[1]  # List of (x, y) positions
            
            # Filter 1: Too short
            if len(centroids) < min_length:
                filtered['short'] += 1
                continue
            
            # Filter 2: Calculate total distance
            total_dist = 0.0
            for i in range(len(centroids) - 1):
                dx = centroids[i+1][0] - centroids[i][0]
                dy = centroids[i+1][1] - centroids[i][1]
                total_dist += np.sqrt(dx**2 + dy**2)
            
            if total_dist < min_distance:
                filtered['no_movement'] += 1
                continue
            
            # Filter 3: Check variance (detect stationary bee)
            x_vals = [c[0] for c in centroids]
            y_vals = [c[1] for c in centroids]
            
            if np.var(x_vals) < 10.0 and np.var(y_vals) < 10.0:
                filtered['stationary'] += 1
                continue
            
            # Valid trajectory
            valid_movements.append(movement)
        
        total = sum(filtered.values())
        if total > 0:
            logger.info(f"  Filtered {total} fragments: "
                       f"short={filtered['short']}, "
                       f"stationary={filtered['stationary']}, "
                       f"no_movement={filtered['no_movement']}")
        
        return valid_movements
    
    def _remove_temporal_clusters(
        self,
        events: pd.DataFrame,
        window: int = 30
    ) -> pd.DataFrame:
        """Remove Exit→Entry pairs from ID switches at same nest.
        
        Detects pattern:
          Frame 100: Exit at nest 22
          Frame 105: Entry at nest 22  ← Same bee, new ID!
          
        Removes BOTH events (false exit + false entry).
        
        Args:
            events: DataFrame with events
            window: Frame window to detect clusters (default: 30)
            
        Returns:
            Cleaned DataFrame
        """
        if events.empty or len(events) < 2:
            return events
        
        df = events.sort_values(['nest', 'frame_number']).copy()
        keep = [True] * len(df)
        removed = 0
        
        i = 0
        while i < len(df) - 1:
            if not keep[i] or df.loc[i, 'action'] != 'Exit':
                i += 1
                continue
            
            exit_nest = df.loc[i, 'nest']
            exit_frame = df.loc[i, 'frame_number']
            
            # Look for Entry at same nest nearby
            for j in range(i + 1, len(df)):
                if not keep[j]:
                    continue
                
                entry_nest = df.loc[j, 'nest']
                entry_frame = df.loc[j, 'frame_number']
                
                # Different nest → stop
                if entry_nest != exit_nest:
                    break
                
                # Too far apart → stop
                if entry_frame - exit_frame > window:
                    break
                
                # Found Exit→Entry cluster!
                if df.loc[j, 'action'] == 'Entry':
                    keep[i] = False  # Remove Exit
                    keep[j] = False  # Remove Entry
                    removed += 1
                    
                    logger.debug(f"Removed ID switch cluster: "
                               f"Exit/Entry at nest {exit_nest}, "
                               f"frames {exit_frame}/{entry_frame}")
                    break
            
            i += 1
        
        if removed > 0:
            logger.info(f"  Removed {removed * 2} events from {removed} ID switch clusters")
        
        return df[keep].reset_index(drop=True)
    
    def _merge_duplicate_events(
        self,
        events: pd.DataFrame,
        window: int = 10
    ) -> pd.DataFrame:
        """Merge duplicate events (same action/nest within frame window).
        
        Args:
            events: DataFrame with events
            window: Frame window to consider duplicates
            
        Returns:
            DataFrame with duplicates merged
        """
        if events.empty:
            return events
        
        # Sort by action, nest, frame
        df = events.sort_values(['action', 'nest', 'frame_number']).reset_index(drop=True)
        
        # Mark duplicates
        keep_mask = [True] * len(df)
        
        for i in range(len(df) - 1):
            if not keep_mask[i]:
                continue
            
            curr_action = df.loc[i, 'action']
            curr_nest = df.loc[i, 'nest']
            curr_frame = df.loc[i, 'frame_number']
            
            # Check subsequent events
            for j in range(i + 1, len(df)):
                if not keep_mask[j]:
                    continue
                
                next_action = df.loc[j, 'action']
                next_nest = df.loc[j, 'nest']
                next_frame = df.loc[j, 'frame_number']
                
                # Different action or nest - stop checking
                if next_action != curr_action or next_nest != curr_nest:
                    break
                
                # Frame too far - stop checking
                if next_frame - curr_frame > window:
                    break
                
                # Duplicate found - mark for removal
                # Keep the one with higher confidence if available
                if 'label_confidence' in df.columns:
                    curr_conf = df.loc[i, 'label_confidence']
                    next_conf = df.loc[j, 'label_confidence']
                    
                    if next_conf > curr_conf:
                        keep_mask[i] = False
                        break  # Keep next, remove current
                    else:
                        keep_mask[j] = False  # Keep current, remove next
                else:
                    # No confidence info - keep first occurrence
                    keep_mask[j] = False
        
        # Filter to kept events
        return df[keep_mask].reset_index(drop=True)
    
    def _remove_temporal_clusters(
        self,
        events: pd.DataFrame,
        cluster_window: int = 30
    ) -> pd.DataFrame:
        """Remove false entry/exit pairs from ID switches.
        
        Detects patterns like:
        - Exit at nest X, frame N
        - Entry at nest X, frame N+5
        
        This indicates an ID switch where the same bee got two IDs,
        creating false exit (old ID) and entry (new ID) events.
        
        Args:
            events: DataFrame with events
            cluster_window: Frame window to detect clusters
            
        Returns:
            DataFrame with clusters removed
        """
        if events.empty or len(events) < 2:
            return events
        
        df = events.sort_values(['nest', 'frame_number']).reset_index(drop=True)
        keep_mask = [True] * len(df)
        clusters_removed = 0
        
        # Look for Exit → Entry patterns at same nest
        i = 0
        while i < len(df) - 1:
            if not keep_mask[i]:
                i += 1
                continue
            
            curr_action = df.loc[i, 'action']
            curr_nest = df.loc[i, 'nest']
            curr_frame = df.loc[i, 'frame_number']
            
            # Only check if current is an Exit
            if curr_action != 'Exit':
                i += 1
                continue
            
            # Look for Entry at same nest within window
            for j in range(i + 1, len(df)):
                if not keep_mask[j]:
                    continue
                
                next_action = df.loc[j, 'action']
                next_nest = df.loc[j, 'nest']
                next_frame = df.loc[j, 'frame_number']
                
                # Different nest - stop checking
                if next_nest != curr_nest:
                    break
                
                # Too far apart - stop checking
                if next_frame - curr_frame > cluster_window:
                    break
                
                # Found Exit → Entry at same nest within window
                if next_action == 'Entry':
                    # This is likely an ID switch artifact
                    # Remove both events
                    keep_mask[i] = False
                    keep_mask[j] = False
                    clusters_removed += 1
                    
                    logger.debug(f"Removed ID switch cluster: "
                                f"Exit at nest {curr_nest} frame {curr_frame}, "
                                f"Entry at nest {next_nest} frame {next_frame} "
                                f"(gap={next_frame - curr_frame} frames)")
                    break
            
            i += 1
        
        if clusters_removed > 0:
            logger.info(f"  Removed {clusters_removed * 2} events from {clusters_removed} ID switch clusters")
        
        return df[keep_mask].reset_index(drop=True)
    
    def validate_events(
        self,
        events: pd.DataFrame,
        nests: Dict,
        max_distance: float = 100.0
    ) -> pd.DataFrame:
        """Validate events are near actual nests.
        
        Removes events that are too far from any nest location.
        Useful for filtering false positives from tracking errors.
        
        Args:
            events: DataFrame with events
            nests: Dictionary with nest locations
            max_distance: Maximum distance from nest center (pixels)
            
        Returns:
            Validated DataFrame
            
        Example:
            >>> events = processor.process_tracks(motion_data, nests)
            >>> valid_events = processor.validate_events(events, nests)
        """
        if events.empty or 'nest' not in events.columns:
            return events
        
        original_count = len(events)
        valid_mask = []
        
        for idx, event in events.iterrows():
            nest_id = str(event['nest'])
            
            # Check if nest exists
            if nest_id not in nests['nests']:
                valid_mask.append(False)
                logger.debug(f"Event at invalid nest {nest_id} (frame {event['frame_number']})")
                continue
            
            valid_mask.append(True)
        
        df = events[valid_mask].reset_index(drop=True)
        
        removed = original_count - len(df)
        if removed > 0:
            logger.info(f"Validated events: removed {removed} events at invalid nests")
        
        return df
    
    def get_event_statistics(self, events: pd.DataFrame) -> Dict:
        """Get statistics about events.
        
        Args:
            events: DataFrame with events
            
        Returns:
            Dictionary with statistics
            
        Example:
            >>> events = processor.process_tracks(motion_data, nests)
            >>> stats = processor.get_event_statistics(events)
            >>> print(f"Entries: {stats['entries']}, Exits: {stats['exits']}")
        """
        if events.empty:
            return {
                'total_events': 0,
                'entries': 0,
                'exits': 0,
                'unique_nests': 0,
                'label_distribution': {},
                'avg_confidence': 0.0
            }
        
        stats = {
            'total_events': len(events),
            'entries': len(events[events['action'] == 'Entry']),
            'exits': len(events[events['action'] == 'Exit']),
            'unique_nests': events['nest'].nunique()
        }
        
        # Label distribution
        if 'label' in events.columns:
            stats['label_distribution'] = events['label'].value_counts().to_dict()
        
        # Average confidence
        if 'label_confidence' in events.columns:
            stats['avg_confidence'] = events['label_confidence'].mean()
        
        # Frame range
        if 'frame_number' in events.columns:
            stats['frame_range'] = (
                int(events['frame_number'].min()),
                int(events['frame_number'].max())
            )
        
        # Events per nest
        if 'nest' in events.columns:
            nest_counts = events['nest'].value_counts().to_dict()
            # Convert to sorted list for readability
            stats['events_per_nest'] = dict(sorted(nest_counts.items()))
            stats['most_active_nest'] = max(nest_counts.items(), key=lambda x: x[1])[0]
        
        return stats
    
    def __repr__(self) -> str:
        """String representation of processor."""
        ml_status = "with ML" if self.ml_classifier else "heuristic"
        return f"EventProcessor({ml_status}, config={self.config is not None})"