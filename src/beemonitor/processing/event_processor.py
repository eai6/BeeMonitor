"""Event processing for bee tracking data with multi-species support.

This module processes bee trajectories to identify entry and exit events
at nest holes, including species classification.
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
    """Processor for identifying bee entry/exit events with species tracking.
    
    This class analyzes bee trajectories to determine when bees enter
    or exit nest holes, creating a timeline of activity events with
    species information.
    
    Attributes:
        config: Configuration object
        trajectory_analyzer: TrajectoryAnalyzer instance
    
    Example:
        >>> processor = EventProcessor(config)
        >>> events = processor.process_tracks(motion_data, nests)
        >>> print(f"Found {len(events)} events")
        >>> print(events['species'].value_counts())
    """
    
    # def __init__(self, config: Optional[Config] = None):
    #     """Initialize EventProcessor.
        
    #     Args:
    #         config: Configuration object (optional)
    #     """
    #     self.config = config if config is not None else Config.default()
    #     self.trajectory_analyzer = TrajectoryAnalyzer(self.config)
        
    #     logger.debug("EventProcessor initialized with species support")


    # REPLACE WITH:
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
        label_map: Optional[Dict[int, str]] = None
    ) -> pd.DataFrame:
        """Process tracking data to identify entry/exit events with species.
        
        Args:
            motion_data: DataFrame with columns: frame_number, tracks, detections
            nests: Dictionary with 'hotel' ROI and 'nests' mapping
            species_map: Optional mapping of class IDs to species names
            
        Returns:
            DataFrame with columns: action, nest, frame_number, species, 
                                   species_class, species_confidence, notes
            
        Example:
            >>> events = processor.process_tracks(motion_data, nests)
            >>> entries = events[events['action'] == 'Entry']
            >>> print(f"Found {len(entries)} entry events")
            >>> # With species
            >>> honeybee_entries = entries[entries['species'] == 'honeybee']
        """
        logger.info("Processing tracks to identify events...")
        
        # Use species_map from config if not provided
        if label_map is None and hasattr(self.config, 'tracking'):
            label_map = self.config.tracking.label_map
        
        # Extract all movements from tracking data
        movements = [] # these are trajectories
        for period in motion_data.tracks:
            for track in period:
                movements.append(track)
        
        logger.debug(f"Processing {len(movements)} trajectories")
        
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
                    padding=entry_padding,
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
                species_counts = df['label'].value_counts()
                logger.info(f"Label distribution: {species_counts.to_dict()}")
            return df
        else:
            return pd.DataFrame(columns=[
                'action', 'nest', 'frame_number', 'label',
                'label_class', 'label_confidence', 'notes'
            ])
    
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
            species_map: Mapping of class IDs to species names
            
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
        if label_map and label_class is not None:
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
    
    def detect_entry(
        self,
        bee_trajectory: List[Point],
        hole_bboxes: Dict[str, BBox],
        window_size: int = 3,
        padding: float = 20
    ) -> int:
        """Detect if bee enters a hole (analyze start of trajectory).
        
        Args:
            bee_trajectory: List of (x, y) positions
            hole_bboxes: Dictionary mapping hole IDs to bounding boxes
            window_size: Number of frames to analyze
            padding: Padding around boxes (already scaled)
            
        Returns:
            Hole ID if entry detected, -1 otherwise
        """
        if len(bee_trajectory) < window_size:
            window_size = max(1, len(bee_trajectory) // 2)
        
        start_trajectory = bee_trajectory[:window_size]
        
        for hole_id, bbox in hole_bboxes.items():
            start_inside = all(
                self._is_inside_bbox(pos, bbox, padding)
                for pos in start_trajectory
            )
            
            if start_inside:
                return hole_id
        
        return -1
    
    def detect_exit(
        self,
        bee_trajectory: List[Point],
        hole_bboxes: Dict[str, BBox],
        window_size: int = 3,
        padding: float = 20
    ) -> int:
        """Detect if bee exits a hole (analyze end of trajectory).
        
        Args:
            bee_trajectory: List of (x, y) positions
            hole_bboxes: Dictionary mapping hole IDs to bounding boxes
            window_size: Number of frames to analyze
            padding: Padding around boxes (already scaled)
            
        Returns:
            Hole ID if exit detected, -1 otherwise
        """
        if len(bee_trajectory) < window_size:
            window_size = max(1, len(bee_trajectory) // 2)
        
        end_trajectory = bee_trajectory[-window_size:]
        
        for hole_id, bbox in hole_bboxes.items():
            end_inside = all(
                self._is_inside_bbox(pos, bbox, padding)
                for pos in end_trajectory
            )
            
            if end_inside:
                return hole_id
        
        return -1
    
    def process_yolo_tracks(
        self,
        movements: List[Tuple],
        nests: Dict,
        species_map: Optional[Dict[int, str]] = None
    ) -> pd.DataFrame:
        """Process YOLO tracking results to identify events with species.
        
        This is an alternative processing method for trajectories from
        Ultralytics YOLO tracking rather than custom BeeTracker.
        
        Args:
            movements: List of trajectories from UltralyticsTracker
            nests: Dictionary with nest locations
            species_map: Optional mapping of class IDs to species names
            
        Returns:
            DataFrame with events
            
        Example:
            >>> from beemonitor.tracking import UltralyticsTracker
            >>> tracker = UltralyticsTracker(model)
            >>> trajectories = tracker.get_tracks("video.mp4")
            >>> events = processor.process_yolo_tracks(trajectories, nests)
        """
        logger.info("Processing YOLO tracks to identify events...")
        
        # Use species_map from config if not provided
        if species_map is None and hasattr(self.config, 'tracking'):
            species_map = self.config.tracking.species_map
        
        # Get resolution for scaled parameters
        res_width = self.config.video.res_width
        res_height = self.config.video.res_height
        
        actions = []
        for movement in movements:
            # Skip short trajectories
            if len(movement[1]) < self.config.processing.min_trajectory_length:
                continue
            
            # Classify movement type
            if self.trajectory_analyzer.is_exit_behavior(movement):
                exit_window = self.config.processing.exit_window_size
                exit_padding = self.config.processing.exit_padding(res_width, res_height)
                
                action = self._get_action(
                    movement,
                    nests,
                    window_size=exit_window,
                    padding=exit_padding,
                    species_map=species_map
                )
            elif self.trajectory_analyzer.is_entry_behavior(movement):
                entry_window = self.config.processing.entry_window_size
                entry_padding = self.config.processing.entry_padding(res_width, res_height)
                
                action = self._get_action(
                    movement,
                    nests,
                    window_size=entry_window,
                    padding=entry_padding,
                    species_map=species_map
                )
            else:
                continue
            
            # Add actions to list
            if action:
                if isinstance(action, list):
                    actions.extend(action)
                else:
                    actions.append(action)
        
        logger.info(f"Identified {len(actions)} events from YOLO tracks")
        
        if actions:
            df = pd.DataFrame(actions)
            # Log species distribution if present
            if 'species' in df.columns:
                species_counts = df['species'].value_counts()
                logger.info(f"Species distribution: {species_counts.to_dict()}")
            return df
        else:
            return pd.DataFrame(columns=[
                'action', 'nest', 'frame_number', 'species',
                'species_class', 'species_confidence', 'notes'
            ])
    

    def process_tracks_ml(
        self,
        motion_data: pd.DataFrame,
        nests: Dict,
        species_map: Optional[Dict[int, str]] = None,
        bee_threshold: float = 0.6,
        event_threshold: float = 0.5
    ) -> pd.DataFrame:
        """Process tracks using ML classifier.
        
        Args:
            motion_data: DataFrame with columns: frame_number, tracks, detections
            nests: Dictionary with 'hotel' ROI and 'nests' mapping
            species_map: Optional mapping of class IDs to species names
            bee_threshold: Confidence threshold for bee classification
            event_threshold: Confidence threshold for event classification
            
        Returns:
            DataFrame with events and ML confidence scores
        """
        if self.ml_classifier is None:
            raise ValueError("ML classifier not available. Use process_tracks() for heuristic classification.")
        
        logger.info("Processing tracks with ML classifier...")
        
        # Use species_map from config if not provided
        if species_map is None and hasattr(self.config, 'tracking'):
            species_map = self.config.tracking.species_map
        
        # Extract all movements
        movements = []
        for period in motion_data.tracks:
            for track in period:
                movements.append(track)
        
        logger.debug(f"Classifying {len(movements)} trajectories with ML")
        
        # Classify each movement
        actions = []
        noise_filtered = 0
        
        for movement in movements:
            # Skip short trajectories
            if len(movement[1]) < self.config.processing.min_trajectory_length:
                continue
            
            # Classify with ML
            result = self.ml_classifier.classify_trajectory(
                movement,
                nests=nests,
                bee_threshold=bee_threshold,
                event_threshold=event_threshold
            )
            
            # Filter out noise
            if not result['is_bee']:
                noise_filtered += 1
                continue
            
            # Skip if event type is uncertain
            if result['event_type'] is None:
                continue
            
            # Get nest assignment
            nest_id = self._find_nearest_nest(
                result['event_location'],
                nests['nests']
            )
            
            if nest_id is None:
                continue
            
            # Get species info
            species_class = movement[4] if len(movement) > 4 else None
            species_votes = movement[5] if len(movement) > 5 else {}
            
            if species_map and species_class is not None:
                species_name = species_map.get(species_class, 'unknown')
            else:
                species_name = 'unknown'
            
            # Calculate species confidence
            species_confidence = 0.0
            if species_votes:
                total_votes = sum(species_votes.values())
                species_confidence = max(species_votes.values()) / total_votes if total_votes > 0 else 0.0
            
            # Determine frame number
            frame_numbers = movement[3]
            if result['event_type'] == 'entry':
                frame_num = frame_numbers[-1]  # Last frame
            elif result['event_type'] == 'exit':
                frame_num = frame_numbers[0]  # First frame
            else:
                frame_num = frame_numbers[len(frame_numbers) // 2]  # Middle
            
            # Create action
            action = {
                'action': result['event_type'].capitalize(),
                'nest': str(nest_id),
                'frame_number': frame_num,
                'species': species_name,
                'species_class': species_class,
                'species_confidence': species_confidence,
                'bee_confidence': result['bee_confidence'],
                'event_confidence': result['event_confidence'],
                'notes': f"ML classified: {result['event_type']}"
            }
            
            actions.append(action)
        
        logger.info(f"ML classifier: {len(actions)} events, {noise_filtered} trajectories filtered as noise")
        
        # Convert to DataFrame
        if actions:
            df = pd.DataFrame(actions)
            if 'species' in df.columns:
                species_counts = df['species'].value_counts()
                logger.info(f"Species distribution: {species_counts.to_dict()}")
            return df
        else:
            return pd.DataFrame(columns=[
                'action', 'nest', 'frame_number', 'species',
                'species_class', 'species_confidence',
                'bee_confidence', 'event_confidence', 'notes'
            ])

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
    

    # REPLACE WITH:
    def __repr__(self) -> str:
        """String representation of processor."""
        ml_status = "with ML" if self.ml_classifier else "heuristic"
        return f"EventProcessor({ml_status}, config={self.config is not None})"
    




# Backward compatibility functions
def is_inside_bbox(bee_position: Tuple[float, float], bbox: Tuple, padding: float = 20) -> bool:
    """Check if a point is inside a bounding box with padding.
    
    Args:
        bee_position: Position as (x, y)
        bbox: Bounding box as (x_min, y_min, x_max, y_max)
        padding: Padding around bbox
        
    Returns:
        True if point is inside padded bbox
    """
    processor = EventProcessor()
    return processor._is_inside_bbox(bee_position, bbox, padding)


def process_tracking(
    motion: pd.DataFrame,
    nest: Dict,
    species_map: Optional[Dict[int, str]] = None,
    config: Optional[Config] = None
) -> pd.DataFrame:
    """Process tracking data into events (backward compatible function).
    
    Args:
        motion: DataFrame with tracking data
        nest: Dictionary with nest information
        species_map: Optional mapping of class IDs to species names
        config: Optional configuration object
        
    Returns:
        DataFrame with event information including species
    """
    processor = EventProcessor(config)
    return processor.process_tracks(motion, nest, species_map)


def process_yolo_tracks(
    movements: List,
    nest: Dict,
    species_map: Optional[Dict[int, str]] = None,
    config: Optional[Config] = None
) -> pd.DataFrame:
    """Process YOLO tracking results (backward compatible function).
    
    Args:
        movements: List of track trajectories
        nest: Dictionary with nest information
        species_map: Mapping of class IDs to species names
        config: Optional configuration object
        
    Returns:
        DataFrame with event information
    """
    processor = EventProcessor(config)
    return processor.process_yolo_tracks(movements, nest, species_map)