# """Trajectory analysis for bee movements.

# This module analyzes bee trajectories to calculate speeds, accelerations,
# and classify movement behaviors.
# """

# import logging
# from typing import List, Tuple, Optional
# import numpy as np

# from beemonitor.core.config import Config


# logger = logging.getLogger(__name__)

# # Type aliases
# Point = Tuple[float, float]


# class TrajectoryAnalyzer:
#     """Analyzer for bee trajectory properties.
    
#     This class provides methods to calculate speed, acceleration, and
#     classify movement patterns from bee trajectories.
    
#     Attributes:
#         config: Configuration object
    
#     Example:
#         >>> analyzer = TrajectoryAnalyzer(config)
#         >>> speeds = analyzer.calculate_speed(trajectory)
#         >>> is_entry = analyzer.is_entry_behavior(movement)
#     """
    
#     def __init__(self, config: Optional[Config] = None):
#         """Initialize TrajectoryAnalyzer.
        
#         Args:
#             config: Configuration object (optional)
#         """
#         self.config = config if config is not None else Config.default()
    
#     def calculate_speed(self, trajectory: List[Point]) -> List[float]:
#         """Calculate speed from trajectory positions.
        
#         Computes the Euclidean distance between consecutive positions,
#         assuming 1 unit of time between frames.
        
#         Args:
#             trajectory: List of (x, y) positions
            
#         Returns:
#             List of speeds (distance per frame)
            
#         Example:
#             >>> trajectory = [(0, 0), (3, 4), (6, 8)]
#             >>> speeds = analyzer.calculate_speed(trajectory)
#             >>> speeds
#             [5.0, 5.0]
#         """
#         if len(trajectory) < 2:
#             return []
        
#         speeds = []
#         for i in range(1, len(trajectory)):
#             x1, y1 = trajectory[i - 1]
#             x2, y2 = trajectory[i]
            
#             # Euclidean distance
#             distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            
#             # Speed (assuming time interval = 1)
#             speed = distance / 1.0
#             speeds.append(speed)
        
#         return speeds
    
#     def calculate_acceleration(self, speeds: List[float]) -> List[float]:
#         """Calculate acceleration from speed values.
        
#         Computes the change in speed between consecutive time steps.
        
#         Args:
#             speeds: List of speed values
            
#         Returns:
#             List of accelerations (change in speed per frame)
            
#         Example:
#             >>> speeds = [5.0, 10.0, 15.0]
#             >>> accelerations = analyzer.calculate_acceleration(speeds)
#             >>> accelerations
#             [5.0, 5.0]
#         """
#         if len(speeds) < 2:
#             return []
        
#         accelerations = []
#         for i in range(1, len(speeds)):
#             acceleration = speeds[i] - speeds[i - 1]
#             accelerations.append(acceleration)
        
#         return accelerations
    
#     def check_start_and_end_speed(
#         self,
#         movement: Tuple
#     ) -> Tuple[float, float]:
#         """Check the start and end speed of a movement.
        
#         Args:
#             movement: Tuple of (track_id, centroids, bboxes, frame_numbers)
            
#         Returns:
#             Tuple of (start_speed, end_speed)
            
#         Example:
#             >>> start_speed, end_speed = analyzer.check_start_and_end_speed(movement)
#             >>> print(f"Start: {start_speed}, End: {end_speed}")
#         """
#         speeds = self.calculate_speed(movement[1])  # movement[1] is centroids
        
#         if not speeds:
#             return 0.0, 0.0
        
#         return speeds[0], speeds[-1]
    
#     def is_entry_behavior(
#         self,
#         movement: Tuple,
#         start_speed_threshold: Optional[float] = None,
#         end_speed_threshold: Optional[float] = None
#     ) -> bool:
#         """Check if movement represents entry behavior.
        
#         Entry behavior is characterized by movement that ends with low speed,
#         indicating the bee is settling into a nest.
        
#         Args:
#             movement: Tuple of (track_id, centroids, bboxes, frame_numbers)
#             start_speed_threshold: Threshold for start speed (optional)
#             end_speed_threshold: Threshold for end speed (optional)
            
#         Returns:
#             True if movement appears to be entry behavior
            
#         Example:
#             >>> if analyzer.is_entry_behavior(movement):
#             ...     print("Bee is entering nest")
#         """
#         if start_speed_threshold is None:
#             start_speed_threshold = self.config.processing.start_speed_threshold
#         if end_speed_threshold is None:
#             end_speed_threshold = self.config.processing.end_speed_threshold
        
#         start_speed, end_speed = self.check_start_and_end_speed(movement)
        
#         # Entry: bee slows down at the end
#         return end_speed < end_speed_threshold
    
#     def is_exit_behavior(
#         self,
#         movement: Tuple,
#         start_speed_threshold: Optional[float] = None,
#         end_speed_threshold: Optional[float] = None
#     ) -> bool:
#         """Check if movement represents exit behavior.
        
#         Exit behavior is characterized by movement that starts with low speed,
#         indicating the bee is leaving from a stationary position in a nest.
        
#         Args:
#             movement: Tuple of (track_id, centroids, bboxes, frame_numbers)
#             start_speed_threshold: Threshold for start speed (optional)
#             end_speed_threshold: Threshold for end speed (optional)
            
#         Returns:
#             True if movement appears to be exit behavior
            
#         Example:
#             >>> if analyzer.is_exit_behavior(movement):
#             ...     print("Bee is exiting nest")
#         """
#         if start_speed_threshold is None:
#             start_speed_threshold = self.config.processing.start_speed_threshold
#         if end_speed_threshold is None:
#             end_speed_threshold = self.config.processing.end_speed_threshold
        
#         start_speed, end_speed = self.check_start_and_end_speed(movement)
        
#         # Exit: bee starts slow (from nest)
#         return start_speed < start_speed_threshold
    
#     def is_entry_and_exit(
#         self,
#         movement: Tuple,
#         start_speed_threshold: Optional[float] = None,
#         end_speed_threshold: Optional[float] = None
#     ) -> bool:
#         """Check if movement represents both entry and exit.
        
#         This might indicate a brief visit or nest-to-nest movement.
        
#         Args:
#             movement: Tuple of (track_id, centroids, bboxes, frame_numbers)
#             start_speed_threshold: Threshold for start speed (optional)
#             end_speed_threshold: Threshold for end speed (optional)
            
#         Returns:
#             True if movement appears to be both entry and exit
#         """
#         if start_speed_threshold is None:
#             start_speed_threshold = self.config.processing.start_speed_threshold
#         if end_speed_threshold is None:
#             end_speed_threshold = self.config.processing.end_speed_threshold
        
#         start_speed, end_speed = self.check_start_and_end_speed(movement)
        
#         # Both slow at start and end
#         return (start_speed < start_speed_threshold and 
#                 end_speed < end_speed_threshold)
    
#     def calculate_trajectory_length(self, trajectory: List[Point]) -> float:
#         """Calculate total path length of trajectory.
        
#         Args:
#             trajectory: List of (x, y) positions
            
#         Returns:
#             Total distance traveled
            
#         Example:
#             >>> trajectory = [(0, 0), (3, 4), (6, 8)]
#             >>> length = analyzer.calculate_trajectory_length(trajectory)
#             >>> length
#             10.0
#         """
#         if len(trajectory) < 2:
#             return 0.0
        
#         total_length = 0.0
#         for i in range(1, len(trajectory)):
#             x1, y1 = trajectory[i - 1]
#             x2, y2 = trajectory[i]
#             distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
#             total_length += distance
        
#         return total_length
    
#     def calculate_displacement(self, trajectory: List[Point]) -> float:
#         """Calculate straight-line displacement from start to end.
        
#         Args:
#             trajectory: List of (x, y) positions
            
#         Returns:
#             Straight-line distance from first to last position
            
#         Example:
#             >>> trajectory = [(0, 0), (3, 4), (6, 8)]
#             >>> displacement = analyzer.calculate_displacement(trajectory)
#             >>> displacement
#             10.0
#         """
#         if len(trajectory) < 2:
#             return 0.0
        
#         x1, y1 = trajectory[0]
#         x2, y2 = trajectory[-1]
        
#         return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
#     def calculate_tortuosity(self, trajectory: List[Point]) -> float:
#         """Calculate trajectory tortuosity (path length / displacement).
        
#         A value of 1.0 indicates straight-line movement.
#         Higher values indicate more tortuous (winding) paths.
        
#         Args:
#             trajectory: List of (x, y) positions
            
#         Returns:
#             Tortuosity value (>= 1.0)
            
#         Example:
#             >>> trajectory = [(0, 0), (1, 1), (2, 0), (3, 1)]
#             >>> tortuosity = analyzer.calculate_tortuosity(trajectory)
#             >>> print(f"Tortuosity: {tortuosity:.2f}")
#         """
#         path_length = self.calculate_trajectory_length(trajectory)
#         displacement = self.calculate_displacement(trajectory)
        
#         if displacement == 0:
#             return float('inf')
        
#         return path_length / displacement
    
#     def get_average_speed(self, trajectory: List[Point]) -> float:
#         """Calculate average speed over trajectory.
        
#         Args:
#             trajectory: List of (x, y) positions
            
#         Returns:
#             Average speed (distance per frame)
#         """
#         speeds = self.calculate_speed(trajectory)
        
#         if not speeds:
#             return 0.0
        
#         return np.mean(speeds)
    
#     def get_max_speed(self, trajectory: List[Point]) -> float:
#         """Calculate maximum speed in trajectory.
        
#         Args:
#             trajectory: List of (x, y) positions
            
#         Returns:
#             Maximum speed value
#         """
#         speeds = self.calculate_speed(trajectory)
        
#         if not speeds:
#             return 0.0
        
#         return max(speeds)
    
#     def analyze_trajectory(
#         self,
#         movement: Tuple
#     ) -> dict:
#         """Comprehensive trajectory analysis.
        
#         Calculates multiple metrics for a trajectory including speeds,
#         accelerations, path properties, and behavior classification.
        
#         Args:
#             movement: Tuple of (track_id, centroids, bboxes, frame_numbers)
            
#         Returns:
#             Dictionary with analysis results
            
#         Example:
#             >>> analysis = analyzer.analyze_trajectory(movement)
#             >>> print(f"Average speed: {analysis['avg_speed']:.2f}")
#             >>> print(f"Behavior: {analysis['behavior']}")
#         """
#         trajectory = movement[1]  # centroids
        
#         speeds = self.calculate_speed(trajectory)
        
#         if not speeds:
#             avg_speed = 0.0
#             max_speed = 0.0
#         else:
#             avg_speed = np.mean(speeds)
#             max_speed = max(speeds)
        
#         # Classify behavior
#         if self.is_entry_behavior(movement):
#             behavior = "entry"
#         elif self.is_exit_behavior(movement):
#             behavior = "exit"
#         elif self.is_entry_and_exit(movement):
#             behavior = "entry_and_exit"
#         else:
#             behavior = "unknown"
        
#         return {
#             "track_id": movement[0],
#             "num_positions": len(trajectory),
#             "num_frames": len(movement[3]),
#             "avg_speed": avg_speed,
#             "max_speed": max_speed,
#             "path_length": self.calculate_trajectory_length(trajectory),
#             "displacement": self.calculate_displacement(trajectory),
#             "tortuosity": self.calculate_tortuosity(trajectory),
#             "behavior": behavior,
#             "start_frame": movement[3][0] if movement[3] else None,
#             "end_frame": movement[3][-1] if movement[3] else None,
#         }
    
#     def __repr__(self) -> str:
#         """String representation of analyzer."""
#         return f"TrajectoryAnalyzer(config={self.config is not None})"




"""Trajectory analysis for bee movements with resolution-aware parameters.

This module analyzes bee trajectories to calculate speeds, accelerations,
and classify movement behaviors with automatic parameter scaling.
"""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np

from beemonitor.core.config import Config


logger = logging.getLogger(__name__)

# Type aliases
Point = Tuple[float, float]


class TrajectoryAnalyzer:
    """Analyzer for bee trajectory properties with resolution-aware thresholds.
    
    This class provides methods to calculate speed, acceleration, and
    classify movement patterns from bee trajectories. Speed thresholds
    automatically scale with video resolution.
    
    Attributes:
        config: Configuration object
    
    Example:
        >>> analyzer = TrajectoryAnalyzer(config)
        >>> speeds = analyzer.calculate_speed(trajectory)
        >>> is_entry = analyzer.is_entry_behavior(movement)
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize TrajectoryAnalyzer.
        
        Args:
            config: Configuration object (optional)
        """
        self.config = config if config is not None else Config.default()
        
        logger.debug("TrajectoryAnalyzer initialized with resolution-aware thresholds")
    
    def calculate_speed(self, trajectory: List[Point]) -> List[float]:
        """Calculate speed from trajectory positions.
        
        Computes the Euclidean distance between consecutive positions,
        assuming 1 unit of time between frames.
        
        Args:
            trajectory: List of (x, y) positions
            
        Returns:
            List of speeds (distance per frame)
            
        Example:
            >>> trajectory = [(0, 0), (3, 4), (6, 8)]
            >>> speeds = analyzer.calculate_speed(trajectory)
            >>> speeds
            [5.0, 5.0]
        """
        if len(trajectory) < 2:
            return []
        
        speeds = []
        for i in range(1, len(trajectory)):
            x1, y1 = trajectory[i - 1]
            x2, y2 = trajectory[i]
            
            # Euclidean distance
            distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            
            # Speed (assuming time interval = 1)
            speed = distance / 1.0
            speeds.append(speed)
        
        return speeds
    
    def calculate_acceleration(self, speeds: List[float]) -> List[float]:
        """Calculate acceleration from speed values.
        
        Computes the change in speed between consecutive time steps.
        
        Args:
            speeds: List of speed values
            
        Returns:
            List of accelerations (change in speed per frame)
            
        Example:
            >>> speeds = [5.0, 10.0, 15.0]
            >>> accelerations = analyzer.calculate_acceleration(speeds)
            >>> accelerations
            [5.0, 5.0]
        """
        if len(speeds) < 2:
            return []
        
        accelerations = []
        for i in range(1, len(speeds)):
            acceleration = speeds[i] - speeds[i - 1]
            accelerations.append(acceleration)
        
        return accelerations
    
    def check_start_and_end_speed(
        self,
        movement: Tuple
    ) -> Tuple[float, float]:
        """Check the start and end speed of a movement.
        
        Args:
            movement: Tuple of (track_id, centroids, bboxes, frame_numbers)
            
        Returns:
            Tuple of (start_speed, end_speed)
            
        Example:
            >>> start_speed, end_speed = analyzer.check_start_and_end_speed(movement)
            >>> print(f"Start: {start_speed}, End: {end_speed}")
        """
        speeds = self.calculate_speed(movement[1])  # movement[1] is centroids
        
        if not speeds:
            return 0.0, 0.0
        
        return speeds[0], speeds[-1]
    
    def is_entry_behavior(
        self,
        movement: Tuple,
        start_speed_threshold: Optional[float] = None,
        end_speed_threshold: Optional[float] = None
    ) -> bool:
        """Check if movement represents entry behavior.
        
        Entry behavior is characterized by movement that ends with low speed,
        indicating the bee is settling into a nest.
        
        Args:
            movement: Tuple of (track_id, centroids, bboxes, frame_numbers)
            start_speed_threshold: Threshold for start speed (optional, overrides config)
            end_speed_threshold: Threshold for end speed (optional, overrides config)
            
        Returns:
            True if movement appears to be entry behavior
            
        Example:
            >>> if analyzer.is_entry_behavior(movement):
            ...     print("Bee is entering nest")
        """
        # Get resolution from config
        res_width = self.config.video.res_width
        res_height = self.config.video.res_height
        
        # Get scaled thresholds if not provided
        if start_speed_threshold is None:
            start_speed_threshold = self.config.processing.start_speed_threshold(res_width, res_height)
        if end_speed_threshold is None:
            end_speed_threshold = self.config.processing.end_speed_threshold(res_width, res_height)
        
        start_speed, end_speed = self.check_start_and_end_speed(movement)
        
        # Entry: bee slows down at the end
        return end_speed < end_speed_threshold
    
    def is_exit_behavior(
        self,
        movement: Tuple,
        start_speed_threshold: Optional[float] = None,
        end_speed_threshold: Optional[float] = None
    ) -> bool:
        """Check if movement represents exit behavior.
        
        Exit behavior is characterized by movement that starts with low speed,
        indicating the bee is leaving from a stationary position in a nest.
        
        Args:
            movement: Tuple of (track_id, centroids, bboxes, frame_numbers)
            start_speed_threshold: Threshold for start speed (optional, overrides config)
            end_speed_threshold: Threshold for end speed (optional, overrides config)
            
        Returns:
            True if movement appears to be exit behavior
            
        Example:
            >>> if analyzer.is_exit_behavior(movement):
            ...     print("Bee is exiting nest")
        """
        # Get resolution from config
        res_width = self.config.video.res_width
        res_height = self.config.video.res_height
        
        # Get scaled thresholds if not provided
        if start_speed_threshold is None:
            start_speed_threshold = self.config.processing.start_speed_threshold(res_width, res_height)
        if end_speed_threshold is None:
            end_speed_threshold = self.config.processing.end_speed_threshold(res_width, res_height)
        
        start_speed, end_speed = self.check_start_and_end_speed(movement)
        
        # Exit: bee starts slow (from nest)
        return start_speed < start_speed_threshold
    
    def is_entry_and_exit(
        self,
        movement: Tuple,
        start_speed_threshold: Optional[float] = None,
        end_speed_threshold: Optional[float] = None
    ) -> bool:
        """Check if movement represents both entry and exit.
        
        This might indicate a brief visit or nest-to-nest movement.
        
        Args:
            movement: Tuple of (track_id, centroids, bboxes, frame_numbers)
            start_speed_threshold: Threshold for start speed (optional, overrides config)
            end_speed_threshold: Threshold for end speed (optional, overrides config)
            
        Returns:
            True if movement appears to be both entry and exit
        """
        # Get resolution from config
        res_width = self.config.video.res_width
        res_height = self.config.video.res_height
        
        # Get scaled thresholds if not provided
        if start_speed_threshold is None:
            start_speed_threshold = self.config.processing.start_speed_threshold(res_width, res_height)
        if end_speed_threshold is None:
            end_speed_threshold = self.config.processing.end_speed_threshold(res_width, res_height)
        
        start_speed, end_speed = self.check_start_and_end_speed(movement)
        
        # Both slow at start and end
        return (start_speed < start_speed_threshold and 
                end_speed < end_speed_threshold)
    
    def calculate_trajectory_length(self, trajectory: List[Point]) -> float:
        """Calculate total path length of trajectory.
        
        Args:
            trajectory: List of (x, y) positions
            
        Returns:
            Total distance traveled
            
        Example:
            >>> trajectory = [(0, 0), (3, 4), (6, 8)]
            >>> length = analyzer.calculate_trajectory_length(trajectory)
            >>> length
            10.0
        """
        if len(trajectory) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(1, len(trajectory)):
            x1, y1 = trajectory[i - 1]
            x2, y2 = trajectory[i]
            distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            total_length += distance
        
        return total_length
    
    def calculate_displacement(self, trajectory: List[Point]) -> float:
        """Calculate straight-line displacement from start to end.
        
        Args:
            trajectory: List of (x, y) positions
            
        Returns:
            Straight-line distance from first to last position
            
        Example:
            >>> trajectory = [(0, 0), (3, 4), (6, 8)]
            >>> displacement = analyzer.calculate_displacement(trajectory)
            >>> displacement
            10.0
        """
        if len(trajectory) < 2:
            return 0.0
        
        x1, y1 = trajectory[0]
        x2, y2 = trajectory[-1]
        
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    def calculate_tortuosity(self, trajectory: List[Point]) -> float:
        """Calculate trajectory tortuosity (path length / displacement).
        
        A value of 1.0 indicates straight-line movement.
        Higher values indicate more tortuous (winding) paths.
        
        Args:
            trajectory: List of (x, y) positions
            
        Returns:
            Tortuosity value (>= 1.0)
            
        Example:
            >>> trajectory = [(0, 0), (1, 1), (2, 0), (3, 1)]
            >>> tortuosity = analyzer.calculate_tortuosity(trajectory)
            >>> print(f"Tortuosity: {tortuosity:.2f}")
        """
        path_length = self.calculate_trajectory_length(trajectory)
        displacement = self.calculate_displacement(trajectory)
        
        if displacement == 0:
            return float('inf')
        
        return path_length / displacement
    
    def get_average_speed(self, trajectory: List[Point]) -> float:
        """Calculate average speed over trajectory.
        
        Args:
            trajectory: List of (x, y) positions
            
        Returns:
            Average speed (distance per frame)
        """
        speeds = self.calculate_speed(trajectory)
        
        if not speeds:
            return 0.0
        
        return np.mean(speeds)
    
    def get_max_speed(self, trajectory: List[Point]) -> float:
        """Calculate maximum speed in trajectory.
        
        Args:
            trajectory: List of (x, y) positions
            
        Returns:
            Maximum speed value
        """
        speeds = self.calculate_speed(trajectory)
        
        if not speeds:
            return 0.0
        
        return max(speeds)
    
    def analyze_trajectory(
        self,
        movement: Tuple
    ) -> dict:
        """Comprehensive trajectory analysis with resolution-aware classification.
        
        Calculates multiple metrics for a trajectory including speeds,
        accelerations, path properties, and behavior classification using
        scaled thresholds.
        
        Args:
            movement: Tuple of (track_id, centroids, bboxes, frame_numbers)
            
        Returns:
            Dictionary with analysis results
            
        Example:
            >>> analysis = analyzer.analyze_trajectory(movement)
            >>> print(f"Average speed: {analysis['avg_speed']:.2f}")
            >>> print(f"Behavior: {analysis['behavior']}")
        """
        trajectory = movement[1]  # centroids
        
        speeds = self.calculate_speed(trajectory)
        
        if not speeds:
            avg_speed = 0.0
            max_speed = 0.0
        else:
            avg_speed = np.mean(speeds)
            max_speed = max(speeds)
        
        # Classify behavior using resolution-aware thresholds
        if self.is_entry_behavior(movement):
            behavior = "entry"
        elif self.is_exit_behavior(movement):
            behavior = "exit"
        elif self.is_entry_and_exit(movement):
            behavior = "entry_and_exit"
        else:
            behavior = "unknown"
        
        return {
            "track_id": movement[0],
            "num_positions": len(trajectory),
            "num_frames": len(movement[3]),
            "avg_speed": avg_speed,
            "max_speed": max_speed,
            "path_length": self.calculate_trajectory_length(trajectory),
            "displacement": self.calculate_displacement(trajectory),
            "tortuosity": self.calculate_tortuosity(trajectory),
            "behavior": behavior,
            "start_frame": movement[3][0] if movement[3] else None,
            "end_frame": movement[3][-1] if movement[3] else None,
        }
    
    def __repr__(self) -> str:
        """String representation of analyzer."""
        return f"TrajectoryAnalyzer(config={self.config is not None}, resolution_aware=True)"
    


    def extract_features(
        self,
        movement: Tuple,
        nests: Optional[Dict] = None
    ) -> Dict[str, float]:
        """Extract features from trajectory for ML classification.
        
        Extracts comprehensive features including speed, acceleration,
        direction changes, trajectory shape, and spatial relationships to nests.
        
        Args:
            movement: Tuple of (track_id, centroids, bboxes, frame_numbers, species, species_votes)
            nests: Optional dictionary with nest locations
            
        Returns:
            Dictionary of feature values for ML model
        """
        trajectory = movement[1]  # centroids
        frame_numbers = movement[3]
        
        if len(trajectory) < 2:
            return self._get_empty_features()
        
        # Speed features
        speeds = self.calculate_speed(trajectory)
        speed_mean = np.mean(speeds) if speeds else 0.0
        speed_std = np.std(speeds) if speeds else 0.0
        speed_min = np.min(speeds) if speeds else 0.0
        speed_max = np.max(speeds) if speeds else 0.0
        
        # Start and end speeds (critical for entry/exit)
        start_speeds = speeds[:3] if len(speeds) >= 3 else speeds
        end_speeds = speeds[-3:] if len(speeds) >= 3 else speeds
        start_speed_mean = np.mean(start_speeds) if start_speeds else 0.0
        end_speed_mean = np.mean(end_speeds) if end_speeds else 0.0
        
        # Acceleration features
        accelerations = self.calculate_acceleration(speeds)
        accel_mean = np.mean(accelerations) if accelerations else 0.0
        accel_std = np.std(accelerations) if accelerations else 0.0
        
        # Direction changes (angular velocity)
        direction_changes = self._calculate_direction_changes(trajectory)
        direction_change_mean = np.mean(direction_changes) if direction_changes else 0.0
        direction_change_std = np.std(direction_changes) if direction_changes else 0.0
        
        # Trajectory shape features
        path_length = self.calculate_trajectory_length(trajectory)
        displacement = self.calculate_displacement(trajectory)
        tortuosity = self.calculate_tortuosity(trajectory)
        
        # Straightness (displacement / path_length)
        straightness = displacement / path_length if path_length > 0 else 0.0
        
        # Trajectory duration
        duration = len(frame_numbers)
        
        # Start/end position features
        start_x, start_y = trajectory[0]
        end_x, end_y = trajectory[-1]
        
        # Movement direction (overall)
        dx = end_x - start_x
        dy = end_y - start_y
        overall_angle = np.arctan2(dy, dx)
        
        # Position variance (how spread out the trajectory is)
        x_coords = [p[0] for p in trajectory]
        y_coords = [p[1] for p in trajectory]
        position_variance = np.std(x_coords) + np.std(y_coords)
        
        # Median positions (more robust than single point)
        start_median_x = np.median([p[0] for p in trajectory[:min(5, len(trajectory))]])
        start_median_y = np.median([p[1] for p in trajectory[:min(5, len(trajectory))]])
        end_median_x = np.median([p[0] for p in trajectory[-min(5, len(trajectory)):]])
        end_median_y = np.median([p[1] for p in trajectory[-min(5, len(trajectory)):]])
        
        features = {
            # Speed features
            'speed_mean': speed_mean,
            'speed_std': speed_std,
            'speed_min': speed_min,
            'speed_max': speed_max,
            'start_speed_mean': start_speed_mean,
            'end_speed_mean': end_speed_mean,
            'speed_ratio': end_speed_mean / start_speed_mean if start_speed_mean > 0 else 0.0,
            
            # Acceleration features
            'accel_mean': accel_mean,
            'accel_std': accel_std,
            
            # Direction features
            'direction_change_mean': direction_change_mean,
            'direction_change_std': direction_change_std,
            
            # Shape features
            'path_length': path_length,
            'displacement': displacement,
            'tortuosity': tortuosity,
            'straightness': straightness,
            'position_variance': position_variance,
            
            # Temporal features
            'duration': duration,
            
            # Spatial features
            'start_x': start_x,
            'start_y': start_y,
            'end_x': end_x,
            'end_y': end_y,
            'start_median_x': start_median_x,
            'start_median_y': start_median_y,
            'end_median_x': end_median_x,
            'end_median_y': end_median_y,
            'overall_angle': overall_angle,
        }
        
        # Add nest-relative features if nests provided
        if nests is not None and 'nests' in nests:
            nest_features = self._calculate_nest_features(
                trajectory,
                nests['nests'],
                start_median_x, start_median_y,
                end_median_x, end_median_y
            )
            features.update(nest_features)
        
        return features

    def _calculate_direction_changes(self, trajectory: List[Point]) -> List[float]:
        """Calculate angular changes between consecutive segments."""
        if len(trajectory) < 3:
            return []
        
        angles = []
        for i in range(len(trajectory) - 2):
            p1, p2, p3 = trajectory[i], trajectory[i+1], trajectory[i+2]
            
            # Vectors
            v1 = (p2[0] - p1[0], p2[1] - p1[1])
            v2 = (p3[0] - p2[0], p3[1] - p2[1])
            
            # Angles
            angle1 = np.arctan2(v1[1], v1[0])
            angle2 = np.arctan2(v2[1], v2[0])
            
            # Angular difference
            diff = angle2 - angle1
            # Normalize to [-pi, pi]
            diff = np.arctan2(np.sin(diff), np.cos(diff))
            angles.append(abs(diff))
        
        return angles

    def _calculate_nest_features(
        self,
        trajectory: List[Point],
        nest_bboxes: Dict,
        start_median_x: float,
        start_median_y: float,
        end_median_x: float,
        end_median_y: float
    ) -> Dict[str, float]:
        """Calculate features related to nest proximity."""
        # Distance to nearest nest at start/end
        min_start_dist = float('inf')
        min_end_dist = float('inf')
        
        for nest_id, bbox in nest_bboxes.items():
            nest_center_x = (bbox[0] + bbox[2]) / 2
            nest_center_y = (bbox[1] + bbox[3]) / 2
            
            # Distance from start
            start_dist = np.sqrt(
                (start_median_x - nest_center_x)**2 + 
                (start_median_y - nest_center_y)**2
            )
            min_start_dist = min(min_start_dist, start_dist)
            
            # Distance from end
            end_dist = np.sqrt(
                (end_median_x - nest_center_x)**2 + 
                (end_median_y - nest_center_y)**2
            )
            min_end_dist = min(min_end_dist, end_dist)
        
        return {
            'min_dist_to_nest_start': min_start_dist,
            'min_dist_to_nest_end': min_end_dist,
            'nest_approach': min_start_dist - min_end_dist,  # Negative means approaching
        }

    def _get_empty_features(self) -> Dict[str, float]:
        """Return default features for invalid trajectories."""
        return {
            'speed_mean': 0.0, 'speed_std': 0.0, 'speed_min': 0.0, 'speed_max': 0.0,
            'start_speed_mean': 0.0, 'end_speed_mean': 0.0, 'speed_ratio': 0.0,
            'accel_mean': 0.0, 'accel_std': 0.0,
            'direction_change_mean': 0.0, 'direction_change_std': 0.0,
            'path_length': 0.0, 'displacement': 0.0, 'tortuosity': 0.0,
            'straightness': 0.0, 'position_variance': 0.0,
            'duration': 0.0,
            'start_x': 0.0, 'start_y': 0.0, 'end_x': 0.0, 'end_y': 0.0,
            'start_median_x': 0.0, 'start_median_y': 0.0,
            'end_median_x': 0.0, 'end_median_y': 0.0,
            'overall_angle': 0.0,
        }
