"""Enhanced configuration with hotel box-aware parameter scaling and adaptive tracker.

This module extends the configuration system to support parameter scaling
based on both video resolution AND the hotel box position/distance from camera.
Now includes adaptive tracker configuration for FPS and resolution-independent tracking.
"""

import yaml
from pathlib import Path
from typing import List, Optional, Tuple, Callable, Dict, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class VideoConfig:
    """Video processing configuration with auto-detection support."""
    res_width: int = 1280
    res_height: int = 720
    fps: int = 30
    auto_detect_from_video: bool = True  # Automatically detect FPS and resolution from video
    annotation_mode: bool = True # Use annotation mode for visualization scaling
    debug_mode : bool = False  # Enable debug mode for verbose logging
    
    def __post_init__(self):
        """Validate video configuration."""
        if self.res_width < 1 or self.res_height < 1:
            raise ValueError("Video resolution must be positive")
        if self.fps < 1:
            raise ValueError("FPS must be at least 1")
    
    def update_from_video(self, video_path: str) -> None:
        """Auto-detect and update FPS and resolution from video file.
        
        Args:
            video_path: Path to video file
            
        Raises:
            ValueError: If video cannot be opened
        """
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        try:
            # Get FPS from video
            detected_fps = cap.get(cv2.CAP_PROP_FPS)
            if detected_fps > 0:
                self.fps = int(round(detected_fps))
            
            # Get resolution from video
            detected_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            detected_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if detected_width > 0 and detected_height > 0:
                self.res_width = detected_width
                self.res_height = detected_height
            
            import logging
            logger = logging.getLogger(__name__)
            logger.info(
                f"Auto-detected video properties: {self.res_width}x{self.res_height} @ {self.fps} FPS"
            )
        
        finally:
            cap.release()
    
    def get_video_properties(self, video_path: str) -> Tuple[int, int, int]:
        """Get video properties without modifying config.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (width, height, fps)
            
        Raises:
            ValueError: If video cannot be opened
        """
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        try:
            fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            return (width, height, fps)
        
        finally:
            cap.release()



@dataclass
class ModelConfig:
    """Model paths configuration."""
    nest_detection: str = "/Users/edwardamoah/Documents/GitHub/BeeMonitor_eai6/models/nest_detection.pt"
    tracking : str = "/Users/edwardamoah/Documents/GitHub/BeeMonitor_eai6/models/bee_tracking.pt"
    blob_noise_classifier: str = "/Users/edwardamoah/Documents/GitHub/BeeMonitor_eai6/models/blob_noise_classifier.pth"
    bee_classifier: Optional[str] = None
    event_classifier: Optional[str] = "/Users/edwardamoah/Documents/GitHub/BeeMonitor_eai6/models/event_classifier_model.pkl"



    # nest_detection: str = "/storage/home/eai6/BeeMonitor_eai6/models/nest_detection.pt"
    # tracking : str = "/storage/home/eai6/BeeMonitor_eai6/models/bee_tracking.pt"
    # blob_noise_classifier: str = "/storage/home/eai6/BeeMonitor_eai6/models/blob_noise_classifier.pth"
    # bee_classifier: Optional[str] = None
    # event_classifier: Optional[str] = "/storage/home/eai6/BeeMonitor_eai6/models/event_classifier_model.pkl"

@dataclass  
class HotelBoxConfig:
    """Hotel box position and distance configuration.
    
    These parameters define the physical hotel box location in the frame
    and are used to scale pixel-based thresholds appropriately.
    """
    # Hotel box position in frame (normalized 0-1 or pixel coordinates)
    x_center: float = 0.5  # Horizontal center position (0-1)
    y_center: float = 0.5  # Vertical center position (0-1)
    width_ratio: float = 0.7  # Width relative to frame width (0-1)
    height_ratio: float = 0.6  # Height relative to frame height (0-1)
    
    # Distance/scale parameters
    distance_factor: float = 1.0  # Scale factor based on camera distance (0.5-2.0)
    # 1.0 = reference distance
    # < 1.0 = hotel is closer (objects appear larger)
    # > 1.0 = hotel is farther (objects appear smaller)

    hotel_box_cords: Optional[Tuple[int, int, int, int]] = None  # (x_min, y_min, x_max, y_max) in pixels
    
    # Auto-detection settings
    auto_detect_box: bool = True  # Auto-detect hotel box from first frames
    min_box_confidence: float = 0.8  # Minimum confidence for auto-detection
    
    def __post_init__(self):
        """Validate hotel box configuration."""
        if not (0 <= self.x_center <= 1):
            raise ValueError("x_center must be between 0 and 1")
        if not (0 <= self.y_center <= 1):
            raise ValueError("y_center must be between 0 and 1")
        if not (0 < self.width_ratio <= 1):
            raise ValueError("width_ratio must be between 0 and 1")
        if not (0 < self.height_ratio <= 1):
            raise ValueError("height_ratio must be between 0 and 1")
        if not (0.1 <= self.distance_factor <= 5.0):
            raise ValueError("distance_factor must be between 0.1 and 5.0")
    
    def get_box_bounds(self, frame_width: int, frame_height: int) -> Tuple[int, int, int, int]:
        """Get pixel coordinates of hotel box.
        
        Args:
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            
        Returns:
            Tuple of (x_min, y_min, x_max, y_max)
        """
        box_width = int(frame_width * self.width_ratio)
        box_height = int(frame_height * self.height_ratio)
        
        x_min = int(frame_width * self.x_center - box_width / 2)
        y_min = int(frame_height * self.y_center - box_height / 2)
        x_max = x_min + box_width
        y_max = y_min + box_height
        
        return (x_min, y_min, x_max, y_max)
    
    def get_scale_factor(self) -> float:
        """Get overall scale factor combining distance.
        
        Returns:
            Combined scale factor
        """
        return self.distance_factor


@dataclass
class NestConfig:
    """Nest detection configuration with hotel-aware scaling."""
    
    # Reference resolution (baseline for parameter tuning)
    reference_width: int = 1920
    reference_height: int = 1080
    reference_distance_factor: float = 1.0
    
    # Detection parameters
    confidence_threshold: float = 0.25
    min_detections: int = 35
    frame_skip: int = 30
    max_detection_attempts: int = 10
    nest_tube_class: int = 1
    hotel_class: int = 0
    
    # Grid parameters
    expected_total_nests: int = 60
    expected_rows: int = 6
    expected_nests_per_row: int = 10
    nest_count_tolerance: int = 3
    
    # Base pixel measurements (at reference resolution and distance)
    nest_width_base: float = 24
    nest_height_base: float = 14
    padding_x_base: float = 5.0
    padding_y_base: float = 7.0
    hotel_padding_x_base: float = 100.0
    hotel_padding_y_base: float = 50.0

    # Quality check tolerances (base values)
    spacing_tolerance_base: float = 25.0
    x_position_tolerance_base: float = 20.0
    y_position_tolerance_base: float = 15.0

    # Dynamic padding options
    use_dynamic_padding: bool = True
    dynamic_padding_ratio: float = 0.15
    min_padding_x: float = 3.0
    min_padding_y: float = 3.0
    
    # Quality check tolerances (base values) - RENAMED FOR CLARITY
    nest_spacing_tolerance_base: float = 15.0
    row_alignment_tolerance_base: float = 70.0
    column_alignment_tolerance_base: float = 12.0
    
    # Clustering thresholds (base values)
    row_threshold_base: float = 30.0
    col_threshold_base: float = 50.0

    min_row_size: int = 6
    min_col_size: int = 10
    
    # Scaling methods for different parameters
    def get_scale_factor(self, width: int, height: int, hotel_box: Optional[HotelBoxConfig] = None) -> Tuple[float, float]:
        """Calculate scale factors based on resolution and hotel position."""
        x_scale = width / self.reference_width
        y_scale = height / self.reference_height
        
        if hotel_box is not None:
            distance_scale = hotel_box.get_scale_factor()
            x_scale *= distance_scale
            y_scale *= distance_scale
        
        return (x_scale, y_scale)
    
    def nest_width(self, width: int, height: int, hotel_box: Optional[HotelBoxConfig] = None) -> int:
        """Get scaled nest width."""
        x_scale, _ = self.get_scale_factor(width, height, hotel_box)
        return int(self.nest_width_base * x_scale)
    
    def nest_height(self, width: int, height: int, hotel_box: Optional[HotelBoxConfig] = None) -> int:
        """Get scaled nest height."""
        _, y_scale = self.get_scale_factor(width, height, hotel_box)
        return int(self.nest_height_base * y_scale)
    
    def padding_x(self, width: int, height: int, hotel_box: Optional[HotelBoxConfig] = None) -> int:
        """Get scaled X padding."""
        x_scale, _ = self.get_scale_factor(width, height, hotel_box)
        return int(self.padding_x_base * x_scale)
    
    def padding_y(self, width: int, height: int, hotel_box: Optional[HotelBoxConfig] = None) -> int:
        """Get scaled Y padding."""
        _, y_scale = self.get_scale_factor(width, height, hotel_box)
        return int(self.padding_y_base * y_scale)
    
    def hotel_padding_x(self, width: int, height: int, hotel_box: Optional[HotelBoxConfig] = None) -> int:
        """Get scaled hotel X padding."""
        x_scale, _ = self.get_scale_factor(width, height, hotel_box)
        return int(self.hotel_padding_x_base * x_scale)
    
    def hotel_padding_y(self, width: int, height: int, hotel_box: Optional[HotelBoxConfig] = None) -> int:
        """Get scaled hotel Y padding."""
        _, y_scale = self.get_scale_factor(width, height, hotel_box)
        return int(self.hotel_padding_y_base * y_scale)
    
    def spacing_tolerance(self, width: int, height: int, hotel_box: Optional[HotelBoxConfig] = None) -> float:
        """Get scaled spacing tolerance."""
        x_scale, y_scale = self.get_scale_factor(width, height, hotel_box)
        avg_scale = (x_scale + y_scale) / 2
        return self.spacing_tolerance_base * avg_scale * 2.5
    
    def x_position_tolerance(self, width: int, height: int, hotel_box: Optional[HotelBoxConfig] = None) -> float:
        """Get scaled X position tolerance."""
        x_scale, _ = self.get_scale_factor(width, height, hotel_box)
        return self.x_position_tolerance_base * x_scale * 3.0
    
    def y_position_tolerance(self, width: int, height: int, hotel_box: Optional[HotelBoxConfig] = None) -> float:
        """Get scaled Y position tolerance."""
        _, y_scale = self.get_scale_factor(width, height, hotel_box)
        return self.y_position_tolerance_base * y_scale * 2.5
    
    def row_threshold(self, width: int, height: int, hotel_box: Optional[HotelBoxConfig] = None) -> float:
        """Get scaled row clustering threshold (Y-axis distance)."""
        _, y_scale = self.get_scale_factor(width, height, hotel_box)
        return self.row_threshold_base * y_scale
    
    def col_threshold(self, width: int, height: int, hotel_box: Optional[HotelBoxConfig] = None) -> float:
        """Get scaled column clustering threshold (X-axis distance)."""
        x_scale, _ = self.get_scale_factor(width, height, hotel_box)
        return self.col_threshold_base * x_scale
    
    def nest_spacing_tolerance(self, width: int, height: int, hotel_box: Optional[HotelBoxConfig] = None) -> float:
        """Get scaled tolerance for horizontal spacing between nests in same row."""
        x_scale, _ = self.get_scale_factor(width, height, hotel_box)
        return self.nest_spacing_tolerance_base * x_scale
    
    def row_alignment_tolerance(self, width: int, height: int, hotel_box: Optional[HotelBoxConfig] = None) -> float:
        """Get scaled tolerance for vertical alignment of nests within same row."""
        _, y_scale = self.get_scale_factor(width, height, hotel_box)
        return self.row_alignment_tolerance_base * y_scale
    
    def column_alignment_tolerance(self, width: int, height: int, hotel_box: Optional[HotelBoxConfig] = None) -> float:
        """Get scaled tolerance for horizontal alignment of nests within same column."""
        x_scale, _ = self.get_scale_factor(width, height, hotel_box)
        return self.column_alignment_tolerance_base * x_scale
    
    def calculate_dynamic_padding(
        self,
        rows: List[List[Tuple[float, float]]],
        width: int,
        height: int,
        hotel_box: Optional[HotelBoxConfig] = None
    ) -> Tuple[int, int]:
        """Calculate dynamic padding based on actual nest spacing in detected rows."""
        if not self.use_dynamic_padding or not rows:
            return self.padding_x(width, height, hotel_box), self.padding_y(height, height, hotel_box)
        
        # Calculate average horizontal spacing between nests in same row
        all_x_spacings = []
        for row in rows:
            if len(row) > 1:
                sorted_row = sorted(row, key=lambda p: p[0])
                x_positions = [p[0] for p in sorted_row]
                spacings = [x_positions[i+1] - x_positions[i] for i in range(len(x_positions)-1)]
                all_x_spacings.extend(spacings)
        
        # Calculate average vertical spacing between rows
        all_y_spacings = []
        if len(rows) > 1:
            row_y_positions = []
            for row in rows:
                if row:
                    avg_y = sum(p[1] for p in row) / len(row)
                    row_y_positions.append(avg_y)
            
            row_y_positions.sort()
            
            for i in range(len(row_y_positions)-1):
                spacing = row_y_positions[i+1] - row_y_positions[i]
                all_y_spacings.append(spacing)
        
        # Calculate padding as a fraction of the spacing
        if all_x_spacings:
            avg_x_spacing = sum(all_x_spacings) / len(all_x_spacings)
            pad_x = int(avg_x_spacing * self.dynamic_padding_ratio)
            pad_x = max(pad_x, int(self.min_padding_x))
        else:
            pad_x = self.padding_x(width, height, hotel_box)
        
        if all_y_spacings:
            avg_y_spacing = sum(all_y_spacings) / len(all_y_spacings)
            pad_y = int(avg_y_spacing * self.dynamic_padding_ratio)
            pad_y = max(pad_y, int(self.min_padding_y))
        else:
            pad_y = self.padding_y(width, height, hotel_box)
        
        return pad_x, pad_y


@dataclass
class DetectionConfig:
    """Detection configuration for GUI compatibility.
    
    These parameters are used by the GUI and can be synced to TrackingConfig.
    Keeps detection and tracking configuration separate while allowing coordination.
    """
    # Basic blob detection parameters
    min_area: float = 120.0  # Minimum blob area in pixels
    min_solidity: float = 0.7  # Minimum solidity (convex hull area / contour area, 0-1)
    max_area: float = 4000.0  # Maximum blob area in pixels
    
    # Morphological operations
    erosion_kernel_size: int = 3  # Kernel size for morphological erosion
    dilation_kernel_size: int = 5  # Kernel size for morphological dilation
    
    # Detection mode
    detection_mode: str = "fgbg_sift_yolo"  # Options: fgbg, sift, yolo, fgbg_sift, fgbg_yolo, sift_yolo, fgbg_sift_yolo
    
    # Noise filtering
    use_noise_filter: bool = False  # Enable CNN-based noise filtering
    noise_filter_threshold: float = 0.9  # Threshold for noise filter (higher = more conservative)
    
    # Background initialization
    background_init_frames: int = 100  # Number of frames to use for background model initialization
    
    # YOLO parameters
    yolo_interval: int = 10  # Run YOLO detection every N frames (0 = every frame)
    yolo_conf_threshold: float = 0.3  # YOLO confidence threshold
    
    def sync_to_tracking(self, tracking_config: 'TrackingConfig') -> None:
        """Sync detection parameters to tracking config.
        
        This allows GUI-controlled detection parameters to flow into the
        tracking configuration while keeping them logically separated.
        
        Args:
            tracking_config: TrackingConfig instance to update
        """
        tracking_config.min_blob_area_pixels = self.min_area
        tracking_config.min_blob_solidity = self.min_solidity
        tracking_config.max_blob_area_pixels = self.max_area
    
    def to_tracking_params(self) -> Dict[str, Any]:
        """Convert detection params to tracking config params.
        
        Returns:
            Dictionary of tracking config parameter mappings
        """
        return {
            'min_blob_area_pixels': self.min_area,
            'min_blob_solidity': self.min_solidity,
            'max_blob_area_pixels': self.max_area,
        }


@dataclass
class TrackingConfig:
    """Enhanced tracking configuration with hotel-aware scaling AND adaptive tracker parameters.
    
    Now includes v2.2 adaptive tracker configuration for FPS and resolution-independent tracking.
    """

    fast_mode : bool = False
    use_gpu: bool = True

    debug_mode: bool = False
    viz_padding_frames: int = 15
    track_full_frame: bool = False
    min_blob_solidity: float = 1.75
    max_blob_distance_from_hotel: float = 1800.0
    max_frames_without_yolo: int = 60
    force_yolo_on_no_blobs: bool = True
    force_yolo_for_tracking: bool = True

    save_crops = True
    crops_per_track = 10
    
    # Two-mode tracking optimization
    enable_two_mode_tracking: bool = True  # Enable adaptive mode switching
    motion_detection_threshold: int = 1  # Min detections to stay in tracking mode
    tracking_to_detection_delay: int = 30  # Frames without tracks before switching to motion mode
    motion_mode_frame_merge: int = 10  # Merge frames in motion detection mode for efficiency

    # ═══════════════════════════════════════════════════════════════════════════
    # ADAPTIVE TRACKER CONFIGURATION (v2.2)
    # Resolution and FPS-independent tracking parameters
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Time-based parameters (in SECONDS, not frames)
    # These are automatically converted to frames based on video FPS
    max_age_seconds: float = 0.5  # Max time without detection before track dies
                                   # Example: 1.0s = 30 frames @ 30fps, 60 frames @ 60fps
    
    min_hits_seconds: float = 0.00  # Min time before track is confirmed
                                    # Example: 0.1s = 3 frames @ 30fps, 6 frames @ 60fps
    
    max_resurrection_seconds: float = 0.3  # Max time to keep dead tracks for resurrection
                                            # Example: 0.5s = 15 frames @ 30fps, 30 frames @ 60fps
    
    # Distance-based parameters (as MULTIPLIERS of bee size, not pixels)
    # These are automatically scaled based on detected bee size
    resurrection_search_multiplier: float = 2.5  # Search radius for resurrections
                                                  # Example: 1.5x40px = 60px, 1.5x80px = 120px
    
    duplicate_distance_multiplier: float = 1.2  # Distance threshold for duplicate detection
                                                 # Example: 1.2x40px = 48px, 1.2x80px = 96px
    
    # Other adaptive parameters
    adaptive_iou_threshold: float = 0.25  # IoU threshold for matching (dimensionless)
    adaptive_bee_size: Optional[float] = None  # Average bee size in pixels
                                               # None = auto-calculate from detections (recommended)
                                               # number = use fixed size (for fine-tuning)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # END ADAPTIVE TRACKER CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════════

    # Adaptive association parameters
    adaptive_association: bool = True
    stationary_speed_threshold: float = 5.0
    max_speed_factor: float = 3.0
    lateral_search_ratio: float = 0.6
    reverse_search_ratio: float = 0.8
    reverse_motion_penalty: float = 1.3
    area_similarity_weight: float = 0.7

    # All optimized values now here
    max_age: int = 10
    association_threshold_base: float = 100.0
    min_blob_aspect_ratio: float = 0.4
    max_blob_aspect_ratio: float = 2.5

    initial_distance_threshold: float = 30.0
    
    min_blob_area_pixels: float = 500
    max_blob_area_pixels: float = 1500.0
    
    new_track_distance_threshold: float = 100.0
    new_track_proximity_check: float = 100.0
    low_res_switch_threshold: int = 50
    kalman_process_noise: float = 0.15
    
    # Reference resolution
    reference_width: int = 1280
    reference_height: int = 720
    
    # Track management
    no_motion_frames: int = 15
    track_start_id: int = 0
    
    # Association thresholds (base values at reference resolution)
    distance_threshold_base: float = 100.0
    
    # Background subtractor parameters (MOG2)
    bg_history: int = 500
    bg_var_threshold: float = 50.0
    bg_detect_shadows: bool = True
    bg_learning_rate: float = -1.0
    
    # Background initialization
    bg_init_max_frames: int = 200
    bg_init_target_frames: int = 50
    bg_init_learning_rate: float = 0.1
    bg_init_enabled: bool = True
    
    # Kalman filter parameters
    kalman_measurement_noise: float = 1.0
    kalman_state_vars: int = 4
    kalman_measure_vars: int = 2
    
    # Foreground/background segmentation
    min_fg_area_base: float = 250.0
    min_bee_blob_area: float = 250.0
    fg_area_method: str = "contours"
    min_contour_area_base: float = 250.0
    max_contour_area_base: float = 10000.0
    
    # Blob filtering
    aspect_ratio_min: float = 0.2
    aspect_ratio_max: float = 5.0
    
    # Low-resolution processing mode
    enable_low_res_mode: bool = True
    low_res_scale_factor: float = 0.5
    low_res_check_interval: int = 5
    
    # YOLO detection parameters
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.4
    
    # YOLO confirmation strategy
    yolo_confirmation_distance: float = 100.0
    require_yolo_for_new_tracks: bool = True
    min_yolo_confirmations: int = 1
    yolo_reconfirm_interval: int = 60
    
    # Species tracking
    enable_species_tracking: bool = False
    label_map: Dict[int, str] = field(default_factory=lambda: {0: 'bee', 1: 'bee'})
    tracking_classes: List[int] = field(default_factory=lambda: [0])
    
    # Motion detection (legacy)
    motion_threshold: int = 25

    # Safety limits for association
    max_association_distance: int = 100
    max_bee_velocity: int = 50
    max_bee_acceleration: int = 30
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ADAPTIVE TRACKER HELPER METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_adaptive_tracker_params(self, fps: float) -> Dict[str, Any]:
        """Get adaptive tracker parameters for BeeTracker initialization.
        
        Args:
            fps: Video frame rate
            
        Returns:
            Dictionary of parameters for BeeTracker.__init__()
        """
        return {
            'fps': fps,
            'bee_size': self.adaptive_bee_size,
            'max_age_seconds': self.max_age_seconds,
            'min_hits_seconds': self.min_hits_seconds,
            'max_resurrection_seconds': self.max_resurrection_seconds,
            'resurrection_search_multiplier': self.resurrection_search_multiplier,
            'duplicate_distance_multiplier': self.duplicate_distance_multiplier,
            'iou_threshold': self.adaptive_iou_threshold
        }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # END ADAPTIVE TRACKER HELPER METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Scaling methods
    def distance_threshold(self, width: int, height: int, hotel_box=None) -> float:
        """Get scaled distance threshold for tracking association."""
        x_scale = width / self.reference_width
        if hotel_box is not None:
            x_scale *= hotel_box.get_scale_factor()
        return self.distance_threshold_base * x_scale
    
    def association_threshold(self, width: int, height: int, hotel_box=None) -> float:
        """Get scaled association threshold for Hungarian matching."""
        x_scale = width / self.reference_width
        if hotel_box is not None:
            x_scale *= hotel_box.get_scale_factor()
        return self.association_threshold_base * x_scale
    
    def min_contour_area(self, width: int, height: int, hotel_box=None) -> float:
        """Get scaled minimum contour area for blob detection."""
        x_scale = width / self.reference_width
        y_scale = height / self.reference_height
        
        if hotel_box is not None:
            distance_scale = hotel_box.get_scale_factor()
            x_scale *= distance_scale
            y_scale *= distance_scale
        
        area_scale = x_scale * y_scale
        return self.min_contour_area_base * area_scale
    
    def max_contour_area(self, width: int, height: int, hotel_box=None) -> float:
        """Get scaled maximum contour area for blob filtering."""
        x_scale = width / self.reference_width
        y_scale = height / self.reference_height
        
        if hotel_box is not None:
            distance_scale = hotel_box.get_scale_factor()
            x_scale *= distance_scale
            y_scale *= distance_scale
        
        area_scale = x_scale * y_scale
        return self.max_contour_area_base * area_scale
    
    def min_fg_area(self, width: int, height: int, hotel_box=None) -> float:
        """Get scaled minimum foreground area to trigger YOLO."""
        x_scale = width / self.reference_width
        y_scale = height / self.reference_height
        
        if hotel_box is not None:
            distance_scale = hotel_box.get_scale_factor()
            x_scale *= distance_scale
            y_scale *= distance_scale
        
        area_scale = x_scale * y_scale
        return self.min_fg_area_base * area_scale
    
    def yolo_match_distance(self, width: int, height: int, hotel_box=None) -> float:
        """Get scaled distance threshold for YOLO-to-blob matching."""
        x_scale = width / self.reference_width
        if hotel_box is not None:
            x_scale *= hotel_box.get_scale_factor()
        return self.yolo_confirmation_distance * x_scale
    
    def validate_blob(self, contour: 'np.ndarray', width: int, height: int, hotel_box=None) -> Tuple[bool, str]:
        """Validate if a blob/contour should be tracked."""
        import cv2
        
        area = cv2.contourArea(contour)
        min_area = self.min_contour_area(width, height, hotel_box)
        max_area = self.max_contour_area(width, height, hotel_box)
        
        if area < min_area:
            return False, "area_too_small"
        if area > max_area:
            return False, "area_too_large"
        
        x, y, w, h = cv2.boundingRect(contour)
        if h == 0:
            return False, "invalid_height"
        
        aspect_ratio = w / h
        if aspect_ratio < self.aspect_ratio_min or aspect_ratio > self.aspect_ratio_max:
            return False, "invalid_aspect_ratio"
        
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            return False, "invalid_hull"
        
        solidity = area / hull_area
        if solidity < self.min_blob_solidity:
            return False, "low_solidity"
        
        return True, "valid"
    
    def get_bg_subtractor_params(self) -> Dict:
        """Get parameters for cv2.createBackgroundSubtractorMOG2."""
        return {
            'history': self.bg_history,
            'varThreshold': self.bg_var_threshold,
            'detectShadows': self.bg_detect_shadows
        }
    
    def get_kalman_params(self) -> Dict:
        """Get parameters for Kalman filter initialization."""
        return {
            'state_vars': self.kalman_state_vars,
            'measure_vars': self.kalman_measure_vars,
            'process_noise': self.kalman_process_noise,
            'measurement_noise': self.kalman_measurement_noise
        }
    
    def get_bg_init_params(self) -> Dict:
        """Get parameters for background initialization."""
        return {
            'max_frames': self.bg_init_max_frames,
            'target_clean_frames': self.bg_init_target_frames,
            'learning_rate': self.bg_init_learning_rate,
            'enabled': self.bg_init_enabled
        }
    
    @classmethod
    def default(cls) -> 'TrackingConfig':
        """Get default configuration for standard bee monitoring."""
        return cls()
    
    @classmethod
    def high_activity(cls) -> 'TrackingConfig':
        """Configuration optimized for high bee activity videos."""
        config = cls()
        config.min_fg_area_base = 100.0
        config.yolo_reconfirm_interval = 30
        config.max_age = 20
        config.bg_init_target_frames = 30
        config.low_res_switch_threshold = 20
        # Adaptive tracker params for high activity
        config.max_age_seconds = 0.7
        config.min_hits_seconds = 0.05
        return config
    
    @classmethod
    def low_activity(cls) -> 'TrackingConfig':
        """Configuration optimized for sparse bee activity."""
        config = cls()
        config.min_fg_area_base = 250.0
        config.yolo_reconfirm_interval = 100
        config.max_age = 50
        config.bg_init_target_frames = 75
        config.low_res_switch_threshold = 50
        # Adaptive tracker params for low activity
        config.max_age_seconds = 1.5
        config.max_resurrection_seconds = 0.8
        return config
    
    @classmethod
    def high_accuracy(cls) -> 'TrackingConfig':
        """Configuration optimized for maximum accuracy."""
        config = cls()
        config.yolo_reconfirm_interval = 20
        config.min_yolo_confirmations = 2
        config.min_blob_solidity = 0.5
        config.confidence_threshold = 0.6
        # Adaptive tracker params for high accuracy
        config.min_hits_seconds = 0.2
        config.duplicate_distance_multiplier = 1.0
        return config
    
    @classmethod
    def high_speed(cls) -> 'TrackingConfig':
        """Configuration optimized for processing speed."""
        config = cls()
        config.yolo_reconfirm_interval = 200
        config.min_yolo_confirmations = 1
        config.enable_low_res_mode = True
        config.low_res_scale_factor = 0.4
        config.low_res_check_interval = 10
        config.min_blob_solidity = 0.2
        # Adaptive tracker params for speed
        config.max_age_seconds = 0.5
        config.resurrection_search_multiplier = 1.0
        return config
    
    def to_dict(self) -> Dict:
        """Export configuration as dictionary."""
        from dataclasses import asdict
        return asdict(self)
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return (
            f"TrackingConfig(\n"
            f"  max_age={self.max_age},\n"
            f"  bg_history={self.bg_history},\n"
            f"  min_fg_area_base={self.min_fg_area_base},\n"
            f"  enable_low_res_mode={self.enable_low_res_mode},\n"
            f"  yolo_reconfirm_interval={self.yolo_reconfirm_interval},\n"
            f"  tracking_classes={self.tracking_classes},\n"
            f"  # Adaptive tracker params:\n"
            f"  max_age_seconds={self.max_age_seconds},\n"
            f"  min_hits_seconds={self.min_hits_seconds},\n"
            f"  max_resurrection_seconds={self.max_resurrection_seconds},\n"
            f"  resurrection_search_multiplier={self.resurrection_search_multiplier},\n"
            f"  duplicate_distance_multiplier={self.duplicate_distance_multiplier}\n"
            f")"
        )


@dataclass
class ProcessingConfig:
    """Processing configuration with hotel-aware scaling."""
    
    reference_width: int = 1920
    reference_height: int = 1080
    
    min_trajectory_length: int = 5
    entry_window_size: int = 6
    exit_window_size: int = 3
    
    entry_padding_base: float = 10.0
    exit_padding_base: float = 20.0
    start_speed_threshold_base: float = 10.0
    end_speed_threshold_base: float = 10.0
    
    def entry_padding(self, width: int, height: int, hotel_box: Optional[HotelBoxConfig] = None) -> float:
        """Get scaled entry padding."""
        x_scale = width / self.reference_width
        if hotel_box is not None:
            x_scale *= hotel_box.get_scale_factor()
        return self.entry_padding_base * x_scale
    
    def exit_padding(self, width: int, height: int, hotel_box: Optional[HotelBoxConfig] = None) -> float:
        """Get scaled exit padding."""
        x_scale = width / self.reference_width
        if hotel_box is not None:
            x_scale *= hotel_box.get_scale_factor()
        return self.exit_padding_base * x_scale
    
    def start_speed_threshold(self, width: int, height: int, hotel_box: Optional[HotelBoxConfig] = None) -> float:
        """Get scaled start speed threshold."""
        x_scale = width / self.reference_width
        if hotel_box is not None:
            x_scale *= hotel_box.get_scale_factor()
        return self.start_speed_threshold_base * x_scale
    
    def end_speed_threshold(self, width: int, height: int, hotel_box: Optional[HotelBoxConfig] = None) -> float:
        """Get scaled end speed threshold."""
        x_scale = width / self.reference_width
        if hotel_box is not None:
            x_scale *= hotel_box.get_scale_factor()
        return self.end_speed_threshold_base * x_scale


@dataclass
class OutputConfig:
    """Output generation configuration."""
    base_folder: str = "output"
    save_visualizations: bool = False
    save_intermediate_frames: bool = False
    video_codec: str = "mp4v"
    csv_include_species: bool = True
    csv_columns: Optional[List[str]] = None


@dataclass
class Config:
    """Main configuration with hotel box-aware parameter scaling and adaptive tracker."""
    
    video: VideoConfig = field(default_factory=VideoConfig)
    hotel_box: HotelBoxConfig = field(default_factory=HotelBoxConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    nest: NestConfig = field(default_factory=NestConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    @property
    def resolution(self) -> Tuple[int, int]:
        """Get current resolution as tuple."""
        return (self.video.res_width, self.video.res_height)
    
    def get_nest_params(self, hotel_box: Optional[HotelBoxConfig] = None) -> Dict[str, Any]:
        """Get all scaled nest detection parameters."""
        if hotel_box is None:
            hotel_box = self.hotel_box
        
        width, height = self.resolution
        return {
            'nest_width': self.nest.nest_width(width, height, hotel_box),
            'nest_height': self.nest.nest_height(width, height, hotel_box),
            'padding_x': self.nest.padding_x(width, height, hotel_box),
            'padding_y': self.nest.padding_y(width, height, hotel_box),
            'hotel_padding_x': self.nest.hotel_padding_x(width, height, hotel_box),
            'hotel_padding_y': self.nest.hotel_padding_y(width, height, hotel_box),
            'spacing_tolerance': self.nest.spacing_tolerance(width, height, hotel_box),
            'x_position_tolerance': self.nest.x_position_tolerance(width, height, hotel_box),
            'y_position_tolerance': self.nest.y_position_tolerance(width, height, hotel_box),
        }
    
    def get_tracking_params(self, hotel_box: Optional[HotelBoxConfig] = None) -> Dict[str, Any]:
        """Get all scaled tracking parameters."""
        if hotel_box is None:
            hotel_box = self.hotel_box
        
        width, height = self.resolution
        return {
            'distance_threshold': self.tracking.distance_threshold(width, height, hotel_box),
            'association_threshold': self.tracking.association_threshold(width, height, hotel_box),
            'max_age': self.tracking.max_age,
            'no_motion_frames': self.tracking.no_motion_frames,
        }
    
    def get_adaptive_tracker_params(self) -> Dict[str, Any]:
        """Get adaptive tracker parameters for BeeTracker initialization.
        
        Returns:
            Dictionary of parameters for BeeTracker.__init__()
        """
        return self.tracking.get_adaptive_tracker_params(self.video.fps)
    
    def get_processing_params(self, hotel_box: Optional[HotelBoxConfig] = None) -> Dict[str, Any]:
        """Get all scaled processing parameters."""
        if hotel_box is None:
            hotel_box = self.hotel_box
        
        width, height = self.resolution
        return {
            'entry_padding': self.processing.entry_padding(width, height, hotel_box),
            'exit_padding': self.processing.exit_padding(width, height, hotel_box),
            'start_speed_threshold': self.processing.start_speed_threshold(width, height, hotel_box),
            'end_speed_threshold': self.processing.end_speed_threshold(width, height, hotel_box),
            'min_trajectory_length': self.processing.min_trajectory_length,
        }
    
    def sync_detection_to_tracking(self) -> None:
        """Sync detection parameters to tracking configuration.
        
        Call this after updating detection parameters from GUI to ensure
        tracking uses the updated values.
        """
        self.detection.sync_to_tracking(self.tracking)
    
    def print_scaled_values(self) -> None:
        """Print all scaled parameter values for current resolution and hotel box."""
        width, height = self.resolution
        print(f"Configuration for resolution: {width}x{height}")
        print(f"Reference resolution: {self.nest.reference_width}x{self.nest.reference_height}")
        print(f"Hotel box distance factor: {self.hotel_box.distance_factor:.2f}")
        
        x_scale, y_scale = self.nest.get_scale_factor(width, height, self.hotel_box)
        print(f"Scale factors: X={x_scale:.3f}, Y={y_scale:.3f}")
        print()
        
        print("Hotel Box Configuration:")
        bounds = self.hotel_box.get_box_bounds(width, height)
        print(f"  Position: ({self.hotel_box.x_center:.2f}, {self.hotel_box.y_center:.2f})")
        print(f"  Size: {self.hotel_box.width_ratio:.2f} x {self.hotel_box.height_ratio:.2f}")
        print(f"  Bounds: {bounds}")
        print()
        
        print("Detection Configuration:")
        print(f"  min_area: {self.detection.min_area}")
        print(f"  min_solidity: {self.detection.min_solidity}")
        print(f"  max_area: {self.detection.max_area}")
        print()
        
        print("Adaptive Tracker Configuration:")
        adaptive_params = self.get_adaptive_tracker_params()
        for key, value in adaptive_params.items():
            print(f"  {key}: {value}")
        print()
        
        print("Nest Detection Parameters:")
        nest_params = self.get_nest_params()
        for key, value in nest_params.items():
            print(f"  {key}: {value}")
        print()
        
        print("Tracking Parameters:")
        tracking_params = self.get_tracking_params()
        for key, value in tracking_params.items():
            print(f"  {key}: {value}")
        print()
        
        print("Processing Parameters:")
        processing_params = self.get_processing_params()
        for key, value in processing_params.items():
            print(f"  {key}: {value}")
    
    @classmethod
    def default(cls) -> 'Config':
        """Create a Config instance with default values."""
        return cls()
    
    @classmethod
    def high_quality(cls) -> 'Config':
        """Create a Config optimized for high-quality detection."""
        config = cls()
        config.nest.confidence_threshold = 0.95
        config.nest.min_detections = 60
        config.nest.nest_count_tolerance = 0
        config.nest.spacing_tolerance_base = 10.0
        config.nest.x_position_tolerance_base = 10.0
        config.nest.y_position_tolerance_base = 8.0
        config.tracking.max_age = 20
        config.tracking.association_threshold_base = 150.0
        # Adaptive tracker for high quality
        config.tracking.min_hits_seconds = 0.2
        config.tracking.duplicate_distance_multiplier = 1.0
        return config
    
    @classmethod
    def fast(cls) -> 'Config':
        """Create a Config optimized for speed."""
        config = cls()
        config.nest.confidence_threshold = 0.8
        config.nest.min_detections = 55
        config.nest.max_detection_attempts = 5
        config.nest.nest_count_tolerance = 5
        config.nest.spacing_tolerance_base = 20.0
        config.tracking.no_motion_frames = 20
        # Adaptive tracker for speed
        config.tracking.max_age_seconds = 0.5
        config.tracking.resurrection_search_multiplier = 1.0
        return config
    
    @classmethod
    def for_distance(cls, distance_factor: float) -> 'Config':
        """Create a Config for a specific camera distance."""
        config = cls()
        config.hotel_box.distance_factor = distance_factor
        return config
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'Config':
        """Create a Config from a dictionary."""
        video_config = VideoConfig(**config_dict.get('video', {}))
        hotel_box_config = HotelBoxConfig(**config_dict.get('hotel_box', {}))
        models_config = ModelConfig(**config_dict.get('models', {}))
        nest_config = NestConfig(**config_dict.get('nest', {}))
        detection_config = DetectionConfig(**config_dict.get('detection', {}))
        tracking_config = TrackingConfig(**config_dict.get('tracking', {}))
        processing_config = ProcessingConfig(**config_dict.get('processing', {}))
        output_config = OutputConfig(**config_dict.get('output', {}))
        
        return cls(
            video=video_config,
            hotel_box=hotel_box_config,
            models=models_config,
            nest=nest_config,
            detection=detection_config,
            tracking=tracking_config,
            processing=processing_config,
            output=output_config
        )
    
    def to_dict(self) -> dict:
        """Convert Config to a dictionary."""
        from dataclasses import asdict
        return asdict(self)
    
    def validate(self) -> bool:
        """Validate the entire configuration."""
        if self.tracking.max_age < 1:
            raise ValueError("tracking.max_age must be at least 1")
        if self.video.fps < 1:
            raise ValueError("video.fps must be at least 1")
        return True
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_file, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    def save_yaml(self, output_path: str) -> None:
        """Save configuration to YAML file."""
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


def get_config_for_scenario(scenario: str, **kwargs) -> Config:
    """Get a pre-configured Config for common scenarios."""
    if scenario == 'default':
        return Config.default()
    elif scenario == 'high_quality':
        return Config.high_quality()
    elif scenario == 'fast':
        return Config.fast()
    elif scenario == 'close':
        distance_factor = kwargs.get('distance_factor', 0.7)
        return Config.for_distance(distance_factor)
    elif scenario == 'far':
        distance_factor = kwargs.get('distance_factor', 1.5)
        return Config.for_distance(distance_factor)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")