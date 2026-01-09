




"""Training data generation for bee detection - SIMPLIFIED VERSION.

This module extracts labeled frames from videos focused on two categories:
1. bee_like - Frames with bee-like insects (based on interested_bee_labels)
2. non_bee_like - Frames with non-bee-like insects (other detections)

Empty frames and nest frames are naturally included in the dataset for balance.
"""

import logging
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
import json
from tqdm import tqdm
import random
import multiprocessing as mp
from functools import partial

from ultralytics import YOLO
from beemonitor.core.config import Config
from beemonitor.detection.motion_tracking import MotionTracking
from beemonitor.detection.nest_detector import NestDetector

logger = logging.getLogger(__name__)


@dataclass
class FrameAnnotation:
    """Annotation for a single frame."""
    video_path: str
    frame_number: int
    frame_path: str
    category: str  # 'bee_like' or 'non_bee_like'
    bboxes: List[Dict]  # List of bbox annotations
    metadata: Dict


# def _process_video_worker(args: Dict) -> Tuple[List, Dict]:
#     """Worker function for parallel video processing.
    
#     This function must be at module level to be picklable for multiprocessing.
    
#     Args:
#         args: Dictionary containing all parameters needed for video processing
        
#     Returns:
#         Tuple of (annotations, stats) for the processed video
#     """
#     # Extract parameters
#     video_path = args['video_path']
#     sample_rate = args['sample_rate']
#     min_motion_area = args['min_motion_area']
#     max_motion_area = args['max_motion_area']
#     diversity_threshold = args['diversity_threshold']
#     quality_threshold = args['quality_threshold']
#     min_frame_gap = args['min_frame_gap']
#     enforce_event_diversity = args['enforce_event_diversity']
#     max_frames_per_session = args['max_frames_per_session']
#     targets = args['targets']
#     bee_detector_model_path = args['bee_detector_model_path']
#     interested_bee_labels = args['interested_bee_labels']
#     min_detection_confidence = args['min_detection_confidence']
#     use_quality_filtering = args['use_quality_filtering']
#     use_motion_validation = args['use_motion_validation']
#     use_temporal_validation = args['use_temporal_validation']
#     output_folder = args['output_folder']
#     config = args['config']
    
#     # Create a temporary generator for this worker
#     # Each worker gets its own YOLO model instance
#     from beemonitor.ml.generate_training_data import TrainingDataGenerator
    
#     temp_generator = TrainingDataGenerator.__new__(TrainingDataGenerator)
#     temp_generator.video_folder = Path(video_path).parent
#     temp_generator.output_folder = output_folder
#     temp_generator.config = config
#     temp_generator.bee_detector_model_path = bee_detector_model_path
#     temp_generator.bee_detector_model = YOLO(bee_detector_model_path)
#     temp_generator.interested_bee_labels = interested_bee_labels
#     temp_generator.min_detection_confidence = min_detection_confidence
#     temp_generator.use_quality_filtering = use_quality_filtering
#     temp_generator.use_motion_validation = use_motion_validation
#     temp_generator.use_temporal_validation = use_temporal_validation
#     temp_generator.detection_history = []
#     temp_generator.history_length = 5
#     temp_generator.motion_detector = MotionDetector(config)
    
#     # Initialize stats for this video
#     video_stats = {
#         'bee_like_frames': 0,
#         'non_bee_like_frames': 0,
#         'empty_frames_included': 0,
#         'videos_processed': 1,
#         'frames_analyzed': 0,
#         'frames_rejected_quality': 0,
#         'frames_rejected_diversity': 0,
#         'frames_rejected_temporal': 0,
#         'unique_event_sessions_sampled': 0,
#         'total_detections': 0
#     }
    
#     # Process video using the temp generator's method
#     annotations = temp_generator._process_video(
#         video_path=video_path,
#         sample_rate=sample_rate,
#         min_motion_area=min_motion_area,
#         max_motion_area=max_motion_area,
#         diversity_threshold=diversity_threshold,
#         quality_threshold=quality_threshold,
#         min_frame_gap=min_frame_gap,
#         enforce_event_diversity=enforce_event_diversity,
#         max_frames_per_session=max_frames_per_session,
#         current_stats=video_stats,
#         targets=targets
#     )
    
#     return annotations, video_stats

def _process_video_worker(args: Dict) -> Tuple[List, Dict]:
    """Worker function for parallel video processing.
    
    This function must be at module level to be picklable for multiprocessing.
    
    Args:
        args: Dictionary containing all parameters needed for video processing
        
    Returns:
        Tuple of (annotations, stats) for the processed video
    """
    # Extract parameters
    video_path = args['video_path']
    sample_rate = args['sample_rate']
    min_motion_area = args['min_motion_area']
    max_motion_area = args['max_motion_area']
    diversity_threshold = args['diversity_threshold']
    quality_threshold = args['quality_threshold']
    min_frame_gap = args['min_frame_gap']
    enforce_event_diversity = args['enforce_event_diversity']
    max_frames_per_session = args['max_frames_per_session']
    targets = args['targets']
    bee_detector_model_path = args['bee_detector_model_path']
    interested_bee_labels = args['interested_bee_labels']
    min_detection_confidence = args['min_detection_confidence']
    use_quality_filtering = args['use_quality_filtering']
    use_motion_validation = args['use_motion_validation']
    use_temporal_validation = args['use_temporal_validation']
    output_folder = args['output_folder']
    config = args['config']
    include_empty_frames = args.get('include_empty_frames', True)  # Add this
    
    # Create a temporary generator for this worker
    # Each worker gets its own YOLO model instance
    from beemonitor.ml.generate_image_training_data import TrainingDataGenerator
    
    temp_generator = TrainingDataGenerator.__new__(TrainingDataGenerator)
    temp_generator.video_folder = Path(video_path).parent
    temp_generator.output_folder = output_folder
    temp_generator.config = config
    temp_generator.bee_detector_model_path = bee_detector_model_path
    temp_generator.bee_detector_model = YOLO(bee_detector_model_path)
    temp_generator.interested_bee_labels = interested_bee_labels
    temp_generator.min_detection_confidence = min_detection_confidence
    temp_generator.use_quality_filtering = use_quality_filtering
    temp_generator.use_motion_validation = use_motion_validation
    temp_generator.use_temporal_validation = use_temporal_validation
    temp_generator.include_empty_frames = include_empty_frames  # Add this
    temp_generator.detection_history = []
    temp_generator.history_length = 5
    temp_generator.motion_detector = MotionDetector(config)
    
    # Initialize trajectory/event tracking attributes (THESE WERE MISSING!)
    temp_generator.last_sampled_frame = -1
    temp_generator.min_frame_gap = min_frame_gap
    temp_generator.sampled_event_sessions = {}  # This was the main missing attribute
    temp_generator.current_event_session_id = None
    temp_generator.event_session_counter = 0
    temp_generator.max_frames_per_session = max_frames_per_session
    temp_generator.enforce_event_diversity = enforce_event_diversity
    
    # Initialize stats for this video
    video_stats = {
        'bee_like_frames': 0,
        'non_bee_like_frames': 0,
        'empty_frames_included': 0,
        'videos_processed': 1,
        'frames_analyzed': 0,
        'frames_rejected_quality': 0,
        'frames_rejected_diversity': 0,
        'frames_rejected_temporal': 0,
        'unique_event_sessions_sampled': 0,
        'total_detections': 0
    }
    
    # Process video using the temp generator's method
    annotations = temp_generator._process_video(
        video_path=video_path,
        sample_rate=sample_rate,
        min_motion_area=min_motion_area,
        max_motion_area=max_motion_area,
        diversity_threshold=diversity_threshold,
        quality_threshold=quality_threshold,
        min_frame_gap=min_frame_gap,
        enforce_event_diversity=enforce_event_diversity,
        max_frames_per_session=max_frames_per_session,
        current_stats=video_stats,
        targets=targets
    )
    
    return annotations, video_stats


class TrainingDataGenerator:
    """Generate training data from videos for bee detection.
    
    Simplified version focusing on two categories:
    - bee_like: Insects matching interested_bee_labels (with motion validation)
    - non_bee_like: Other insects (hard negatives)
    
    Empty frames and nest context are naturally included in both categories.
    
    Attributes:
        video_folder: Path to folder containing videos
        output_folder: Path where training data will be saved
        config: Configuration object
        bee_detector_model: YOLO model for detection
        interested_bee_labels: List of class IDs considered as bees
        
    Example:
        >>> generator = TrainingDataGenerator(
        ...     video_folder="data/videos",
        ...     output_folder="data/training",
        ...     interested_bee_labels=[0, 1, 2]
        ... )
        >>> generator.generate_dataset(
        ...     num_bee_like_frames=1000,
        ...     num_non_bee_like_frames=200
        ... )
    """
    
    def __init__(
        self,
        video_folder: str,
        output_folder: str,
        config: Optional[Config] = None, 
        bee_detector_model: Optional[str] = None,
        interested_bee_labels: Optional[List[int]] = None,
        min_detection_confidence: float = 0.25,
        use_quality_filtering: bool = True,
        include_empty_frames: bool = True,
        use_motion_validation: bool = False,
        use_temporal_validation: bool = False,
        num_workers: int = 1,
        sampled_event_sessions: Optional[Set[int]] = None
    ):
        """Initialize TrainingDataGenerator.
        
        Args:
            video_folder: Path to folder containing videos
            output_folder: Path where training data will be saved
            config: Configuration object (optional)
            bee_detector_model: Path to YOLO model (optional)
            interested_bee_labels: List of class IDs to consider as bees (optional)
            min_detection_confidence: Minimum confidence for detections
            use_quality_filtering: Whether to filter low-quality frames
            include_empty_frames: Whether to include frames without detections
            use_motion_validation: Whether to validate detections with motion (default: False)
            use_temporal_validation: Whether to check temporal consistency (default: False)
            num_workers: Number of parallel workers for video processing (default: 1)
        """
        self.video_folder = Path(video_folder)
        self.output_folder = Path(output_folder)
        self.config = config if config is not None else Config.default()
        
        # Load bee detector model
        if bee_detector_model is None:
            bee_detector_model = getattr(self.config.ml.bee_detector, 'model_path', 'yolov8n.pt')
        
        # Store model path for multiprocessing
        self.bee_detector_model_path = bee_detector_model
        
        logger.info(f"Loading bee detector model: {bee_detector_model}")
        self.bee_detector_model = YOLO(bee_detector_model)
        
        # Set interested bee labels
        if interested_bee_labels is None:
            self.interested_bee_labels = list(range(len(self.bee_detector_model.names)))
        else:
            self.interested_bee_labels = interested_bee_labels
        
        self.min_detection_confidence = min_detection_confidence
        self.use_quality_filtering = use_quality_filtering
        self.include_empty_frames = include_empty_frames
        self.use_motion_validation = use_motion_validation
        self.use_temporal_validation = use_temporal_validation
        
        # Multiprocessing settings
        self.num_workers = max(1, min(num_workers, mp.cpu_count()))
        
        logger.info(f"Interested bee labels: {self.interested_bee_labels}")
        logger.info(f"Model classes: {self.bee_detector_model.names}")
        logger.info(f"Minimum detection confidence: {min_detection_confidence}")
        logger.info(f"Include empty frames: {include_empty_frames}")
        logger.info(f"Use motion validation: {use_motion_validation}")
        logger.info(f"Use temporal validation: {use_temporal_validation}")
        logger.info(f"Parallel workers: {self.num_workers} (CPU cores: {mp.cpu_count()})")
        
        # Validate paths
        if not self.video_folder.exists():
            raise FileNotFoundError(f"Video folder not found: {video_folder}")
        
        # Create output directories (simplified structure)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        (self.output_folder / 'images').mkdir(exist_ok=True)
        (self.output_folder / 'annotations').mkdir(exist_ok=True)
        (self.output_folder / 'bee_like_frames').mkdir(exist_ok=True)
        (self.output_folder / 'non_bee_like_frames').mkdir(exist_ok=True)
        
        # Initialize motion detector
        self.motion_detector = MotionDetector(config)
        
        # Tracking state for temporal consistency
        self.detection_history = []
        self.history_length = 5
        
        # Trajectory/event session tracking for diversity
        self.last_sampled_frame = -1
        self.min_frame_gap = 90  # Minimum frames between samples (e.g., 3 sec at 30fps)
        self.sampled_event_sessions = {}  # Track frames sampled per session: {session_id: count}
        self.current_event_session_id = None
        self.event_session_counter = 0
        self.max_frames_per_session = None  # Will be calculated dynamically
        
        logger.info(f"TrainingDataGenerator initialized")
        logger.info(f"  Video folder: {self.video_folder}")
        logger.info(f"  Output folder: {self.output_folder}")
        logger.info(f"  Focus: bee_like vs non_bee_like detection")
    
    def generate_dataset(
        self,
        num_bee_like_frames: int = 1000,
        num_non_bee_like_frames: int = 200,
        sample_rate: int = 30,
        min_motion_area: int = 50,
        max_motion_area: int = 5000,
        diversity_threshold: float = 0.3,
        quality_threshold: float = 0.5,
        min_frame_gap: int = 90,
        enforce_event_diversity: bool = True,
        max_frames_per_session: int = None
    ) -> Dict:
        """Generate a balanced training dataset from videos.
        
        Args:
            num_bee_like_frames: Target number of frames with bee-like insects
            num_non_bee_like_frames: Target number of frames with non-bee insects
            sample_rate: Sample every Nth frame
            min_motion_area: Minimum area for motion detection (pixels)
            max_motion_area: Maximum area for motion detection (pixels)
            diversity_threshold: Threshold for frame diversity (0-1)
            quality_threshold: Minimum quality score (0-1)
            min_frame_gap: Minimum frames between samples to avoid same trajectory (default: 90)
            enforce_event_diversity: Ensure frames come from different event sessions
            max_frames_per_session: Maximum frames to sample per event session (None = auto-calculate)
            
        Returns:
            Dictionary with dataset statistics
        """
        logger.info("="*60)
        logger.info("Starting dataset generation (SIMPLIFIED)")
        logger.info("="*60)
        logger.info(f"  Target bee-like frames: {num_bee_like_frames}")
        logger.info(f"  Target non-bee-like frames: {num_non_bee_like_frames}")
        logger.info(f"  Quality threshold: {quality_threshold}")
        logger.info(f"  Min frame gap: {min_frame_gap} frames")
        logger.info(f"  Enforce event diversity: {enforce_event_diversity}")
        
        if max_frames_per_session is None:
            logger.info(f"  Max frames per session: AUTO (will balance across sessions)")
        else:
            logger.info(f"  Max frames per session: {max_frames_per_session}")
        
        # Collect all videos
        video_paths = self._get_video_paths()
        logger.info(f"  Found {len(video_paths)} videos")
        
        if not video_paths:
            raise ValueError("No videos found in the specified folder")
        
        # Initialize counters
        stats = {
            'bee_like_frames': 0,
            'non_bee_like_frames': 0,
            'empty_frames_included': 0,
            'videos_processed': 0,
            'frames_analyzed': 0,
            'frames_rejected_quality': 0,
            'frames_rejected_diversity': 0,
            'frames_rejected_temporal': 0,  # New: rejected due to temporal spacing
            'unique_event_sessions_sampled': 0,  # New: number of unique events sampled
            'total_detections': 0
        }
        
        annotations = []
        
        # Store parameters for tracking
        self.min_frame_gap = min_frame_gap
        self.enforce_event_diversity = enforce_event_diversity
        self.max_frames_per_session = max_frames_per_session
        
        # First pass: estimate number of event sessions if auto-calculating
        if enforce_event_diversity and max_frames_per_session is None:
            logger.info("  Performing quick scan to estimate event sessions...")
            estimated_sessions = self._estimate_event_sessions(
                video_paths, sample_rate, min_motion_area, max_motion_area
            )
            
            if estimated_sessions > 0:
                total_target = num_bee_like_frames + num_non_bee_like_frames
                calculated_max = max(1, int(total_target / estimated_sessions * 1.5))
                self.max_frames_per_session = calculated_max
                logger.info(f"  Estimated sessions: {estimated_sessions}")
                logger.info(f"  Auto-calculated max frames per session: {calculated_max}")
            else:
                self.max_frames_per_session = 15  # Fallback default
                logger.info(f"  Using fallback max frames per session: 15")
        
        # Process videos
        if self.num_workers > 1:
            # Parallel processing
            logger.info(f"  Using {self.num_workers} parallel workers")
            annotations, stats = self._process_videos_parallel(
                video_paths=video_paths,
                sample_rate=sample_rate,
                min_motion_area=min_motion_area,
                max_motion_area=max_motion_area,
                diversity_threshold=diversity_threshold,
                quality_threshold=quality_threshold,
                min_frame_gap=min_frame_gap,
                enforce_event_diversity=enforce_event_diversity,
                max_frames_per_session=self.max_frames_per_session,
                targets={
                    'bee_like': num_bee_like_frames,
                    'non_bee_like': num_non_bee_like_frames
                }
            )
        else:
            # Sequential processing (original behavior)
            for video_path in tqdm(video_paths, desc="Processing videos"):
                if self._is_dataset_complete(stats, num_bee_like_frames, num_non_bee_like_frames):
                    break
                
                video_annotations = self._process_video(
                    video_path=video_path,
                    sample_rate=sample_rate,
                    min_motion_area=min_motion_area,
                    max_motion_area=max_motion_area,
                    diversity_threshold=diversity_threshold,
                    quality_threshold=quality_threshold,
                    min_frame_gap=min_frame_gap,
                    enforce_event_diversity=enforce_event_diversity,
                    max_frames_per_session=self.max_frames_per_session,
                    current_stats=stats,
                    targets={
                        'bee_like': num_bee_like_frames,
                        'non_bee_like': num_non_bee_like_frames
                    }
                )
                
                annotations.extend(video_annotations)
                stats['videos_processed'] += 1
        
        # Save annotations
        self._save_annotations(annotations)
        
        # Generate summary report
        self._generate_report(stats, annotations)
        
        logger.info("="*60)
        logger.info("Dataset generation complete!")
        logger.info("="*60)
        logger.info(f"  Bee-like frames: {stats['bee_like_frames']}")
        logger.info(f"  Non-bee-like frames: {stats['non_bee_like_frames']}")
        logger.info(f"  Empty frames included: {stats['empty_frames_included']}")
        logger.info(f"  Videos processed: {stats['videos_processed']}")
        logger.info(f"  Total detections: {stats['total_detections']}")
        logger.info(f"  Unique event sessions sampled: {stats['unique_event_sessions_sampled']}")
        logger.info(f"  Frames rejected (temporal spacing): {stats['frames_rejected_temporal']}")
        
        return stats
    
    def _get_video_paths(self) -> List[Path]:
        """Get all video paths from the video folder."""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV']
        video_paths = []
        
        for ext in video_extensions:
            video_paths.extend(self.video_folder.glob(f'*{ext}'))
        
        random.shuffle(video_paths)
        return video_paths
    
    def _estimate_event_sessions(
        self,
        video_paths: List[Path],
        sample_rate: int,
        min_motion_area: int,
        max_motion_area: int
    ) -> int:
        """Estimate the number of event sessions in videos with a quick scan.
        
        This does a fast pass through videos to estimate how many unique bee
        trajectories (event sessions) exist, so we can auto-calculate frames per session.
        """
        total_sessions = 0
        
        # Sample first video or first few minutes of first video for estimation
        for video_path in video_paths[:1]:  # Just check first video for speed
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                continue
            
            frame_idx = 0
            prev_frame_gray = None
            current_session = None
            no_detection_count = 0
            session_counter = 0
            max_frames_to_check = 6000  # Check first ~3 minutes at 30fps
            
            while cap.isOpened() and frame_idx < max_frames_to_check:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sample frames
                if frame_idx % (sample_rate * 3) != 0:  # Sample less frequently for estimation
                    frame_idx += 1
                    continue
                
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Run detection
                detections = self._detect_objects_with_model(frame)
                
                # Track sessions
                if detections:
                    if current_session is None:
                        session_counter += 1
                        current_session = session_counter
                    no_detection_count = 0
                else:
                    no_detection_count += 1
                    if no_detection_count > 10:  # End session faster for estimation
                        current_session = None
                
                prev_frame_gray = frame_gray
                frame_idx += 1
            
            cap.release()
            total_sessions += session_counter
            
            # Extrapolate for full video length
            if frame_idx > 0:
                total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_video_frames > frame_idx:
                    scaling_factor = total_video_frames / frame_idx
                    total_sessions = int(total_sessions * scaling_factor)
        
        # Multiply by number of videos
        total_sessions *= len(video_paths)
        
        return max(1, total_sessions)  # At least 1
    
    # def _process_videos_parallel(
    #     self,
    #     video_paths: List[Path],
    #     sample_rate: int,
    #     min_motion_area: int,
    #     max_motion_area: int,
    #     diversity_threshold: float,
    #     quality_threshold: float,
    #     min_frame_gap: int,
    #     enforce_event_diversity: bool,
    #     max_frames_per_session: int,
    #     targets: Dict
    # ) -> Tuple[List[FrameAnnotation], Dict]:
    #     """Process multiple videos in parallel using multiprocessing.
        
    #     Returns:
    #         Tuple of (annotations, aggregated_stats)
    #     """
    #     # Prepare arguments for each video
    #     process_args = []
    #     for video_path in video_paths:
    #         process_args.append({
    #             'video_path': video_path,
    #             'sample_rate': sample_rate,
    #             'min_motion_area': min_motion_area,
    #             'max_motion_area': max_motion_area,
    #             'diversity_threshold': diversity_threshold,
    #             'quality_threshold': quality_threshold,
    #             'min_frame_gap': min_frame_gap,
    #             'enforce_event_diversity': enforce_event_diversity,
    #             'max_frames_per_session': max_frames_per_session,
    #             'targets': targets.copy(),
    #             'bee_detector_model_path': self.bee_detector_model_path,
    #             'interested_bee_labels': self.interested_bee_labels,
    #             'min_detection_confidence': self.min_detection_confidence,
    #             'use_quality_filtering': self.use_quality_filtering,
    #             'use_motion_validation': self.use_motion_validation,
    #             'use_temporal_validation': self.use_temporal_validation,
    #             'output_folder': self.output_folder,
    #             'config': self.config
    #         })
    def _process_videos_parallel(
        self,
        video_paths: List[Path],
        sample_rate: int,
        min_motion_area: int,
        max_motion_area: int,
        diversity_threshold: float,
        quality_threshold: float,
        min_frame_gap: int,
        enforce_event_diversity: bool,
        max_frames_per_session: int,
        targets: Dict
    ) -> Tuple[List[FrameAnnotation], Dict]:
        """Process multiple videos in parallel using multiprocessing.
        
        Returns:
            Tuple of (annotations, aggregated_stats)
        """
        # Prepare arguments for each video
        process_args = []
        for video_path in video_paths:
            process_args.append({
                'video_path': video_path,
                'sample_rate': sample_rate,
                'min_motion_area': min_motion_area,
                'max_motion_area': max_motion_area,
                'diversity_threshold': diversity_threshold,
                'quality_threshold': quality_threshold,
                'min_frame_gap': min_frame_gap,
                'enforce_event_diversity': enforce_event_diversity,
                'max_frames_per_session': max_frames_per_session,
                'targets': targets.copy(),
                'bee_detector_model_path': self.bee_detector_model_path,
                'interested_bee_labels': self.interested_bee_labels,
                'min_detection_confidence': self.min_detection_confidence,
                'use_quality_filtering': self.use_quality_filtering,
                'use_motion_validation': self.use_motion_validation,
                'use_temporal_validation': self.use_temporal_validation,
                'include_empty_frames': self.include_empty_frames,  # Add this line
                'output_folder': self.output_folder,
                'config': self.config
            })


        
        # Process videos in parallel
        with mp.Pool(processes=self.num_workers) as pool:
            # Use imap_unordered for better progress tracking
            results = list(tqdm(
                pool.imap_unordered(_process_video_worker, process_args),
                total=len(process_args),
                desc=f"Processing videos ({self.num_workers} workers)"
            ))
        
        # Aggregate results
        all_annotations = []
        aggregated_stats = {
            'bee_like_frames': 0,
            'non_bee_like_frames': 0,
            'empty_frames_included': 0,
            'videos_processed': 0,
            'frames_analyzed': 0,
            'frames_rejected_quality': 0,
            'frames_rejected_diversity': 0,
            'frames_rejected_temporal': 0,
            'unique_event_sessions_sampled': 0,
            'total_detections': 0
        }
        
        for annotations, stats in results:
            all_annotations.extend(annotations)
            # Aggregate stats
            for key in aggregated_stats:
                aggregated_stats[key] += stats[key]
        
        # Check if we've collected enough frames
        total_collected = aggregated_stats['bee_like_frames'] + aggregated_stats['non_bee_like_frames']
        total_target = targets['bee_like'] + targets['non_bee_like']
        
        if total_collected > total_target:
            # Trim excess frames
            all_annotations = all_annotations[:total_target]
            logger.info(f"  Collected {total_collected} frames, trimmed to target {total_target}")
        
        return all_annotations, aggregated_stats
    
    def _process_video(
        self,
        video_path: Path,
        sample_rate: int,
        min_motion_area: int,
        max_motion_area: int,
        diversity_threshold: float,
        quality_threshold: float,
        min_frame_gap: int,
        enforce_event_diversity: bool,
        max_frames_per_session: int,
        current_stats: Dict,
        targets: Dict
    ) -> List[FrameAnnotation]:
        """Process a single video to extract training frames."""
        logger.info(f"Processing: {video_path.name}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning(f"Could not open video: {video_path}")
            return []
        
        annotations = []
        frame_idx = 0
        prev_frame_gray = None
        sampled_frames = []
        
        # Reset detection history for new video
        self.detection_history = []
        
        # Reset trajectory tracking for new video  
        self.last_sampled_frame = -1
        # Note: Keep sampled_event_sessions as dict across videos for proper counting
        self.current_event_session_id = None
        self.event_session_counter = 0
        no_detection_count = 0  # Track frames without detection
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            current_stats['frames_analyzed'] += 1
            
            # Sample frames
            if frame_idx % sample_rate != 0:
                frame_idx += 1
                continue
            
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Quality check
            if self.use_quality_filtering:
                quality_score, _ = self._assess_frame_quality(frame)
                if quality_score < quality_threshold:
                    current_stats['frames_rejected_quality'] += 1
                    frame_idx += 1
                    continue
            
            # Diversity check
            if not self._is_diverse_frame(frame_gray, sampled_frames, diversity_threshold):
                current_stats['frames_rejected_diversity'] += 1
                frame_idx += 1
                continue
            
            # Run YOLO detection
            detections = self._detect_objects_with_model(frame)
            current_stats['total_detections'] += len(detections)
            
            # Detect motion for validation
            motion_detected = False
            motion_contours = []
            if prev_frame_gray is not None:
                motion_detected, motion_contours = self._detect_motion(
                    prev_frame_gray, frame_gray, min_motion_area, max_motion_area
                )
            
            # Classify frame
            category, matched_bboxes = self._classify_frame(
                frame, detections, motion_contours
            )
            
            # Track event sessions (trajectories)
            # An event session is a continuous sequence of frames with detections
            if category != 'empty' and len(matched_bboxes) > 0:
                # We have a detection
                if self.current_event_session_id is None:
                    # Start new event session
                    self.event_session_counter += 1
                    self.current_event_session_id = f"{video_path.stem}_event_{self.event_session_counter}"
                no_detection_count = 0
            else:
                # No detection
                no_detection_count += 1
                # End event session after 30 frames (~1 second) without detection
                if no_detection_count > 30:
                    self.current_event_session_id = None
            
            # Temporal spacing check: ensure minimum frame gap between samples
            temporal_spacing_ok = (frame_idx - self.last_sampled_frame) >= min_frame_gap
            if not temporal_spacing_ok and category != 'empty':
                current_stats['frames_rejected_temporal'] += 1
                frame_idx += 1
                prev_frame_gray = frame_gray
                continue
            
            # Event diversity check: limit frames per session
            if enforce_event_diversity and self.current_event_session_id is not None and category != 'empty':
                frames_from_this_session = self.sampled_event_sessions.get(
                    self.current_event_session_id, 0
                )
                
                # Check if we've already sampled enough from this session
                if frames_from_this_session >= max_frames_per_session:
                    # Skip this frame - already got enough from this session
                    frame_idx += 1
                    prev_frame_gray = frame_gray
                    continue
            
            # Save based on category and targets
            if category == 'bee_like' and current_stats['bee_like_frames'] < targets['bee_like']:
                annotation = self._save_frame(
                    frame, frame_idx, video_path, matched_bboxes, 'bee_like'
                )
                if annotation:
                    annotations.append(annotation)
                    current_stats['bee_like_frames'] += 1
                    sampled_frames.append(frame_gray)
                    self.last_sampled_frame = frame_idx
                    # Increment session frame count
                    if self.current_event_session_id:
                        if self.current_event_session_id not in self.sampled_event_sessions:
                            current_stats['unique_event_sessions_sampled'] += 1
                            self.sampled_event_sessions[self.current_event_session_id] = 0
                        self.sampled_event_sessions[self.current_event_session_id] += 1
            
            elif category == 'non_bee_like' and current_stats['non_bee_like_frames'] < targets['non_bee_like']:
                annotation = self._save_frame(
                    frame, frame_idx, video_path, matched_bboxes, 'non_bee_like'
                )
                if annotation:
                    annotations.append(annotation)
                    current_stats['non_bee_like_frames'] += 1
                    sampled_frames.append(frame_gray)
                    self.last_sampled_frame = frame_idx
                    # Increment session frame count
                    if self.current_event_session_id:
                        if self.current_event_session_id not in self.sampled_event_sessions:
                            current_stats['unique_event_sessions_sampled'] += 1
                            self.sampled_event_sessions[self.current_event_session_id] = 0
                        self.sampled_event_sessions[self.current_event_session_id] += 1
            
            elif category == 'empty' and self.include_empty_frames:
                # Distribute empty frames across both categories for balance
                if current_stats['bee_like_frames'] < targets['bee_like']:
                    annotation = self._save_frame(
                        frame, frame_idx, video_path, [], 'bee_like'
                    )
                    if annotation:
                        annotations.append(annotation)
                        current_stats['bee_like_frames'] += 1
                        current_stats['empty_frames_included'] += 1
                        sampled_frames.append(frame_gray)
                        self.last_sampled_frame = frame_idx
                elif current_stats['non_bee_like_frames'] < targets['non_bee_like']:
                    annotation = self._save_frame(
                        frame, frame_idx, video_path, [], 'non_bee_like'
                    )
                    if annotation:
                        annotations.append(annotation)
                        current_stats['non_bee_like_frames'] += 1
                        current_stats['empty_frames_included'] += 1
                        sampled_frames.append(frame_gray)
                        self.last_sampled_frame = frame_idx
            
            prev_frame_gray = frame_gray
            frame_idx += 1
            
            # Check if targets met
            if self._is_dataset_complete(current_stats, targets['bee_like'], targets['non_bee_like']):
                break
        
        cap.release()
        return annotations
    
    def _detect_objects_with_model(self, frame: np.ndarray) -> List[Dict]:
        """Detect objects in frame using YOLO model."""
        detections = []
        
        results = self.bee_detector_model(frame, verbose=False)[0]
        
        for box in results.boxes:
            confidence = float(box.conf[0])
            if confidence >= self.min_detection_confidence:
                class_id = int(box.cls[0])
                xyxy = box.xyxy[0].cpu().numpy()
                
                detections.append({
                    'bbox': [int(xyxy[0]), int(xyxy[1]), 
                            int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1])],
                    'class_id': class_id,
                    'class_name': self.bee_detector_model.names[class_id],
                    'confidence': confidence,
                    'is_bee_like': class_id in self.interested_bee_labels
                })
        
        return detections
    
    def _classify_frame(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        motion_contours: List
    ) -> Tuple[str, List[Dict]]:
        """Classify frame as bee_like, non_bee_like, or empty.
        
        Classification logic:
        1. bee_like: Has detections in interested_bee_labels
        2. non_bee_like: Has detections NOT in interested_bee_labels
        3. empty: No detections
        
        Optional validation (if enabled):
        - Motion validation: Match detections with motion contours
        - Temporal validation: Check consistency across frames
        """
        matched_bboxes = []
        
        # Add to detection history
        self.detection_history.append(detections)
        if len(self.detection_history) > self.history_length:
            self.detection_history.pop(0)
        
        # Separate bee-like and non-bee-like detections
        bee_like_detections = [d for d in detections if d['is_bee_like']]
        non_bee_like_detections = [d for d in detections if not d['is_bee_like']]
        
        # Process bee-like detections
        if bee_like_detections:
            validated = bee_like_detections
            
            # Optional: Validate with motion
            if self.use_motion_validation and motion_contours:
                validated = self._match_detections_with_motion(validated, motion_contours)
            
            # Optional: Check temporal consistency
            if self.use_temporal_validation:
                if validated and not self._check_temporal_consistency(validated, is_bee_like=True):
                    validated = []
            
            if validated:
                matched_bboxes = self._format_bboxes(validated, 'bee_like')
                return 'bee_like', matched_bboxes
        
        # Process non-bee-like detections
        if non_bee_like_detections:
            validated = non_bee_like_detections
            
            # Optional: Validate with motion
            if self.use_motion_validation and motion_contours:
                validated = self._match_detections_with_motion(validated, motion_contours)
            
            # Optional: Check temporal consistency
            if self.use_temporal_validation:
                if validated and not self._check_temporal_consistency(validated, is_bee_like=False):
                    validated = []
            
            if validated:
                matched_bboxes = self._format_bboxes(validated, 'non_bee_like')
                return 'non_bee_like', matched_bboxes
        
        # No valid detections
        return 'empty', []
    
    def _format_bboxes(self, detections: List[Dict], category: str) -> List[Dict]:
        """Format detections as bbox dictionaries."""
        return [
            {
                'x': d['bbox'][0],
                'y': d['bbox'][1],
                'width': d['bbox'][2],
                'height': d['bbox'][3],
                'class': category,
                'class_name': d['class_name'],
                'class_id': d['class_id'],
                'confidence': d['confidence']
            }
            for d in detections
        ]
    
    def _match_detections_with_motion(
        self,
        detections: List[Dict],
        motion_contours: List
    ) -> List[Dict]:
        """Match detections with motion contours using IoU."""
        if not motion_contours:
            return detections
        
        matched = []
        for detection in detections:
            x, y, w, h = detection['bbox']
            det_bbox = (x, y, x + w, y + h)
            
            for contour in motion_contours:
                cx, cy, cw, ch = cv2.boundingRect(contour)
                motion_bbox = (cx, cy, cx + cw, cy + ch)
                
                if self._calculate_iou(det_bbox, motion_bbox) > 0.1:
                    matched.append(detection)
                    break
        
        return matched
    
    def _calculate_iou(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Calculate IoU between two bboxes."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = bbox1_area + bbox2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _check_temporal_consistency(
        self,
        current_detections: List[Dict],
        is_bee_like: bool,
        consistency_threshold: float = 0.4
    ) -> bool:
        """Check temporal consistency of detections."""
        if len(self.detection_history) < 2:
            return True
        
        consistent_frames = 0
        for past_detections in self.detection_history[:-1]:
            past_matching = [d for d in past_detections if d['is_bee_like'] == is_bee_like]
            if past_matching:
                consistent_frames += 1
        
        consistency_ratio = consistent_frames / max(1, len(self.detection_history) - 1)
        return consistency_ratio >= consistency_threshold
    
    def _assess_frame_quality(self, frame: np.ndarray) -> Tuple[float, Dict]:
        """Assess frame quality (blur, brightness, contrast)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Blur detection
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = min(1.0, laplacian_var / 500.0)
        
        # Brightness
        mean_intensity = np.mean(gray)
        brightness_score = 1.0 - abs(mean_intensity - 128) / 128.0
        
        # Contrast
        contrast = np.std(gray)
        contrast_score = min(1.0, contrast / 64.0)
        
        quality_score = 0.5 * blur_score + 0.3 * contrast_score + 0.2 * brightness_score
        
        metrics = {
            'blur_score': blur_score,
            'brightness_score': brightness_score,
            'contrast_score': contrast_score,
            'laplacian_var': laplacian_var,
            'mean_intensity': mean_intensity,
            'contrast': contrast
        }
        
        return quality_score, metrics
    
    def _detect_motion(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        min_area: int,
        max_area: int
    ) -> Tuple[bool, List]:
        """Detect motion between frames."""
        diff = cv2.absdiff(prev_frame, curr_frame)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=2)
        thresh = cv2.erode(thresh, kernel, iterations=1)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_contours = [c for c in contours if min_area < cv2.contourArea(c) < max_area]
        
        return len(valid_contours) > 0, valid_contours
    
    def _is_diverse_frame(
        self,
        frame_gray: np.ndarray,
        sampled_frames: List[np.ndarray],
        threshold: float
    ) -> bool:
        """Check if frame is diverse enough."""
        if len(sampled_frames) == 0:
            return True
        
        recent_frames = sampled_frames[-10:]
        frame_small = cv2.resize(frame_gray, (64, 64))
        
        for sampled_frame in recent_frames:
            sampled_small = cv2.resize(sampled_frame, (64, 64))
            diff = cv2.absdiff(frame_small, sampled_small)
            similarity = 1.0 - (np.mean(diff) / 255.0)
            
            if similarity > (1.0 - threshold):
                return False
        
        return True
    
    def _save_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        video_path: Path,
        bboxes: List[Dict],
        category: str
    ) -> Optional[FrameAnnotation]:
        """Save a frame to the appropriate category folder."""
        frame_filename = f"{category}_{video_path.stem}_frame_{frame_idx:06d}.jpg"
        frame_path = self.output_folder / f'{category}_frames' / frame_filename
        
        cv2.imwrite(str(frame_path), frame)
        
        return FrameAnnotation(
            video_path=str(video_path),
            frame_number=frame_idx,
            frame_path=str(frame_path),
            category=category,
            bboxes=bboxes,
            metadata={
                'num_detections': len(bboxes),
                'has_objects': len(bboxes) > 0
            }
        )
    
    def _is_dataset_complete(
        self,
        stats: Dict,
        target_bee_like: int,
        target_non_bee_like: int
    ) -> bool:
        """Check if dataset targets are met."""
        return (stats['bee_like_frames'] >= target_bee_like and
                stats['non_bee_like_frames'] >= target_non_bee_like)
    
    def _save_annotations(self, annotations: List[FrameAnnotation]) -> None:
        """Save annotations to JSON and COCO format."""
        # JSON format
        json_path = self.output_folder / 'annotations' / 'annotations.json'
        with open(json_path, 'w') as f:
            json.dump([vars(ann) for ann in annotations], f, indent=2)
        
        # COCO format
        coco_annotations = self._convert_to_coco(annotations)
        coco_path = self.output_folder / 'annotations' / 'coco_annotations.json'
        with open(coco_path, 'w') as f:
            json.dump(coco_annotations, f, indent=2)
        
        logger.info(f"Saved {len(annotations)} annotations")
    
    def _convert_to_coco(self, annotations: List[FrameAnnotation]) -> Dict:
        """Convert annotations to COCO format."""
        coco = {
            'images': [],
            'annotations': [],
            'categories': [
                {'id': 1, 'name': 'bee_like'},
                {'id': 2, 'name': 'non_bee_like'}
            ]
        }
        
        category_map = {'bee_like': 1, 'non_bee_like': 2}
        annotation_id = 1
        
        for idx, ann in enumerate(annotations):
            coco['images'].append({
                'id': idx + 1,
                'file_name': Path(ann.frame_path).name,
                'video_path': ann.video_path,
                'frame_number': ann.frame_number
            })
            
            for bbox in ann.bboxes:
                coco['annotations'].append({
                    'id': annotation_id,
                    'image_id': idx + 1,
                    'category_id': category_map.get(bbox.get('class', 'bee_like'), 1),
                    'bbox': [bbox['x'], bbox['y'], bbox['width'], bbox['height']],
                    'area': bbox['width'] * bbox['height'],
                    'iscrowd': 0,
                    'confidence': bbox.get('confidence', 1.0),
                    'class_name': bbox.get('class_name', 'unknown')
                })
                annotation_id += 1
        
        return coco
    
    def _generate_report(self, stats: Dict, annotations: List[FrameAnnotation]) -> None:
        """Generate a summary report."""
        report_path = self.output_folder / 'dataset_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("Training Dataset Generation Report (SIMPLIFIED)\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("Configuration:\n")
            f.write(f"  Model: {self.bee_detector_model.ckpt_path}\n")
            f.write(f"  Interested bee labels: {self.interested_bee_labels}\n")
            f.write(f"  Model classes: {self.bee_detector_model.names}\n")
            f.write(f"  Min confidence: {self.min_detection_confidence}\n")
            f.write(f"  Quality filtering: {self.use_quality_filtering}\n")
            f.write(f"  Include empty frames: {self.include_empty_frames}\n")
            f.write(f"  Motion validation: {self.use_motion_validation}\n")
            f.write(f"  Temporal validation: {self.use_temporal_validation}\n")
            f.write(f"  Min frame gap: {self.min_frame_gap} frames\n")
            f.write(f"  Enforce event diversity: {self.enforce_event_diversity}\n")
            f.write(f"  Max frames per session: {self.max_frames_per_session}\n\n")
            
            f.write("Dataset Statistics:\n")
            f.write(f"  Total frames collected: {len(annotations)}\n")
            f.write(f"  Bee-like frames: {stats['bee_like_frames']}\n")
            f.write(f"  Non-bee-like frames: {stats['non_bee_like_frames']}\n")
            f.write(f"  Empty frames included: {stats['empty_frames_included']}\n")
            f.write(f"  Videos processed: {stats['videos_processed']}\n")
            f.write(f"  Frames analyzed: {stats['frames_analyzed']}\n")
            f.write(f"  Total detections: {stats['total_detections']}\n")
            f.write(f"  Unique event sessions sampled: {stats['unique_event_sessions_sampled']}\n")
            f.write(f"  Frames rejected (quality): {stats['frames_rejected_quality']}\n")
            f.write(f"  Frames rejected (diversity): {stats['frames_rejected_diversity']}\n")
            f.write(f"  Frames rejected (temporal spacing): {stats['frames_rejected_temporal']}\n\n")
            
            f.write("Dataset Balance:\n")
            total = len(annotations)
            if total > 0:
                f.write(f"  Bee-like: {stats['bee_like_frames']/total*100:.1f}%\n")
                f.write(f"  Non-bee-like: {stats['non_bee_like_frames']/total*100:.1f}%\n")
                f.write(f"  Empty (distributed): {stats['empty_frames_included']/total*100:.1f}%\n")
            
            f.write("\nQuality Metrics:\n")
            if stats['frames_analyzed'] > 0:
                acceptance_rate = (total / stats['frames_analyzed'] * 100)
                quality_rejection = (stats['frames_rejected_quality'] / stats['frames_analyzed'] * 100)
                diversity_rejection = (stats['frames_rejected_diversity'] / stats['frames_analyzed'] * 100)
                temporal_rejection = (stats['frames_rejected_temporal'] / stats['frames_analyzed'] * 100)
                
                f.write(f"  Overall acceptance rate: {acceptance_rate:.1f}%\n")
                f.write(f"  Quality rejection rate: {quality_rejection:.1f}%\n")
                f.write(f"  Diversity rejection rate: {diversity_rejection:.1f}%\n")
                f.write(f"  Temporal spacing rejection rate: {temporal_rejection:.1f}%\n")
            
            f.write("\nTrajectory Diversity:\n")
            if total > 0:
                f.write(f"  Unique event sessions sampled: {stats['unique_event_sessions_sampled']}\n")
                f.write(f"  Avg frames per event session: {total/max(stats['unique_event_sessions_sampled'], 1):.1f}\n")
                f.write(f"  This ensures frames are from DIFFERENT bee trajectories!\n")
        
        logger.info(f"Report saved to: {report_path}")
    
    def generate_yolo_format(self, train_split: float = 0.8) -> None:
        """Convert annotations to YOLO format."""
        annotations_path = self.output_folder / 'annotations' / 'annotations.json'
        
        if not annotations_path.exists():
            logger.error("Annotations file not found. Run generate_dataset first.")
            return
        
        with open(annotations_path, 'r') as f:
            annotations = json.load(f)
        
        # Create YOLO directory structure
        yolo_dir = self.output_folder / 'yolo_format'
        (yolo_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
        (yolo_dir / 'images' / 'val').mkdir(parents=True, exist_ok=True)
        (yolo_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
        (yolo_dir / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
        
        # Split data
        random.shuffle(annotations)
        split_idx = int(len(annotations) * train_split)
        train_annotations = annotations[:split_idx]
        val_annotations = annotations[split_idx:]
        
        # Convert to YOLO format
        self._convert_to_yolo_format(train_annotations, yolo_dir, 'train')
        self._convert_to_yolo_format(val_annotations, yolo_dir, 'val')
        
        # Create data.yaml
        data_yaml = {
            'train': str(yolo_dir / 'images' / 'train'),
            'val': str(yolo_dir / 'images' / 'val'),
            'nc': 2,  # Two classes
            'names': ['bee_like', 'non_bee_like']
        }
        
        import yaml
        with open(yolo_dir / 'data.yaml', 'w') as f:
            yaml.dump(data_yaml, f)
        
        logger.info(f"YOLO format dataset created at: {yolo_dir}")
        logger.info(f"  Train samples: {len(train_annotations)}")
        logger.info(f"  Val samples: {len(val_annotations)}")
    
    def _convert_to_yolo_format(
        self,
        annotations: List[Dict],
        yolo_dir: Path,
        split: str
    ) -> None:
        """Convert annotations to YOLO format."""
        category_map = {'bee_like': 0, 'non_bee_like': 1}
        
        for ann in annotations:
            src_image = Path(ann['frame_path'])
            dst_image = yolo_dir / 'images' / split / src_image.name
            
            if src_image.exists():
                import shutil
                shutil.copy2(src_image, dst_image)
                
                img = cv2.imread(str(src_image))
                img_h, img_w = img.shape[:2]
                
                label_file = yolo_dir / 'labels' / split / (src_image.stem + '.txt')
                
                with open(label_file, 'w') as f:
                    for bbox in ann['bboxes']:
                        class_id = category_map.get(bbox.get('class', 'bee_like'), 0)
                        
                        x_center = (bbox['x'] + bbox['width'] / 2) / img_w
                        y_center = (bbox['y'] + bbox['height'] / 2) / img_h
                        width = bbox['width'] / img_w
                        height = bbox['height'] / img_h
                        
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    def visualize_annotations(
        self,
        annotations: Optional[List[FrameAnnotation]] = None,
        output_subfolder: str = 'visualizations',
        max_frames: Optional[int] = None,
        draw_confidence: bool = True,
        draw_class_names: bool = True,
        bbox_thickness: int = 2,
        font_scale: float = 0.5,
        show_metadata: bool = False
    ) -> None:
        """Create annotated frames for manual visualization of annotations.
        
        This method draws bounding boxes, labels, and confidence scores on frames
        to help manually verify the quality of annotations.
        
        Args:
            annotations: List of FrameAnnotation objects. If None, loads from saved annotations.
            output_subfolder: Subfolder name within output_folder for visualizations
            max_frames: Maximum number of frames to visualize (None for all)
            draw_confidence: Whether to draw confidence scores on bboxes
            draw_class_names: Whether to draw class names on bboxes
            bbox_thickness: Thickness of bounding box lines
            font_scale: Scale of text font
            show_metadata: Whether to add metadata text to the top of the frame
            
        Example:
            >>> generator.visualize_annotations(max_frames=50)
            >>> # Or visualize specific annotations
            >>> generator.visualize_annotations(
            ...     annotations=my_annotations,
            ...     draw_confidence=True,
            ...     show_metadata=True
            ... )
        """
        logger.info("="*60)
        logger.info("Creating annotated frames for visualization")
        logger.info("="*60)
        
        # Load annotations if not provided
        if annotations is None:
            annotations_path = self.output_folder / 'annotations' / 'annotations.json'
            if not annotations_path.exists():
                logger.error("No annotations found. Run generate_dataset first.")
                return
            
            logger.info(f"Loading annotations from: {annotations_path}")
            with open(annotations_path, 'r') as f:
                annotations_data = json.load(f)
            
            # Convert to FrameAnnotation objects
            annotations = [
                FrameAnnotation(
                    video_path=ann['video_path'],
                    frame_number=ann['frame_number'],
                    frame_path=ann['frame_path'],
                    category=ann['category'],
                    bboxes=ann['bboxes'],
                    metadata=ann['metadata']
                )
                for ann in annotations_data
            ]
        
        # Limit number of frames if specified
        if max_frames is not None and len(annotations) > max_frames:
            logger.info(f"Limiting visualization to {max_frames} frames (out of {len(annotations)})")
            annotations = random.sample(annotations, max_frames)
        
        # Create output directory
        viz_dir = self.output_folder / output_subfolder
        viz_dir.mkdir(exist_ok=True)
        
        # Create category-specific subdirectories
        (viz_dir / 'bee_like').mkdir(exist_ok=True)
        (viz_dir / 'non_bee_like').mkdir(exist_ok=True)
        (viz_dir / 'all').mkdir(exist_ok=True)
        
        # Define colors for different categories (BGR format)
        colors = {
            'bee_like': (0, 255, 0),      # Green
            'non_bee_like': (0, 0, 255),  # Red
            'default': (255, 255, 0)       # Cyan
        }
        
        logger.info(f"Processing {len(annotations)} annotated frames...")
        
        successful_count = 0
        failed_count = 0
        
        for ann in tqdm(annotations, desc="Creating visualizations"):
            try:
                # Load the frame
                frame_path = Path(ann.frame_path)
                if not frame_path.exists():
                    logger.warning(f"Frame not found: {frame_path}")
                    failed_count += 1
                    continue
                
                frame = cv2.imread(str(frame_path))
                if frame is None:
                    logger.warning(f"Could not read frame: {frame_path}")
                    failed_count += 1
                    continue
                
                # Create a copy for annotation
                annotated_frame = frame.copy()
                
                # Add metadata overlay if requested
                if show_metadata:
                    overlay = annotated_frame.copy()
                    cv2.rectangle(overlay, (0, 0), (annotated_frame.shape[1], 80), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)
                    
                    video_name = Path(ann.video_path).name
                    text_lines = [
                        f"Video: {video_name}",
                        f"Frame: {ann.frame_number} | Category: {ann.category}",
                        f"Detections: {len(ann.bboxes)}"
                    ]
                    
                    y_offset = 20
                    for line in text_lines:
                        cv2.putText(
                            annotated_frame, line, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1
                        )
                        y_offset += 20
                
                # Draw bounding boxes
                for bbox in ann.bboxes:
                    x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    
                    # Get color based on class
                    bbox_class = bbox.get('class', ann.category)
                    color = colors.get(bbox_class, colors['default'])
                    
                    # Draw rectangle
                    cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, bbox_thickness)
                    
                    # Prepare label text
                    label_parts = []
                    if draw_class_names:
                        class_name = bbox.get('class_name', bbox_class)
                        label_parts.append(class_name)
                    
                    if draw_confidence and 'confidence' in bbox:
                        conf = bbox['confidence']
                        label_parts.append(f"{conf:.2f}")
                    
                    if label_parts:
                        label = ' '.join(label_parts)
                        
                        # Calculate label size for background
                        (text_w, text_h), baseline = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
                        )
                        
                        # Draw label background
                        cv2.rectangle(
                            annotated_frame,
                            (x, y - text_h - baseline - 5),
                            (x + text_w, y),
                            color,
                            -1
                        )
                        
                        # Draw label text
                        cv2.putText(
                            annotated_frame, label,
                            (x, y - baseline - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1
                        )
                
                # Generate output filename
                frame_basename = frame_path.stem
                video_name = Path(ann.video_path).stem
                output_filename = f"{video_name}_frame{ann.frame_number:06d}_{frame_basename}.jpg"
                
                # Save to appropriate directories
                # Save to category-specific folder
                category_path = viz_dir / ann.category / output_filename
                cv2.imwrite(str(category_path), annotated_frame)
                
                # Save to 'all' folder
                all_path = viz_dir / 'all' / output_filename
                cv2.imwrite(str(all_path), annotated_frame)
                
                successful_count += 1
                
            except Exception as e:
                logger.error(f"Error visualizing frame {ann.frame_path}: {e}")
                failed_count += 1
        
        logger.info("="*60)
        logger.info("Visualization complete!")
        logger.info(f"  Successfully visualized: {successful_count} frames")
        logger.info(f"  Failed: {failed_count} frames")
        logger.info(f"  Output directory: {viz_dir}")
        logger.info(f"  Organized by:")
        logger.info(f"    - {viz_dir / 'bee_like'} (bee-like insects)")
        logger.info(f"    - {viz_dir / 'non_bee_like'} (non-bee insects)")
        logger.info(f"    - {viz_dir / 'all'} (all frames)")
        logger.info("="*60)
        
        # Create a summary HTML file for easy browsing
        self._create_visualization_html(viz_dir, annotations, successful_count)
    
    def _create_visualization_html(
        self,
        viz_dir: Path,
        annotations: List[FrameAnnotation],
        successful_count: int
    ) -> None:
        """Create an HTML page for browsing visualizations."""
        html_path = viz_dir / 'index.html'
        
        # Organize annotations by category
        bee_like = [a for a in annotations if a.category == 'bee_like']
        non_bee_like = [a for a in annotations if a.category == 'non_bee_like']
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Annotation Visualizations</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
        }}
        .stats {{
            background: white;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .category {{
            margin-bottom: 30px;
        }}
        .category h2 {{
            color: #555;
            border-bottom: 2px solid #ddd;
            padding-bottom: 10px;
        }}
        .image-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 15px;
        }}
        .image-item {{
            background: white;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .image-item img {{
            width: 100%;
            height: auto;
            border-radius: 3px;
        }}
        .image-info {{
            margin-top: 8px;
            font-size: 12px;
            color: #666;
        }}
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 11px;
            font-weight: bold;
            margin-right: 5px;
        }}
        .badge-bee {{
            background-color: #d4edda;
            color: #155724;
        }}
        .badge-nonbee {{
            background-color: #f8d7da;
            color: #721c24;
        }}
    </style>
</head>
<body>
    <h1>Training Data Annotation Visualizations</h1>
    
    <div class="stats">
        <h3>Summary Statistics</h3>
        <p><strong>Total Frames Visualized:</strong> {successful_count}</p>
        <p><strong>Bee-like Frames:</strong> {len(bee_like)} ({len(bee_like)/successful_count*100:.1f}%)</p>
        <p><strong>Non-bee-like Frames:</strong> {len(non_bee_like)} ({len(non_bee_like)/successful_count*100:.1f}%)</p>
    </div>
    
    <div class="category">
        <h2><span class="badge badge-bee">Bee-like Insects</span> ({len(bee_like)} frames)</h2>
        <div class="image-grid">
"""
        
        # Add bee-like images
        for ann in bee_like[:50]:  # Limit to first 50 for HTML performance
            video_name = Path(ann.video_path).stem
            frame_basename = Path(ann.frame_path).stem
            img_filename = f"{video_name}_frame{ann.frame_number:06d}_{frame_basename}.jpg"
            img_path = f"bee_like/{img_filename}"
            
            html_content += f"""
            <div class="image-item">
                <img src="{img_path}" alt="Frame {ann.frame_number}">
                <div class="image-info">
                    <strong>Frame {ann.frame_number}</strong><br>
                    Video: {video_name}<br>
                    Detections: {len(ann.bboxes)}
                </div>
            </div>
"""
        
        html_content += """
        </div>
    </div>
    
    <div class="category">
        <h2><span class="badge badge-nonbee">Non-bee-like Insects</span> (""" + f"{len(non_bee_like)} frames)</h2>"
        html_content += """
        <div class="image-grid">
"""
        
        # Add non-bee-like images
        for ann in non_bee_like[:50]:  # Limit to first 50 for HTML performance
            video_name = Path(ann.video_path).stem
            frame_basename = Path(ann.frame_path).stem
            img_filename = f"{video_name}_frame{ann.frame_number:06d}_{frame_basename}.jpg"
            img_path = f"non_bee_like/{img_filename}"
            
            html_content += f"""
            <div class="image-item">
                <img src="{img_path}" alt="Frame {ann.frame_number}">
                <div class="image-info">
                    <strong>Frame {ann.frame_number}</strong><br>
                    Video: {video_name}<br>
                    Detections: {len(ann.bboxes)}
                </div>
            </div>
"""
        
        html_content += """
        </div>
    </div>
    
    <p style="margin-top: 40px; color: #999; text-align: center;">
        Generated by BeeMonitor Training Data Generator
    </p>
</body>
</html>
"""
        
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"  HTML viewer created: {html_path}")
        logger.info(f"  Open in browser to browse all visualizations")
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"TrainingDataGenerator(video_folder='{self.video_folder}', "
                f"output_folder='{self.output_folder}', "
                f"categories=['bee_like', 'non_bee_like'])")


