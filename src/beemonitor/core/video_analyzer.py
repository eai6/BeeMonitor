"""Main video analyzer class for bee monitoring.

This module provides the BeeMonitor class, which orchestrates the entire
bee detection and tracking pipeline.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any
import pandas as pd
import numpy as np
from ultralytics import YOLO
import os

from beemonitor.core.config import Config

from pathlib import Path
from typing import Dict, Optional
import cv2

import re


logger = logging.getLogger(__name__)
from concurrent.futures import ThreadPoolExecutor, as_completed


class AnalysisResults:
    """Container for video analysis results.
    
    This class holds all the outputs from video analysis and provides
    convenient methods for exporting and accessing results.
    
    Attributes:
        events: DataFrame containing entry/exit events with timestamps
        tracks: List of bee trajectories
        nests: Dictionary mapping nest IDs to bounding boxes
        video_path: Path to the analyzed video
        nest_detections: Raw nest detection DataFrame
        motion_data: Motion detection DataFrame
    
    Example:
        >>> results = monitor.analyze_video("video.mp4")
        >>> results.to_csv("output.csv")
        >>> print(f"Found {len(results.events)} events")
        >>> stats = results.get_statistics()
    """
    
    def __init__(
        self,
        events: pd.DataFrame,
        tracks: List, # trajectories
        nests: Dict,
        video_path: str,
        motion_data: Optional[pd.DataFrame] = None, 
        config: Optional[Config] = None
    ):
        """Initialize analysis results.
        
        Args:
            events: DataFrame with processed events
            tracks: List of bee trajectories
            nests: Dictionary of nest locations
            video_path: Path to analyzed video
            nest_detections: Raw nest detection data (optional)
            motion_data: Motion detection data (optional)
        """
        self.events = events
        self.tracks = tracks
        self.nests = nests
        self.video_path = video_path
        self.motion_data = motion_data
        self.config = config
    
    def to_csv(self, output_folder: str = "output", columns: Optional[List[str]] = None) -> None:
        """Export events and tracking results to CSV files.
        
        Saves two files:
        - *_events.csv: Entry/exit events
        - *_tracking_results.csv: Frame-by-frame tracking data
        
        Args:
            output_folder: Directory for output files
            columns: Columns to include in events CSV (default: all)
            
        Example:
            >>> results.to_csv("output")
            >>> results.to_csv("output", columns=["timestamp", "nest", "action"])
        """
        # Create output folder
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        # Base filename from video
        base_filename = self.video_path.replace(".mp4", "")
        base_filename = Path(base_filename).name
        
        # === SAVE EVENTS ===
        events_filename = str(Path(output_folder) / f"{base_filename}_events.csv")
        
        if columns is None:
            self.events.to_csv(events_filename, index=False)
        else:
            available_cols = [col for col in columns if col in self.events.columns]
            self.events[available_cols].to_csv(events_filename, index=False)
        
        logger.info(f"Saved {len(self.events)} events to {events_filename}")
        
        # === SAVE TRACKING RESULTS ===
        tracking_filename = str(Path(output_folder) / f"{base_filename}_tracking_results.csv")
        
        if self.tracks is not None and isinstance(self.tracks, pd.DataFrame) and not self.tracks.empty:
            # Tracks is already a flat DataFrame - just save it!
            self.tracks.to_csv(tracking_filename, index=False)
            logger.info(f"Saved {len(self.tracks)} tracking records to {tracking_filename}")
        else:
            logger.warning("No tracking results to save (tracks is not a valid DataFrame)")
    
    def save_video(self, output_folder: str = "output") -> None:
        """Save annotated video with tracking visualization.
        
        Args:
            filename: Output video file path
            output_folder: Directory for output (default: "output")
            
        Example:
            >>> results.save_video("annotated.mp4")
        """
        from beemonitor.output.video_synthesizer import VideoSynthesizer
        from beemonitor.core.config import Config
        
        synthesizer = VideoSynthesizer(self.config)
        
        output_path = synthesizer.synthesize(
            self.video_path,
            self.events,
            self.motion_data,
            self.nests,
            output_folder
        )
        
        logger.info(f"Saved annotated video to {output_path}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary format.
        
        Returns:
            Dictionary representation of analysis results
            
        Example:
            >>> results_dict = results.to_dict()
            >>> print(results_dict['events'])
        """
        return {
            "events": self.events,
            "tracks": self.tracks,
            "nests": self.nests,
            "video_path": self.video_path,
            "motion_data": self.motion_data
        }       
    
    
    def get_statistics(self) -> Dict[str, Any]:
        """Calculate summary statistics from the analysis.
        
        Returns:
            Dictionary containing analysis statistics
            
        Example:
            >>> stats = results.get_statistics()
            >>> print(f"Total entries: {stats['total_entries']}")
        """
        if self.events.empty:
            return {
                "total_events": 0,
                "total_entries": 0,
                "total_exits": 0,
                "active_nests": 0,
                "total_tracks": 0,
            }
        
        stats = {
            "total_events": len(self.events),
            "total_entries": len(self.events[self.events['action'] == 'Entry']),
            "total_exits": len(self.events[self.events['action'] == 'Exit']),
            "active_nests": len(self.events['nest'].unique()),
            "total_nests": len(self.nests.get('nests', {})),
            "total_tracks": len(self.tracks) if isinstance(self.tracks, list) else 0,
        }
        
        # Add per-nest statistics
        if 'nest' in self.events.columns:
            nest_counts = self.events.groupby('nest')['action'].value_counts().unstack(fill_value=0)
            stats['nest_activity'] = nest_counts.to_dict()
        
        return stats
    
    def __repr__(self) -> str:
        """String representation of results."""
        return (
            f"AnalysisResults(events={len(self.events)}, "
            f"tracks={len(self.tracks) if isinstance(self.tracks, list) else 0}, "
            f"nests={len(self.nests.get('nests', {}))})"
        )


class BeeMonitor:
    """Main interface for bee monitoring video analysis.
    
    This class provides a high-level API for analyzing bee hotel videos,
    including nest detection, motion tracking, and event processing.
    
    Attributes:
        nest_model: YOLO model for nest detection
        tracking_model: YOLO model for bee tracking
        classification_model: YOLO model for bee species classification
        config: Configuration object with all settings
        res_height: Video resolution height
        res_width: Video resolution width
    
    Example:
        >>> # Method 1: From configuration file
        >>> monitor = BeeMonitor.from_config("config/default_config.yaml")
        >>> results = monitor.analyze_video("video.mp4")
        
        >>> # Method 2: With explicit models
        >>> monitor = BeeMonitor(
        ...     nest_model_path="models/nest.pt",
        ...     tracking_model_path="models/tracking.pt"
        ... )
        >>> results = monitor.analyze_video("video.mp4")
    """
    
    def __init__(
        self,
        config: Optional[Config] = None
    ):
        """Initialize BeeMonitor with model paths and configuration.
        
        Args:
            config: Configuration object (default: None, uses default config)

            use config defualt to initialize settings if config is None
        
        Raises:
            FileNotFoundError: If model files don't exist
            ValueError: If resolution values are invalid
        """
        if config is None:
            config = Config.default()
        self.config = config
        self.res_height = config.video.res_height
        self.res_width = config.video.res_width
        self.nest_model = YOLO(config.models.nest_detection)
        self.tracking_model = YOLO(config.models.tracking)
        # self.classification_model = YOLO(config.models.bee_classification)

        
        logger.info("BeeMonitor initialized successfully")
    
    @classmethod
    def from_config(cls, config_path: str) -> "BeeMonitor":
        """Create BeeMonitor from configuration file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Initialized BeeMonitor instance
            
        Example:
            >>> monitor = BeeMonitor()
            >>> results = monitor.analyze_video("video.mp4")
        """
        config = Config.from_yaml(config_path)
        
        return cls(
            nest_model_path=config.models.nest_detection,
            tracking_model_path=config.models.tracking,
            classification_model_path=config.models.bee_classification,
            res_height=config.video.height,
            res_width=config.video.width,
            config=config
        )
    
    def visualize_motion(self, motion_data: pd.DataFrame) -> None:
        """Visualize motion tracking data for debugging.
        
        Args:
            motion_data: DataFrame with motion tracking results
        """

        # Simple Test to Isolate Event Detection Issue

        res_width = self.res_width
        res_height = self.res_height
        config = self.config

        events = motion_data.get('events', None)

        print("Testing motion tracking output...")

        # Test 1: Check if motion_data is empty
        if len(motion_data) == 0:
            print("❌ PROBLEM: motion_data is EMPTY - no tracks detected!")
            print("   Check:")
            print("   - Is there activity in the video?")
            print("   - Are blob filters too strict?")
            print("   - Check visualization video for tracks")
            exit()

        print(f"✓ Motion data has {len(motion_data)} periods")

        # Test 2: Check if tracks exist
        first_period = motion_data.iloc[0]
        num_tracks = len(first_period['tracks'])
        print(f"✓ First period has {num_tracks} tracks")

        if num_tracks == 0:
            print("❌ PROBLEM: Tracks list is EMPTY!")
            exit()

        # Test 3: Check track coordinates
        first_track = first_period['tracks'][0]
        track_id, centroids, bboxes, frame_nums = first_track

        print(f"\n✓ First track:")
        print(f"  - ID: {track_id}")
        print(f"  - Length: {len(centroids)} positions")
        print(f"  - First centroid: {centroids[0]}")
        print(f"  - Frame range: {frame_nums[0]} - {frame_nums[-1]}")

        # Test 4: Check coordinate system
        x, y = centroids[0]
        print(f"\n✓ Coordinate check:")
        print(f"  - Frame size: {res_width}x{res_height}")
        print(f"  - Track at: ({x:.1f}, {y:.1f})")

        if config.hotel_box:
            hx1, hy1, hx2, hy2 = config.hotel_box.hotel_box_cords
            print(f"  - Hotel box: ({hx1:.0f}, {hy1:.0f}) to ({hx2:.0f}, {hy2:.0f})")
            
            # Check if track is near hotel
            in_hotel = hx1 <= x <= hx2 and hy1 <= y <= hy2
            print(f"  - Track in hotel: {in_hotel}")
            
            if not in_hotel:
                print(f"\n❌ PROBLEM: Track is OUTSIDE hotel box!")
                print(f"   This will prevent event detection!")

        # Test 5: Check if event processor was called
        print(f"\n✓ Event data:")
        if 'events' not in globals():
            print("❌ PROBLEM: 'events' variable doesn't exist!")
            print("   Was event processor called?")
        elif events is None:
            print("❌ PROBLEM: events is None!")
        elif len(events) == 0:
            print("❌ PROBLEM: Events DataFrame is EMPTY!")
            print("   Tracks exist but no events detected.")
            print("   Possible causes:")
            print("   - Tracks not near nests")
            print("   - Trajectory too short")
            print("   - Event detection parameters too strict")
        else:
            print(f"✓ Found {len(events)} events!")
            print(events.head())
    
    def analyze_video(
        self,
        video_path: str,
        nest_video_path: Optional[str] = None,
        output_folder: Optional[str] = None,
        visualize: Optional[bool] = None,
        detection_mode: str = 'fgbg'
    ) -> AnalysisResults:
        """Analyze a video to detect and track bee activity.
        
        This is the main method that orchestrates the entire analysis pipeline:
        1. Detect nests in the video
        2. Detect motion and track bees
        3. Process tracks to identify entry/exit events
        
        Args:
            video_path: Path to input video file
            output_folder: Directory for output files (default: from config)
            visualize: Whether to save visualization videos (default: False)
            
        Returns:
            AnalysisResults object containing all analysis outputs
            
        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If video cannot be opened
            
        Example:
            >>> monitor = BeeMonitor()
            >>> results = monitor.analyze_video("video.mp4")
            >>> results.to_csv("output.csv")
            >>> print(f"Found {len(results.events)} events")
        """
        # Validate video path
        video_file = Path(video_path)
        if not video_file.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
         # NEW: Use config settings if not explicitly provided
        if visualize is None:
            visualize = self.config.output.save_tracking_visualizations
        
        # Set output folder
        if output_folder is None:
            output_folder = self.config.output.base_folder
        
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting analysis of {video_path}")
        
        # Step 1: Detect nests
        if nest_video_path is None:
            nest_video_path = video_path
        logger.info("Step 1/3: Detecting nests...")
        nests = self.get_nest_detections(nest_video_path)

        if nests is None:
            logger.warning("No nests detected, skipping video analysis")
            return None
        
        # Step 3: Detect motion and track bees
        logger.info("Step 2/3: Detecting motion and tracking bees...")
        flat_tracking_df, grouped_tracking_df = self.get_motion_tracking(
            video_path,
            nests['hotel'],
            output_folder,
            visualize=visualize,
            detection_mode=detection_mode
        )

        # visualize
        #self.visualize_motion(motion_data)

        # Validate motion_data
        if flat_tracking_df is None or flat_tracking_df.empty:
            logger.warning("No motion tracking data returned, skipping video analysis")
            return None
        
        # Step 4: Process tracks to get events (uses grouped format)
        logger.info("Step 3/3: Processing tracks to identify events...")
        events = self.process_motion_tracking(grouped_tracking_df, nests)

        # Synthesize CSV
        events = self.synthesize_csv(
            events,
            video_path
        )
        
        # Create results object (use flat DataFrame for saving!)
        results = AnalysisResults(
            events=events,
            tracks=flat_tracking_df,  # Flat DataFrame for CSV export
            nests=nests,
            video_path=video_path,
            motion_data=grouped_tracking_df,  # Grouped for compatibility
            config=self.config
        )

        if visualize: # save synthesized video and synthesized csv only if visualize is true
            # Synthesize CSV
            results.to_csv(output_folder=output_folder)

            results.save_video(output_folder=output_folder)
        
        logger.info(f"Analysis complete: {len(events)} events detected")

    
        
        return results
    
    def analyze_video_with_relative_nests(
        self,
        video_path: str,
        video_files: List[str],
        output_folder: Optional[str] = None,
        visualize: Optional[bool] = None
    )-> AnalysisResults:
        """Analyze a video using nest detections from adjacent videos.
        
        Args:
            video_path: Path to input video file
            video_files: List of all video files in the directory
            output_folder: Directory for output files (default: from config)
            visualize: Whether to save visualization videos (default: False)
        Returns:
            AnalysisResults object containing all analysis outputs
        """ 
        # Get previous and next videos
        prev_video = self._get_prev_video(video_path, video_files)
        next_video = self._get_next_video(video_path, video_files)
        
        # Try previous video first
        if prev_video is not None:
            logger.info(f"Using nest detections from previous video: {prev_video}")
            nests = self.get_nest_detections(prev_video)
            if nests is not None:
                return self.analyze_video(
                    video_path,
                    prev_video,
                    output_folder=output_folder,
                    visualize=visualize
                )
        
        # Fallback to next video
        if next_video is not None:
            logger.info(f"Using nest detections from next video: {next_video}")
            nests = self.get_nest_detections(next_video)
            if nests is not None:
                return self.analyze_video(
                    video_path,
                    next_video,
                    output_folder=output_folder,
                    visualize=visualize
                )
        
        logger.warning("No adjacent videos with valid nest detections found")
        return None
    

    def analyze_videos_in_folder(
        self,
        video_folder: str,
        output_folder: Optional[str] = None,
        visualize: bool = False,
        max_workers: int = 4  # Number of parallel videos
    ) -> Dict[str, AnalysisResults]:
        """Process multiple videos in parallel."""

        if output_folder is None:
            output_folder = self.config.output.base_folder

        video_files = [str(Path(video_folder) / vf) for vf in os.listdir(video_folder) if vf.endswith(('.mp4', '.avi', '.mov'))]
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_video = {
                executor.submit(
                    self.analyze_video,
                    video_path,
                    output_folder=output_folder,
                    visualize=visualize
                ): video_path
                for video_path in video_files
            }
            
            for future in as_completed(future_to_video):
                video_path = future_to_video[future]
                try:
                    results[video_path] = future.result()
                    logger.info(f"Completed: {video_path}")
                except Exception as e:
                    logger.error(f"Failed {video_path}: {e}")
                    results[video_path] = None
        
        return results
    

    def get_nest_detections(self, video_path: str) -> pd.DataFrame:
        """Detect nests using improved robust detector."""
        from beemonitor.detection.nest_detector import NestDetector
        
        logger.info("Starting nest detection with improved detector")
        
        # Initialize detector
        detector = NestDetector(
            model=self.nest_model,
            config=self.config
        )
            
        # Detect and assign IDs
        nests = detector.get_nests_and_hotel_detections(
            video_path=video_path,
        )
    
        return nests

    def _get_next_video(
            self,
            video_path: str,
            video_files: List[str],
    ) -> str:
        """Get the next consecutive video file after the current video

        1. Get the timestamp from the filename (i.e.,"2024-05-11_10_50_00")
        2. Extract the date from the timestamp
        3. Filter files in video_files to those with the same date as the timestamp from the current video
        4. Sort the filtered files
        6. Find the index of the current video file
        7. Return the next file if it exists, else return None
        """

        def extract_ts(fp: str) -> Optional[str]:
            m = re.search(r"\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2}", Path(fp).stem)
            return m.group(0) if m else None

        current_path = Path(video_path)
        current_ts = extract_ts(current_path.name)

        # If no timestamp found, fall back to lexicographic ordering of all files
        if current_ts is None:
            logger.debug("No timestamp found in current filename; falling back to filename ordering")
            sorted_files = sorted(video_files)
        else:
            date_part = current_ts.split("_")[0]
            # Filter files that contain the same date in their timestamp
            same_date_files = []
            for f in video_files:
                ts = extract_ts(f)
                if ts and ts.startswith(date_part):
                    same_date_files.append(f)
            if not same_date_files:
                logger.debug("No same-date files found for %s", video_path)
                return None
            # Sort by full timestamp (lexicographic sort works because format is sortable)
            sorted_files = sorted(same_date_files, key=lambda p: extract_ts(p) or "")

        # Normalize names for matching
        sorted_paths = [str(Path(p)) for p in sorted_files]
        try:
            idx = next(i for i, p in enumerate(sorted_paths) if Path(p).name == current_path.name)
        except StopIteration:
            logger.debug("Current video %s not found in filtered list", video_path)
            return None

        next_idx = idx + 1
        if next_idx < len(sorted_paths):
            return sorted_paths[next_idx]
        return None

    def _get_prev_video(
            self,
            video_path: str,
            video_files: List[str],
    ) -> str:
        """Get the previous consecutive video file before the current video

        1. Get the timestamp from the filename (i.e.,"2024-05-11_10_50_00")
        2. Extract the date from the timestamp
        3. Filter files in video_files to those with the same date as the timestamp from the current video
        4. Sort the filtered files
        6. Find the index of the current video file
        7. Return the prev file if it exists, else return None
        """

        def extract_ts(fp: str) -> Optional[str]:
            m = re.search(r"\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2}", Path(fp).stem)
            return m.group(0) if m else None

        current_path = Path(video_path)
        current_ts = extract_ts(current_path.name)

        # If no timestamp found, fall back to lexicographic ordering of all files
        if current_ts is None:
            logger.debug("No timestamp found in current filename; falling back to filename ordering")
            sorted_files = sorted(video_files)
        else:
            date_part = current_ts.split("_")[0]
            # Filter files that contain the same date in their timestamp
            same_date_files = []
            for f in video_files:
                ts = extract_ts(f)
                if ts and ts.startswith(date_part):
                    same_date_files.append(f)
            if not same_date_files:
                logger.debug("No same-date files found for %s", video_path)
                return None
            # Sort by full timestamp (lexicographic sort works because format is sortable)
            sorted_files = sorted(same_date_files, key=lambda p: extract_ts(p) or "")

        # Normalize names for matching
        sorted_paths = [str(Path(p)) for p in sorted_files]
        try:
            idx = next(i for i, p in enumerate(sorted_paths) if Path(p).name == current_path.name)
        except StopIteration:
            logger.debug("Current video %s not found in filtered list", video_path)
            return None

        prev_idx = idx - 1
        if prev_idx >= 0:
            return sorted_paths[prev_idx]
        return None

        
    def get_motion_tracking(
        self,
        video_path: str,
        hotel_roi: Tuple[float, float, float, float],
        output_folder: str,
        visualize: bool = False,
        detection_mode: str = 'fgbg'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Detect motion and track bees.
        
        CNN noise filter + learned solidity are AUTOMATICALLY applied 
        to all blob-based modes (fgbg, fgbg_yolo).
        
        Args:
            video_path: Path to video file
            hotel_roi: Hotel region of interest (x1, y1, x2, y2)
            output_folder: Directory for output files
            visualize: Whether to save visualization
            detection_mode: 'fgbg', 'fgbg_yolo', or 'yolo_only'
            
        Returns:
            Tuple of (flat_df, grouped_df)
        """
        from beemonitor.tracking.bee_tracking import BeeTracking, DetectionMode
        from beemonitor.tracking.mot.bee_tracker import BeeTracker
        from beemonitor.detection import BlobDetector, YOLODetector
        
        # Map string detection mode to enum
        mode_map = {
            'fgbg': DetectionMode.FGBG_ONLY,
            'fgbg_yolo': DetectionMode.FGBG_YOLO,
            'yolo': DetectionMode.YOLO_ONLY,
            'yolo_only': DetectionMode.YOLO_ONLY,
        }
        detection_mode_enum = mode_map.get(detection_mode, DetectionMode.FGBG_ONLY)
        
        logger.info("\n" + "="*70)
        logger.info(f"DETECTION MODE: {detection_mode.upper()}")
        logger.info(f"  CNN noise filter: {'AUTO' if detection_mode in ['fgbg', 'fgbg_yolo'] else 'N/A'}")
        logger.info(f"  Learned solidity: {'AUTO' if detection_mode in ['fgbg', 'fgbg_yolo'] else 'N/A'}")
        logger.info("="*70)
        
        # =====================================================================
        # =====================================================================
        # PHASE 1: Initialize with Researched Optimal Defaults
        # =====================================================================
        print("\nPhase 1: Background Initialization (Researched Optimal Defaults)")
        print("-" * 70)
        
        # Use RESEARCHED OPTIMAL values from ablation study
        # These work for 95% of videos immediately
        RESEARCHED_MIN_AREA = 30.0        # Conservative - catches most bees
        RESEARCHED_MIN_SOLIDITY = 0.56    # 80% of typical 0.7 (proven F1=53.0%)
        
        print(f"Using researched optimal thresholds:")
        print(f"  min_area: {RESEARCHED_MIN_AREA} (from ablation study)")
        print(f"  min_solidity: {RESEARCHED_MIN_SOLIDITY} (80% scaling, proven optimal)")
        print(f"  These will adapt during tracking as bees are confirmed")
        
        blob_detector = BlobDetector(
            min_area=RESEARCHED_MIN_AREA,
            min_solidity=RESEARCHED_MIN_SOLIDITY
        )
        
        try:
            blob_detector.initialize_from_video(
                video_path=video_path,
                num_frames=100,
                start_frame=0
            )
            print("✓ Background model initialized (100 frames)")
            print("✓ Starting with researched optimal thresholds")
            print("  (will adapt online during tracking if YOLO enabled)")
        except Exception as e:
            print(f"⚠ Background initialization failed: {e}")
            print("Continuing with default background model")
        
        # =====================================================================
        # PHASE 2: Full Tracking
        # =====================================================================
        print("\nPhase 2: Full Video Analysis")
        print("-" * 70)
        print(f"Detection mode: {detection_mode.upper()} ({detection_mode_enum.value})")
        print(f"  Blob detection: {'YES' if detection_mode in ['fgbg', 'fgbg_yolo'] else 'NO'}")
        print(f"  YOLO detection: {'YES' if detection_mode in ['fgbg_yolo', 'yolo_only'] else 'NO'}")
        print(f"  CNN filtering: {'WILL TRY TO ENABLE' if detection_mode in ['fgbg', 'fgbg_yolo'] else 'NOT NEEDED'}")
        print("")
        
        tracking_class_names = []
        if hasattr(self.config.tracking, 'label_map'):
            for class_id in self.config.tracking.tracking_classes:
                class_name = self.config.tracking.label_map.get(class_id, f'class_{class_id}')
                tracking_class_names.append(class_name)
        else:
            tracking_class_names = [str(cid) for cid in self.config.tracking.tracking_classes]
        
        yolo_detector = YOLODetector(
            model=self.tracking_model,
            conf_threshold=0.25,
            tracking_classes= None #tracking_class_names
        )
        
        # Initialize MOT algorithm
        # Enable YOLO confirmation if YOLO is in the detection mode
        use_yolo_confirmation = detection_mode_enum in [
            DetectionMode.FGBG_YOLO, 
            DetectionMode.YOLO_ONLY,
            DetectionMode.SIFT_YOLO,
            DetectionMode.FGBG_SIFT_YOLO
        ]
        
        mot_algorithm = BeeTracker(
            config=self.config,
            tracking_classes=tracking_class_names,
            require_yolo_confirmation=use_yolo_confirmation,  # Enable for YOLO modes
        )
        
        if use_yolo_confirmation:
            print(f"  ✓ YOLO confirmation ENABLED (new tracks require YOLO verification)")
        else:
            print(f"  ℹ YOLO confirmation DISABLED (legacy mode, any detection can create track)")
        
        # =====================================================================
        # Enable noise filter for ALL blob-based modes (AUTOMATIC)
        # =====================================================================
        print("\n" + "="*70)
        print("CNN NOISE FILTER INITIALIZATION")
        print("="*70)
        
        use_noise_filter = detection_mode_enum in [DetectionMode.FGBG_ONLY, DetectionMode.FGBG_YOLO]
        noise_filter_model = None

        logger.info(f"Detection mode: {detection_mode_enum.value}")
        logger.info(f"Should enable CNN filter: {use_noise_filter}")
        
        if use_noise_filter:
            logger.info("Attempting to load CNN noise filter...")
            try:
                from beemonitor.detection.noise_filter import BeeNoiseFilter
                logger.info("  ✓ BeeNoiseFilter class imported successfully")
                
                # Check config
                model_path = self.config.models.blob_noise_classifier if hasattr(self.config.models, 'blob_noise_classifier') else None
                logger.info(f"  Config path: {model_path if model_path else 'NOT CONFIGURED'}")
                
                if model_path and Path(model_path).exists():
                    print(f"  ✓ Model file exists: {model_path}")
                    print(f"  Loading CNN model...")
                    noise_filter_model = BeeNoiseFilter(
                        model_path=model_path,
                        noise_threshold=0.9  # High threshold = strict filtering
                    )
                    print(f"  ✓✓✓ CNN NOISE FILTER ENABLED (AUTOMATIC) ✓✓✓")
                    print(f"  Model: {model_path}")
                    print(f"  Threshold: 0.9 (strict)")
                    print(f"  Expected: ~66% reduction in false positives")
                else:
                    if model_path:
                        print(f"  ✗ Model file NOT FOUND: {model_path}")
                    else:
                        print(f"  ✗ Model path NOT CONFIGURED in config.models.blob_noise_classifier")
                    print("  CNN filter will NOT be used")
                    print("  Continuing with morphological filtering only")
                    use_noise_filter = False
            except Exception as e:
                print(f"  ✗ CNN filter loading FAILED: {e}")
                import traceback
                print(traceback.format_exc())
                print("  Continuing with morphological filtering only")
                use_noise_filter = False
        else:
            print(f"CNN filter not needed for mode: {detection_mode}")
        
        print("="*70 + "\n")
        
        # Initialize BeeTracking system
        print("\n" + "="*70)
        print("BEETRACKING SYSTEM INITIALIZATION")
        print("="*70)
        print(f"Detection mode: {detection_mode_enum.value}")
        print(f"Use noise filter: {use_noise_filter}")
        print(f"Noise filter model: {'LOADED' if noise_filter_model else 'NONE'}")
        print(f"Online learning: ENABLED (thresholds adapt during tracking)")
        
        tracker = BeeTracking(
            mot_algorithm=mot_algorithm,
            yolo_model=self.tracking_model,
            detection_mode=detection_mode_enum,  # Use enum from mode mapping
            use_noise_filter=use_noise_filter,   # Auto-set based on mode
            noise_filter_model=noise_filter_model,
            config=self.config,
            enable_online_learning=True  # Enable adaptive threshold learning
        )
        
        # *** Use our initialized detectors ***
        tracker.blob_detector = blob_detector
        tracker.yolo_detector = yolo_detector
        
        logger.info(f"✓ BeeTracking initialized with:")
        logger.info(f"  Blob detector: {'SET' if tracker.blob_detector else 'NONE'}")
        logger.info(f"  YOLO detector: {'SET' if tracker.yolo_detector else 'NONE'}")
        logger.info(f"  Noise filter: {'SET' if tracker.noise_filter else 'NONE'}")
        logger.info(f"  Detection mode: {tracker.detection_mode}")
        logger.info("="*70 + "\n")
        
        logger.info("✓ Tracking system initialized")
        logger.info("="*70 + "\n")
        
        # Process video with progress logging
        print("\n" + "="*70)
        print("PROCESSING VIDEO")
        print("="*70)
        print(f"Video: {video_path}")
        print(f"ROI: {hotel_roi}")
        print(f"Detection mode: {tracker.detection_mode.value}")
        print(f"Starting video processing...")
        print("="*70 + "\n")
        
        import time
        start_time = time.time()
        
        tracking_df = tracker.process_video(
            video_path=video_path,
            roi=hotel_roi
        )
        
        elapsed = time.time() - start_time
        
        print(f"\n{'='*70}")
        print(f"VIDEO PROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"Processing time: {elapsed:.1f} seconds")
        print(f"Total detections: {len(tracking_df)}")
        print(f"Unique tracks: {tracking_df['track_id'].nunique() if not tracking_df.empty else 0}")
        print(f"{'='*70}\n")
        
        # Convert to grouped format
        grouped_df = self._convert_tracking_to_grouped_format(tracking_df)
        
        logger.info(f"\n✓ Tracking complete:")
        logger.info(f"  Total detections: {len(tracking_df)}")
        logger.info(f"  Unique tracks: {tracking_df['track_id'].nunique() if not tracking_df.empty else 0}")
        
        # Print online learning statistics
        if hasattr(tracker, 'blob_detector') and tracker.blob_detector:
            if hasattr(tracker.blob_detector, 'get_learning_stats'):
                stats = tracker.blob_detector.get_learning_stats()
                print(f"\n{'='*70}")
                print("ONLINE LEARNING SUMMARY")
                print(f"{'='*70}")
                print(f"Confirmed bees collected: {stats['samples_collected']}")
                print(f"Threshold updates: {stats['updates_count']}")
                print(f"\nFinal thresholds:")
                print(f"  min_area: {stats['current_min_area']:.1f} "
                      f"(started: {stats['researched_min_area']:.1f}, "
                      f"adapted: {stats['area_adaptation']:+.1f}%)")
                print(f"  min_solidity: {stats['current_min_solidity']:.3f} "
                      f"(started: {stats['researched_min_solidity']:.3f}, "
                      f"adapted: {stats['solidity_adaptation']:+.1f}%)")
                print(f"{'='*70}\n")
        
        # Print YOLO confirmation statistics (if enabled)
        if hasattr(tracker, 'mot') and hasattr(tracker.mot, 'get_statistics'):
            mot_stats = tracker.mot.get_statistics()
            
            # Only show if YOLO confirmation was enabled
            if mot_stats.get('yolo_confirmed_tracks', 0) > 0 or mot_stats.get('pending_aged_out', 0) > 0:
                print(f"\n{'='*70}")
                print("YOLO CONFIRMATION STATISTICS")
                print(f"{'='*70}")
                print(f"Total tracks created: {mot_stats['total_tracks_created']}")
                print(f"YOLO-confirmed tracks: {mot_stats['yolo_confirmed_tracks']} "
                      f"({mot_stats['confirmation_rate']*100:.1f}% confirmation rate)")
                if mot_stats['pending_confirmed'] > 0:
                    print(f"  - Pending confirmed: {mot_stats['pending_confirmed']}")
                print(f"Pending aged out: {mot_stats['pending_aged_out']} (noise filtered!)")
                print(f"\n✓ Track quality: {mot_stats['confirmation_rate']*100:.1f}% of tracks were YOLO-verified")
                print(f"✓ Noise reduction: {mot_stats['pending_aged_out']} false positives prevented")
                print(f"{'='*70}\n")
        
        return tracking_df, grouped_df

    
    def _convert_tracking_to_grouped_format(self, tracking_df: pd.DataFrame) -> pd.DataFrame:
        """Convert flat tracking DataFrame to grouped format expected by event_processor.
        
        Args:
            tracking_df: DataFrame with columns: frame, track_id, x1, y1, x2, y2, species, confidence
            
        Returns:
            DataFrame with columns: frame_number, tracks, detections
            - frame_number: tuple of (min_frame, max_frame) for the period
            - tracks: list of tuples (track_id, centroids, bboxes, frame_numbers)
            - detections: dict mapping frame_num to {'boxes': [...], 'label': [...]}
        """
        if tracking_df.empty:
            return pd.DataFrame(columns=['frame_number', 'tracks', 'detections'])
        
        # Group into periods based on gaps
        gap_threshold = int(self.config.tracking.max_age * 1.1)
        periods = self._split_into_periods(tracking_df, gap_threshold)
        
        result_rows = []
        for period_df in periods:
            track_groups = {}
            
            # Group by track_id
            for track_id in period_df['track_id'].unique():
                track_df = period_df[period_df['track_id'] == track_id].sort_values('frame')
                
                # Split track by gaps
                segments = self._split_track_by_gaps(track_df, gap_threshold=self.config.tracking.max_age)
                
                for seg_idx, seg_df in enumerate(segments):
                    unique_id = f"{track_id}_{seg_idx}" if len(segments) > 1 else track_id
                    
                    # Extract centroids, bboxes, frame_numbers
                    centroids = [((row['x1'] + row['x2']) / 2, (row['y1'] + row['y2']) / 2)
                                for _, row in seg_df.iterrows()]
                    bboxes = [(row['x1'], row['y1'], row['x2'], row['y2'])
                             for _, row in seg_df.iterrows()]
                    frame_numbers = seg_df['frame'].tolist()
                    
                    # Extract species information (for event processor)
                    if 'species' in seg_df.columns:
                        # Get most common species for this track
                        species_list = seg_df['species'].tolist()
                        species_votes = {}
                        for species in species_list:
                            species_votes[species] = species_votes.get(species, 0) + 1
                        
                        # Most voted species
                        most_common_species = max(species_votes, key=species_votes.get)
                    else:
                        most_common_species = 'unknown'
                        species_votes = {'unknown': len(frame_numbers)}
                    
                    # Only include tracks that meet minimum length
                    if len(frame_numbers) >= self.config.tracking.min_track_length:
                        # Event processor expects: (track_id, centroids, bboxes, frame_numbers, species, species_votes)
                        track_groups[unique_id] = (
                            unique_id, 
                            centroids, 
                            bboxes, 
                            frame_numbers,
                            most_common_species,  # Species as string (not class ID)
                            species_votes
                        )
            
            if not track_groups:
                continue
            
            all_tracks = list(track_groups.values())
            min_frame = int(period_df['frame'].min())
            max_frame = int(period_df['frame'].max())
            
            # Create frame detections dictionary
            frame_detections = {}
            for frame_num in period_df['frame'].unique():
                frame_df = period_df[period_df['frame'] == frame_num]
                frame_detections[int(frame_num)] = {
                    'boxes': [(row['x1'], row['y1'], row['x2'], row['y2'])
                             for _, row in frame_df.iterrows()],
                    'label': frame_df['species'].tolist() if 'species' in frame_df.columns else ['bee'] * len(frame_df)
                }
            
            result_rows.append({
                'frame_number': (min_frame, max_frame),
                'tracks': all_tracks,
                'detections': frame_detections
            })
        
        return pd.DataFrame(result_rows) if result_rows else pd.DataFrame(columns=['frame_number', 'tracks', 'detections'])
    
    def _split_into_periods(self, df: pd.DataFrame, gap_threshold: int = 100) -> List[pd.DataFrame]:
        """Split detections into activity periods based on frame gaps.
        
        Args:
            df: Tracking DataFrame
            gap_threshold: Maximum gap between frames to consider same period
            
        Returns:
            List of DataFrames, one per period
        """
        df = df.sort_values('frame')
        frames = df['frame'].tolist()
        
        if not frames:
            return []
        
        periods = []
        current_start = 0
        
        for i in range(len(frames) - 1):
            gap = frames[i + 1] - frames[i]
            if gap > gap_threshold:
                periods.append(df.iloc[current_start:i+1].copy())
                current_start = i + 1
        
        if current_start < len(df):
            periods.append(df.iloc[current_start:].copy())
        
        return periods if periods else [df]
    
    def _split_track_by_gaps(self, track_df: pd.DataFrame, gap_threshold: int = 30) -> List[pd.DataFrame]:
        """Split track into segments by gaps.
        
        Args:
            track_df: DataFrame for a single track
            gap_threshold: Maximum gap between frames to consider same segment
            
        Returns:
            List of DataFrames, one per segment
        """
        frames = track_df['frame'].tolist()
        
        if not frames:
            return []
        
        segments = []
        current_start = 0
        
        for i in range(len(frames) - 1):
            gap = frames[i + 1] - frames[i]
            if gap > gap_threshold:
                segments.append(track_df.iloc[current_start:i+1].copy())
                current_start = i + 1
        
        if current_start < len(track_df):
            segments.append(track_df.iloc[current_start:].copy())
        
        return segments if segments else [track_df]
    
    def process_motion_tracking(
        self,
        motion_data: pd.DataFrame,
        nests: Dict
    ) -> pd.DataFrame:
        """Process tracking data to identify entry/exit events.
        
        Args:
            motion_data: DataFrame from get_motion_tracking
            nests: Dictionary from process_nest_detection
            
        Returns:
            DataFrame with events (timestamp, nest, action)
        """
        # Validate input
        if motion_data is None:
            logger.warning("motion_data is None, returning empty events DataFrame")
            return pd.DataFrame(columns=['timestamp', 'nest_id', 'action', 'track_id', 'species'])
        
        # Import here to avoid circular imports
        from beemonitor.processing.event_processor import EventProcessor
        
        processor = EventProcessor(config=self.config)
        
        return processor.process_tracks(
            motion_data=motion_data,
            nests=nests
        )
    
    def synthesize_csv(
        self,
        events: pd.DataFrame,
        video_path: str
    ) -> pd.DataFrame:
        """Generate CSV with timestamps from events.
        
        Args:
            events: DataFrame with events
            video_path: Path to video file (for timestamp calculation)
            
        Returns:
            DataFrame with timestamps added
        """
        # Import here to avoid circular imports
        from beemonitor.output.csv_generator import CSVGenerator
        
        generator = CSVGenerator(config=self.config)
        
        return generator.generate_csv(
            events=events,
            video_path=video_path
        )
    
    def synthesize_video(
        self,
        video_path: str,
        events: pd.DataFrame,
        motion_data: pd.DataFrame,
        nests: Dict,
        output_folder: str
    ) -> str:
        """Generate annotated video with tracking visualization.
        
        Args:
            video_path: Path to input video
            events: DataFrame with events
            motion_data: DataFrame with tracking data
            nests: Dictionary with nest locations
            output_folder: Directory for output
            
        Returns:
            Path to generated video file
        """
        # Import here to avoid circular imports
        from beemonitor.output.video_synthesizer import VideoSynthesizer
        synthesizer = VideoSynthesizer(config=self.config)
        
        return synthesizer.synthesize(
            video_path=video_path,
            events=events,
            motion_data=motion_data,
            nests=nests,
            output_folder=output_folder,
            res_height=self.res_height,
            res_width=self.res_width
        )
    
    def __repr__(self) -> str:
        """String representation of BeeMonitor."""
        return (
            f"BeeMonitor(resolution={self.res_width}x{self.res_height}, "
            f"config={self.config is not None})"
        )