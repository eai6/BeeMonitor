








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
        detection_mode: str = 'yolo_only'
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

        # ALWAYS save CSV and video results (needed for analysis)
        results.to_csv(output_folder=output_folder)
        logger.info(f"✓ Saved CSV results to {output_folder}")

        results.save_video(output_folder=output_folder)
        logger.info(f"✓ Saved visualization video to {output_folder}")
        
        # Only save visualization video if requested
        # if visualize:
        #     results.save_video(output_folder=output_folder)
        #     logger.info(f"✓ Saved visualization video to {output_folder}")
        
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
        detection_mode: str = 'yolo_only'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Detect motion and track bees using specified detection mode.
        
        Args:
            video_path: Path to video file
            hotel_roi: Hotel region of interest (x1, y1, x2, y2)
            output_folder: Directory for output files
            visualize: Whether to save visualization
            detection_mode: Detection mode to use ('yolo_only', 'fgbg_yolo', 'fgbg_only')
            
        Returns:
            Tuple of (flat_df, grouped_df)
        """
        from beemonitor.tracking.bee_tracking import BeeTracking, DetectionMode
        from beemonitor.tracking.mot.bee_tracker import BeeTracker
        from beemonitor.detection import BlobDetector, YOLODetector
        
        # Map string detection mode to enum
        mode_map = {
            'fgbg': DetectionMode.FGBG_ONLY,
            'fgbg_only': DetectionMode.FGBG_ONLY,
            'sift': DetectionMode.SIFT_ONLY,
            'sift_only': DetectionMode.SIFT_ONLY,
            'yolo': DetectionMode.YOLO_ONLY,
            'yolo_only': DetectionMode.YOLO_ONLY,
            'fgbg_sift': DetectionMode.FGBG_SIFT,
            'fgbg_yolo': DetectionMode.FGBG_YOLO,
            'sift_yolo': DetectionMode.SIFT_YOLO,
            'fgbg_sift_yolo': DetectionMode.FGBG_SIFT_YOLO
        }
        detection_mode_enum = mode_map.get(detection_mode, DetectionMode.FGBG_SIFT_YOLO)  # Default to comprehensive
        
        logger.info("\n" + "="*70)
        logger.info(f"DETECTION MODE: {detection_mode.upper()} ({detection_mode_enum.value})")
        logger.info("="*70)
        
        # =====================================================================
        # PHASE 1: Initialize Background Model (Blob Detector)
        # =====================================================================
        
        # Skip background initialization for YOLO_ONLY mode (saves 10-20 seconds!)
        if detection_mode_enum == DetectionMode.YOLO_ONLY:
            logger.info("\nPhase 1: Background Initialization")
            logger.info("-" * 70)
            logger.info("⏭️  Skipping background initialization (YOLO_ONLY mode)")
            logger.info("   YOLO does not use background subtraction")
            blob_detector = None
        else:
            logger.info("\nPhase 1: Background Initialization")
            logger.info("-" * 70)
            
            # Use stricter parameters for FGBG modes (reduces noise)
            if detection_mode_enum in [DetectionMode.FGBG_ONLY, DetectionMode.FGBG_YOLO, 
                                         DetectionMode.FGBG_SIFT, DetectionMode.FGBG_SIFT_YOLO]:
                blob_min_area = max(100.0, self.config.detection.min_area)
                blob_max_area = min(2000.0, self.config.detection.max_area)  # Cap at 2000 for FGBG
                blob_min_solidity = max(0.75, self.config.detection.min_solidity)
                blob_min_circularity = 0.3  # Bees are reasonably circular
                blob_min_aspect_ratio = 0.4  # Not too elongated (min width/height)
                blob_max_aspect_ratio = 2.5  # Not too elongated (max width/height)
                blob_min_extent = 0.3  # Fills at least 30% of bounding box
                blob_morph_kernel = 9
                blob_morph_iters = 4
                blob_var_threshold = 25  # Higher = less sensitive to small changes
                logger.info("Using strict blob filtering for FGBG mode")
                logger.info(f"  Area: {blob_min_area}-{blob_max_area} px")
                logger.info(f"  Solidity: ≥{blob_min_solidity} (compact shape)")
                logger.info(f"  Circularity: ≥{blob_min_circularity} (roundness)")
                logger.info(f"  Aspect ratio: {blob_min_aspect_ratio}-{blob_max_aspect_ratio} (not elongated)")
                logger.info(f"  Extent: ≥{blob_min_extent} (fills bounding box)")
            else:
                blob_min_area = self.config.detection.min_area if hasattr(self.config.detection, 'min_area') else 50.0
                blob_max_area = self.config.detection.max_area if hasattr(self.config.detection, 'max_area') else 5000.0
                blob_min_solidity = self.config.detection.min_solidity if hasattr(self.config.detection, 'min_solidity') else 0.5
                blob_min_circularity = 0.0  # No circularity filter for YOLO mode
                blob_min_aspect_ratio = 0.0
                blob_max_aspect_ratio = 10.0
                blob_min_extent = 0.0
                blob_morph_kernel = 5
                blob_morph_iters = 2
                blob_var_threshold = 16  # Default sensitivity
            
            blob_detector = BlobDetector(
                min_area=blob_min_area,
                max_area=blob_max_area,
                min_solidity=blob_min_solidity,
                min_circularity=blob_min_circularity,
                min_aspect_ratio=blob_min_aspect_ratio,
                max_aspect_ratio=blob_max_aspect_ratio,
                min_extent=blob_min_extent,
                morph_kernel_size=blob_morph_kernel,
                morph_iterations=blob_morph_iters,
                var_threshold=blob_var_threshold
            )
            
            try:
                blob_detector.initialize_from_video(
                    video_path=video_path,
                    num_frames=100,
                    start_frame=0  # Start from beginning
                )
                logger.info("✓ Background model initialized (frames 0-100)")
            except Exception as e:
                logger.warning(f"Background initialization failed: {e}")
                logger.warning("Continuing with default background model")
        
        # =====================================================================
        # PHASE 1b: Learn Geometric Filter Thresholds (if FGBG mode)
        # =====================================================================
        if detection_mode_enum in [DetectionMode.FGBG_ONLY, DetectionMode.FGBG_YOLO,
                                     DetectionMode.FGBG_SIFT, DetectionMode.FGBG_SIFT_YOLO]:
            logger.info("\nPhase 1b: Learning Geometric Filter Thresholds")
            logger.info("-" * 70)
            
            try:
                # Convert class IDs to class names from config
                tracking_class_names = []
                if hasattr(self.config.tracking, 'label_map'):
                    for class_id in self.config.tracking.tracking_classes:
                        class_name = self.config.tracking.label_map.get(class_id, f'class_{class_id}')
                        tracking_class_names.append(class_name)
                else:
                    tracking_class_names = [str(cid) for cid in self.config.tracking.tracking_classes]
                
                logger.info(f"  Using tracking classes: {tracking_class_names}")
                
                # Create temporary YOLO detector for threshold learning
                # NOTE: Use 'bee' class name (what fine-tuned model has)
                from beemonitor.detection import YOLODetector
                temp_yolo = YOLODetector(
                    model=self.tracking_model,
                    conf_threshold=0.4,  # Moderate confidence for learning (was 0.5)
                    tracking_classes= None #['bee']  # Match fine-tuned model class names!
                )
                
                # Learn thresholds from frames with bee activity
                learned_thresholds = blob_detector.learn_geometric_thresholds_from_video(
                    video_path=video_path,
                    yolo_detector=temp_yolo,
                    num_frames=200,  # Use more frames
                    start_frame=0,   # Start from beginning
                    min_detections=20,
                    percentile_low=5.0,
                    percentile_high=95.0
                )
                
                logger.info("✓ Geometric thresholds learned from tracked object detections (frames 0-200)")
                
            except Exception as e:
                logger.warning(f"Geometric threshold learning failed: {e}")
                logger.warning("Using default geometric thresholds")
        else:
            logger.info("\nSkipping geometric threshold learning (YOLO_ONLY mode)")
        
        # =====================================================================
        # PHASE 2: Full Tracking with Blob + YOLO
        # =====================================================================
        logger.info("\nPhase 2: Full Video Analysis")
        logger.info("-" * 70)
        logger.info(f"Detection mode: {detection_mode_enum.value}")
        logger.info("")
        
        # Convert class IDs to class names
        tracking_class_names = []
        if hasattr(self.config.tracking, 'label_map'):
            for class_id in self.config.tracking.tracking_classes:
                class_name = self.config.tracking.label_map.get(class_id, f'class_{class_id}')
                tracking_class_names.append(class_name)
        else:
            tracking_class_names = [str(cid) for cid in self.config.tracking.tracking_classes]
        
        # Create YOLO detector with correct class names
        # NOTE: Use 'bee' class name (what fine-tuned model has)
        yolo_detector = YOLODetector(
            model=self.tracking_model,
            conf_threshold=0.25,
            tracking_classes=['bee']  # Match fine-tuned model class names!
        )
        
        # Initialize MOT algorithm
        mot_algorithm = BeeTracker(
            config=self.config,
            tracking_classes=tracking_class_names
        )
        
        # Enable noise filter for FGBG modes (they need it to reduce false positives)
        use_noise_filter = detection_mode_enum in [DetectionMode.FGBG_ONLY, DetectionMode.FGBG_YOLO,
                                                     DetectionMode.FGBG_SIFT, DetectionMode.FGBG_SIFT_YOLO]
        noise_filter_model = None
        
        if use_noise_filter:
            try:
                from beemonitor.detection.noise_filter import BeeNoiseFilter
                # Get model path from config.models.blob_noise_classifier
                model_path = self.config.models.blob_noise_classifier if hasattr(self.config.models, 'blob_noise_classifier') else None
                if model_path and Path(model_path).exists():
                    noise_filter_model = BeeNoiseFilter(
                        model_path=model_path,
                        noise_threshold=0.9  # High threshold = strict filtering
                    )
                    logger.info(f"✓ CNN noise filter enabled: {model_path}")
                else:
                    if model_path:
                        logger.warning(f"Noise filter model not found: {model_path}")
                    else:
                        logger.info("⚠ Noise filter model not configured in config.models.blob_noise_classifier")
                    logger.info("Continuing with morphological filtering only")
                    use_noise_filter = False
            except Exception as e:
                logger.warning(f"Could not load noise filter: {e}")
                logger.info("Continuing with morphological filtering only")
                use_noise_filter = False
        
        # =====================================================================
        # PHASE 1c: SIFT Template Learning (if SIFT mode)
        # =====================================================================
        sift_detector = None
        if detection_mode_enum in [DetectionMode.SIFT_ONLY, DetectionMode.FGBG_SIFT,
                                     DetectionMode.SIFT_YOLO, DetectionMode.FGBG_SIFT_YOLO]:
            logger.info("\nPhase 1c: SIFT Template Learning")
            logger.info("-" * 70)
            
            try:
                from beemonitor.detection import SIFTDetector, YOLODetector
                
                # Create SIFT detector
                sift_detector = SIFTDetector(
                    min_keypoints=3,
                    use_templates=True,
                    require_movement=False  # DISABLE movement filter - too strict!
                )
                
                # Create YOLO detector for template learning
                # NOTE: Use 'bee' class name (what fine-tuned model has)
                # NOT config classes (which might be 'osmia_cornifrons')
                temp_yolo = YOLODetector(
                    model=self.tracking_model,
                    conf_threshold=0.25,  # Match tracking threshold (was 0.7 - too strict!)
                    tracking_classes=['bee']  # Match fine-tuned model class names!
                )
                
                logger.info(f"  SIFT YOLO detector configured:")
                logger.info(f"    Confidence: 0.25")
                logger.info(f"    Classes: ['bee'] (fine-tuned model)")
                logger.info(f"    Model: {temp_yolo.model}")
                
                # TEST: Does YOLO work at all?
                logger.info(f"\n  Testing YOLO detector on first frame...")
                import cv2
                cap = cv2.VideoCapture(video_path)
                ret, test_frame = cap.read()
                cap.release()
                if ret:
                    test_dets = temp_yolo.detect(test_frame)
                    logger.info(f"  ✓ YOLO test: {len(test_dets)} detections on frame 0")
                else:
                    logger.warning(f"  ✗ Could not read test frame from video")
                
                # Learn templates from video
                logger.info(f"\n  Learning bee templates from frames 0-200...")
                num_templates = sift_detector.initialize_from_video(
                    video_path=video_path,
                    yolo_detector=temp_yolo,
                    num_frames=200,  # Use more frames
                    start_frame=0,   # Start from beginning
                    min_confidence=0.25  # MATCH YOLO conf (was 0.5 - too strict!)
                )
                
                logger.info(f"✓ Learned {num_templates} SIFT templates (frames 0-200)")
                
                # Save templates for reuse
                video_name = Path(video_path).stem
                templates_path = os.path.join(output_folder, f'{video_name}_sift_templates.pkl')
                sift_detector.save_templates(templates_path)
                logger.info(f"✓ Templates saved to: {templates_path}")
                
                # VISUALIZE TEMPLATES (save images to folder)
                if num_templates > 0:
                    viz_folder = os.path.join(output_folder, 'sift_templates_viz')
                    logger.info(f"\n  Visualizing templates...")
                    num_viz = sift_detector.visualize_templates(
                        video_path=video_path,
                        output_dir=viz_folder,
                        max_templates=20  # Save top 20 templates
                    )
                    logger.info(f"✓ Saved {num_viz} template visualizations to: {viz_folder}/")
                    logger.info(f"  Each image shows:")
                    logger.info(f"    • Bee crop with SIFT keypoints (green circles)")
                    logger.info(f"    • Template quality score")
                    logger.info(f"    • Source frame number")
                
            except Exception as e:
                logger.warning(f"SIFT initialization failed: {e}")
                logger.warning("Continuing without SIFT detection")
                # Fall back to mode without SIFT
                if detection_mode_enum == DetectionMode.SIFT_ONLY:
                    logger.error("Cannot continue - SIFT_ONLY mode requires SIFT initialization")
                    return None, None
                elif detection_mode_enum == DetectionMode.FGBG_SIFT:
                    detection_mode_enum = DetectionMode.FGBG_ONLY
                elif detection_mode_enum == DetectionMode.SIFT_YOLO:
                    detection_mode_enum = DetectionMode.YOLO_ONLY
                elif detection_mode_enum == DetectionMode.FGBG_SIFT_YOLO:
                    detection_mode_enum = DetectionMode.FGBG_YOLO
        
        # Initialize BeeTracking system
        tracker = BeeTracking(
            mot_algorithm=mot_algorithm,
            yolo_model=self.tracking_model,
            detection_mode=detection_mode_enum,  # Use parameter instead of hardcoded
            use_noise_filter=use_noise_filter,  # Enable for FGBG modes
            noise_filter_model=noise_filter_model,
            config=self.config
        )
        
        # *** CRITICAL: Use our initialized detectors ***
        tracker.blob_detector = blob_detector
        tracker.yolo_detector = yolo_detector  # Add this line!
        if sift_detector is not None:
            tracker.sift_detector = sift_detector  # Add SIFT detector
            logger.info(f"✓ SIFT detector added ({sift_detector.get_num_templates()} templates)")
        
        logger.info("✓ Tracking system initialized")
        logger.info("="*70 + "\n")
        
        # Process video
        tracking_df = tracker.process_video(
            video_path=video_path,
            roi=hotel_roi
        )
        
        # Convert to grouped format
        grouped_df = self._convert_tracking_to_grouped_format(tracking_df)
        
        logger.info(f"\n✓ Tracking complete:")
        logger.info(f"  Total detections: {len(tracking_df)}")
        logger.info(f"  Unique tracks: {tracking_df['track_id'].nunique() if not tracking_df.empty else 0}")
        
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