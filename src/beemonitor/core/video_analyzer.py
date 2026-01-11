"""Main video analyzer class for bee monitoring.

This module provides the BeeMonitor class, which orchestrates the entire
bee detection and tracking pipeline with FALLBACK SUPPORT for nest detection
across multiple videos from the same bee hotel.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any
import pandas as pd
import numpy as np
from ultralytics import YOLO
import os
import re

from beemonitor.core.config import Config
from concurrent.futures import ThreadPoolExecutor, as_completed


logger = logging.getLogger(__name__)


class AnalysisResults:
    """Container for video analysis results.
    
    This class holds all the outputs from video analysis and provides
    convenient methods for exporting and accessing results.
    """
    
    def __init__(
        self,
        events: pd.DataFrame,
        tracks: List,
        nests: Dict,
        video_path: str,
        motion_data: Optional[pd.DataFrame] = None,
        config: Optional[Config] = None
    ):
        """Initialize analysis results."""
        self.events = events
        self.tracks = tracks
        self.nests = nests
        self.video_path = video_path
        self.motion_data = motion_data
        self.config = config
    
    def to_csv(self, output_folder: str = "output", columns: Optional[List[str]] = None) -> None:
        """Export events and tracking results to CSV files."""
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        base_filename = Path(self.video_path).stem
        
        # Save events
        events_filename = str(Path(output_folder) / f"{base_filename}_events.csv")
        if columns is None:
            self.events.to_csv(events_filename, index=False)
        else:
            available_cols = [col for col in columns if col in self.events.columns]
            self.events[available_cols].to_csv(events_filename, index=False)
        
        logger.info(f"Saved {len(self.events)} events to {events_filename}")
        
        # Save tracking results
        tracking_filename = str(Path(output_folder) / f"{base_filename}_tracking_results.csv")
        if self.tracks is not None and isinstance(self.tracks, pd.DataFrame) and not self.tracks.empty:
            self.tracks.to_csv(tracking_filename, index=False)
            logger.info(f"Saved {len(self.tracks)} tracking records to {tracking_filename}")
    
    def save_video(self, output_folder: str = "output") -> None:
        """Save annotated video with tracking visualization."""
        from beemonitor.output.video_synthesizer import VideoSynthesizer
        
        synthesizer = VideoSynthesizer(self.config)
        output_path = synthesizer.synthesize(
            self.video_path,
            self.events,
            self.motion_data,
            self.nests,
            output_folder
        )
        logger.info(f"Saved annotated video to {output_path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Calculate summary statistics from the analysis."""
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
        
        if 'nest' in self.events.columns:
            nest_counts = self.events.groupby('nest')['action'].value_counts().unstack(fill_value=0)
            stats['nest_activity'] = nest_counts.to_dict()
        
        return stats
    
    def __repr__(self) -> str:
        return (
            f"AnalysisResults(events={len(self.events)}, "
            f"tracks={len(self.tracks) if isinstance(self.tracks, list) else 0}, "
            f"nests={len(self.nests.get('nests', {}))})"
        )


class BeeMonitor:
    """Main interface for bee monitoring video analysis.
    
    Supports NEST DETECTION FALLBACK for processing multiple videos
    from the same bee hotel - if nest detection fails on one video,
    it automatically tries using nests from adjacent videos.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize BeeMonitor with configuration."""
        if config is None:
            config = Config.default()
        self.config = config
        self.res_height = config.video.res_height
        self.res_width = config.video.res_width
        self.nest_model = YOLO(config.models.nest_detection)
        self.tracking_model = YOLO(config.models.tracking)
        
        logger.info("BeeMonitor initialized successfully")
    
    @classmethod
    def from_config(cls, config_path: str) -> "BeeMonitor":
        """Create BeeMonitor from configuration file."""
        config = Config.from_yaml(config_path)
        return cls(config=config)
    
    def analyze_video(
        self,
        video_path: str,
        nest_video_path: Optional[str] = None,
        output_folder: Optional[str] = None,
        visualize: Optional[bool] = None,
        detection_mode: str = 'fgbg_yolo'
    ) -> AnalysisResults:
        """Analyze a single video.
        
        Args:
            video_path: Path to input video file
            nest_video_path: Optional path to video for nest detection (defaults to video_path)
            output_folder: Directory for output files
            visualize: Whether to save visualization videos
            detection_mode: 'fgbg', 'fgbg_yolo', or 'yolo_only'
            
        Returns:
            AnalysisResults object
        """
        video_file = Path(video_path)
        if not video_file.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        if visualize is None:
            visualize = self.config.output.save_tracking_visualizations
        
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
        
        # Step 2: Track bees
        logger.info("Step 2/3: Detecting motion and tracking bees...")
        flat_tracking_df, grouped_tracking_df = self.get_motion_tracking(
            video_path,
            nests['hotel'],
            output_folder,
            visualize=visualize,
            detection_mode=detection_mode
        )
        
        if flat_tracking_df is None or flat_tracking_df.empty:
            logger.warning("No motion tracking data returned, skipping video analysis")
            return None
        
        # Step 3: Process events
        logger.info("Step 3/3: Processing tracks to identify events...")
        events = self.process_motion_tracking(grouped_tracking_df, nests)
        events = self.synthesize_csv(events, video_path)
        
        # Create results
        results = AnalysisResults(
            events=events,
            tracks=flat_tracking_df,
            nests=nests,
            video_path=video_path,
            motion_data=grouped_tracking_df,
            config=self.config
        )
        
        if visualize:
            results.to_csv(output_folder=output_folder)
            results.save_video(output_folder=output_folder)
        
        logger.info(f"Analysis complete: {len(events)} events detected")
        return results
    
    def analyze_video_with_relative_nests(
        self,
        video_path: str,
        video_files: List[str],
        output_folder: Optional[str] = None,
        visualize: Optional[bool] = None,
        detection_mode: str = 'fgbg_yolo'
    ) -> AnalysisResults:
        """Analyze video with FALLBACK to adjacent videos for nest detection.
        
        This method implements the fallback logic:
        1. Try detecting nests in current video
        2. If fails, try previous video's nests
        3. If fails, try next video's nests
        4. If all fail, return None
        
        Args:
            video_path: Path to input video file
            video_files: List of all video files (for finding adjacent videos)
            output_folder: Directory for output files
            visualize: Whether to save visualization videos
            detection_mode: 'fgbg', 'fgbg_yolo', or 'yolo_only'
            
        Returns:
            AnalysisResults object or None if nest detection fails
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"ANALYZING VIDEO WITH FALLBACK SUPPORT")
        logger.info(f"{'='*70}")
        logger.info(f"Video: {Path(video_path).name}")
        
        # Try current video first
        logger.info("Attempting nest detection on current video...")
        try:
            nests = self.get_nest_detections(video_path)
            if nests is not None:
                logger.info("✓ Nest detection successful on current video")
                return self.analyze_video(
                    video_path,
                    nest_video_path=video_path,
                    output_folder=output_folder,
                    visualize=visualize,
                    detection_mode=detection_mode
                )
        except Exception as e:
            logger.warning(f"Nest detection failed on current video: {e}")
        
        # Get adjacent videos
        prev_video = self._get_prev_video(video_path, video_files)
        next_video = self._get_next_video(video_path, video_files)
        
        # Try previous video
        if prev_video is not None:
            logger.info(f"⟳ FALLBACK: Trying previous video: {Path(prev_video).name}")
            try:
                nests = self.get_nest_detections(prev_video)
                if nests is not None:
                    logger.info(f"✓ Using nest detections from PREVIOUS video")
                    return self.analyze_video(
                        video_path,
                        nest_video_path=prev_video,
                        output_folder=output_folder,
                        visualize=visualize,
                        detection_mode=detection_mode
                    )
            except Exception as e:
                logger.warning(f"Nest detection failed on previous video: {e}")
        
        # Try next video
        if next_video is not None:
            logger.info(f"⟳ FALLBACK: Trying next video: {Path(next_video).name}")
            try:
                nests = self.get_nest_detections(next_video)
                if nests is not None:
                    logger.info(f"✓ Using nest detections from NEXT video")
                    return self.analyze_video(
                        video_path,
                        nest_video_path=next_video,
                        output_folder=output_folder,
                        visualize=visualize,
                        detection_mode=detection_mode
                    )
            except Exception as e:
                logger.warning(f"Nest detection failed on next video: {e}")
        
        logger.error(f"✗ NEST DETECTION FAILED - No valid nests from current, previous, or next video")
        logger.error(f"  Current: {Path(video_path).name}")
        logger.error(f"  Previous: {Path(prev_video).name if prev_video else 'None'}")
        logger.error(f"  Next: {Path(next_video).name if next_video else 'None'}")
        logger.info(f"{'='*70}\n")
        
        return None
    
    def analyze_videos_in_folder(
        self,
        video_folder: str,
        output_folder: Optional[str] = None,
        visualize: bool = False,
        detection_mode: str = 'fgbg_yolo',
        use_fallback: bool = True,
        max_workers: int = 4
    ) -> Dict[str, AnalysisResults]:
        """Process multiple videos in parallel with FALLBACK support.
        
        Args:
            video_folder: Directory containing video files
            output_folder: Directory for output files
            visualize: Whether to save visualization videos
            detection_mode: 'fgbg', 'fgbg_yolo', or 'yolo_only'
            use_fallback: Enable nest detection fallback (DEFAULT: True)
            max_workers: Number of parallel video processing threads
            
        Returns:
            Dictionary mapping video paths to AnalysisResults
        """
        if output_folder is None:
            output_folder = self.config.output.base_folder
        
        # Get all video files
        video_files = sorted([
            str(Path(video_folder) / vf) 
            for vf in os.listdir(video_folder) 
            if vf.endswith(('.mp4', '.avi', '.mov'))
        ])
        
        logger.info(f"\n{'='*70}")
        logger.info(f"BATCH VIDEO ANALYSIS")
        logger.info(f"{'='*70}")
        logger.info(f"Folder: {video_folder}")
        logger.info(f"Videos found: {len(video_files)}")
        logger.info(f"Fallback mode: {'ENABLED' if use_fallback else 'DISABLED'}")
        logger.info(f"Parallel workers: {max_workers}")
        logger.info(f"Detection mode: {detection_mode}")
        logger.info(f"{'='*70}\n")
        
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            if use_fallback:
                # USE FALLBACK - adjacent videos for nest detection
                future_to_video = {
                    executor.submit(
                        self.analyze_video_with_relative_nests,
                        video_path,
                        video_files,  # Pass all videos for fallback
                        output_folder=output_folder,
                        visualize=visualize,
                        detection_mode=detection_mode
                    ): video_path
                    for video_path in video_files
                }
            else:
                # NO FALLBACK - independent analysis
                future_to_video = {
                    executor.submit(
                        self.analyze_video,
                        video_path,
                        output_folder=output_folder,
                        visualize=visualize,
                        detection_mode=detection_mode
                    ): video_path
                    for video_path in video_files
                }
            
            for future in as_completed(future_to_video):
                video_path = future_to_video[future]
                try:
                    result = future.result()
                    results[video_path] = result
                    
                    if result is not None:
                        logger.info(f"✓ Completed: {Path(video_path).name} "
                                   f"({len(result.events)} events)")
                    else:
                        logger.warning(f"✗ Failed: {Path(video_path).name} (no results)")
                        
                except Exception as e:
                    logger.error(f"✗ Error processing {Path(video_path).name}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    results[video_path] = None
        
        # Summary
        successful = sum(1 for r in results.values() if r is not None)
        failed = len(results) - successful
        
        logger.info(f"\n{'='*70}")
        logger.info(f"BATCH ANALYSIS COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"Total videos: {len(video_files)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"{'='*70}\n")
        
        return results
    
    def get_nest_detections(self, video_path: str) -> Optional[Dict]:
        """Detect nests using NestDetector."""
        from beemonitor.detection.nest_detector import NestDetector
        
        logger.info(f"Detecting nests in: {Path(video_path).name}")
        
        detector = NestDetector(
            model=self.nest_model,
            config=self.config
        )
        
        nests = detector.get_nests_and_hotel_detections(video_path=video_path)
        
        if nests is not None:
            logger.info(f"✓ Detected {len(nests.get('nests', {}))} nests")
        else:
            logger.warning(f"✗ No nests detected")
        
        return nests
    
    def _get_next_video(
        self,
        video_path: str,
        video_files: List[str],
    ) -> Optional[str]:
        """Get the next consecutive video file after the current video.
        
        Uses timestamp from filename to find same-day videos in sequence.
        """
        def extract_ts(fp: str) -> Optional[str]:
            m = re.search(r"\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2}", Path(fp).stem)
            return m.group(0) if m else None
        
        current_path = Path(video_path)
        current_ts = extract_ts(current_path.name)
        
        if current_ts is None:
            logger.debug("No timestamp found; falling back to filename ordering")
            sorted_files = sorted(video_files)
        else:
            date_part = current_ts.split("_")[0]
            same_date_files = []
            for f in video_files:
                ts = extract_ts(f)
                if ts and ts.startswith(date_part):
                    same_date_files.append(f)
            
            if not same_date_files:
                logger.debug(f"No same-date files found for {video_path}")
                return None
            
            sorted_files = sorted(same_date_files, key=lambda p: extract_ts(p) or "")
        
        sorted_paths = [str(Path(p)) for p in sorted_files]
        try:
            idx = next(i for i, p in enumerate(sorted_paths) if Path(p).name == current_path.name)
        except StopIteration:
            logger.debug(f"Current video {video_path} not found in filtered list")
            return None
        
        next_idx = idx + 1
        if next_idx < len(sorted_paths):
            return sorted_paths[next_idx]
        return None
    
    def _get_prev_video(
        self,
        video_path: str,
        video_files: List[str],
    ) -> Optional[str]:
        """Get the previous consecutive video file before the current video.
        
        Uses timestamp from filename to find same-day videos in sequence.
        """
        def extract_ts(fp: str) -> Optional[str]:
            m = re.search(r"\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2}", Path(fp).stem)
            return m.group(0) if m else None
        
        current_path = Path(video_path)
        current_ts = extract_ts(current_path.name)
        
        if current_ts is None:
            logger.debug("No timestamp found; falling back to filename ordering")
            sorted_files = sorted(video_files)
        else:
            date_part = current_ts.split("_")[0]
            same_date_files = []
            for f in video_files:
                ts = extract_ts(f)
                if ts and ts.startswith(date_part):
                    same_date_files.append(f)
            
            if not same_date_files:
                logger.debug(f"No same-date files found for {video_path}")
                return None
            
            sorted_files = sorted(same_date_files, key=lambda p: extract_ts(p) or "")
        
        sorted_paths = [str(Path(p)) for p in sorted_files]
        try:
            idx = next(i for i, p in enumerate(sorted_paths) if Path(p).name == current_path.name)
        except StopIteration:
            logger.debug(f"Current video {video_path} not found in filtered list")
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
        detection_mode: str = 'fgbg_yolo'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Detect motion and track bees with automatic CNN filtering."""
        from beemonitor.tracking.bee_tracking import BeeTracking, DetectionMode
        from beemonitor.tracking.mot.bee_tracker import BeeTracker
        from beemonitor.detection import BlobDetector, YOLODetector
        
        # Map detection mode
        mode_map = {
            'fgbg': DetectionMode.FGBG_ONLY,
            'fgbg_yolo': DetectionMode.FGBG_YOLO,
            'yolo': DetectionMode.YOLO_ONLY,
            'yolo_only': DetectionMode.YOLO_ONLY,
        }
        detection_mode_enum = mode_map.get(detection_mode, DetectionMode.FGBG_YOLO)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"DETECTION MODE: {detection_mode.upper()}")
        logger.info(f"  CNN noise filter: {'AUTO' if detection_mode in ['fgbg', 'fgbg_yolo'] else 'N/A'}")
        logger.info(f"  Learned solidity: {'AUTO' if detection_mode in ['fgbg', 'fgbg_yolo'] else 'N/A'}")
        logger.info(f"{'='*70}")
        
        # Initialize blob detector with researched optimal defaults
        RESEARCHED_MIN_AREA = 30.0
        RESEARCHED_MIN_SOLIDITY = 0.56
        
        logger.info(f"\nPhase 1: Background Initialization")
        logger.info(f"  min_area: {RESEARCHED_MIN_AREA} (researched optimal)")
        logger.info(f"  min_solidity: {RESEARCHED_MIN_SOLIDITY} (researched optimal)")
        
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
            logger.info("✓ Background model initialized")
        except Exception as e:
            logger.warning(f"Background initialization failed: {e}")
        
        # Initialize YOLO detector
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
            tracking_classes=None
        )
        
        # Initialize MOT algorithm
        use_yolo_confirmation = detection_mode_enum in [
            DetectionMode.FGBG_YOLO,
            DetectionMode.YOLO_ONLY
        ]
        
        mot_algorithm = BeeTracker(
            config=self.config,
            tracking_classes=tracking_class_names,
            require_yolo_confirmation=use_yolo_confirmation,
        )
        
        if use_yolo_confirmation:
            logger.info("✓ YOLO confirmation ENABLED")
        else:
            logger.info("ℹ YOLO confirmation DISABLED")
        
        # Enable CNN noise filter
        use_noise_filter = detection_mode_enum in [DetectionMode.FGBG_ONLY, DetectionMode.FGBG_YOLO]
        noise_filter_model = None
        
        if use_noise_filter:
            try:
                from beemonitor.detection.noise_filter import BeeNoiseFilter
                
                model_path = self.config.models.blob_noise_classifier if hasattr(self.config.models, 'blob_noise_classifier') else None
                
                if model_path and Path(model_path).exists():
                    logger.info(f"✓ CNN noise filter ENABLED")
                    noise_filter_model = BeeNoiseFilter(
                        model_path=model_path,
                        noise_threshold=0.9
                    )
                else:
                    logger.warning("CNN filter model not found - continuing without it")
                    use_noise_filter = False
            except Exception as e:
                logger.warning(f"CNN filter loading failed: {e}")
                use_noise_filter = False
        
        # Initialize tracking system
        tracker = BeeTracking(
            mot_algorithm=mot_algorithm,
            yolo_model=self.tracking_model,
            detection_mode=detection_mode_enum,
            use_noise_filter=use_noise_filter,
            noise_filter_model=noise_filter_model,
            config=self.config,
            enable_online_learning=True
        )
        
        tracker.blob_detector = blob_detector
        tracker.yolo_detector = yolo_detector
        
        logger.info("✓ Tracking system initialized")
        
        # Process video
        logger.info(f"\nProcessing video: {Path(video_path).name}")
        
        import time
        start_time = time.time()
        
        tracking_df = tracker.process_video(
            video_path=video_path,
            roi=hotel_roi
        )
        
        elapsed = time.time() - start_time
        
        logger.info(f"\n✓ Video processing complete ({elapsed:.1f}s)")
        logger.info(f"  Total detections: {len(tracking_df)}")
        logger.info(f"  Unique tracks: {tracking_df['track_id'].nunique() if not tracking_df.empty else 0}")
        
        # Convert to grouped format
        grouped_df = self._convert_tracking_to_grouped_format(tracking_df)
        
        return tracking_df, grouped_df
    
    def _convert_tracking_to_grouped_format(self, tracking_df: pd.DataFrame) -> pd.DataFrame:
        """Convert flat tracking DataFrame to grouped format for event processor."""
        if tracking_df.empty:
            return pd.DataFrame(columns=['frame_number', 'tracks', 'detections'])
        
        gap_threshold = int(self.config.tracking.max_age * 1.1)
        periods = self._split_into_periods(tracking_df, gap_threshold)
        
        result_rows = []
        for period_df in periods:
            track_groups = {}
            
            for track_id in period_df['track_id'].unique():
                track_df = period_df[period_df['track_id'] == track_id].sort_values('frame')
                segments = self._split_track_by_gaps(track_df, gap_threshold=self.config.tracking.max_age)
                
                for seg_idx, seg_df in enumerate(segments):
                    unique_id = f"{track_id}_{seg_idx}" if len(segments) > 1 else track_id
                    
                    centroids = [((row['x1'] + row['x2']) / 2, (row['y1'] + row['y2']) / 2)
                                for _, row in seg_df.iterrows()]
                    bboxes = [(row['x1'], row['y1'], row['x2'], row['y2'])
                             for _, row in seg_df.iterrows()]
                    frame_numbers = seg_df['frame'].tolist()
                    
                    if 'species' in seg_df.columns:
                        species_list = seg_df['species'].tolist()
                        species_votes = {}
                        for species in species_list:
                            species_votes[species] = species_votes.get(species, 0) + 1
                        most_common_species = max(species_votes, key=species_votes.get)
                    else:
                        most_common_species = 'unknown'
                        species_votes = {'unknown': len(frame_numbers)}
                    
                    if len(frame_numbers) >= self.config.tracking.min_track_length:
                        track_groups[unique_id] = (
                            unique_id,
                            centroids,
                            bboxes,
                            frame_numbers,
                            most_common_species,
                            species_votes
                        )
            
            if not track_groups:
                continue
            
            all_tracks = list(track_groups.values())
            min_frame = int(period_df['frame'].min())
            max_frame = int(period_df['frame'].max())
            
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
        """Split detections into activity periods based on frame gaps."""
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
        """Split track into segments by gaps."""
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
    
    def process_motion_tracking(self, motion_data: pd.DataFrame, nests: Dict) -> pd.DataFrame:
        """Process tracking data to identify entry/exit events."""
        if motion_data is None:
            logger.warning("motion_data is None, returning empty events DataFrame")
            return pd.DataFrame(columns=['timestamp', 'nest_id', 'action', 'track_id', 'species'])
        
        from beemonitor.processing.event_processor import EventProcessor
        
        processor = EventProcessor(config=self.config)
        return processor.process_tracks(motion_data=motion_data, nests=nests)
    
    def synthesize_csv(self, events: pd.DataFrame, video_path: str) -> pd.DataFrame:
        """Generate CSV with timestamps from events."""
        from beemonitor.output.csv_generator import CSVGenerator
        
        generator = CSVGenerator(config=self.config)
        return generator.generate_csv(events=events, video_path=video_path)
    
    def __repr__(self) -> str:
        return (
            f"BeeMonitor(resolution={self.res_width}x{self.res_height}, "
            f"config={self.config is not None})"
        )