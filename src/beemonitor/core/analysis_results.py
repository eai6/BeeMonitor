"""Analysis results container for bee monitoring.

This module provides the AnalysisResults class which holds all outputs
from video analysis and provides convenient methods for exporting results.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, List, Any
import pandas as pd

logger = logging.getLogger(__name__)


class AnalysisResults:
    """Container for video analysis results.
    
    This class holds all the outputs from video analysis and provides
    convenient methods for exporting and accessing results.
    
    Attributes:
        events: DataFrame containing entry/exit events with timestamps
        tracks: List of bee trajectories
        nests: Dictionary mapping nest IDs to bounding boxes
        video_path: Path to the analyzed video
        motion_data: Motion detection DataFrame
        config: Configuration object
    
    Example:
        >>> results = monitor.analyze_video("video.mp4")
        >>> results.to_csv("output.csv")
        >>> print(f"Found {len(results.events)} events")
        >>> stats = results.get_statistics()
    """
    
    def __init__(
        self,
        events: pd.DataFrame,
        tracks: List,
        nests: Dict,
        video_path: str,
        motion_data: Optional[pd.DataFrame] = None, 
        config: Optional[Any] = None
    ):
        """Initialize analysis results.
        
        Args:
            events: DataFrame with processed events
            tracks: List of bee trajectories
            nests: Dictionary of nest locations
            video_path: Path to analyzed video
            motion_data: Motion detection data (optional)
            config: Configuration object (optional)
        """
        self.events = events
        self.tracks = tracks
        self.nests = nests
        self.video_path = video_path
        self.motion_data = motion_data
        self.config = config
    
    def to_csv(self, output_folder: str = "output", columns: Optional[List[str]] = None) -> None:
        """Export events to CSV file.
        
        Args:
            output_folder: Directory for output files
            columns: Columns to include (default: all)
            
        Example:
            >>> results.to_csv("output")
            >>> results.to_csv("output", columns=["timestamp", "nest", "action"])
        """
        filename = self.video_path.replace(".mp4", "_events.csv") 
        filename = filename.split("/")[-1]
        filename = str(Path(output_folder) / Path(filename).name)
        
        if columns is None:
            self.events.to_csv(filename, index=False)
        else:
            available_cols = [col for col in columns if col in self.events.columns]
            self.events[available_cols].to_csv(filename, index=False)
        
        logger.info(f"Saved {len(self.events)} events to {filename}")
    
    def save_video(self, output_folder: str = "output") -> None:
        """Save annotated video with tracking visualization.
        
        Args:
            output_folder: Directory for output (default: "output")
            
        Example:
            >>> results.save_video("output")
        """
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
        