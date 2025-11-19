
"""CSV synthesis module with resolution-adaptive configuration.

This module handles the conversion of tracking events to CSV format with timestamps.
While CSV generation is mostly resolution-independent, this module is updated to
work seamlessly with the Config system for consistency.
"""

from datetime import time, datetime, timedelta
from typing import Optional
import pandas as pd


class CSVSynthesizer:
    """Handles synthesis of tracking events into CSV format.
    
    This class converts tracking events with frame numbers into timestamped
    CSV records suitable for analysis and reporting.
    
    Attributes:
        fps: Frames per second of the source video
        columns: List of column names to include in output CSV
    
    Example:
        >>> from config import Config
        >>> config = Config.default()
        >>> synthesizer = CSVSynthesizer(config)
        >>> csv_df = synthesizer.process(events, "site_2024-01-15_14_30_00.mp4")
    """
    
    def __init__(self, config):
        """Initialize CSV synthesizer with configuration.
        
        Args:
            config: Config object containing video and output settings
        """
        self.fps = config.video.fps
        # Set default columns if not specified in config
        if config.output.csv_columns is None:
            self.columns = ['timestamp', 'nest_id', 'action', 'track_id', 'species']
        else:
            self.columns = config.output.csv_columns
    
    def extract_start_time(self, filename: str) -> datetime:
        """Extract the start time from a video filename.
        
        Expected filename format: site_YYYY-MM-DD_HH_MM_SS.mp4
        
        Args:
            filename: Video filename (with or without path)
            
        Returns:
            datetime object representing the video start time
            
        Raises:
            ValueError: If filename format is invalid
            
        Example:
            >>> synthesizer = CSVSynthesizer(config)
            >>> dt = synthesizer.extract_start_time("site_2024-01-15_14_30_00.mp4")
            >>> print(dt)
            2024-01-15 14:30:00
        """
        try:
            # Remove path and extension
            basename = filename.split('/')[-1].split('.')[0]
            
            # Parse components
            parts = basename.split('_')
            if len(parts) < 5:
                raise ValueError(f"Filename must have format: site_YYYY-MM-DD_HH_MM_SS, got: {basename}")
            
            site = parts[0]
            date_str = parts[1]
            hour = int(parts[2])
            minute = int(parts[3])
            second = int(parts[4])
            
            # Parse date
            date = datetime.strptime(date_str, "%Y-%m-%d")
            
            # Create time object
            time_obj = time(hour, minute, second)
            
            # Combine into datetime
            base_datetime = datetime.combine(date, time_obj)
            
            return base_datetime
            
        except (ValueError, IndexError) as e:
            raise ValueError(
                f"Invalid filename format: {filename}. "
                f"Expected format: site_YYYY-MM-DD_HH_MM_SS.mp4"
            ) from e
    
    def frame_to_timestamp(self, frame_number: int, base_datetime: datetime) -> datetime:
        """Convert a frame number to a timestamp.
        
        Args:
            frame_number: Frame number in the video
            base_datetime: Video start time
            
        Returns:
            datetime object for the frame
            
        Example:
            >>> base = datetime(2024, 1, 15, 14, 30, 0)
            >>> timestamp = synthesizer.frame_to_timestamp(90, base)  # Frame 90 at 30fps = 3 seconds
            >>> print(timestamp)
            2024-01-15 14:30:03
        """
        seconds = frame_number / self.fps
        return base_datetime + timedelta(seconds=seconds)
    
    def process(self, events: pd.DataFrame, filename: str, 
                output_filename: Optional[str] = None) -> pd.DataFrame:
        """Process events DataFrame to add timestamps and format for output.
        
        Args:
            events: DataFrame with columns ['frame_number', 'nest', 'action', ...]
            filename: Source video filename for timestamp calculation
            output_filename: Optional custom output filename
            
        Returns:
            DataFrame with timestamps and configured columns
            
        Example:
            >>> events = pd.DataFrame({
            ...     'frame_number': [100, 200, 300],
            ...     'nest': ['1', '2', '1'],
            ...     'action': ['Exit', 'Entry', 'Entry']
            ... })
            >>> csv_df = synthesizer.process(events, "site_2024-01-15_14_30_00.mp4")
        """
        if events.empty:
            # Return empty DataFrame with correct columns
            if self.columns is None:
                # If no columns specified, use default event columns
                default_columns = ['timestamp', 'nest_id', 'action', 'track_id', 'species', 'filename']
                empty_df = pd.DataFrame(columns=default_columns)
            else:
                empty_df = pd.DataFrame(columns=self.columns + ['filename'])
            return empty_df
        
        # Extract base datetime from filename
        base_datetime = self.extract_start_time(filename)
        
        # Add timestamps
        events = events.copy()
        events['timestamp'] = events['frame_number'].apply(
            lambda x: self.frame_to_timestamp(x, base_datetime)
        )

        #events['site'] = filename.split('/')[-1].split('_')[0]
        
        # Add filename
        #events['filename'] = output_filename if output_filename else filename
        
        # # Select and order columns
        # output_columns = self.columns + ['filename']
        # available_columns = [col for col in output_columns if col in events.columns]
        
        return events   #[available_columns]
    
    def save(self, events: pd.DataFrame, filename: str, output_path: str) -> str:
        """Process events and save to CSV file.
        
        Args:
            events: DataFrame with tracking events
            filename: Source video filename
            output_path: Path where CSV should be saved
            
        Returns:
            Path to saved CSV file
            
        Example:
            >>> csv_path = synthesizer.save(events, "site_2024-01-15_14_30_00.mp4", 
            ...                              "/output/results.csv")
        """
        processed_df = self.process(events, filename)
        processed_df.to_csv(output_path, index=False)
        return output_path
    
    def generate_csv(self, events: pd.DataFrame, video_path: str) -> pd.DataFrame:
        """Generate CSV with timestamps from events.
        
        This method is an alias for process() provided for API compatibility.
        
        Args:
            events: DataFrame with tracking events  
            video_path: Source video path for timestamp calculation
            
        Returns:
            Processed DataFrame with timestamps
            
        Example:
            >>> csv_df = synthesizer.generate_csv(events, "video.mp4")
        """
        return self.process(events, video_path)


# Alias for backward compatibility with BeeMonitor codebase
CSVGenerator = CSVSynthesizer


# Legacy function for backward compatibility
def processCSV(events: pd.DataFrame, filename: str, fps: int = 30) -> pd.DataFrame:
    """Legacy function for backward compatibility.
    
    Args:
        events: DataFrame with events
        filename: Video filename
        fps: Frames per second
        
    Returns:
        Processed DataFrame
        
    Note:
        This function is deprecated. Use CSVSynthesizer class instead.
    """
    # Create a minimal config for legacy support
    from dataclasses import dataclass, field
    
    @dataclass
    class LegacyVideoConfig:
        fps: int = fps
    
    @dataclass  
    class LegacyOutputConfig:
        csv_columns: list = field(default_factory=lambda: ["timestamp", "nest", "action"])
    
    @dataclass
    class LegacyConfig:
        video: LegacyVideoConfig = field(default_factory=LegacyVideoConfig)
        output: LegacyOutputConfig = field(default_factory=LegacyOutputConfig)
    
    config = LegacyConfig()
    synthesizer = CSVSynthesizer(config)
    return synthesizer.process(events, filename)


# Convenience function
def synthesize_csv(events: pd.DataFrame, filename: str, config, 
                   output_path: Optional[str] = None) -> pd.DataFrame:
    """Convenience function to synthesize CSV with config.
    
    Args:
        events: DataFrame with tracking events
        filename: Source video filename
        config: Config object
        output_path: Optional path to save CSV
        
    Returns:
        Processed DataFrame
        
    Example:
        >>> from config import Config
        >>> config = Config.default()
        >>> csv_df = synthesize_csv(events, "video.mp4", config, "output.csv")
    """
    synthesizer = CSVSynthesizer(config)
    processed_df = synthesizer.process(events, filename)
    
    if output_path:
        processed_df.to_csv(output_path, index=False)
    
    return processed_df