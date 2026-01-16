"""
Utility Functions
=================

Helper functions for the GUI.
"""

from pathlib import Path
import pandas as pd
from typing import Optional, Tuple

from .constants import FRAME_COLUMN_NAMES, POSITION_COLUMN_SETS


def find_tracking_file(events_filepath: str) -> Optional[str]:
    """Try to find the tracking results file in the same directory."""
    directory = Path(events_filepath).parent
    
    possible_names = [
        'tracking_results.csv',
        'tracks.csv',
        'tracking.csv',
        'trajectories.csv'
    ]
    
    for name in possible_names:
        path = directory / name
        if path.exists():
            return str(path)
    
    for file in directory.glob('*_tracks.csv'):
        return str(file)
    for file in directory.glob('*_tracking.csv'):
        return str(file)
    
    return None


def validate_tracking_csv(df: pd.DataFrame) -> Tuple[bool, str]:
    """Validate tracking CSV has required columns."""
    if 'track_id' not in df.columns:
        return False, "Missing 'track_id' column"
    
    frame_col = None
    for col in FRAME_COLUMN_NAMES:
        if col in df.columns:
            frame_col = col
            break
    
    if frame_col is None:
        return False, f"Missing frame column (tried: {', '.join(FRAME_COLUMN_NAMES)})"
    
    has_positions = False
    for col_set in POSITION_COLUMN_SETS:
        if all(col in df.columns for col in col_set):
            has_positions = True
            break
    
    if not has_positions:
        return False, "Missing position columns (need x1,y1,x2,y2 or x,y or centroid_x,centroid_y)"
    
    return True, ""


def get_position_from_row(row: pd.Series) -> Optional[Tuple[int, int]]:
    """Extract position from a DataFrame row."""
    if all(col in row.index for col in ['x1', 'y1', 'x2', 'y2']):
        cx = int((row['x1'] + row['x2']) / 2)
        cy = int((row['y1'] + row['y2']) / 2)
        return (cx, cy)
    
    if all(col in row.index for col in ['x', 'y']):
        return (int(row['x']), int(row['y']))
    
    if all(col in row.index for col in ['centroid_x', 'centroid_y']):
        return (int(row['centroid_x']), int(row['centroid_y']))
    
    return None


def format_time(seconds: float) -> str:
    """Format seconds as MM:SS."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def format_file_size(bytes: int) -> str:
    """Format bytes as KB/MB/GB."""
    if bytes < 1024:
        return f"{bytes} B"
    elif bytes < 1024 * 1024:
        return f"{bytes / 1024:.1f} KB"
    elif bytes < 1024 * 1024 * 1024:
        return f"{bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{bytes / (1024 * 1024 * 1024):.1f} GB"