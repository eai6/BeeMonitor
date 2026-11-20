#!/usr/bin/env python3
"""Aggregate all BeeMonitor events CSVs into a single analysis file.

Parses video filenames to extract date/time, combines with event data,
and creates a comprehensive CSV for foraging analysis.

Input:  /storage/home/eai6/CVPR_Output/*_events.csv
Output: /storage/home/eai6/CVPR_Output/aggregated_events.csv

Usage:
    python aggregate_events.py
"""

import sys
import re
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd


def parse_video_filename(filename: str) -> dict:
    """Extract date and time from video filename.
    
    Expected formats:
        mendels_2024-04-30_09_00_00.mp4
        mendels_2024-04-30_09_00_00_events.csv
    
    Returns dict with: date, time, datetime, hour, video_name
    """
    # Pattern: site_YYYY-MM-DD_HH_MM_SS
    pattern = r'(\w+)_(\d{4})-(\d{2})-(\d{2})_(\d{2})_(\d{2})_(\d{2})'
    match = re.search(pattern, filename)
    
    if not match:
        return None
    
    site, year, month, day, hour, minute, second = match.groups()
    
    try:
        dt = datetime(int(year), int(month), int(day), 
                      int(hour), int(minute), int(second))
        return {
            'site': site,
            'date': dt.date(),
            'time': dt.time(),
            'datetime': dt,
            'hour': int(hour),
            'video_name': f"{site}_{year}-{month}-{day}_{hour}_{minute}_{second}"
        }
    except ValueError:
        return None


def load_events_csv(filepath: Path) -> pd.DataFrame:
    """Load an events CSV and add video metadata."""
    try:
        df = pd.read_csv(filepath)
        
        if df.empty:
            return None
        
        # Parse video info from filename
        video_info = parse_video_filename(filepath.name)
        
        if video_info is None:
            print(f"  ⚠ Could not parse filename: {filepath.name}")
            return None
        
        # Add video metadata columns
        df['site'] = video_info['site']
        df['video_date'] = video_info['date']
        df['video_time'] = video_info['time']
        df['video_datetime'] = video_info['datetime']
        df['video_hour'] = video_info['hour']
        df['video_name'] = video_info['video_name']
        df['source_file'] = filepath.name
        
        # Calculate actual event timestamp if frame info available
        if 'frame_number' in df.columns:
            # Assume 30 fps, calculate seconds from video start
            fps = 30
            df['seconds_in_video'] = df['frame_number'] / fps
            df['event_datetime'] = df.apply(
                lambda row: video_info['datetime'] + timedelta(seconds=row['seconds_in_video']),
                axis=1
            )
        else:
            df['event_datetime'] = video_info['datetime']
        
        return df
        
    except Exception as e:
        print(f"  ✗ Error loading {filepath.name}: {e}")
        return None


def main():
    output_folder = Path("/storage/group/cmg25/default/osmia_data_processing_output")
    
    if not output_folder.exists():
        print(f"Error: Output folder not found: {output_folder}")
        sys.exit(1)
    
    # Find all events CSVs
    events_files = sorted(output_folder.glob('*_events.csv'))
    
    print(f"{'='*70}")
    print(f"BeeMonitor Events Aggregator")
    print(f"{'='*70}")
    print(f"\nInput folder: {output_folder}")
    print(f"Events files found: {len(events_files)}")
    
    if not events_files:
        print("No events files found!")
        sys.exit(1)
    
    # Load and combine all events
    print(f"\nLoading events files...")
    
    all_events = []
    files_loaded = 0
    files_empty = 0
    files_failed = 0
    total_events = 0
    
    for filepath in events_files:
        df = load_events_csv(filepath)
        
        if df is not None and len(df) > 0:
            all_events.append(df)
            files_loaded += 1
            total_events += len(df)
        elif df is not None and len(df) == 0:
            files_empty += 1
        else:
            files_failed += 1
    
    print(f"\nFiles processed:")
    print(f"  Loaded with events: {files_loaded}")
    print(f"  Empty (0 events):   {files_empty}")
    print(f"  Failed to load:     {files_failed}")
    print(f"  Total events:       {total_events:,}")
    
    if not all_events:
        print("\nNo events to aggregate!")
        sys.exit(1)
    
    # Combine all DataFrames
    print(f"\nCombining events...")
    combined = pd.concat(all_events, ignore_index=True)
    
    # Sort by event datetime
    if 'event_datetime' in combined.columns:
        combined = combined.sort_values('event_datetime')
    else:
        combined = combined.sort_values(['video_date', 'video_time'])
    
    # Create summary by date
    print(f"\n{'='*70}")
    print(f"Daily Summary")
    print(f"{'='*70}")
    
    daily_summary = combined.groupby('video_date').agg({
        'video_name': 'nunique',
        'event_datetime': 'count'
    }).rename(columns={
        'video_name': 'n_videos',
        'event_datetime': 'n_events'
    })
    
    print(f"\n{'Date':<12} {'Videos':>8} {'Events':>10}")
    print("-" * 32)
    for date, row in daily_summary.iterrows():
        print(f"{str(date):<12} {row['n_videos']:>8} {row['n_events']:>10}")
    print("-" * 32)
    print(f"{'TOTAL':<12} {daily_summary['n_videos'].sum():>8} {daily_summary['n_events'].sum():>10}")
    
    # Identify missing dates
    if len(daily_summary) > 0:
        all_dates = pd.date_range(
            start=daily_summary.index.min(),
            end=daily_summary.index.max(),
            freq='D'
        ).date
        
        present_dates = set(daily_summary.index)
        missing_dates = [d for d in all_dates if d not in present_dates]
        
        if missing_dates:
            print(f"\n⚠ Missing dates (no data/corrupted videos):")
            for d in missing_dates:
                print(f"  - {d}")
    
    # Save aggregated CSV
    output_file = "/storage/work/eai6/eai6/work/eai6/BeeMonitor/cloud_scripts/output/aggregated_events.csv"
    combined.to_csv(output_file, index=False)
    print(f"\n✓ Saved: {output_file}")
    print(f"  Rows: {len(combined):,}")
    print(f"  Columns: {list(combined.columns)}")
    
    # Also save daily summary
    summary_file = "/storage/work/eai6/eai6/work/eai6/BeeMonitor/cloud_scripts/output/daily_summary.csv"
    daily_summary.to_csv(summary_file)
    print(f"✓ Saved: {summary_file}")
    
    # Create hourly summary
    if 'video_hour' in combined.columns:
        hourly_summary = combined.groupby(['video_date', 'video_hour']).size().reset_index(name='events')
        hourly_file = "/storage/work/eai6/eai6/work/eai6/BeeMonitor/cloud_scripts/output/hourly_events.csv"
        hourly_summary.to_csv(hourly_file, index=False)
        print(f"✓ Saved: {hourly_file}")


if __name__ == '__main__':
    main()