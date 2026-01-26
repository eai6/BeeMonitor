#!/usr/bin/env python3
"""Complete batch processor for BeeMonitor - CVPR evaluation.

Uses manual multithreading with explicit detection_mode for two-mode tracking.

Generates:
  1. Tracking CSV (frame-by-frame detections)
  2. Events CSV (entry/exit events at nests)
  3. Annotated video (with tracks visualized)

Usage:
    python batch_process_cvpr.py
"""

import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from beemonitor import BeeMonitor
from beemonitor.core.config import Config as BeeMonitorConfig


def process_video(monitor, video_path, output_folder):
    """Process a single video with two-mode tracking."""
    try:
        result = monitor.analyze_video(
            str(video_path),
            output_folder=str(output_folder),
            visualize=False,
            detection_mode='yolo_only'  # User's default
        )
        return video_path, result
    except Exception as e:
        print(f"  ✗ Error processing {video_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return video_path, None


def main():
    # Hardcoded paths for CVPR evaluation
    # input_folder = Path("/storage/home/eai6/CVPR_Evaluation_Video_Data")
    # output_folder = Path("/storage/home/eai6/CVPR_Output")


    input_folder = Path("/Users/edwardamoah/Documents/GitHub/BeeMonitor_eai6/data/short_videos")
    output_folder = Path("/Users/edwardamoah/Documents/GitHub/BeeMonitor_eai6/CVPR_Cloud_Output/CVPR_local")
    
    # Check input folder
    if not input_folder.exists():
        print(f"Error: Folder not found: {input_folder}")
        sys.exit(1)
    
    # Create output folder
    output_folder.mkdir(exist_ok=True, parents=True)
    
    # Count videos
    video_extensions = ('.mp4', '.MP4', '.avi', '.AVI', '.mov', '.MOV')
    videos = [f for f in input_folder.iterdir() if f.suffix in video_extensions]
    
    if not videos:
        print(f"No videos found in: {input_folder}")
        sys.exit(1)
    
    print(f"{'='*70}")
    print(f"BeeMonitor Batch Processing - CVPR Evaluation")
    print(f"{'='*70}")
    print(f"\nConfiguration:")
    print(f"  Input:  {input_folder}")
    print(f"  Output: {output_folder}")
    print(f"  Videos: {len(videos)}")
    print(f"\nOutputs per video:")
    print(f"  1. *_tracking.csv - Frame-by-frame detections")
    print(f"  2. *_events.csv - Entry/exit events at nests")
    print(f"  3. *_annotated.mp4 - Video with tracks visualized")
    
    # Load config and enable two-mode tracking
    print(f"\nLoading configuration...")
    config = BeeMonitorConfig()
    
    # Enable two-mode tracking for optimal performance
    config.tracking.enable_two_mode_tracking = True
    config.tracking.motion_detection_threshold = 1
    config.tracking.tracking_to_detection_delay = 30
    
    print(f"  ✓ Two-mode tracking: ENABLED")
    print(f"    - Motion mode: Fast detection (~20ms/frame)")
    print(f"    - Tracking mode: Full YOLO (~150ms/frame)")
    print(f"    - Switches adaptively based on activity")
    print(f"  ✓ Detection mode: yolo_only (default)")
    
    # Initialize BeeMonitor
    print(f"\nInitializing BeeMonitor...")
    monitor = BeeMonitor(config=config)
    print(f"  ✓ Nest detection model loaded")
    print(f"  ✓ Bee tracking model loaded")
    
    # Determine number of parallel workers
    max_workers = 3
    print(f"\nParallel processing:")
    print(f"  Workers: {max_workers} (processes {max_workers} videos simultaneously)")
    print(f"  Total videos: {len(videos)}")
    print(f"  Batches: ~{(len(videos) + max_workers - 1) // max_workers}")
    
    # Process videos in parallel with explicit detection_mode
    print(f"\n{'='*70}")
    print(f"Starting parallel video analysis...")
    print(f"{'='*70}\n")
    
    results = {}
    completed = 0
    
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all videos
            future_to_video = {
                executor.submit(process_video, monitor, video_path, output_folder): video_path
                for video_path in videos
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_video):
                video_path = future_to_video[future]
                video_name = video_path.name
                
                try:
                    _, result = future.result()
                    results[str(video_path)] = result
                    completed += 1
                    
                    if result is not None:
                        n_events = len(result.events)
                        n_tracks = len(result.tracks)
                        n_nests = len(result.nests.get('nests', {}))
                        print(f"[{completed}/{len(videos)}] ✓ {video_name}")
                        print(f"    Events: {n_events}, Tracks: {n_tracks}, Nests: {n_nests}")
                    else:
                        print(f"[{completed}/{len(videos)}] ⚠ {video_name} - No results")
                        
                except Exception as e:
                    results[str(video_path)] = None
                    completed += 1
                    print(f"[{completed}/{len(videos)}] ✗ {video_name} - Failed: {e}")
        
        # Summary
        print(f"\n{'='*70}")
        print(f"Processing Complete!")
        print(f"{'='*70}")
        
        successful = sum(1 for r in results.values() if r is not None)
        failed = sum(1 for r in results.values() if r is None)
        
        print(f"\nResults:")
        print(f"  Total videos:   {len(results)}")
        print(f"  Successful:     {successful}")
        print(f"  Failed:         {failed}")
        
        # File summary
        print(f"\nGenerated files:")
        tracking_csvs = list(output_folder.glob('*_tracking.csv'))
        events_csvs = list(output_folder.glob('*_events.csv'))
        annotated_videos = list(output_folder.glob('*_annotated.mp4'))
        
        print(f"  Tracking CSVs:      {len(tracking_csvs)}")
        print(f"  Events CSVs:        {len(events_csvs)}")
        print(f"  Annotated videos:   {len(annotated_videos)}")
        
        print(f"\nOutput directory: {output_folder}")
        print(f"Total size: ", end='')
        import subprocess
        subprocess.run(['du', '-sh', str(output_folder)])
        
    except Exception as e:
        print(f"\n✗ Error during batch processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()