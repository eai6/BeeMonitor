#!/usr/bin/env python3
"""Batch process bee hotel videos.

Process all videos in a folder and save tracking results.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional
import time
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from beemonitor.core.config import BeeMonitorConfig
from beemonitor.detection import BlobDetector, YOLODetector
from beemonitor.tracking import BeeTracking, DetectionMode
from beemonitor.tracking.mot import BeeTracker


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_videos(folder: Path, extensions: List[str] = None) -> List[Path]:
    """Find all video files in folder.
    
    Args:
        folder: Folder to search
        extensions: Video file extensions (default: common formats)
        
    Returns:
        List of video file paths
    """
    if extensions is None:
        extensions = ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV']
    
    videos = []
    for ext in extensions:
        videos.extend(folder.glob(f'*{ext}'))
    
    return sorted(videos)


def process_single_video(
    video_path: Path,
    output_folder: Path,
    config: BeeMonitorConfig,
    yolo_model,
    blob_detector: Optional[BlobDetector] = None,
    roi: Optional[tuple] = None
) -> dict:
    """Process a single video.
    
    Args:
        video_path: Path to video
        output_folder: Folder for results
        config: Configuration
        yolo_model: YOLO model instance
        blob_detector: Pre-initialized blob detector (optional)
        roi: Region of interest (optional)
        
    Returns:
        Dictionary with results summary
    """
    start_time = time.time()
    video_name = video_path.stem
    
    logger.info(f"Processing: {video_name}")
    
    try:
        # Create output folder for this video
        video_output = output_folder / video_name
        video_output.mkdir(parents=True, exist_ok=True)
        
        # Initialize blob detector if not provided
        if blob_detector is None:
            logger.info(f"  Initializing blob detector...")
            blob_detector = BlobDetector()
            yolo = YOLODetector(yolo_model, tracking_classes=['bee'])
            
            num_clean = blob_detector.initialize_from_video_with_verification(
                video_path=str(video_path),
                yolo_detector=yolo,
                num_frames=100,
                max_detections=0
            )
            logger.info(f"  Background initialized ({num_clean} clean frames)")
        
        # Create tracking system
        mot = BeeTracker(config, tracking_classes=['bee'])
        
        tracker = BeeTracking(
            mot_algorithm=mot,
            yolo_model=yolo_model,
            detection_mode=DetectionMode.FGBG_YOLO,
            config=config
        )
        
        tracker.blob_detector = blob_detector
        
        # Process video
        logger.info(f"  Tracking bees...")
        results = tracker.process_video(
            video_path=str(video_path),
            roi=roi
        )
        
        # Save results
        csv_path = video_output / f"{video_name}_tracking.csv"
        results.to_csv(csv_path, index=False)
        logger.info(f"  Saved: {csv_path}")
        
        # Get statistics
        stats = tracker.get_statistics()
        processing_time = time.time() - start_time
        
        summary = {
            'video': video_name,
            'success': True,
            'total_frames': stats['total_frames'],
            'total_detections': stats['total_detections'],
            'unique_tracks': results['track_id'].nunique() if len(results) > 0 else 0,
            'processing_time': processing_time,
            'csv_path': str(csv_path)
        }
        
        if 'mode_switches' in stats:
            summary['mode_switches'] = stats['mode_switches']
            summary['frames_in_motion_mode'] = stats.get('frames_in_motion_mode', 0)
            summary['frames_in_tracking_mode'] = stats.get('frames_in_tracking_mode', 0)
        
        logger.info(f"  ✓ Complete: {stats['total_frames']} frames, "
                   f"{summary['unique_tracks']} tracks, "
                   f"{processing_time:.1f}s")
        
        return summary
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"  ✗ Failed: {e}")
        return {
            'video': video_name,
            'success': False,
            'error': str(e),
            'processing_time': processing_time
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Batch process bee hotel videos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all videos in folder
  python batch_process.py /path/to/videos
  
  # Process with specific ROI
  python batch_process.py /path/to/videos --roi 100 100 800 600
  
  # Use custom output folder
  python batch_process.py /path/to/videos --output /path/to/results
  
  # Shared background (all videos from same camera)
  python batch_process.py /path/to/videos --shared-background
        """
    )
    
    parser.add_argument(
        'input_folder',
        type=str,
        help='Folder containing video files'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output folder (default: input_folder/results)'
    )
    
    parser.add_argument(
        '--roi',
        type=int,
        nargs=4,
        metavar=('X1', 'Y1', 'X2', 'Y2'),
        default=None,
        help='Region of interest (x1 y1 x2 y2)'
    )
    
    parser.add_argument(
        '--shared-background',
        action='store_true',
        help='Use same background model for all videos (same camera/location)'
    )
    
    parser.add_argument(
        '--yolo-model',
        type=str,
        default=None,
        help='YOLO model path (default: uses config.models.tracking)'
    )
    
    parser.add_argument(
        '--extensions',
        type=str,
        nargs='+',
        default=None,
        help='Video file extensions (default: .mp4 .avi .mov .mkv)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        logging.getLogger('beemonitor').setLevel(logging.DEBUG)
    
    # Validate input folder
    input_folder = Path(args.input_folder)
    if not input_folder.exists():
        logger.error(f"Input folder not found: {input_folder}")
        sys.exit(1)
    
    # Setup output folder
    if args.output:
        output_folder = Path(args.output)
    else:
        output_folder = input_folder / 'results'
    
    output_folder.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output folder: {output_folder}")
    
    # Find videos
    videos = find_videos(input_folder, args.extensions)
    
    if not videos:
        logger.error(f"No video files found in: {input_folder}")
        sys.exit(1)
    
    logger.info(f"Found {len(videos)} video(s)")
    
    # Convert ROI
    roi = tuple(args.roi) if args.roi else None
    if roi:
        logger.info(f"ROI: {roi}")
    
    # Create config
    config = BeeMonitorConfig()
    
    # Load YOLO model
    if args.yolo_model:
        yolo_model_path = args.yolo_model
        logger.info(f"Using specified YOLO model: {yolo_model_path}")
    else:
        yolo_model_path = config.models.tracking
        logger.info(f"Using config YOLO model: {yolo_model_path}")
    
    logger.info(f"Loading YOLO model...")
    try:
        from ultralytics import YOLO
        yolo_model = YOLO(yolo_model_path)
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        sys.exit(1)
    
    # Initialize shared background if requested
    shared_blob_detector = None
    if args.shared_background and len(videos) > 1:
        logger.info("Initializing shared background model...")
        shared_blob_detector = BlobDetector()
        yolo = YOLODetector(yolo_model, tracking_classes=['bee'])
        
        # Use first video for background
        num_clean = shared_blob_detector.initialize_from_video_with_verification(
            video_path=str(videos[0]),
            yolo_detector=yolo,
            num_frames=100,
            max_detections=0
        )
        logger.info(f"Shared background ready ({num_clean} clean frames)")
    
    # Process videos
    logger.info("\n" + "="*70)
    logger.info("BATCH PROCESSING")
    logger.info("="*70)
    
    all_results = []
    
    for video_path in tqdm(videos, desc="Processing videos"):
        result = process_single_video(
            video_path=video_path,
            output_folder=output_folder,
            config=config,
            yolo_model=yolo_model,
            blob_detector=shared_blob_detector,
            roi=roi
        )
        all_results.append(result)
    
    # Generate summary
    logger.info("\n" + "="*70)
    logger.info("SUMMARY")
    logger.info("="*70)
    
    successful = [r for r in all_results if r['success']]
    failed = [r for r in all_results if not r['success']]
    
    logger.info(f"Total videos: {len(videos)}")
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(failed)}")
    
    if successful:
        total_frames = sum(r['total_frames'] for r in successful)
        total_tracks = sum(r['unique_tracks'] for r in successful)
        total_time = sum(r['processing_time'] for r in successful)
        
        logger.info(f"\nTotal frames processed: {total_frames}")
        logger.info(f"Total unique tracks: {total_tracks}")
        logger.info(f"Total processing time: {total_time:.1f}s ({total_time/60:.1f} min)")
        logger.info(f"Average per video: {total_time/len(successful):.1f}s")
        
        if total_frames > 0:
            logger.info(f"Average per frame: {total_time/total_frames*1000:.1f}ms")
    
    if failed:
        logger.info("\nFailed videos:")
        for r in failed:
            logger.info(f"  ✗ {r['video']}: {r['error']}")
    
    # Save summary CSV
    summary_df = pd.DataFrame(all_results)
    summary_path = output_folder / f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"\nSummary saved: {summary_path}")
    
    logger.info("\n" + "="*70)
    logger.info("COMPLETE")
    logger.info("="*70)
    
    return 0 if len(failed) == 0 else 1


if __name__ == '__main__':
    sys.exit(main())