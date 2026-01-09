"""Performance Benchmarks for Detection and Tracking

Run: python tests/benchmark_performance.py
"""

import time
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List

from beemonitor.tracking import BeeTracking, DetectionMode
from beemonitor.tracking.mot import BeeTracker
from beemonitor.core.config import Config


def create_benchmark_video(output_path: str, num_frames: int = 100) -> str:
    """Create test video for benchmarking.
    
    Args:
        output_path: Path to save video
        num_frames: Number of frames to generate
        
    Returns:
        Path to created video
    """
    print(f"Creating benchmark video with {num_frames} frames...")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (1280, 720))
    
    # Create multiple moving objects
    num_objects = 5
    positions = [(np.random.randint(100, 1180), np.random.randint(100, 620)) 
                 for _ in range(num_objects)]
    velocities = [(np.random.randint(-5, 5), np.random.randint(-5, 5)) 
                  for _ in range(num_objects)]
    
    for frame_num in range(num_frames):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # Draw moving circles
        for i in range(num_objects):
            x, y = positions[i]
            vx, vy = velocities[i]
            
            # Update position
            x = (x + vx) % 1280
            y = (y + vy) % 720
            positions[i] = (x, y)
            
            # Draw bee-like object
            cv2.circle(frame, (int(x), int(y)), 25, (255, 255, 255), -1)
            cv2.circle(frame, (int(x), int(y)), 20, (200, 200, 200), -1)
            cv2.circle(frame, (int(x), int(y)), 10, (150, 150, 150), -1)
        
        out.write(frame)
    
    out.release()
    print(f"Benchmark video created: {output_path}")
    return output_path


def benchmark_detection_mode(
    video_path: str,
    mode: DetectionMode,
    config: Config,
    yolo_model = None
) -> Dict:
    """Benchmark a specific detection mode.
    
    Args:
        video_path: Path to test video
        mode: Detection mode to test
        config: Configuration object
        yolo_model: YOLO model (if required for mode)
        
    Returns:
        Dictionary with benchmark results
    """
    print(f"\nBenchmarking {mode.value}...")
    
    # Create tracker
    mot = BeeTracker(config, ['bee'])
    
    tracker = BeeTracking(
        mot_algorithm=mot,
        yolo_model=yolo_model if 'YOLO' in mode.value.upper() else None,
        detection_mode=mode,
        use_noise_filter=False,  # Disable for pure performance test
        config=config
    )
    
    # Benchmark
    start_time = time.time()
    
    try:
        results = tracker.process_video(
            video_path,
            roi=(0, 0, 1280, 720)
        )
        
        elapsed = time.time() - start_time
        stats = tracker.get_statistics()
        
        fps = stats['total_frames'] / elapsed if elapsed > 0 else 0
        
        return {
            'mode': mode.value,
            'success': True,
            'elapsed_time': elapsed,
            'fps': fps,
            'total_frames': stats['total_frames'],
            'total_detections': stats['total_detections'],
            'total_tracks': stats['total_tracks'],
            'avg_detections_per_frame': stats['total_detections'] / max(stats['total_frames'], 1)
        }
    
    except Exception as e:
        return {
            'mode': mode.value,
            'success': False,
            'error': str(e),
            'elapsed_time': time.time() - start_time
        }


def run_full_benchmark(video_path: str = None, num_frames: int = 100):
    """Run complete performance benchmark.
    
    Args:
        video_path: Path to test video (creates one if None)
        num_frames: Number of frames if creating video
    """
    print("="*80)
    print("BEEMONITOR PERFORMANCE BENCHMARK")
    print("="*80)
    
    # Create or use video
    if video_path is None:
        video_path = '/tmp/benchmark_video.mp4'
        create_benchmark_video(video_path, num_frames)
    
    # Setup
    config = Config.default()
    
    # Detection modes to test
    modes = [
        DetectionMode.FGBG_ONLY,
        DetectionMode.SIFT_ONLY,
        DetectionMode.FGBG_SIFT,
        # DetectionMode.FGBG_YOLO,  # Requires YOLO model
        # DetectionMode.FGBG_SIFT_YOLO,  # Requires YOLO model
    ]
    
    results = []
    
    # Run benchmarks
    for mode in modes:
        result = benchmark_detection_mode(video_path, mode, config)
        results.append(result)
        
        if result['success']:
            print(f"  ✓ Time: {result['elapsed_time']:.2f}s")
            print(f"  ✓ FPS: {result['fps']:.2f}")
            print(f"  ✓ Detections/frame: {result['avg_detections_per_frame']:.2f}")
        else:
            print(f"  ✗ Error: {result.get('error', 'Unknown')}")
    
    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(f"{'Mode':<20} {'Time (s)':<12} {'FPS':<10} {'Det/Frame':<12} {'Status'}")
    print("-"*80)
    
    for r in results:
        if r['success']:
            print(f"{r['mode']:<20} {r['elapsed_time']:<12.2f} {r['fps']:<10.2f} "
                  f"{r['avg_detections_per_frame']:<12.2f} ✓")
        else:
            print(f"{r['mode']:<20} {r['elapsed_time']:<12.2f} {'N/A':<10} "
                  f"{'N/A':<12} ✗")
    
    # Find fastest
    successful = [r for r in results if r['success']]
    if successful:
        fastest = min(successful, key=lambda x: x['elapsed_time'])
        print(f"\n🏆 Fastest: {fastest['mode']} ({fastest['fps']:.2f} FPS)")
        
        # Speed comparison
        print(f"\nSpeed Comparison (relative to {fastest['mode']}):")
        for r in successful:
            speedup = fastest['elapsed_time'] / r['elapsed_time']
            print(f"  {r['mode']:<20} {speedup:.2f}x")
    
    return results


def benchmark_mot_algorithms(video_path: str = None):
    """Compare different MOT algorithms.
    
    Args:
        video_path: Path to test video (creates one if None)
    """
    print("\n" + "="*80)
    print("MOT ALGORITHM COMPARISON")
    print("="*80)
    
    # Create or use video
    if video_path is None:
        video_path = '/tmp/mot_benchmark_video.mp4'
        create_benchmark_video(video_path, num_frames=100)
    
    config = Config.default()
    
    # MOT algorithms to test
    mot_configs = [
        ("BeeTracker", BeeTracker(config, ['bee'])),
        # ("ByteTrack", UltralyticsTracker(tracker_type='bytetrack.yaml')),  # If available
    ]
    
    results = []
    
    for name, mot in mot_configs:
        print(f"\nTesting {name}...")
        
        tracker = BeeTracking(
            mot_algorithm=mot,
            detection_mode=DetectionMode.FGBG_ONLY,  # Same detection for fair comparison
            config=config
        )
        
        start_time = time.time()
        
        try:
            df = tracker.process_video(video_path, roi=(0, 0, 1280, 720))
            elapsed = time.time() - start_time
            stats = tracker.get_statistics()
            
            results.append({
                'algorithm': name,
                'success': True,
                'elapsed_time': elapsed,
                'fps': stats['total_frames'] / elapsed,
                'total_tracks': stats['total_tracks']
            })
            
            print(f"  ✓ Time: {elapsed:.2f}s")
            print(f"  ✓ FPS: {stats['total_frames'] / elapsed:.2f}")
            print(f"  ✓ Total tracks: {stats['total_tracks']}")
            
        except Exception as e:
            results.append({
                'algorithm': name,
                'success': False,
                'error': str(e)
            })
            print(f"  ✗ Error: {e}")
    
    # Print comparison
    if results:
        print(f"\n{'Algorithm':<20} {'Time (s)':<12} {'FPS':<10} {'Tracks':<10} {'Status'}")
        print("-"*60)
        for r in results:
            if r['success']:
                print(f"{r['algorithm']:<20} {r['elapsed_time']:<12.2f} "
                      f"{r['fps']:<10.2f} {r['total_tracks']:<10} ✓")
            else:
                print(f"{r['algorithm']:<20} {'N/A':<12} {'N/A':<10} {'N/A':<10} ✗")


def benchmark_frame_processing():
    """Benchmark individual frame processing speed."""
    print("\n" + "="*80)
    print("FRAME PROCESSING SPEED")
    print("="*80)
    
    config = Config.default()
    
    # Create test frame
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    for i in range(10):
        x, y = np.random.randint(100, 1180), np.random.randint(100, 620)
        cv2.circle(frame, (x, y), 25, (255, 255, 255), -1)
    
    modes = [
        DetectionMode.FGBG_ONLY,
        DetectionMode.SIFT_ONLY,
        DetectionMode.FGBG_SIFT,
    ]
    
    num_iterations = 100
    
    print(f"\nProcessing {num_iterations} frames per mode...\n")
    
    for mode in modes:
        mot = BeeTracker(config, ['bee'])
        tracker = BeeTracking(
            mot_algorithm=mot,
            detection_mode=mode,
            config=config
        )
        
        start = time.time()
        
        for i in range(num_iterations):
            tracker.process_frame(frame, i)
        
        elapsed = time.time() - start
        fps = num_iterations / elapsed
        
        print(f"{mode.value:<20} {elapsed:.3f}s  ({fps:.1f} FPS)")


if __name__ == '__main__':
    import sys
    
    # Parse arguments
    video_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Run benchmarks
    print("\n🚀 Starting Performance Benchmarks...\n")
    
    # 1. Full detection mode benchmark
    results = run_full_benchmark(video_path, num_frames=100)
    
    # 2. MOT algorithm comparison
    benchmark_mot_algorithms(video_path)
    
    # 3. Frame processing speed
    benchmark_frame_processing()
    
    print("\n✅ All benchmarks complete!")
