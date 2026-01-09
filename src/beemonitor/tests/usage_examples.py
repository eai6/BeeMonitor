"""Usage examples for BeeMonitor tracking system."""

from motion_tracking import MotionTracking
from bee_tracker import BeeTracker
from ultralytics_tracker import UltralyticsTracker, create_bytetrack, create_botsort
from beemonitor.core.config import Config
from ultralytics import YOLO


# ============================================================================
# Example 1: Basic usage with default Kalman tracker
# ============================================================================
def example_kalman_tracker():
    """Use default Kalman-based BeeTracker."""
    
    config = Config.load('config.yaml')
    model = YOLO('yolov8n.pt')
    
    # Initialize with default BeeTracker (Kalman)
    tracker = MotionTracking(model, config)
    
    # Process video
    results = tracker.process_video(
        'bee_video.mp4',
        roi_coords=(100, 100, 800, 600),
        visualize=True
    )
    
    print(f"Tracked {len(results)} tracks")
    return results


# ============================================================================
# Example 2: ByteTrack (fast, good for occlusion)
# ============================================================================
def example_bytetrack():
    """Use ByteTrack from Ultralytics."""
    
    config = Config.load('config.yaml')
    model = YOLO('yolov8n.pt')
    
    # Create ByteTrack tracker
    bytetrack = create_bytetrack(
        model=model,
        tracking_classes=['bee', 'wasp'],
        conf=0.25,
        iou=0.7
    )
    
    # Use with motion detection
    tracker = MotionTracking(model, config, mot_algorithm=bytetrack)
    
    results = tracker.process_video('bee_video.mp4')
    
    print(f"ByteTrack: {len(results)} tracks")
    return results


# ============================================================================
# Example 3: BoT-SORT (appearance features, re-identification)
# ============================================================================
def example_botsort():
    """Use BoT-SORT from Ultralytics."""
    
    config = Config.load('config.yaml')
    model = YOLO('yolov8n.pt')
    
    # Create BoT-SORT tracker
    botsort = create_botsort(
        model=model,
        tracking_classes=['bee'],
        conf=0.3,
        iou=0.7
    )
    
    # Use with motion detection
    tracker = MotionTracking(model, config, mot_algorithm=botsort)
    
    results = tracker.process_video('bee_video.mp4')
    
    print(f"BoT-SORT: {len(results)} tracks")
    return results


# ============================================================================
# Example 4: Custom Kalman tracker configuration
# ============================================================================
def example_custom_kalman():
    """Create custom BeeTracker with specific settings."""
    
    config = Config.load('config.yaml')
    model = YOLO('yolov8n.pt')
    
    # Create custom BeeTracker
    custom_tracker = BeeTracker(
        config=config,
        tracking_classes=['bee']  # Only track bees
    )
    
    # Use with motion detection
    tracker = MotionTracking(model, config, mot_algorithm=custom_tracker)
    
    results = tracker.process_video('bee_video.mp4')
    
    print(f"Custom Kalman: {len(results)} tracks")
    return results


# ============================================================================
# Example 5: Custom Ultralytics tracker config
# ============================================================================
def example_custom_ultralytics():
    """Use custom Ultralytics tracker configuration."""
    
    config = Config.load('config.yaml')
    model = YOLO('yolov8n.pt')
    
    # Use custom tracker YAML
    custom_tracker = UltralyticsTracker(
        model=model,
        tracker_config='path/to/custom_tracker.yaml',
        tracking_classes=['bee', 'wasp'],
        conf_threshold=0.25,
        iou_threshold=0.7
    )
    
    tracker = MotionTracking(model, config, mot_algorithm=custom_tracker)
    
    results = tracker.process_video('bee_video.mp4')
    
    print(f"Custom Ultralytics: {len(results)} tracks")
    return results


# ============================================================================
# Example 6: A/B testing different trackers
# ============================================================================
def example_compare_trackers():
    """Compare performance of different trackers."""
    
    config = Config.load('config.yaml')
    model = YOLO('yolov8n.pt')
    video_path = 'bee_video.mp4'
    
    # Define trackers to compare
    trackers = {
        'Kalman (default)': None,  # Will use default BeeTracker
        'ByteTrack': create_bytetrack(model, ['bee']),
        'BoT-SORT': create_botsort(model, ['bee']),
        'Custom Kalman': BeeTracker(config, ['bee'])
    }
    
    results = {}
    
    for name, mot_algorithm in trackers.items():
        print(f"\nTesting {name}...")
        
        tracker = MotionTracking(
            model,
            config,
            mot_algorithm=mot_algorithm,
            fast_mode=True
        )
        
        result_df = tracker.process_video(video_path)
        
        num_tracks = len(result_df)
        results[name] = {
            'tracks': num_tracks,
            'dataframe': result_df
        }
        
        print(f"{name}: {num_tracks} tracks detected")
    
    # Summary
    print("\n" + "="*50)
    print("COMPARISON SUMMARY")
    print("="*50)
    for name, data in results.items():
        print(f"{name:20s}: {data['tracks']} tracks")
    
    return results


# ============================================================================
# Example 7: Device-specific configuration
# ============================================================================
def example_device_config():
    """Configure specific device for inference."""
    
    config = Config.load('config.yaml')
    model = YOLO('yolov8n.pt')
    
    # Auto-detect (default)
    tracker_auto = MotionTracking(model, config)  # Will use MPS/CUDA/CPU
    
    # Force GPU
    tracker_gpu = MotionTracking(model, config, use_gpu=True)
    
    # Force CPU
    tracker_cpu = MotionTracking(model, config, use_gpu=False)
    
    print(f"Auto device: {tracker_auto.device}")
    print(f"GPU device: {tracker_gpu.device}")
    print(f"CPU device: {tracker_cpu.device}")
    
    # Process with auto-detected device
    results = tracker_auto.process_video('bee_video.mp4')
    
    return results


# ============================================================================
# Example 8: Fast mode vs quality mode
# ============================================================================
def example_speed_modes():
    """Compare fast mode vs quality mode."""
    
    config = Config.load('config.yaml')
    model = YOLO('yolov8n.pt')
    
    # Fast mode (default)
    tracker_fast = MotionTracking(
        model, config,
        fast_mode=True  # 0.5x resolution, simplified morphology
    )
    
    # Quality mode
    tracker_quality = MotionTracking(
        model, config,
        fast_mode=False  # Full resolution, more iterations
    )
    
    print("Processing with fast mode...")
    results_fast = tracker_fast.process_video('bee_video.mp4')
    
    print("Processing with quality mode...")
    results_quality = tracker_quality.process_video('bee_video.mp4')
    
    print(f"Fast mode: {len(results_fast)} tracks")
    print(f"Quality mode: {len(results_quality)} tracks")
    
    return results_fast, results_quality


# ============================================================================
# Example 9: Implement custom MOT algorithm
# ============================================================================
def example_custom_mot():
    """Implement custom tracking algorithm."""
    
    from base_mot import BaseMOT, Detection, Track
    from typing import Dict, List
    
    class SimpleCentroidTracker(BaseMOT):
        """Simple centroid-based tracker."""
        
        def __init__(self, max_distance=50.0):
            self.tracks = {}
            self.next_id = 0
            self.max_distance = max_distance
        
        def predict(self, frame_num):
            return self.get_tracks()
        
        def update(self, detections: List[Detection], frame_num: int):
            # Simple nearest-neighbor matching
            if not self.tracks:
                # Initialize tracks
                for det in detections:
                    self.tracks[self.next_id] = Track(
                        track_id=self.next_id,
                        bbox=det.bbox,
                        centroid=det.centroid,
                        label=det.label,
                        age=0,
                        frames_without_detection=0,
                        last_confirmation_frame=frame_num,
                        trajectory=[(frame_num, det.centroid)]
                    )
                    self.next_id += 1
            else:
                # Match to existing tracks (simplified)
                # In real implementation, use Hungarian algorithm
                pass
            
            return self.get_tracks()
        
        def get_tracks(self):
            return self.tracks.copy()
        
        def reset(self):
            self.tracks.clear()
            self.next_id = 0
        
        def get_search_regions(self, frame_num):
            return {}
    
    # Use custom tracker
    config = Config.load('config.yaml')
    model = YOLO('yolov8n.pt')
    
    custom_mot = SimpleCentroidTracker(max_distance=50.0)
    tracker = MotionTracking(model, config, mot_algorithm=custom_mot)
    
    results = tracker.process_video('bee_video.mp4')
    return results


if __name__ == '__main__':
    # Run examples
    print("Example 1: Kalman Tracker")
    example_kalman_tracker()
    
    print("\nExample 2: ByteTrack")
    example_bytetrack()
    
    print("\nExample 3: BoT-SORT")
    example_botsort()
    
    print("\nExample 6: Compare All Trackers")
    example_compare_trackers()
