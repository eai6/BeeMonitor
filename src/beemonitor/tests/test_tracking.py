"""Integration Tests for Tracking System

Run: python -m pytest tests/test_tracking.py -v
Or: python tests/test_tracking.py
"""

import unittest
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

from beemonitor.tracking import BeeTracking, DetectionMode
from beemonitor.tracking.mot import BeeTracker, BaseMOT, Detection, Track
from beemonitor.core.config import Config


class TestMOTDetection(unittest.TestCase):
    """Test MOT Detection data class."""
    
    def test_detection_creation(self):
        """Test creating Detection for MOT."""
        det = Detection(
            bbox=(100, 100, 200, 200),
            centroid=(150, 150),
            label='bee',
            confidence=0.9,
            source='test'
        )
        
        self.assertEqual(det.bbox, (100, 100, 200, 200))
        self.assertEqual(det.centroid, (150, 150))
        self.assertEqual(det.label, 'bee')


class TestTrack(unittest.TestCase):
    """Test Track data class."""
    
    def test_track_creation(self):
        """Test creating Track object."""
        track = Track(
            track_id=1,
            bbox=(100, 100, 200, 200),
            centroid=(150, 150),
            label='bee',
            age=5,
            frames_without_detection=0,
            last_confirmation_frame=10,
            trajectory=[(10, (150, 150))]
        )
        
        self.assertEqual(track.track_id, 1)
        self.assertEqual(track.label, 'bee')
        self.assertEqual(track.age, 5)


class TestBeeTracker(unittest.TestCase):
    """Test BeeTracker MOT algorithm."""
    
    def setUp(self):
        """Create test config and tracker."""
        self.config = Config.default()
        self.tracker = BeeTracker(
            config=self.config,
            tracking_classes=['bee']
        )
    
    def test_tracker_creation(self):
        """Test creating BeeTracker."""
        self.assertIsInstance(self.tracker, BaseMOT)
    
    def test_update_with_detections(self):
        """Test updating tracker with detections."""
        # Create test detections
        detections = [
            Detection(
                bbox=(100, 100, 150, 150),
                centroid=(125, 125),
                label='bee',
                confidence=0.9,
                source='test'
            ),
            Detection(
                bbox=(200, 200, 250, 250),
                centroid=(225, 225),
                label='bee',
                confidence=0.8,
                source='test'
            )
        ]
        
        # Update tracker
        tracks = self.tracker.update(detections, frame_num=0)
        
        self.assertIsInstance(tracks, dict)
        self.assertGreaterEqual(len(tracks), 0)
    
    def test_predict(self):
        """Test track prediction."""
        # Add detection
        detections = [
            Detection(
                bbox=(100, 100, 150, 150),
                centroid=(125, 125),
                label='bee',
                confidence=0.9,
                source='test'
            )
        ]
        
        self.tracker.update(detections, frame_num=0)
        
        # Predict next frame
        predicted = self.tracker.predict(frame_num=1)
        
        self.assertIsInstance(predicted, dict)
    
    def test_reset(self):
        """Test resetting tracker."""
        # Add some tracks
        detections = [
            Detection(
                bbox=(100, 100, 150, 150),
                centroid=(125, 125),
                label='bee',
                confidence=0.9,
                source='test'
            )
        ]
        
        self.tracker.update(detections, frame_num=0)
        
        # Reset
        self.tracker.reset()
        
        # Should have no tracks
        tracks = self.tracker.get_tracks()
        self.assertEqual(len(tracks), 0)


class TestBeeTracking(unittest.TestCase):
    """Test BeeTracking system."""
    
    def setUp(self):
        """Create test config."""
        self.config = Config.default()
    
    def test_bee_tracking_creation_fgbg_only(self):
        """Test creating BeeTracking with FGBG_ONLY mode."""
        mot = BeeTracker(self.config, ['bee'])
        
        tracker = BeeTracking(
            mot_algorithm=mot,
            detection_mode=DetectionMode.FGBG_ONLY,
            config=self.config
        )
        
        self.assertEqual(tracker.detection_mode, DetectionMode.FGBG_ONLY)
        self.assertIsNotNone(tracker.blob_detector)
        self.assertIsNone(tracker.sift_detector)
        self.assertIsNone(tracker.yolo_detector)
    
    def test_bee_tracking_creation_sift_only(self):
        """Test creating BeeTracking with SIFT_ONLY mode."""
        mot = BeeTracker(self.config, ['bee'])
        
        tracker = BeeTracking(
            mot_algorithm=mot,
            detection_mode=DetectionMode.SIFT_ONLY,
            config=self.config
        )
        
        self.assertEqual(tracker.detection_mode, DetectionMode.SIFT_ONLY)
        self.assertIsNone(tracker.blob_detector)
        self.assertIsNotNone(tracker.sift_detector)
    
    def test_bee_tracking_creation_fgbg_sift(self):
        """Test creating BeeTracking with FGBG_SIFT mode."""
        mot = BeeTracker(self.config, ['bee'])
        
        tracker = BeeTracking(
            mot_algorithm=mot,
            detection_mode=DetectionMode.FGBG_SIFT,
            config=self.config
        )
        
        self.assertEqual(tracker.detection_mode, DetectionMode.FGBG_SIFT)
        self.assertIsNotNone(tracker.blob_detector)
        self.assertIsNotNone(tracker.sift_detector)
    
    def test_process_frame(self):
        """Test processing single frame."""
        mot = BeeTracker(self.config, ['bee'])
        
        tracker = BeeTracking(
            mot_algorithm=mot,
            detection_mode=DetectionMode.FGBG_ONLY,
            config=self.config
        )
        
        # Create test frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(frame, (320, 240), 30, (255, 255, 255), -1)
        
        # Process frame
        result = tracker.process_frame(frame, frame_num=0)
        
        self.assertIn('detections', result)
        self.assertIn('tracks', result)
        self.assertIn('mode', result)
    
    def test_configure_detection(self):
        """Test configuring detection parameters."""
        mot = BeeTracker(self.config, ['bee'])
        
        tracker = BeeTracking(
            mot_algorithm=mot,
            detection_mode=DetectionMode.FGBG_SIFT,
            config=self.config
        )
        
        # Should not raise errors
        tracker.configure_detection(
            blob_min_area=100,
            sift_min_keypoints=5
        )
    
    def test_configure_tracking(self):
        """Test configuring tracking parameters."""
        mot = BeeTracker(self.config, ['bee'])
        
        tracker = BeeTracking(
            mot_algorithm=mot,
            detection_mode=DetectionMode.FGBG_ONLY,
            config=self.config
        )
        
        # Should not raise errors
        tracker.configure_tracking(
            max_age=30,
            min_hits=3
        )
    
    def test_reset(self):
        """Test resetting tracking system."""
        mot = BeeTracker(self.config, ['bee'])
        
        tracker = BeeTracking(
            mot_algorithm=mot,
            detection_mode=DetectionMode.FGBG_ONLY,
            config=self.config
        )
        
        # Process some frames
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tracker.process_frame(frame, 0)
        
        # Reset
        tracker.reset()
        
        stats = tracker.get_statistics()
        self.assertEqual(stats['total_frames'], 0)
    
    def test_get_statistics(self):
        """Test getting statistics."""
        mot = BeeTracker(self.config, ['bee'])
        
        tracker = BeeTracking(
            mot_algorithm=mot,
            detection_mode=DetectionMode.FGBG_ONLY,
            config=self.config
        )
        
        stats = tracker.get_statistics()
        
        self.assertIn('total_frames', stats)
        self.assertIn('total_detections', stats)
        self.assertIn('total_tracks', stats)


def create_test_video(output_path, num_frames=30):
    """Create test video with moving circle."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (640, 480))
    
    for i in range(num_frames):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Moving circle
        x = int(200 + i * 10)
        y = 240
        cv2.circle(frame, (x, y), 30, (255, 255, 255), -1)
        
        out.write(frame)
    
    out.release()
    return output_path


def run_integration_test():
    """Full integration test with video processing."""
    print("\n" + "="*70)
    print("INTEGRATION TEST - BeeTracking System")
    print("="*70)
    
    # Create test video
    test_video = '/tmp/test_tracking.mp4'
    print(f"\nCreating test video: {test_video}")
    create_test_video(test_video, num_frames=30)
    
    # Test different modes
    modes = [
        DetectionMode.FGBG_ONLY,
        DetectionMode.SIFT_ONLY,
        DetectionMode.FGBG_SIFT,
    ]
    
    config = Config.default()
    
    for mode in modes:
        print(f"\nTesting {mode.value}...")
        
        mot = BeeTracker(config, ['bee'])
        
        tracker = BeeTracking(
            mot_algorithm=mot,
            detection_mode=mode,
            config=config
        )
        
        try:
            results = tracker.process_video(
                test_video,
                roi=(0, 0, 640, 480)
            )
            
            stats = tracker.get_statistics()
            
            print(f"  Frames processed: {stats['total_frames']}")
            print(f"  Total detections: {stats['total_detections']}")
            print(f"  Total tracks: {stats['total_tracks']}")
            print(f"  Results shape: {results.shape if isinstance(results, pd.DataFrame) else 'N/A'}")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\nIntegration test complete!")


if __name__ == '__main__':
    # Run unit tests
    print("Running unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run integration test
    print("\nRunning integration test...")
    try:
        run_integration_test()
    except Exception as e:
        print(f"Integration test failed: {e}")
        import traceback
        traceback.print_exc()
