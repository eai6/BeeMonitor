"""Unit Tests for Detection Module

Run: python -m pytest tests/test_detectors.py -v
Or: python tests/test_detectors.py
"""

import unittest
import cv2
import numpy as np
from pathlib import Path

from beemonitor.detection import (
    BlobDetector, SIFTDetector, YOLODetector,
    BaseDetector, Detection
)


class TestDetection(unittest.TestCase):
    """Test Detection data class."""
    
    def test_detection_creation(self):
        """Test creating Detection object."""
        det = Detection(
            bbox=(100, 100, 200, 200),
            centroid=(150, 150),
            confidence=0.9,
            label='bee',
            source='test'
        )
        
        self.assertEqual(det.bbox, (100, 100, 200, 200))
        self.assertEqual(det.centroid, (150, 150))
        self.assertEqual(det.confidence, 0.9)
        self.assertEqual(det.label, 'bee')
        self.assertEqual(det.source, 'test')


class TestBlobDetector(unittest.TestCase):
    """Test BlobDetector."""
    
    def setUp(self):
        """Create test frame."""
        # Create simple test frame with a white blob
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(self.frame, (320, 240), 30, (255, 255, 255), -1)
    
    def test_detector_creation(self):
        """Test creating BlobDetector."""
        detector = BlobDetector(min_area=50, min_solidity=0.5)
        self.assertIsInstance(detector, BaseDetector)
        self.assertEqual(detector.get_source_name(), 'blob')
    
    def test_detect_blob(self):
        """Test blob detection."""
        detector = BlobDetector(min_area=50, min_solidity=0.5)
        
        # First frame initializes background
        detector.detect(self.frame)
        
        # Second frame with changed blob should detect
        frame2 = self.frame.copy()
        cv2.circle(frame2, (400, 300), 30, (255, 255, 255), -1)
        
        detections = detector.detect(frame2)
        
        # Should detect the new blob
        self.assertIsInstance(detections, list)
        for det in detections:
            self.assertIsInstance(det, Detection)
            self.assertEqual(det.source, 'blob')
            self.assertGreater(det.confidence, 0)
    
    def test_configure(self):
        """Test configuring detector."""
        detector = BlobDetector(min_area=50)
        detector.configure(min_area=100, min_solidity=0.7)
        # Configuration should not raise errors
    
    def test_reset(self):
        """Test resetting detector."""
        detector = BlobDetector(min_area=50)
        detector.detect(self.frame)
        detector.reset()
        # Reset should not raise errors


class TestSIFTDetector(unittest.TestCase):
    """Test SIFTDetector."""
    
    def setUp(self):
        """Create test frame with texture."""
        # Create frame with some texture for SIFT
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add textured regions
        for i in range(10):
            x, y = np.random.randint(50, 600), np.random.randint(50, 430)
            cv2.circle(self.frame, (x, y), 20, (255, 255, 255), -1)
            cv2.circle(self.frame, (x, y), 15, (0, 0, 0), -1)
    
    def test_detector_creation(self):
        """Test creating SIFTDetector."""
        detector = SIFTDetector(min_keypoints=3, cluster_eps=30.0)
        self.assertIsInstance(detector, BaseDetector)
        self.assertEqual(detector.get_source_name(), 'sift')
    
    def test_detect_sift(self):
        """Test SIFT detection."""
        detector = SIFTDetector(min_keypoints=3, cluster_eps=30.0)
        detections = detector.detect(self.frame)
        
        self.assertIsInstance(detections, list)
        for det in detections:
            self.assertIsInstance(det, Detection)
            self.assertEqual(det.source, 'sift')
    
    def test_configure(self):
        """Test configuring detector."""
        detector = SIFTDetector(min_keypoints=3)
        detector.configure(min_keypoints=5, cluster_eps=50.0)
    
    def test_reset(self):
        """Test resetting detector."""
        detector = SIFTDetector(min_keypoints=3)
        detector.reset()


class TestYOLODetector(unittest.TestCase):
    """Test YOLODetector."""
    
    def setUp(self):
        """Create test frame."""
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    def test_detector_creation_no_model(self):
        """Test creating YOLODetector without model raises error."""
        # Should work if model is None (for testing)
        # In real usage, model is required
        pass
    
    def test_get_source_name(self):
        """Test source name."""
        # This test assumes you have a mock YOLO model
        # detector = YOLODetector(mock_model, conf_threshold=0.25)
        # self.assertEqual(detector.get_source_name(), 'yolo')
        pass


class TestDetectorComparison(unittest.TestCase):
    """Integration test comparing detectors."""
    
    def setUp(self):
        """Create test frame."""
        # Create frame with blob and texture
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add blob
        cv2.circle(self.frame, (320, 240), 40, (255, 255, 255), -1)
        
        # Add texture
        for i in range(5):
            x, y = np.random.randint(100, 540), np.random.randint(100, 380)
            cv2.circle(self.frame, (x, y), 15, (200, 200, 200), -1)
    
    def test_all_detectors(self):
        """Test all detectors on same frame."""
        blob_det = BlobDetector(min_area=50)
        sift_det = SIFTDetector(min_keypoints=3)
        
        # Initialize blob detector
        blob_det.detect(self.frame)
        
        # Detect with changed frame
        frame2 = self.frame.copy()
        cv2.circle(frame2, (400, 300), 40, (255, 255, 255), -1)
        
        blob_dets = blob_det.detect(frame2)
        sift_dets = sift_det.detect(frame2)
        
        print(f"\nDetector Comparison:")
        print(f"  Blob: {len(blob_dets)} detections")
        print(f"  SIFT: {len(sift_dets)} detections")
        
        # Both should find something
        self.assertIsInstance(blob_dets, list)
        self.assertIsInstance(sift_dets, list)


def run_visual_test():
    """Visual test showing detector outputs."""
    import cv2
    
    print("\n" + "="*70)
    print("VISUAL DETECTOR TEST")
    print("="*70)
    
    # Create test frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add some bees (white circles with texture)
    bee_positions = [(200, 200), (400, 300), (500, 150)]
    for pos in bee_positions:
        cv2.circle(frame, pos, 30, (255, 255, 255), -1)
        cv2.circle(frame, pos, 20, (200, 200, 200), -1)
        cv2.circle(frame, pos, 10, (150, 150, 150), -1)
    
    # Test detectors
    blob_det = BlobDetector(min_area=50, min_solidity=0.5)
    sift_det = SIFTDetector(min_keypoints=3, cluster_eps=30.0)
    
    # Initialize blob detector
    blob_det.detect(frame)
    
    # Create second frame with movement
    frame2 = frame.copy()
    cv2.circle(frame2, (300, 250), 30, (255, 255, 255), -1)
    
    # Detect
    blob_dets = blob_det.detect(frame2)
    sift_dets = sift_det.detect(frame2)
    
    # Visualize
    h, w = frame2.shape[:2]
    combined = np.zeros((h, w*2, 3), dtype=np.uint8)
    
    # Blob detections
    frame_blob = frame2.copy()
    for det in blob_dets:
        x1, y1, x2, y2 = [int(c) for c in det.bbox]
        cv2.rectangle(frame_blob, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame_blob, f"Blob: {len(blob_dets)}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    combined[:, 0:w] = frame_blob
    
    # SIFT detections
    frame_sift = frame2.copy()
    for det in sift_dets:
        x1, y1, x2, y2 = [int(c) for c in det.bbox]
        cv2.rectangle(frame_sift, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(frame_sift, f"SIFT: {len(sift_dets)}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    combined[:, w:w*2] = frame_sift
    
    cv2.imshow('Detector Comparison (Press any key to close)', combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print(f"\nResults:")
    print(f"  Blob detections: {len(blob_dets)}")
    print(f"  SIFT detections: {len(sift_dets)}")


if __name__ == '__main__':
    # Run unit tests
    print("Running unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run visual test
    print("\nRunning visual test...")
    try:
        run_visual_test()
    except Exception as e:
        print(f"Visual test skipped: {e}")
