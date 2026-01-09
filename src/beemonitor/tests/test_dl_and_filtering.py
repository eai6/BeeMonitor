"""Tests for Deep Learning Detection and Noise Filtering

Tests the YOLO detector and CNN noise filter components.

Run: python tests/test_dl_and_filtering.py
"""

import unittest
import cv2
import numpy as np
from pathlib import Path
from typing import List

from beemonitor.detection import YOLODetector, NoiseFilter, BlobDetector, Detection
from beemonitor.core.config import Config


class TestYOLODetector(unittest.TestCase):
    """Test YOLO-based deep learning detection."""
    
    @classmethod
    def setUpClass(cls):
        """Load YOLO model once for all tests."""
        try:
            from ultralytics import YOLO
            # Try to get model path from config or find available model
            config = Config.default()
            model_path = None
            
            # Try config path first
            if hasattr(config.models, 'tracking'):
                config_path = config.models.tracking
                if Path(config_path).exists():
                    model_path = config_path
            
            # If config path doesn't exist, try relative paths
            if model_path is None:
                alt_paths = [
                    'models/bee_tracking.pt',
                    'models/bee_tracking_back_up.pt',
                    'models/bee_tracking_back_up_Full_Mode.pt',
                    'models/bee_detection.pt',
                ]
                for path in alt_paths:
                    if Path(path).exists():
                        model_path = path
                        break
            
            if model_path is None:
                raise FileNotFoundError("No YOLO model file found. Please ensure a model file exists in models/ directory")
            
            cls.yolo_model = YOLO(model_path)
            cls.has_model = True
            print(f"Loaded YOLO model from: {model_path}")
        except Exception as e:
            cls.yolo_model = None
            cls.has_model = False
            print(f"WARNING: YOLO model not available ({e}), tests will be skipped")
    
    def setUp(self):
        """Create test frame."""
        # Create frame with bee-like objects
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw some bee-like objects
        for i in range(3):
            x, y = 200 + i * 150, 240
            cv2.circle(self.frame, (x, y), 25, (255, 255, 255), -1)
            cv2.circle(self.frame, (x, y), 20, (200, 200, 200), -1)
            cv2.circle(self.frame, (x, y), 10, (150, 150, 150), -1)
    
    def test_yolo_detector_creation(self):
        """Test creating YOLO detector."""
        if not self.has_model:
            self.skipTest("YOLO model not available")
        
        detector = YOLODetector(
            model=self.yolo_model,
            conf_threshold=0.25,
            tracking_classes=['bee', 'wasp']
        )
        
        self.assertEqual(detector.get_source_name(), 'yolo')
        self.assertIsNotNone(detector.model)
    
    def test_yolo_detection(self):
        """Test YOLO detection on frame."""
        if not self.has_model:
            self.skipTest("YOLO model not available")
        
        detector = YOLODetector(
            model=self.yolo_model,
            conf_threshold=0.25,
            tracking_classes=['bee']
        )
        
        detections = detector.detect(self.frame)
        
        # Verify detection format
        self.assertIsInstance(detections, list)
        for det in detections:
            self.assertIsInstance(det, Detection)
            self.assertEqual(det.source, 'yolo')
            self.assertGreater(det.confidence, 0.25)
            self.assertIn(det.label, ['bee', 'wasp'])
    
    def test_yolo_confidence_filtering(self):
        """Test YOLO confidence threshold filtering."""
        if not self.has_model:
            self.skipTest("YOLO model not available")
        
        # Low threshold
        detector_low = YOLODetector(
            model=self.yolo_model,
            conf_threshold=0.1,
            tracking_classes=['bee']
        )
        
        # High threshold
        detector_high = YOLODetector(
            model=self.yolo_model,
            conf_threshold=0.8,
            tracking_classes=['bee']
        )
        
        dets_low = detector_low.detect(self.frame)
        dets_high = detector_high.detect(self.frame)
        
        # High threshold should filter more
        self.assertGreaterEqual(len(dets_low), len(dets_high))
    
    def test_yolo_class_filtering(self):
        """Test YOLO class filtering."""
        if not self.has_model:
            self.skipTest("YOLO model not available")
        
        # Only track bees
        detector = YOLODetector(
            model=self.yolo_model,
            conf_threshold=0.25,
            tracking_classes=['bee']  # Not 'wasp'
        )
        
        detections = detector.detect(self.frame)
        
        # All detections should be 'bee'
        for det in detections:
            self.assertEqual(det.label, 'bee')
    
    def test_yolo_configure(self):
        """Test reconfiguring YOLO detector."""
        if not self.has_model:
            self.skipTest("YOLO model not available")
        
        detector = YOLODetector(
            model=self.yolo_model,
            conf_threshold=0.25
        )
        
        # Reconfigure
        detector.configure(
            conf_threshold=0.5,
            tracking_classes=['bee', 'wasp', 'fly']
        )
        
        # Should not raise errors
        detections = detector.detect(self.frame)
        self.assertIsInstance(detections, list)


class TestNoiseFilter(unittest.TestCase):
    """Test CNN-based noise filtering."""
    
    def setUp(self):
        """Create test frame and detections."""
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Create mock detections (some real, some noise)
        self.detections = [
            Detection(
                bbox=(100, 100, 150, 150),
                centroid=(125, 125),
                confidence=0.9,
                label='bee',
                source='blob'
            ),
            Detection(
                bbox=(200, 200, 220, 220),  # Small noise
                centroid=(210, 210),
                confidence=0.8,
                label='bee',
                source='blob'
            ),
            Detection(
                bbox=(300, 300, 350, 350),
                centroid=(325, 325),
                confidence=0.85,
                label='bee',
                source='blob'
            ),
        ]
    
    def test_noise_filter_creation(self):
        """Test creating noise filter."""
        # Mock classifier
        class MockClassifier:
            def predict(self, crops):
                # Return random probabilities
                return np.random.rand(len(crops))
        
        filter = NoiseFilter(
            classifier=MockClassifier(),
            threshold=0.7
        )
        
        self.assertIsNotNone(filter.classifier)
        self.assertEqual(filter.threshold, 0.7)
    
    def test_noise_filter_filtering(self):
        """Test noise filtering on detections."""
        # Mock classifier that marks second detection as noise
        class MockClassifier:
            def predict(self, crops):
                # First and third are bees, second is noise
                return np.array([0.9, 0.3, 0.85])
        
        filter = NoiseFilter(
            classifier=MockClassifier(),
            threshold=0.7
        )
        
        filtered = filter.filter_detections(self.frame, self.detections)
        
        # Should keep first and third, remove second
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0].centroid, (125, 125))
        self.assertEqual(filtered[1].centroid, (325, 325))
    
    def test_noise_filter_threshold(self):
        """Test different threshold values."""
        class MockClassifier:
            def predict(self, crops):
                return np.array([0.9, 0.6, 0.8])
        
        # Low threshold - keep more
        filter_low = NoiseFilter(
            classifier=MockClassifier(),
            threshold=0.5
        )
        
        # High threshold - keep fewer
        filter_high = NoiseFilter(
            classifier=MockClassifier(),
            threshold=0.75
        )
        
        filtered_low = filter_low.filter_detections(self.frame, self.detections)
        filtered_high = filter_high.filter_detections(self.frame, self.detections)
        
        self.assertGreaterEqual(len(filtered_low), len(filtered_high))
    
    def test_noise_filter_empty_input(self):
        """Test noise filter with no detections."""
        class MockClassifier:
            def predict(self, crops):
                return np.array([])
        
        filter = NoiseFilter(
            classifier=MockClassifier(),
            threshold=0.7
        )
        
        filtered = filter.filter_detections(self.frame, [])
        
        self.assertEqual(len(filtered), 0)


class TestDLWithNoiseFilterPipeline(unittest.TestCase):
    """Test integration of YOLO detection + noise filtering."""
    
    @classmethod
    def setUpClass(cls):
        """Load models."""
        try:
            from ultralytics import YOLO
            # Try to get model path from config or find available model
            config = Config.default()
            model_path = None
            
            # Try config path first
            if hasattr(config.models, 'tracking'):
                config_path = config.models.tracking
                if Path(config_path).exists():
                    model_path = config_path
            
            # If config path doesn't exist, try relative paths
            if model_path is None:
                alt_paths = [
                    'models/bee_tracking.pt',
                    'models/bee_tracking_back_up.pt',
                    'models/bee_tracking_back_up_Full_Mode.pt',
                    'models/bee_detection.pt',
                ]
                for path in alt_paths:
                    if Path(path).exists():
                        model_path = path
                        break
            
            if model_path is None:
                raise FileNotFoundError("No YOLO model file found. Please ensure a model file exists in models/ directory")
            
            cls.yolo_model = YOLO(model_path)
            cls.has_yolo = True
            print(f"Loaded YOLO model from: {model_path}")
        except Exception as e:
            cls.yolo_model = None
            cls.has_yolo = False
            print(f"WARNING: YOLO model not available ({e})")
    
    def setUp(self):
        """Create test frame."""
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for i in range(5):
            x, y = 100 + i * 100, 240
            cv2.circle(self.frame, (x, y), 20, (255, 255, 255), -1)
    
    def test_dl_then_noise_filter(self):
        """Test YOLO detection followed by noise filtering."""
        if not self.has_yolo:
            self.skipTest("YOLO model not available")
        
        # YOLO detection
        yolo_det = YOLODetector(
            model=self.yolo_model,
            conf_threshold=0.25,
            tracking_classes=['bee']
        )
        
        detections = yolo_det.detect(self.frame)
        
        print(f"YOLO found {len(detections)} detections")
        
        # Apply noise filter
        class MockClassifier:
            def predict(self, crops):
                # Randomly mark some as noise
                return np.random.rand(len(crops))
        
        noise_filter = NoiseFilter(
            classifier=MockClassifier(),
            threshold=0.7
        )
        
        filtered = noise_filter.filter_detections(self.frame, detections)
        
        print(f"After filtering: {len(filtered)} detections")
        
        # Filtered should be <= original
        self.assertLessEqual(len(filtered), len(detections))


def run_visual_test():
    """Visual test comparing YOLO with and without noise filter."""
    print("\n" + "="*70)
    print("VISUAL TEST - DL Detection + Noise Filtering")
    print("="*70)
    
    try:
        from ultralytics import YOLO
        # Try to get model path from config or find available model
        config = Config.default()
        model_path = None
        
        # Try config path first
        if hasattr(config.models, 'tracking'):
            config_path = config.models.tracking
            if Path(config_path).exists():
                model_path = config_path
        
        # If config path doesn't exist, try relative paths
        if model_path is None:
            alt_paths = [
                'models/bee_tracking.pt',
                'models/bee_tracking_back_up.pt',
                'models/bee_tracking_back_up_Full_Mode.pt',
                'models/bee_detection.pt',
            ]
            for path in alt_paths:
                if Path(path).exists():
                    model_path = path
                    break
        
        if model_path is None:
            raise FileNotFoundError("No YOLO model file found. Please ensure a model file exists in models/ directory")
        
        yolo_model = YOLO(model_path)
        print(f"Loaded YOLO model from: {model_path}")
    except Exception as e:
        print(f"YOLO model not available ({e}), skipping visual test")
        return
    
    # Create test frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add bees and noise
    for i in range(3):
        # Real bees
        x, y = 100 + i * 150, 200
        cv2.circle(frame, (x, y), 25, (255, 255, 255), -1)
        cv2.circle(frame, (x, y), 20, (200, 200, 200), -1)
    
    # Add noise (small blobs)
    for i in range(5):
        x, y = np.random.randint(50, 600), np.random.randint(300, 450)
        cv2.circle(frame, (x, y), 8, (180, 180, 180), -1)
    
    # YOLO detection
    yolo_det = YOLODetector(
        model=yolo_model,
        conf_threshold=0.25,
        tracking_classes=['bee']
    )
    
    detections = yolo_det.detect(frame)
    
    # Create mock noise filter
    class MockClassifier:
        def predict(self, crops):
            # Simulate: larger objects = bees, small = noise
            probs = []
            for crop in crops:
                h, w = crop.shape[:2]
                area = h * w
                prob = min(area / 1000.0, 1.0)  # Larger = higher prob
                probs.append(prob)
            return np.array(probs)
    
    noise_filter = NoiseFilter(
        classifier=MockClassifier(),
        threshold=0.6
    )
    
    filtered = noise_filter.filter_detections(frame, detections)
    
    # Visualize
    vis = np.hstack([frame.copy(), frame.copy()])
    h, w = frame.shape[:2]
    
    # Left: All YOLO detections
    for det in detections:
        x1, y1, x2, y2 = [int(c) for c in det.bbox]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(vis, f"YOLO: {len(detections)}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Right: After noise filtering
    for det in filtered:
        x1, y1, x2, y2 = [int(c) for c in det.bbox]
        x1 += w  # Offset for right side
        x2 += w
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(vis, f"Filtered: {len(filtered)}", (w + 10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    cv2.imshow('DL Detection + Noise Filter', vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print(f"\nResults:")
    print(f"  YOLO detections: {len(detections)}")
    print(f"  After filtering: {len(filtered)}")
    print(f"  Removed: {len(detections) - len(filtered)}")


if __name__ == '__main__':
    # Run unit tests
    print("Running DL and noise filter tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run visual test
    print("\nRunning visual test...")
    try:
        run_visual_test()
    except Exception as e:
        print(f"Visual test failed: {e}")
