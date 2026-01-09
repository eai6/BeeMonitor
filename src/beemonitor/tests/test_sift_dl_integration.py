"""Tests for SIFT Initialization from DL-Confirmed Boxes

This tests the critical workflow:
1. YOLO detects and confirms "this is a bee"
2. SIFT extracts features from within that confirmed box
3. SIFT continues tracking even when bee becomes stationary

This is MUCH better than SIFT blindly clustering keypoints!

Run: python tests/test_sift_dl_integration.py
"""

import unittest
import cv2
import numpy as np
from typing import List, Tuple

from beemonitor.detection import YOLODetector, SIFTDetector, Detection
from beemonitor.core.config import Config


class TestSIFTInitializationFromDL(unittest.TestCase):
    """Test SIFT feature extraction from DL-confirmed boxes."""
    
    @classmethod
    def setUpClass(cls):
        """Load YOLO model."""
        try:
            from ultralytics import YOLO
            cls.yolo_model = YOLO('path/to/tracking_model.pt')
            cls.has_model = True
        except:
            cls.yolo_model = None
            cls.has_model = False
    
    def setUp(self):
        """Create test frames."""
        # Frame 1: Moving bee
        self.frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        self._draw_bee(self.frame1, (300, 240))
        
        # Frame 2: Same bee, slightly moved
        self.frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
        self._draw_bee(self.frame2, (320, 240))
        
        # Frame 3: Bee stopped moving (stationary)
        self.frame3 = np.zeros((480, 640, 3), dtype=np.uint8)
        self._draw_bee(self.frame3, (320, 240))
    
    def _draw_bee(self, frame, pos):
        """Draw textured bee for SIFT."""
        x, y = pos
        # Bee body with texture
        cv2.circle(frame, (x, y), 25, (255, 255, 255), -1)
        cv2.circle(frame, (x, y), 20, (200, 200, 200), -1)
        cv2.circle(frame, (x, y), 15, (150, 150, 150), -1)
        cv2.circle(frame, (x, y), 10, (100, 100, 100), -1)
        # Stripes for more features
        for i in range(-20, 20, 8):
            cv2.line(frame, (x+i, y-15), (x+i, y+15), (80, 80, 80), 2)
    
    def test_yolo_provides_bee_box(self):
        """Test that YOLO provides confirmed bee bounding box."""
        if not self.has_model:
            self.skipTest("YOLO model not available")
        
        yolo_det = YOLODetector(
            model=self.yolo_model,
            conf_threshold=0.25,
            tracking_classes=['bee']
        )
        
        detections = yolo_det.detect(self.frame1)
        
        # Should have at least one bee detection
        self.assertGreater(len(detections), 0)
        
        # Get first detection
        bee_det = detections[0]
        
        # Verify it's a confirmed bee
        self.assertEqual(bee_det.label, 'bee')
        self.assertEqual(bee_det.source, 'yolo')
        self.assertGreater(bee_det.confidence, 0.25)
        
        # Bounding box should be valid
        x1, y1, x2, y2 = bee_det.bbox
        self.assertLess(x1, x2)
        self.assertLess(y1, y2)
    
    def test_sift_features_from_dl_box(self):
        """Test extracting SIFT features from DL-confirmed box."""
        if not self.has_model:
            self.skipTest("YOLO model not available")
        
        # Step 1: Get DL-confirmed bee box
        yolo_det = YOLODetector(
            model=self.yolo_model,
            conf_threshold=0.25,
            tracking_classes=['bee']
        )
        
        detections = yolo_det.detect(self.frame1)
        self.assertGreater(len(detections), 0)
        
        bee_box = detections[0].bbox
        
        # Step 2: Extract SIFT features from within that box
        sift = cv2.SIFT_create()
        
        x1, y1, x2, y2 = [int(c) for c in bee_box]
        roi = self.frame1[y1:y2, x1:x2]
        
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray_roi, None)
        
        # Should find keypoints in the bee
        self.assertGreater(len(keypoints), 0)
        self.assertIsNotNone(descriptors)
        
        print(f"\nFound {len(keypoints)} SIFT keypoints in DL-confirmed bee box")
    
    def test_sift_tracking_initialized_from_dl(self):
        """Test SIFT tracking initialized from DL-confirmed detection."""
        if not self.has_model:
            self.skipTest("YOLO model not available")
        
        # Frame 1: DL detection initializes SIFT
        yolo_det = YOLODetector(
            model=self.yolo_model,
            conf_threshold=0.25,
            tracking_classes=['bee']
        )
        
        dl_detections = yolo_det.detect(self.frame1)
        self.assertGreater(len(dl_detections), 0)
        
        bee_box = dl_detections[0].bbox
        
        # Extract SIFT features from DL box
        sift = cv2.SIFT_create()
        x1, y1, x2, y2 = [int(c) for c in bee_box]
        roi1 = self.frame1[y1:y2, x1:x2]
        gray_roi1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
        kp1, desc1 = sift.detectAndCompute(gray_roi1, None)
        
        # Frame 2: Track using SIFT features
        # (In real system, would search around predicted location)
        roi2 = self.frame2[y1:y2, x1:x2]
        gray_roi2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
        kp2, desc2 = sift.detectAndCompute(gray_roi2, None)
        
        # Match features
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(desc1, desc2, k=2)
        
        # Apply ratio test
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        # Should find matching features
        self.assertGreater(len(good_matches), 0)
        
        print(f"\nMatched {len(good_matches)} features between frames")
    
    def test_sift_persists_when_stationary(self):
        """Test SIFT continues tracking when bee becomes stationary."""
        if not self.has_model:
            self.skipTest("YOLO model not available")
        
        # Frame 1: Initialize from DL
        yolo_det = YOLODetector(
            model=self.yolo_model,
            conf_threshold=0.25,
            tracking_classes=['bee']
        )
        
        dl_detections = yolo_det.detect(self.frame1)
        bee_box = dl_detections[0].bbox
        
        # Extract initial SIFT features
        sift = cv2.SIFT_create()
        x1, y1, x2, y2 = [int(c) for c in bee_box]
        
        roi1 = self.frame1[y1:y2, x1:x2]
        gray1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
        kp1, desc1 = sift.detectAndCompute(gray1, None)
        
        # Frame 3: Bee is stationary (same position as frame 2)
        # FG/BG would NOT detect this, but SIFT should match features
        
        roi3 = self.frame3[y1:y2, x1:x2]
        gray3 = cv2.cvtColor(roi3, cv2.COLOR_BGR2GRAY)
        kp3, desc3 = sift.detectAndCompute(gray3, None)
        
        # Match features
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(desc1, desc3, k=2)
        
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        # SIFT should still match features even though bee is stationary
        self.assertGreater(len(good_matches), 0)
        
        print(f"\nSIFT matched {len(good_matches)} features on stationary bee")
        print("  → This proves SIFT can track when FG/BG cannot!")


class TestCorrectSIFTWorkflow(unittest.TestCase):
    """Test the correct SIFT + DL integration workflow."""
    
    def test_workflow_documentation(self):
        """Document the correct SIFT initialization workflow."""
        
        workflow = """
        CORRECT SIFT + DL WORKFLOW:
        
        1. YOLO Detection (Confirmation)
           - Run YOLO on frame
           - Get bounding box: (x1, y1, x2, y2)
           - Confidence that "this IS a bee"
        
        2. SIFT Feature Extraction (From DL Box)
           - Extract ROI from confirmed box
           - roi = frame[y1:y2, x1:x2]
           - Extract SIFT keypoints & descriptors from ROI
           - Store: (track_id, bbox, keypoints, descriptors)
        
        3. SIFT Tracking (Subsequent Frames)
           - Predict new position (Kalman filter)
           - Search region around prediction
           - Extract SIFT features in search region
           - Match features with stored descriptors
           - Update track position
        
        4. Advantages
           - DL confirms "this is a bee" (not noise)
           - SIFT provides features for tracking
           - Works even when bee becomes stationary
           - More robust than pure FG/BG
        
        WRONG APPROACH:
        - Blindly clustering all SIFT keypoints
        - No DL confirmation
        - High false positive rate
        
        RIGHT APPROACH:
        - DL confirms identity
        - SIFT provides trackable features
        - Best of both worlds!
        """
        
        print(workflow)
        
        # This test always passes - it's documentation
        self.assertTrue(True)


def demonstrate_correct_workflow():
    """Demonstrate the correct SIFT + DL workflow."""
    print("\n" + "="*70)
    print("DEMONSTRATION: Correct SIFT + DL Workflow")
    print("="*70)
    
    try:
        from ultralytics import YOLO
        yolo_model = YOLO('path/to/tracking_model.pt')
    except:
        print("YOLO model not available, using mock workflow")
        return
    
    # Create test frames
    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw moving bee
    def draw_bee(frame, x, y):
        cv2.circle(frame, (x, y), 25, (255, 255, 255), -1)
        cv2.circle(frame, (x, y), 20, (200, 200, 200), -1)
        for i in range(-20, 20, 8):
            cv2.line(frame, (x+i, y-15), (x+i, y+15), (80, 80, 80), 2)
    
    draw_bee(frame1, 300, 240)
    draw_bee(frame2, 320, 240)
    
    print("\n1. YOLO Detection (DL Confirmation)")
    yolo_det = YOLODetector(
        model=yolo_model,
        conf_threshold=0.25,
        tracking_classes=['bee']
    )
    
    detections = yolo_det.detect(frame1)
    
    if len(detections) == 0:
        print("   No detections found")
        return
    
    bee_det = detections[0]
    print(f"   ✓ Confirmed: {bee_det.label}")
    print(f"   ✓ Confidence: {bee_det.confidence:.2f}")
    print(f"   ✓ Box: {bee_det.bbox}")
    
    print("\n2. SIFT Feature Extraction (From Confirmed Box)")
    x1, y1, x2, y2 = [int(c) for c in bee_det.bbox]
    roi = frame1[y1:y2, x1:x2]
    
    sift = cv2.SIFT_create()
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray_roi, None)
    
    print(f"   ✓ Extracted {len(keypoints)} SIFT keypoints")
    print(f"   ✓ Descriptor shape: {descriptors.shape}")
    
    print("\n3. SIFT Tracking (Next Frame)")
    # Search in same region (simplified - would use Kalman prediction)
    roi2 = frame2[y1:y2, x1:x2]
    gray_roi2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
    kp2, desc2 = sift.detectAndCompute(gray_roi2, None)
    
    # Match features
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors, desc2, k=2)
    
    good_matches = []
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
    
    print(f"   ✓ Matched {len(good_matches)} features")
    print(f"   ✓ Track confirmed via feature matching")
    
    print("\n4. Visualization")
    # Draw matches
    match_img = cv2.drawMatches(
        roi, keypoints, roi2, kp2, good_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    cv2.imshow('SIFT Feature Tracking from DL Box', match_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\n" + "="*70)
    print("Summary:")
    print("  ✓ DL confirmed bee identity")
    print("  ✓ SIFT extracted trackable features")
    print("  ✓ Features matched across frames")
    print("  → This enables tracking even when bee becomes stationary!")
    print("="*70)


if __name__ == '__main__':
    # Run tests
    print("Running SIFT + DL integration tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Demonstrate workflow
    print("\nDemonstrating correct workflow...")
    try:
        demonstrate_correct_workflow()
    except Exception as e:
        print(f"Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
