#!/usr/bin/env python3
"""Quick test to verify CNN filter works on one frame."""

import sys
import cv2
from pathlib import Path

VIDEO_PATH = "mendels_2024-05-23_18_20_25.mp4"
FRAME_NUM = 100

print("="*70)
print("QUICK FILTER TEST")
print("="*70)

print(f"\n1. Loading frame {FRAME_NUM}...")
if not Path(VIDEO_PATH).exists():
    print(f"   ✗ Video not found: {VIDEO_PATH}")
    print("   Update VIDEO_PATH in this script")
    sys.exit(1)

cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_NUM)
ret, frame = cap.read()
cap.release()

if not ret:
    print(f"   ✗ Could not read frame")
    sys.exit(1)
print(f"   ✓ Frame loaded: {frame.shape}")

print("\n2. Detecting blobs...")
try:
    from beemonitor.detection import BlobDetector
    
    blob = BlobDetector(min_area=200, min_solidity=0.5)
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    frames = []
    for i in range(30):
        ret, f = cap.read()
        if ret:
            frames.append(f)
    cap.release()
    blob.initialize_background(frames)
    
    detections = blob.detect(frame)
    print(f"   ✓ Raw blobs: {len(detections)}")
except Exception as e:
    print(f"   ✗ Blob detection failed: {e}")
    sys.exit(1)

print("\n3. Testing CNN filter...")
try:
    from beemonitor.detection.noise_filter import BeeNoiseFilter
    
    model_path = "models/blob_noise_classifier.pth"
    if not Path(model_path).exists():
        print(f"   ✗ CNN model not found: {model_path}")
        print("\n   You need the CNN model file!")
        sys.exit(1)
    
    print(f"   ✓ Model exists: {model_path}")
    
    cnn = BeeNoiseFilter(model_path=model_path, noise_threshold=0.9)
    print(f"   ✓ CNN loaded (device: {cnn.device})")
    
    filtered = cnn.filter_detections(frame, detections)
    removed = len(detections) - len(filtered)
    reduction_pct = (removed / len(detections) * 100) if len(detections) > 0 else 0
    
    print(f"\n   ✓✓✓ CNN FILTER WORKING ✓✓✓")
    print(f"   Before: {len(detections)}")
    print(f"   After:  {len(filtered)}")
    print(f"   Removed: {removed} ({reduction_pct:.1f}%)")
    
    if reduction_pct > 50:
        print("\n   ✓ Filter working well (>50% reduction)")
    else:
        print("\n   ⚠ Lower than expected (~66% expected)")
        
except Exception as e:
    print(f"   ✗ CNN filter failed: {e}")
    import traceback
    print(traceback.format_exc())
    sys.exit(1)

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)
print(f"""
✓ Blob detection: {len(detections)} raw blobs
✓ CNN filter: {len(filtered)} after filtering ({reduction_pct:.1f}% reduction)

If this works but analysis doesn't:
1. Install updated video_analyzer.py
2. Use detection_mode='fgbg'
3. Restart Python completely
4. Check logs for "CNN NOISE FILTER ENABLED"
""")