#!/usr/bin/env python3
"""
Direct YOLO test - bypass all initialization logic
"""

import sys
from pathlib import Path

project_root = Path('/Users/edwardamoah/Documents/GitHub/BeeMonitor_eai6')
sys.path.insert(0, str(project_root))

from ultralytics import YOLO
from beemonitor.detection import YOLODetector
import cv2

video_path = project_root / 'data/mendels_2024-05-16_18_10_00.mp4'

print("="*70)
print("Direct YOLO Test")
print("="*70)

# Test 1: Can we open the video?
print(f"\n1. Testing video file: {video_path}")
cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    print("❌ Cannot open video!")
    exit(1)

ret, frame = cap.read()
if not ret:
    print("❌ Cannot read frame!")
    exit(1)

print(f"✓ Video opened: {frame.shape}")
cap.release()

# Test 2: Load YOLO model
print("\n2. Loading YOLO model...")
try:
    yolo_model = YOLO('/Users/edwardamoah/Documents/GitHub/BeeMonitor_eai6/models/bee_tracking_back_up_Full_Mode.pt')
    print(f"✓ YOLO model loaded: {yolo_model}")
except Exception as e:
    print(f"❌ Cannot load YOLO: {e}")
    exit(1)

# Test 3: Test different confidence thresholds
print("\n3. Testing YOLO with different configurations...")

configs = [
    {'conf': 0.25, 'classes': None, 'desc': 'conf=0.25, all classes'},
    {'conf': 0.25, 'classes': [0], 'desc': 'conf=0.25, class 0 (person)'},
    {'conf': 0.1, 'classes': None, 'desc': 'conf=0.1, all classes'},
    {'conf': 0.01, 'classes': None, 'desc': 'conf=0.01, all classes (very low)'},
]

cap = cv2.VideoCapture(str(video_path))
ret, frame = cap.read()
cap.release()

for config in configs:
    print(f"\n  Testing: {config['desc']}")
    
    # Direct YOLO prediction
    results = yolo_model.predict(
        frame,
        conf=config['conf'],
        classes=config['classes'],
        verbose=False
    )
    
    if len(results) > 0 and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        print(f"    ✓ Found {len(boxes)} detections")
        
        # Show first few detections
        for i, box in enumerate(boxes[:3]):
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = yolo_model.names[cls]
            print(f"      Detection {i+1}: class={class_name} (id={cls}), conf={conf:.3f}")
    else:
        print(f"    ✗ Found 0 detections")

# Test 4: Test with YOLODetector wrapper
print("\n4. Testing YOLODetector wrapper...")

test_configs = [
    {'classes': ['bee'], 'conf': 0.15},
    {'classes': ['osmia_cornifrons'], 'conf': 0.15},
    {'classes': None, 'conf': 0.15},
    {'classes': ['person'], 'conf': 0.15},
    {'classes': [0], 'conf': 0.15},  # person class ID
]

cap = cv2.VideoCapture(str(video_path))
ret, frame = cap.read()
cap.release()

for tc in test_configs:
    desc = f"classes={tc['classes']}, conf={tc['conf']}"
    print(f"\n  Testing: {desc}")
    
    try:
        detector = YOLODetector(
            model=yolo_model,
            conf_threshold=tc['conf'],
            tracking_classes=tc['classes']
        )
        
        dets = detector.detect(frame)
        print(f"    ✓ Found {len(dets)} detections")
        
        for i, det in enumerate(dets[:3]):
            print(f"      Detection {i+1}: label={det.label}, conf={det.confidence:.3f}")
            
    except Exception as e:
        print(f"    ❌ Error: {e}")

print("\n" + "="*70)
print("Summary:")
print("  If all tests show 0 detections → Video has no detectable objects")
print("  If some tests work → Configuration issue (wrong classes)")
print("="*70)