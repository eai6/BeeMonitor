#!/usr/bin/env python3
"""
Filter Order Comparison Test

Tests whether geometric filtering should happen before or after noise filtering.

Order 1: Geometric → Noise (current)
Order 2: Noise → Geometric (reversed)

Measures: Recall, Precision, F1, Speed
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import time

project_root = Path('/Users/edwardamoah/Documents/GitHub/BeeMonitor_eai6')
sys.path.insert(0, str(project_root / 'src'))

from ultralytics import YOLO
from beemonitor.detection import BlobDetector, YOLODetector, BeeNoiseFilter

def calculate_iou(bbox1, bbox2):
    """Calculate IoU between two bboxes."""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def test_geometric_then_noise(
    blob_detector, noise_filter, yolo_detector,
    video_path, test_frames, start_frame, iou_threshold,
    min_solidity
):
    """Order 1: Geometric filter → Noise filter (CURRENT)"""
    
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    total_raw = 0
    total_after_geometric = 0
    total_after_noise = 0
    total_yolo = 0
    total_matched = 0
    total_time = 0.0
    
    for i in range(test_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.time()
        
        # Get YOLO ground truth
        yolo_dets = yolo_detector.detect(frame)
        total_yolo += len(yolo_dets)
        
        # Step 1: Blob detection with geometric filter
        blob_dets_raw = blob_detector.detect(frame, min_solidity=0.0)  # No filter
        total_raw += len(blob_dets_raw)
        
        # Apply geometric filter
        blob_dets_geometric = [d for d in blob_dets_raw if d.metadata.get('solidity', 0) >= min_solidity]
        total_after_geometric += len(blob_dets_geometric)
        
        # Step 2: Apply noise filter
        blob_dets_final = noise_filter.filter_detections(frame, blob_dets_geometric)
        total_after_noise += len(blob_dets_final)
        
        elapsed = time.time() - start_time
        total_time += elapsed
        
        # Match with YOLO
        for blob_det in blob_dets_final:
            for yolo_det in yolo_dets:
                if calculate_iou(blob_det.bbox, yolo_det.bbox) >= iou_threshold:
                    total_matched += 1
                    break
    
    cap.release()
    
    recall = (total_matched / total_yolo * 100) if total_yolo > 0 else 0
    precision = (total_matched / total_after_noise * 100) if total_after_noise > 0 else 0
    f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) > 0 else 0
    
    return {
        'order': 'Geometric → Noise',
        'raw': total_raw,
        'after_geometric': total_after_geometric,
        'after_noise': total_after_noise,
        'yolo': total_yolo,
        'matched': total_matched,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'time_sec': total_time,
        'time_per_frame': total_time / test_frames if test_frames > 0 else 0
    }


def test_noise_then_geometric(
    blob_detector, noise_filter, yolo_detector,
    video_path, test_frames, start_frame, iou_threshold,
    min_solidity
):
    """Order 2: Noise filter → Geometric filter (REVERSED)"""
    
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    total_raw = 0
    total_after_noise = 0
    total_after_geometric = 0
    total_yolo = 0
    total_matched = 0
    total_time = 0.0
    
    for i in range(test_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.time()
        
        # Get YOLO ground truth
        yolo_dets = yolo_detector.detect(frame)
        total_yolo += len(yolo_dets)
        
        # Step 1: Blob detection (no geometric filter)
        blob_dets_raw = blob_detector.detect(frame, min_solidity=0.0)
        total_raw += len(blob_dets_raw)
        
        # Step 2: Apply noise filter FIRST
        blob_dets_noise = noise_filter.filter_detections(frame, blob_dets_raw)
        total_after_noise += len(blob_dets_noise)
        
        # Step 3: Apply geometric filter
        blob_dets_final = [d for d in blob_dets_noise if d.metadata.get('solidity', 0) >= min_solidity]
        total_after_geometric += len(blob_dets_final)
        
        elapsed = time.time() - start_time
        total_time += elapsed
        
        # Match with YOLO
        for blob_det in blob_dets_final:
            for yolo_det in yolo_dets:
                if calculate_iou(blob_det.bbox, yolo_det.bbox) >= iou_threshold:
                    total_matched += 1
                    break
    
    cap.release()
    
    recall = (total_matched / total_yolo * 100) if total_yolo > 0 else 0
    precision = (total_matched / total_after_geometric * 100) if total_after_geometric > 0 else 0
    f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) > 0 else 0
    
    return {
        'order': 'Noise → Geometric',
        'raw': total_raw,
        'after_noise': total_after_noise,
        'after_geometric': total_after_geometric,
        'yolo': total_yolo,
        'matched': total_matched,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'time_sec': total_time,
        'time_per_frame': total_time / test_frames if test_frames > 0 else 0
    }


print("="*70)
print("Filter Order Comparison Test")
print("="*70)

# Configuration
video_path = project_root / 'data/mendels_2024-05-23_18_20_25.mp4'
model_path = 'models/bee_tracking_back_up_Full_Mode.pt'

BLOB_INIT_FRAMES = 30
TEST_FRAMES = 150
TEST_START_FRAME = 180
IOU_THRESHOLD = 0.15
NOISE_THRESHOLD = 0.6
MIN_SOLIDITY = 0.613  # Winner from ablation study (80% of learned 0.767)

print(f"\nConfiguration:")
print(f"  Video: {video_path.name}")
print(f"  Test frames: {TEST_START_FRAME}-{TEST_START_FRAME + TEST_FRAMES}")
print(f"  IoU threshold: {IOU_THRESHOLD}")
print(f"  Noise threshold: {NOISE_THRESHOLD}")
print(f"  Min solidity: {MIN_SOLIDITY}")

# Initialize
print(f"\nInitializing...")
yolo_model = YOLO(model_path)
yolo_detector = YOLODetector(yolo_model, tracking_classes=['bee'])

noise_filter = BeeNoiseFilter(
    model_path='models/blob_noise_classifier.pth',
    noise_threshold=NOISE_THRESHOLD
)

blob_detector = BlobDetector()
blob_detector.initialize_from_video_with_verification(
    video_path=str(video_path),
    yolo_detector=yolo_detector,
    num_frames=BLOB_INIT_FRAMES
)

print(f"\n{'='*70}")
print("Testing Filter Orders")
print(f"{'='*70}\n")

# Test Order 1: Geometric → Noise (CURRENT)
print("Order 1: Geometric filter → Noise filter (CURRENT)")
result1 = test_geometric_then_noise(
    blob_detector=blob_detector,
    noise_filter=noise_filter,
    yolo_detector=yolo_detector,
    video_path=str(video_path),
    test_frames=TEST_FRAMES,
    start_frame=TEST_START_FRAME,
    iou_threshold=IOU_THRESHOLD,
    min_solidity=MIN_SOLIDITY
)

print(f"  Raw blobs: {result1['raw']}")
print(f"  After geometric: {result1['after_geometric']} (removed {result1['raw'] - result1['after_geometric']})")
print(f"  After noise: {result1['after_noise']} (removed {result1['after_geometric'] - result1['after_noise']})")
print(f"  Matched: {result1['matched']}/{result1['yolo']}")
print(f"  Recall: {result1['recall']:.1f}%")
print(f"  Precision: {result1['precision']:.1f}%")
print(f"  F1 Score: {result1['f1']:.1f}%")
print(f"  Time: {result1['time_sec']:.2f}s ({result1['time_per_frame']*1000:.1f}ms/frame)")

print(f"\nOrder 2: Noise filter → Geometric filter (REVERSED)")
result2 = test_noise_then_geometric(
    blob_detector=blob_detector,
    noise_filter=noise_filter,
    yolo_detector=yolo_detector,
    video_path=str(video_path),
    test_frames=TEST_FRAMES,
    start_frame=TEST_START_FRAME,
    iou_threshold=IOU_THRESHOLD,
    min_solidity=MIN_SOLIDITY
)

print(f"  Raw blobs: {result2['raw']}")
print(f"  After noise: {result2['after_noise']} (removed {result2['raw'] - result2['after_noise']})")
print(f"  After geometric: {result2['after_geometric']} (removed {result2['after_noise'] - result2['after_geometric']})")
print(f"  Matched: {result2['matched']}/{result2['yolo']}")
print(f"  Recall: {result2['recall']:.1f}%")
print(f"  Precision: {result2['precision']:.1f}%")
print(f"  F1 Score: {result2['f1']:.1f}%")
print(f"  Time: {result2['time_sec']:.2f}s ({result2['time_per_frame']*1000:.1f}ms/frame)")

# Comparison
print(f"\n{'='*70}")
print("COMPARISON")
print(f"{'='*70}\n")

print(f"Metric                  | Geometric→Noise | Noise→Geometric | Difference")
print(f"-" * 70)
print(f"Recall                  | {result1['recall']:14.1f}% | {result2['recall']:14.1f}% | {result2['recall']-result1['recall']:+9.1f}%")
print(f"Precision               | {result1['precision']:14.1f}% | {result2['precision']:14.1f}% | {result2['precision']-result1['precision']:+9.1f}%")
print(f"F1 Score                | {result1['f1']:14.1f}% | {result2['f1']:14.1f}% | {result2['f1']-result1['f1']:+9.1f}%")
print(f"Time per frame          | {result1['time_per_frame']*1000:13.1f}ms | {result2['time_per_frame']*1000:13.1f}ms | {(result2['time_per_frame']-result1['time_per_frame'])*1000:+8.1f}ms")
print(f"Final blob count        | {result1['after_noise']:15d} | {result2['after_geometric']:15d} | {result2['after_geometric']-result1['after_noise']:+9d}")

# Determine winner
print(f"\n{'='*70}")
print("RECOMMENDATION")
print(f"{'='*70}\n")

if result1['f1'] > result2['f1']:
    winner = "Order 1: Geometric → Noise (CURRENT)"
    improvement = result1['f1'] - result2['f1']
    print(f"✓ Use current order: {winner}")
    print(f"  F1 advantage: +{improvement:.1f}%")
elif result2['f1'] > result1['f1']:
    winner = "Order 2: Noise → Geometric (REVERSED)"
    improvement = result2['f1'] - result1['f1']
    print(f"✓ Switch to reversed order: {winner}")
    print(f"  F1 advantage: +{improvement:.1f}%")
else:
    print(f"✓ Order doesn't matter - same F1 score!")

# Speed comparison
speed_diff = (result2['time_per_frame'] - result1['time_per_frame']) * 1000
if abs(speed_diff) > 1.0:
    if result1['time_per_frame'] < result2['time_per_frame']:
        print(f"  Speed advantage: Geometric→Noise is {-speed_diff:.1f}ms faster per frame")
    else:
        print(f"  Speed advantage: Noise→Geometric is {speed_diff:.1f}ms faster per frame")

print(f"\n{'='*70}")