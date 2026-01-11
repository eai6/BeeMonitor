#!/usr/bin/env python3
"""
Blob Filter Ablation Study

Test different geometric filters independently to see which ones
actually improve detection performance.

Tests:
1. No filtering (baseline)
2. Area only
3. Solidity only  
4. Circularity only
5. Aspect ratio only
6. Extent only
7. Best combinations

Goal: Find which filters help vs hurt recall/precision
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

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


def test_filter_configuration(
    video_path: str,
    blob_detector: BlobDetector,
    yolo_detector: YOLODetector,
    noise_filter,
    test_frames: int,
    start_frame: int,
    iou_threshold: float,
    filter_config: dict,
    config_name: str
):
    """Test a specific filter configuration."""
    
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    blob_raw_count = 0
    blob_filtered_count = 0
    yolo_count = 0
    matched_count = 0
    
    for i in range(test_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get YOLO ground truth
        yolo_dets = yolo_detector.detect(frame)
        yolo_count += len(yolo_dets)
        
        # Get blob detections with specific filter config
        blob_dets_raw = blob_detector.detect(frame, **filter_config)
        blob_raw_count += len(blob_dets_raw)
        
        # Apply noise filter
        if noise_filter:
            blob_dets = noise_filter.filter_detections(frame, blob_dets_raw)
        else:
            blob_dets = blob_dets_raw
        
        blob_filtered_count += len(blob_dets)
        
        # Match with YOLO
        for blob_det in blob_dets:
            for yolo_det in yolo_dets:
                if calculate_iou(blob_det.bbox, yolo_det.bbox) >= iou_threshold:
                    matched_count += 1
                    break
    
    cap.release()
    
    # Calculate metrics
    recall = (matched_count / yolo_count * 100) if yolo_count > 0 else 0
    precision = (matched_count / blob_filtered_count * 100) if blob_filtered_count > 0 else 0
    f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) > 0 else 0
    noise_reduction = ((blob_raw_count - blob_filtered_count) / blob_raw_count * 100) if blob_raw_count > 0 else 0
    
    return {
        'config_name': config_name,
        'blob_raw': blob_raw_count,
        'blob_filtered': blob_filtered_count,
        'yolo_total': yolo_count,
        'matched': matched_count,
        'recall': recall,
        'precision': precision,
        'f1_score': f1,
        'noise_reduction': noise_reduction
    }


print("="*70)
print("Blob Filter Ablation Study")
print("="*70)

# Configuration
video_path = project_root / 'data/mendels_2024-05-23_18_20_25.mp4'
model_path = 'models/bee_tracking_back_up_Full_Mode.pt'

BLOB_INIT_FRAMES = 30
TEST_FRAMES = 150
TEST_START_FRAME = 180
IOU_THRESHOLD = 0.15
NOISE_THRESHOLD = 0.6

print(f"\nTest configuration:")
print(f"  Video: {video_path.name}")
print(f"  Test frames: {TEST_START_FRAME}-{TEST_START_FRAME + TEST_FRAMES}")
print(f"  IoU threshold: {IOU_THRESHOLD}")
print(f"  Noise threshold: {NOISE_THRESHOLD}")

# Initialize YOLO
print(f"\nLoading YOLO...")
yolo_model = YOLO(model_path)
yolo_detector = YOLODetector(yolo_model, tracking_classes=['bee'])

# Initialize noise filter
print(f"Loading noise filter...")
noise_filter = BeeNoiseFilter(
    model_path='models/blob_noise_classifier.pth',
    noise_threshold=NOISE_THRESHOLD
)

# Initialize blob detector
print(f"Initializing blob detector...")
blob_detector = BlobDetector()
blob_detector.initialize_from_video_with_verification(
    video_path=str(video_path),
    yolo_detector=yolo_detector,
    num_frames=BLOB_INIT_FRAMES
)

print(f"\n{'='*70}")
print("Testing Filter Configurations")
print(f"{'='*70}\n")

print(f"\n{'='*70}")
print("STEP 1: Learning Blob Characteristics from Video")
print(f"{'='*70}\n")

# Learn actual blob parameters
print(f"Learning from actual foreground blobs...")
learned_params = blob_detector.learn_from_foreground_blobs(
    video_path=str(video_path),
    yolo_detector=yolo_detector,
    num_frames=100,
    start_frame=BLOB_INIT_FRAMES,
    percentile_low=5.0,
    percentile_high=95.0
)

print(f"\n{'='*70}")
print("LEARNED BLOB STATISTICS")
print(f"{'='*70}\n")

print(f"These are the ACTUAL characteristics of bee foreground blobs:")
print(f"  Area: {learned_params['min_area']:.1f} - {learned_params['max_area']:.1f} px²")
print(f"  Solidity: ≥{learned_params['min_solidity']:.3f}")
print(f"  Circularity: ≥{learned_params.get('min_circularity', 0.0):.3f}")
print(f"  Aspect Ratio: {learned_params.get('min_aspect_ratio', 0.0):.2f} - {learned_params.get('max_aspect_ratio', 3.0):.2f}")
print(f"  Extent: ≥{learned_params.get('min_extent', 0.0):.3f}")

print(f"\nWe'll test these learned values at different strictness levels:")
print(f"  - Very Lenient: Use lower percentiles (catches more bees)")
print(f"  - Moderate: Use learned percentiles (5th/95th)")
print(f"  - Strict: Use higher percentiles (filters more noise)")

print(f"\n{'='*70}")
print("STEP 2: Testing Filter Configurations")
print(f"{'='*70}\n")

# Define test configurations using LEARNED values
configs = [
    # Baseline - no filtering
    {
        'name': '1. No Filtering',
        'params': {
            'min_area': 0.0,
            'max_area': 100000.0,
            'min_solidity': 0.0,
            'min_circularity': 0.0,
            'min_aspect_ratio': 0.0,
            'max_aspect_ratio': 100.0,
            'min_extent': 0.0
        }
    },
    
    # Area only - using learned values at different strictness
    {
        'name': '2a. Area (Learned 5th %ile)',
        'params': {
            'min_area': learned_params['min_area'],  # 5th percentile
            'max_area': learned_params['max_area'],  # 95th percentile
            'min_solidity': 0.0,
            'min_circularity': 0.0,
            'min_aspect_ratio': 0.0,
            'max_aspect_ratio': 100.0,
            'min_extent': 0.0
        }
    },
    {
        'name': '2b. Area (50% of Learned)',
        'params': {
            'min_area': learned_params['min_area'] * 0.5,  # More lenient
            'max_area': learned_params['max_area'],
            'min_solidity': 0.0,
            'min_circularity': 0.0,
            'min_aspect_ratio': 0.0,
            'max_aspect_ratio': 100.0,
            'min_extent': 0.0
        }
    },
    {
        'name': '2c. Area (30% of Learned)',
        'params': {
            'min_area': learned_params['min_area'] * 0.3,  # Very lenient
            'max_area': learned_params['max_area'],
            'min_solidity': 0.0,
            'min_circularity': 0.0,
            'min_aspect_ratio': 0.0,
            'max_aspect_ratio': 100.0,
            'min_extent': 0.0
        }
    },
    
    # Solidity only - using learned values
    {
        'name': '3a. Solidity (Learned 5th %ile)',
        'params': {
            'min_area': 0.0,
            'max_area': 100000.0,
            'min_solidity': learned_params['min_solidity'],
            'min_circularity': 0.0,
            'min_aspect_ratio': 0.0,
            'max_aspect_ratio': 100.0,
            'min_extent': 0.0
        }
    },
    {
        'name': '3b. Solidity (80% of Learned)',
        'params': {
            'min_area': 0.0,
            'max_area': 100000.0,
            'min_solidity': learned_params['min_solidity'] * 0.8,
            'min_circularity': 0.0,
            'min_aspect_ratio': 0.0,
            'max_aspect_ratio': 100.0,
            'min_extent': 0.0
        }
    },
    {
        'name': '3c. Solidity (60% of Learned)',
        'params': {
            'min_area': 0.0,
            'max_area': 100000.0,
            'min_solidity': learned_params['min_solidity'] * 0.6,
            'min_circularity': 0.0,
            'min_aspect_ratio': 0.0,
            'max_aspect_ratio': 100.0,
            'min_extent': 0.0
        }
    },
    
    # Circularity only - using learned values
    {
        'name': '4a. Circularity (Learned 5th %ile)',
        'params': {
            'min_area': 0.0,
            'max_area': 100000.0,
            'min_solidity': 0.0,
            'min_circularity': learned_params.get('min_circularity', 0.2),
            'min_aspect_ratio': 0.0,
            'max_aspect_ratio': 100.0,
            'min_extent': 0.0
        }
    },
    {
        'name': '4b. Circularity (80% of Learned)',
        'params': {
            'min_area': 0.0,
            'max_area': 100000.0,
            'min_solidity': 0.0,
            'min_circularity': learned_params.get('min_circularity', 0.2) * 0.8,
            'min_aspect_ratio': 0.0,
            'max_aspect_ratio': 100.0,
            'min_extent': 0.0
        }
    },
    
    # Aspect ratio only - using learned values
    {
        'name': '5a. Aspect Ratio (Learned)',
        'params': {
            'min_area': 0.0,
            'max_area': 100000.0,
            'min_solidity': 0.0,
            'min_circularity': 0.0,
            'min_aspect_ratio': learned_params.get('min_aspect_ratio', 0.3),
            'max_aspect_ratio': learned_params.get('max_aspect_ratio', 3.0),
            'min_extent': 0.0
        }
    },
    {
        'name': '5b. Aspect Ratio (Relaxed)',
        'params': {
            'min_area': 0.0,
            'max_area': 100000.0,
            'min_solidity': 0.0,
            'min_circularity': 0.0,
            'min_aspect_ratio': learned_params.get('min_aspect_ratio', 0.3) * 0.7,
            'max_aspect_ratio': learned_params.get('max_aspect_ratio', 3.0) * 1.3,
            'min_extent': 0.0
        }
    },
    
    # Extent only - using learned values
    {
        'name': '6a. Extent (Learned 5th %ile)',
        'params': {
            'min_area': 0.0,
            'max_area': 100000.0,
            'min_solidity': 0.0,
            'min_circularity': 0.0,
            'min_aspect_ratio': 0.0,
            'max_aspect_ratio': 100.0,
            'min_extent': learned_params.get('min_extent', 0.2)
        }
    },
    {
        'name': '6b. Extent (80% of Learned)',
        'params': {
            'min_area': 0.0,
            'max_area': 100000.0,
            'min_solidity': 0.0,
            'min_circularity': 0.0,
            'min_aspect_ratio': 0.0,
            'max_aspect_ratio': 100.0,
            'min_extent': learned_params.get('min_extent', 0.2) * 0.8
        }
    },
    
    # Combinations - using learned values
    {
        'name': '7a. Area + Solidity (Learned)',
        'params': {
            'min_area': learned_params['min_area'],
            'max_area': learned_params['max_area'],
            'min_solidity': learned_params['min_solidity'],
            'min_circularity': 0.0,
            'min_aspect_ratio': 0.0,
            'max_aspect_ratio': 100.0,
            'min_extent': 0.0
        }
    },
    {
        'name': '7b. Area + Solidity (50% Area, 80% Solidity)',
        'params': {
            'min_area': learned_params['min_area'] * 0.5,
            'max_area': learned_params['max_area'],
            'min_solidity': learned_params['min_solidity'] * 0.8,
            'min_circularity': 0.0,
            'min_aspect_ratio': 0.0,
            'max_aspect_ratio': 100.0,
            'min_extent': 0.0
        }
    },
    {
        'name': '7c. Area + Aspect Ratio (50% Area, Relaxed AR)',
        'params': {
            'min_area': learned_params['min_area'] * 0.5,
            'max_area': learned_params['max_area'],
            'min_solidity': 0.0,
            'min_circularity': 0.0,
            'min_aspect_ratio': learned_params.get('min_aspect_ratio', 0.3) * 0.7,
            'max_aspect_ratio': learned_params.get('max_aspect_ratio', 3.0) * 1.3,
            'min_extent': 0.0
        }
    },
    {
        'name': '7d. All Filters (Learned)',
        'params': {
            'min_area': learned_params['min_area'],
            'max_area': learned_params['max_area'],
            'min_solidity': learned_params['min_solidity'],
            'min_circularity': learned_params.get('min_circularity', 0.2),
            'min_aspect_ratio': learned_params.get('min_aspect_ratio', 0.3),
            'max_aspect_ratio': learned_params.get('max_aspect_ratio', 3.0),
            'min_extent': learned_params.get('min_extent', 0.2)
        }
    },
    {
        'name': '7e. All Filters (Lenient: 50% Area, 80% Others)',
        'params': {
            'min_area': learned_params['min_area'] * 0.5,
            'max_area': learned_params['max_area'],
            'min_solidity': learned_params['min_solidity'] * 0.8,
            'min_circularity': learned_params.get('min_circularity', 0.2) * 0.8,
            'min_aspect_ratio': learned_params.get('min_aspect_ratio', 0.3) * 0.8,
            'max_aspect_ratio': learned_params.get('max_aspect_ratio', 3.0) * 1.2,
            'min_extent': learned_params.get('min_extent', 0.2) * 0.8
        }
    },
    
    # Manual lenient (for comparison)
    {
        'name': '8. Manual Lenient (area=30, solidity=0.35)',
        'params': {
            'min_area': 30.0,
            'max_area': 8000.0,
            'min_solidity': 0.35,
            'min_circularity': 0.0,
            'min_aspect_ratio': 0.0,
            'max_aspect_ratio': 10.0,
            'min_extent': 0.0
        }
    }
]

# Run tests
results = []
for i, config in enumerate(configs):
    print(f"Testing {config['name']}...")
    result = test_filter_configuration(
        video_path=str(video_path),
        blob_detector=blob_detector,
        yolo_detector=yolo_detector,
        noise_filter=noise_filter,
        test_frames=TEST_FRAMES,
        start_frame=TEST_START_FRAME,
        iou_threshold=IOU_THRESHOLD,
        filter_config=config['params'],
        config_name=config['name']
    )
    results.append(result)
    print(f"  Recall: {result['recall']:.1f}%, Precision: {result['precision']:.1f}%, F1: {result['f1_score']:.1f}%")

# Create results DataFrame
df = pd.DataFrame(results)

print(f"\n{'='*70}")
print("RESULTS SUMMARY")
print(f"{'='*70}\n")

# Sort by F1 score
df_sorted = df.sort_values('f1_score', ascending=False)

print("Ranked by F1 Score:")
print(df_sorted[['config_name', 'recall', 'precision', 'f1_score']].to_string(index=False))

print(f"\n{'='*70}")
print("DETAILED RESULTS")
print(f"{'='*70}\n")

for _, row in df_sorted.iterrows():
    print(f"{row['config_name']}:")
    print(f"  Blob raw: {row['blob_raw']}")
    print(f"  Blob filtered: {row['blob_filtered']} (removed {row['blob_raw'] - row['blob_filtered']})")
    print(f"  Noise reduction: {row['noise_reduction']:.1f}%")
    print(f"  Matched: {row['matched']}/{row['yolo_total']}")
    print(f"  Recall: {row['recall']:.1f}%")
    print(f"  Precision: {row['precision']:.1f}%")
    print(f"  F1 Score: {row['f1_score']:.1f}%")
    print()

# Find best configuration
best_f1 = df_sorted.iloc[0]
best_recall = df.loc[df['recall'].idxmax()]
best_precision = df.loc[df['precision'].idxmax()]

print(f"{'='*70}")
print("RECOMMENDATIONS")
print(f"{'='*70}\n")

print(f"Best F1 Score ({best_f1['f1_score']:.1f}%):")
print(f"  {best_f1['config_name']}")
print(f"  Recall: {best_f1['recall']:.1f}%, Precision: {best_f1['precision']:.1f}%")

print(f"\nBest Recall ({best_recall['recall']:.1f}%):")
print(f"  {best_recall['config_name']}")
print(f"  Precision: {best_recall['precision']:.1f}%, F1: {best_recall['f1_score']:.1f}%")

print(f"\nBest Precision ({best_precision['precision']:.1f}%):")
print(f"  {best_precision['config_name']}")
print(f"  Recall: {best_precision['recall']:.1f}%, F1: {best_precision['f1_score']:.1f}%")

# Save results
output_file = project_root / 'blob_filter_ablation_results.csv'
df_sorted.to_csv(output_file, index=False)
print(f"\n✓ Results saved to: {output_file}")