#!/usr/bin/env python3
"""
Simplified Tracking Pipeline - No SIFT

Architecture:
1. Blob detection (motion)
2. CNN noise filter (primary - 66% filtering)
3. Solidity filter (fallback - learned from actual bees)
4. MOT tracking

SIFT REMOVED - it contributed only 2-3% of detections!
"""

import sys
from pathlib import Path
import cv2
import pandas as pd

project_root = Path('/Users/edwardamoah/Documents/GitHub/BeeMonitor_eai6')
sys.path.insert(0, str(project_root / 'src'))

from ultralytics import YOLO
from beemonitor.detection import BlobDetector, YOLODetector, BeeNoiseFilter
from beemonitor.tracking import BeeTracking, DetectionMode
from beemonitor.tracking.mot import BeeTracker
from beemonitor.core.config import Config

print("="*70)
print("Simplified Tracking Pipeline: Blob + CNN + Learned Morphology")
print("="*70)

# Configuration
video_path = project_root / 'data/mendels_2024-05-23_18_20_25.mp4'
model_path = 'models/bee_tracking_back_up_Full_Mode.pt'

BLOB_INIT_FRAMES = 30      # Initialize background
MORPHOLOGY_LEARN_FRAMES = 100  # Learn blob characteristics
PROCESS_START_FRAME = 200  # Start tracking after initialization
PROCESS_FRAMES = 150       # Process N frames

# Filter settings (from ablation study)
NOISE_THRESHOLD = 0.6      # CNN threshold
USE_SOLIDITY_FALLBACK = True  # Safety net
SOLIDITY_SCALE = 0.8       # Use 80% of learned (proven winner)

print(f"\nConfiguration:")
print(f"  Video: {video_path.name}")
print(f"  Background init: 0-{BLOB_INIT_FRAMES}")
print(f"  Morphology learning: {BLOB_INIT_FRAMES}-{BLOB_INIT_FRAMES + MORPHOLOGY_LEARN_FRAMES}")
print(f"  Tracking: {PROCESS_START_FRAME}-{PROCESS_START_FRAME + PROCESS_FRAMES}")
print(f"  Noise threshold: {NOISE_THRESHOLD}")
print(f"  Solidity fallback: {USE_SOLIDITY_FALLBACK}")

# ============================================================================
# PHASE 1: INITIALIZATION
# ============================================================================

print(f"\n{'='*70}")
print("PHASE 1: Detector Initialization")
print(f"{'='*70}\n")

# 1.1: Load YOLO
print("1. Loading YOLO model...")
yolo_model = YOLO(model_path)
yolo_detector = YOLODetector(yolo_model, tracking_classes=['bee'])
print("   ✓ YOLO ready")

# 1.2: Load CNN noise filter
print("\n2. Loading CNN noise filter...")
noise_filter = BeeNoiseFilter(
    model_path='models/blob_noise_classifier.pth',
    noise_threshold=NOISE_THRESHOLD
)
print("   ✓ CNN noise filter ready")

# 1.3: Initialize blob detector (background subtraction)
print(f"\n3. Initializing blob detector (frames 0-{BLOB_INIT_FRAMES})...")
blob_detector = BlobDetector(min_area=50, min_solidity=0.5)
num_clean = blob_detector.initialize_from_video_with_verification(
    video_path=str(video_path),
    yolo_detector=yolo_detector,
    num_frames=BLOB_INIT_FRAMES,
    max_detections=0  # Require bee-free frames
)
print(f"   ✓ Background model initialized ({num_clean} verified clean frames)")

# 1.4: Learn morphology from actual bee blobs
print(f"\n4. Learning morphology from actual foreground blobs...")
print(f"   Analyzing frames {BLOB_INIT_FRAMES}-{BLOB_INIT_FRAMES + MORPHOLOGY_LEARN_FRAMES}...")

learned_params = blob_detector.learn_from_foreground_blobs(
    video_path=str(video_path),
    yolo_detector=yolo_detector,
    num_frames=MORPHOLOGY_LEARN_FRAMES,
    start_frame=BLOB_INIT_FRAMES,
    percentile_low=5.0,
    percentile_high=95.0
)

print(f"\n   Learned characteristics from actual bee foreground blobs:")
print(f"     Area: {learned_params['min_area']:.1f} - {learned_params['max_area']:.1f} px²")
print(f"     Solidity: ≥{learned_params['min_solidity']:.3f}")
print(f"     Circularity: ≥{learned_params.get('min_circularity', 0.0):.3f}")
print(f"     Aspect Ratio: {learned_params.get('min_aspect_ratio', 0.0):.2f} - {learned_params.get('max_aspect_ratio', 3.0):.2f}")
print(f"     Extent: ≥{learned_params.get('min_extent', 0.0):.3f}")

# Apply solidity fallback (80% of learned - proven optimal)
if USE_SOLIDITY_FALLBACK:
    fallback_solidity = learned_params['min_solidity'] * SOLIDITY_SCALE
    print(f"\n   Applying solidity fallback:")
    print(f"     Min solidity: {fallback_solidity:.3f} (80% of learned {learned_params['min_solidity']:.3f})")
else:
    fallback_solidity = 0.0
    print(f"\n   No solidity fallback - CNN only")

print("   ✓ Morphology learned and configured")

# ============================================================================
# PHASE 2: TRACKING SETUP
# ============================================================================

print(f"\n{'='*70}")
print("PHASE 2: Tracking System Setup")
print(f"{'='*70}\n")

print("1. Creating MOT algorithm...")
config = Config()
mot = BeeTracker(config, tracking_classes=['bee'])
print("   ✓ BeeTracker (Kalman + Hungarian) ready")

print("\n2. Creating tracking system...")
# Use FGBG_ONLY mode (no SIFT, no YOLO in detection loop)
tracker = BeeTracking(
    mot_algorithm=mot,
    yolo_model=yolo_model,
    detection_mode=DetectionMode.FGBG_ONLY,  # Blob only!
    use_noise_filter=True,
    noise_filter_model=noise_filter,
    config=config
)

# Assign initialized blob detector
tracker.blob_detector = blob_detector

print("   ✓ Tracking system ready")
print(f"   Detection mode: FGBG_ONLY (Blob → CNN → Solidity)")
print(f"   Order: Noise filter first (proven +1.5% F1 advantage)")

# ============================================================================
# PHASE 3: CUSTOM DETECTION WITH LEARNED MORPHOLOGY
# ============================================================================

print(f"\n{'='*70}")
print("PHASE 3: Video Processing with Learned Filters")
print(f"{'='*70}\n")

def detect_with_learned_filters(frame, blob_detector, noise_filter, min_solidity):
    """
    Custom detection pipeline:
    1. Blob detection (motion)
    2. CNN noise filter (primary - 66% reduction)
    3. Solidity filter (fallback - learned)
    
    Order: Noise → Geometric (proven winner, +9% precision)
    """
    
    # Stage 1: Get raw blobs
    blob_dets = blob_detector.detect(frame)
    
    # Stage 2: CNN noise filter (PRIMARY)
    if noise_filter:
        blob_dets = noise_filter.filter_detections(frame, blob_dets)
    
    # Stage 3: Solidity filter (FALLBACK - safety net)
    if min_solidity > 0:
        blob_dets = [d for d in blob_dets 
                     if d.metadata.get('solidity', 0.0) >= min_solidity]
    
    return blob_dets


# Process video with custom detection
print(f"Processing frames {PROCESS_START_FRAME}-{PROCESS_START_FRAME + PROCESS_FRAMES}...")

cap = cv2.VideoCapture(str(video_path))
cap.set(cv2.CAP_PROP_POS_FRAMES, PROCESS_START_FRAME)

tracking_results = []
frame_num = PROCESS_START_FRAME

stats = {
    'total_blobs_raw': 0,
    'total_blobs_after_cnn': 0,
    'total_blobs_after_solidity': 0,
    'total_tracks': 0
}

for i in range(PROCESS_FRAMES):
    ret, frame = cap.read()
    if not ret:
        break
    
    # Custom detection with learned filters
    blob_dets_raw = blob_detector.detect(frame)
    stats['total_blobs_raw'] += len(blob_dets_raw)
    
    # CNN filter
    blob_dets_cnn = noise_filter.filter_detections(frame, blob_dets_raw)
    stats['total_blobs_after_cnn'] += len(blob_dets_cnn)
    
    # Solidity fallback
    if USE_SOLIDITY_FALLBACK:
        blob_dets_final = [d for d in blob_dets_cnn 
                          if d.metadata.get('solidity', 0.0) >= fallback_solidity]
        stats['total_blobs_after_solidity'] += len(blob_dets_final)
    else:
        blob_dets_final = blob_dets_cnn
        stats['total_blobs_after_solidity'] += len(blob_dets_final)
    
    # Convert to MOT format and update tracking
    from beemonitor.tracking.mot.base_mot import Detection as MOTDetection
    mot_detections = [
        MOTDetection(
            bbox=d.bbox,
            centroid=d.centroid,
            label=d.label,
            confidence=d.confidence,
            source=d.source
        )
        for d in blob_dets_final
    ]
    
    # Update MOT
    tracks = mot.update(mot_detections, frame_num)
    stats['total_tracks'] = max(stats['total_tracks'], len(tracks))
    
    # Record results
    for track_id, track in tracks.items():
        tracking_results.append({
            'frame': frame_num,
            'track_id': track_id,
            'x1': track.bbox[0],
            'y1': track.bbox[1],
            'x2': track.bbox[2],
            'y2': track.bbox[3],
            'confidence': track.confidence if hasattr(track, 'confidence') else 1.0,
            'species': track.label
        })
    
    frame_num += 1
    
    if (i + 1) % 50 == 0:
        print(f"   Processed {i+1}/{PROCESS_FRAMES} frames...")

cap.release()

print(f"   ✓ Processing complete!")

# ============================================================================
# PHASE 4: RESULTS & STATISTICS
# ============================================================================

print(f"\n{'='*70}")
print("RESULTS & STATISTICS")
print(f"{'='*70}\n")

# Convert to DataFrame
df_results = pd.DataFrame(tracking_results)

print("Filtering Statistics:")
print(f"  Raw blobs detected: {stats['total_blobs_raw']}")
print(f"  After CNN filter: {stats['total_blobs_after_cnn']} "
      f"(removed {stats['total_blobs_raw'] - stats['total_blobs_after_cnn']}, "
      f"{(stats['total_blobs_raw'] - stats['total_blobs_after_cnn'])/stats['total_blobs_raw']*100:.1f}%)")
print(f"  After solidity filter: {stats['total_blobs_after_solidity']} "
      f"(removed {stats['total_blobs_after_cnn'] - stats['total_blobs_after_solidity']}, "
      f"{(stats['total_blobs_after_cnn'] - stats['total_blobs_after_solidity'])/stats['total_blobs_after_cnn']*100 if stats['total_blobs_after_cnn'] > 0 else 0:.1f}%)")

print(f"\nTracking Statistics:")
if len(df_results) > 0:
    print(f"  Total tracks created: {df_results['track_id'].nunique()}")
    print(f"  Total detections: {len(df_results)}")
    print(f"  Frames processed: {df_results['frame'].nunique()}")
    print(f"  Max simultaneous tracks: {stats['total_tracks']}")
else:
    print(f"  No tracks detected!")

# Save results
output_file = project_root / 'tracking_results_simplified.csv'
if len(df_results) > 0:
    df_results.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")
else:
    print(f"\n⚠️  No results to save")

print(f"\n{'='*70}")
print("Pipeline Summary")
print(f"{'='*70}\n")

print("Architecture:")
print("  1. Blob Detection (MOG2 background subtraction)")
print("  2. CNN Noise Filter (primary - does 66% of filtering)")
print("  3. Solidity Filter (fallback - learned from actual bees)")
print("  4. BeeTracker MOT (Kalman + Hungarian)")

print(f"\nRemoved Components:")
print("  ✗ SIFT detection (contributed only 2-3%)")
print("  ✗ YOLO in detection loop (use for initialization only)")

print(f"\nKey Improvements:")
print("  ✓ Noise→Geometric order (+1.5% F1, +9% precision)")
print("  ✓ Learned morphology (from actual bee blobs)")
print("  ✓ Simpler pipeline (faster, easier to debug)")
print("  ✓ Robust fallback (CNN fails → solidity catches)")

print(f"\n{'='*70}")