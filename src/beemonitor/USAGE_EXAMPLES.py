"""Usage Examples for BeeMonitor Detection and Tracking

This file demonstrates how to use the modular detection and tracking architecture.
"""

import cv2
import numpy as np
from ultralytics import YOLO

# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE 1: Basic Detection
# ═══════════════════════════════════════════════════════════════════════════

from beemonitor.detection import BlobDetector, SIFTDetector, YOLODetector

# Load frame
frame = cv2.imread('frame.jpg')

# Create detectors
blob_detector = BlobDetector(min_area=50, min_solidity=0.5)
sift_detector = SIFTDetector(min_keypoints=3, cluster_eps=30.0)
yolo_detector = YOLODetector(
    model=YOLO('models/tracking.pt'),
    conf_threshold=0.25,
    tracking_classes=['bee', 'wasp']
)

# Detect with each method
blob_detections = blob_detector.detect(frame)
sift_detections = sift_detector.detect(frame)
yolo_detections = yolo_detector.detect(frame)

print(f"Blob: {len(blob_detections)} detections")
print(f"SIFT: {len(sift_detections)} detections")
print(f"YOLO: {len(yolo_detections)} detections")

# Each detection has:
for det in blob_detections[:3]:
    print(f"{det.label} at {det.centroid} confidence={det.confidence:.2f}")


# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE 2: Noise Filtering
# ═══════════════════════════════════════════════════════════════════════════

from beemonitor.detection import NoiseFilter

# Create noise filter
noise_filter = NoiseFilter(
    classifier=None,  # Your CNN classifier
    threshold=0.7
)

# Filter blob detections
filtered_detections = noise_filter.filter_detections(frame, blob_detections)

print(f"Before filter: {len(blob_detections)}")
print(f"After filter: {len(filtered_detections)}")


# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE 3: Basic Tracking with BeeTracker
# ═══════════════════════════════════════════════════════════════════════════

from beemonitor.tracking.mot import BeeTracker
from beemonitor.core.config import Config

# Create config
config = Config.default()

# Create MOT algorithm
mot = BeeTracker(
    config=config,
    tracking_classes=['bee', 'wasp']
)

# Process detections
from beemonitor.tracking.mot import Detection as MOTDetection

mot_detections = [
    MOTDetection(
        bbox=det.bbox,
        centroid=det.centroid,
        label=det.label,
        confidence=det.confidence,
        source=det.source
    )
    for det in blob_detections
]

# Update tracker
tracks = mot.update(mot_detections, frame_num=0)

print(f"Active tracks: {len(tracks)}")
for track_id, track in tracks.items():
    print(f"Track {track_id}: {track.label} at {track.centroid}")


# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE 4: BeeTracking System - Motion Only (Fast)
# ═══════════════════════════════════════════════════════════════════════════

from beemonitor.tracking import BeeTracking, DetectionMode

# Create bee tracking with motion detection only
tracker = BeeTracking(
    mot_algorithm=mot,
    detection_mode=DetectionMode.FGBG_ONLY,  # Fast!
    config=config
)

# Process video
results = tracker.process_video(
    video_path='video.mp4',
    roi=(100, 100, 800, 600)  # Hotel box region
)

print(results.head())


# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE 5: BeeTracking System - Stationary Detection (SIFT)
# ═══════════════════════════════════════════════════════════════════════════

# Create bee tracking with SIFT detection (finds stationary bees!)
tracker = BeeTracking(
    mot_algorithm=mot,
    detection_mode=DetectionMode.SIFT_ONLY,  # Stationary detection
    config=config
)

results = tracker.process_video('video.mp4', roi=(100, 100, 800, 600))


# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE 6: BeeTracking System - Comprehensive Detection
# ═══════════════════════════════════════════════════════════════════════════

yolo_model = YOLO('models/tracking.pt')

# Create bee tracking with ALL methods
tracker = BeeTracking(
    mot_algorithm=mot,
    yolo_model=yolo_model,
    detection_mode=DetectionMode.FGBG_SIFT_YOLO,  # Comprehensive!
    use_noise_filter=True,
    noise_filter_model=None,  # Your CNN model
    config=config
)

results = tracker.process_video(
    video_path='video.mp4',
    roi=(100, 100, 800, 600)
)

stats = tracker.get_statistics()
print(f"Processed {stats['total_frames']} frames")
print(f"Total detections: {stats['total_detections']}")
print(f"Total tracks: {stats['total_tracks']}")


# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE 7: Motion + YOLO Confirmation (Default)
# ═══════════════════════════════════════════════════════════════════════════

# This is the recommended default - fast FG/BG with YOLO confirmation
tracker = BeeTracking(
    mot_algorithm=mot,
    yolo_model=yolo_model,
    detection_mode=DetectionMode.FGBG_YOLO,  # Balanced
    use_noise_filter=True,
    config=config
)

results = tracker.process_video('video.mp4', roi=(100, 100, 800, 600))


# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE 8: Configuring Detection and Tracking
# ═══════════════════════════════════════════════════════════════════════════

tracker = BeeTracking(
    mot_algorithm=mot,
    yolo_model=yolo_model,
    detection_mode=DetectionMode.FGBG_SIFT_YOLO,
    config=config
)

# Configure detection parameters
tracker.configure_detection(
    blob_min_area=100,          # Larger blobs only
    sift_min_keypoints=5,        # More keypoints required
    yolo_conf=0.5                # Higher YOLO confidence
)

# Configure tracking parameters
tracker.configure_tracking(
    max_age=30,                  # Tracks survive 30 frames without detection
    min_hits=3,                  # 3 detections to confirm track
    iou_threshold=0.3            # IoU threshold for matching
)

results = tracker.process_video('video.mp4', roi=(100, 100, 800, 600))


# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE 9: Using ByteTrack Instead of BeeTracker
# ═══════════════════════════════════════════════════════════════════════════

from beemonitor.tracking.mot import UltralyticsTracker

# Use ByteTrack algorithm
byte_tracker = UltralyticsTracker(tracker_type='bytetrack.yaml')

tracker = BeeTracking(
    mot_algorithm=byte_tracker,  # Different MOT!
    yolo_model=yolo_model,
    detection_mode=DetectionMode.FGBG_YOLO,
    config=config
)

results = tracker.process_video('video.mp4', roi=(100, 100, 800, 600))


# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE 10: Frame-by-Frame Processing with Visualization
# ═══════════════════════════════════════════════════════════════════════════

tracker = BeeTracking(
    mot_algorithm=mot,
    yolo_model=yolo_model,
    detection_mode=DetectionMode.FGBG_SIFT,
    config=config
)

cap = cv2.VideoCapture('video.mp4')
frame_num = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process single frame
    result = tracker.process_frame(frame, frame_num)
    
    # Get detections and tracks
    detections = result['detections']
    tracks = result['tracks']
    
    # Visualize
    vis_frame = frame.copy()
    
    # Draw detections
    for det in detections:
        x1, y1, x2, y2 = [int(c) for c in det.bbox]
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Draw tracks
    for track_id, track in tracks.items():
        x1, y1, x2, y2 = [int(c) for c in track.bbox]
        cx, cy = [int(c) for c in track.centroid]
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(vis_frame, f"ID:{track_id}", (cx, cy),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    cv2.imshow('Tracking', vis_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    frame_num += 1

cap.release()
cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE 11: Testing Different Detection Modes
# ═══════════════════════════════════════════════════════════════════════════

modes = [
    (DetectionMode.FGBG_ONLY, "Fast motion only"),
    (DetectionMode.SIFT_ONLY, "Stationary detection"),
    (DetectionMode.FGBG_SIFT, "Motion + stationary"),
    (DetectionMode.FGBG_YOLO, "Motion + DL confirmation"),
    (DetectionMode.FGBG_SIFT_YOLO, "Comprehensive"),
]

for mode, description in modes:
    print(f"\nTesting {description}...")
    
    tracker = BeeTracking(
        mot_algorithm=BeeTracker(config, ['bee']),
        yolo_model=yolo_model if 'YOLO' in mode.value.upper() else None,
        detection_mode=mode,
        config=config
    )
    
    results = tracker.process_video('test_video.mp4', roi=(100, 100, 800, 600))
    stats = tracker.get_statistics()
    
    print(f"  Frames: {stats['total_frames']}")
    print(f"  Detections: {stats['total_detections']}")
    print(f"  Tracks: {stats['total_tracks']}")


# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE 12: Integration with Full BeeMonitor Pipeline
# ═══════════════════════════════════════════════════════════════════════════

from beemonitor.core import BeeMonitor

# The BeeMonitor uses this architecture internally:

monitor = BeeMonitor(config=config)

# analyze_video() internally:
# 1. Uses NestDetector to find nests
# 2. Uses BeeTracking (with configured mode) to track bees
# 3. Uses EventProcessor to identify entry/exit events

results = monitor.analyze_video('video.mp4')

# Results contain:
# - events: Entry/exit events DataFrame
# - tracks: Full trajectory data
# - nests: Nest locations
# - motion_data: Motion tracking details

results.to_csv('output')
results.save_video('output')
"""
