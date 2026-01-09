"""TESTING GUIDE - BeeMonitor Detection & Tracking Systems

This guide shows you how to test the modular detection and tracking architecture.

═══════════════════════════════════════════════════════════════════════════
TESTING STRATEGY
═══════════════════════════════════════════════════════════════════════════

1. Unit Tests - Test individual components in isolation
2. Integration Tests - Test components working together
3. Visual Tests - See what's being detected/tracked
4. Performance Tests - Benchmark speed and accuracy
5. Comparison Tests - Compare different modes/algorithms

═══════════════════════════════════════════════════════════════════════════
1. UNIT TESTING - Individual Components
═══════════════════════════════════════════════════════════════════════════

Test each detector independently:

```python
# test_detectors.py
import cv2
from beemonitor.detection import BlobDetector, SIFTDetector, YOLODetector

# Load test frame
frame = cv2.imread('test_data/frame_001.jpg')

# Test BlobDetector
print("Testing BlobDetector...")
blob_det = BlobDetector(min_area=50, min_solidity=0.5)
blob_dets = blob_det.detect(frame)
print(f"  Found {len(blob_dets)} blobs")
for det in blob_dets[:3]:
    print(f"    - {det.label} at {det.centroid}, conf={det.confidence:.2f}")

# Test SIFTDetector
print("\nTesting SIFTDetector...")
sift_det = SIFTDetector(min_keypoints=3, cluster_eps=30.0)
sift_dets = sift_det.detect(frame)
print(f"  Found {len(sift_dets)} SIFT clusters")

# Test YOLODetector
print("\nTesting YOLODetector...")
yolo_det = YOLODetector(model, conf_threshold=0.25, tracking_classes=['bee'])
yolo_dets = yolo_det.detect(frame)
print(f"  Found {len(yolo_dets)} YOLO detections")
```

Test MOT algorithms:

```python
# test_mot.py
from beemonitor.tracking.mot import BeeTracker, Detection

# Create tracker
tracker = BeeTracker(config, tracking_classes=['bee'])

# Create mock detections
detections = [
    Detection(
        bbox=(100, 100, 150, 150),
        centroid=(125, 125),
        label='bee',
        confidence=0.9,
        source='test'
    )
]

# Update tracker
tracks = tracker.update(detections, frame_num=0)
print(f"Active tracks: {len(tracks)}")

# Test prediction
predicted_tracks = tracker.predict(frame_num=1)
print(f"Predicted tracks: {len(predicted_tracks)}")
```

═══════════════════════════════════════════════════════════════════════════
2. INTEGRATION TESTING - Full Pipeline
═══════════════════════════════════════════════════════════════════════════

Test BeeTracking system with different modes:

```python
# test_bee_tracking.py
from beemonitor.tracking import BeeTracking, DetectionMode
from beemonitor.tracking.mot import BeeTracker

modes = [
    DetectionMode.FGBG_ONLY,
    DetectionMode.SIFT_ONLY,
    DetectionMode.FGBG_SIFT,
    DetectionMode.FGBG_YOLO,
]

for mode in modes:
    print(f"\nTesting {mode.value}...")
    
    mot = BeeTracker(config, ['bee'])
    tracker = BeeTracking(
        mot_algorithm=mot,
        yolo_model=yolo_model if 'YOLO' in mode.value.upper() else None,
        detection_mode=mode,
        config=config
    )
    
    results = tracker.process_video('test_video.mp4', roi=(100, 100, 800, 600))
    stats = tracker.get_statistics()
    
    print(f"  Frames: {stats['total_frames']}")
    print(f"  Detections: {stats['total_detections']}")
    print(f"  Tracks: {stats['total_tracks']}")
```

═══════════════════════════════════════════════════════════════════════════
3. VISUAL TESTING - See What's Happening
═══════════════════════════════════════════════════════════════════════════

Visualize detections and tracks:

```python
# test_visual.py
import cv2
from beemonitor.tracking import BeeTracking, DetectionMode
from beemonitor.tracking.mot import BeeTracker

# Setup tracker
mot = BeeTracker(config, ['bee'])
tracker = BeeTracking(
    mot_algorithm=mot,
    yolo_model=yolo_model,
    detection_mode=DetectionMode.FGBG_SIFT_YOLO,
    config=config
)

# Process frame-by-frame
cap = cv2.VideoCapture('test_video.mp4')
frame_num = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    result = tracker.process_frame(frame, frame_num)
    
    vis_frame = frame.copy()
    
    # Draw detections (green)
    for det in result['detections']:
        x1, y1, x2, y2 = [int(c) for c in det.bbox]
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis_frame, det.source, (x1, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Draw tracks (blue)
    for track_id, track in result['tracks'].items():
        x1, y1, x2, y2 = [int(c) for c in track.bbox]
        cx, cy = [int(c) for c in track.centroid]
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(vis_frame, f"ID:{track_id}", (cx, cy),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw trajectory
        if len(track.trajectory) > 1:
            points = [pt[1] for pt in track.trajectory[-10:]]  # Last 10 points
            for i in range(len(points)-1):
                pt1 = tuple(map(int, points[i]))
                pt2 = tuple(map(int, points[i+1]))
                cv2.line(vis_frame, pt1, pt2, (255, 0, 0), 2)
    
    # Show info
    cv2.putText(vis_frame, f"Frame: {frame_num}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(vis_frame, f"Detections: {len(result['detections'])}", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(vis_frame, f"Tracks: {len(result['tracks'])}", (10, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(vis_frame, f"Mode: {result['mode']}", (10, 120),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imshow('Tracking Test', vis_frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
    
    frame_num += 1

cap.release()
cv2.destroyAllWindows()
```

═══════════════════════════════════════════════════════════════════════════
4. PERFORMANCE TESTING - Speed & Accuracy
═══════════════════════════════════════════════════════════════════════════

Benchmark different modes:

```python
# test_performance.py
import time
from beemonitor.tracking import BeeTracking, DetectionMode

modes = [
    DetectionMode.FGBG_ONLY,
    DetectionMode.SIFT_ONLY,
    DetectionMode.FGBG_SIFT,
    DetectionMode.FGBG_YOLO,
    DetectionMode.FGBG_SIFT_YOLO,
]

results = []

for mode in modes:
    print(f"\nBenchmarking {mode.value}...")
    
    tracker = BeeTracking(
        mot_algorithm=BeeTracker(config, ['bee']),
        yolo_model=yolo_model if 'YOLO' in mode.value.upper() else None,
        detection_mode=mode,
        config=config
    )
    
    start = time.time()
    df = tracker.process_video('test_video.mp4', roi=(100, 100, 800, 600))
    elapsed = time.time() - start
    
    stats = tracker.get_statistics()
    fps = stats['total_frames'] / elapsed
    
    results.append({
        'mode': mode.value,
        'time': elapsed,
        'fps': fps,
        'detections': stats['total_detections'],
        'tracks': stats['total_tracks']
    })
    
    print(f"  Time: {elapsed:.2f}s")
    print(f"  FPS: {fps:.2f}")
    print(f"  Detections: {stats['total_detections']}")
    print(f"  Tracks: {stats['total_tracks']}")

# Print comparison
print("\n" + "="*70)
print("PERFORMANCE COMPARISON")
print("="*70)
print(f"{'Mode':<20} {'Time (s)':<12} {'FPS':<10} {'Detections':<12} {'Tracks'}")
print("-"*70)
for r in results:
    print(f"{r['mode']:<20} {r['time']:<12.2f} {r['fps']:<10.2f} {r['detections']:<12} {r['tracks']}")
```

═══════════════════════════════════════════════════════════════════════════
5. COMPARISON TESTING - Different Algorithms
═══════════════════════════════════════════════════════════════════════════

Compare MOT algorithms:

```python
# test_mot_comparison.py
from beemonitor.tracking import BeeTracking, DetectionMode
from beemonitor.tracking.mot import BeeTracker, UltralyticsTracker

mot_algorithms = [
    ("BeeTracker", BeeTracker(config, ['bee'])),
    ("ByteTrack", UltralyticsTracker(tracker_type='bytetrack.yaml')),
    ("BoT-SORT", UltralyticsTracker(tracker_type='botsort.yaml')),
]

for name, mot in mot_algorithms:
    print(f"\nTesting {name}...")
    
    tracker = BeeTracking(
        mot_algorithm=mot,
        yolo_model=yolo_model,
        detection_mode=DetectionMode.FGBG_YOLO,
        config=config
    )
    
    results = tracker.process_video('test_video.mp4', roi=(100, 100, 800, 600))
    stats = tracker.get_statistics()
    
    print(f"  Tracks: {stats['total_tracks']}")
    print(f"  Detections: {stats['total_detections']}")
```

═══════════════════════════════════════════════════════════════════════════
6. DETECTOR COMPARISON - Side-by-Side
═══════════════════════════════════════════════════════════════════════════

Compare what each detector finds:

```python
# test_detector_comparison.py
import cv2
from beemonitor.detection import BlobDetector, SIFTDetector, YOLODetector

frame = cv2.imread('test_frame.jpg')

# Create detectors
blob = BlobDetector(min_area=50)
sift = SIFTDetector(min_keypoints=3)
yolo = YOLODetector(model, conf_threshold=0.25)

# Detect
blob_dets = blob.detect(frame)
sift_dets = sift.detect(frame)
yolo_dets = yolo.detect(frame)

# Visualize side-by-side
h, w = frame.shape[:2]
combined = np.zeros((h, w*3, 3), dtype=np.uint8)

# Blob detections
frame_blob = frame.copy()
for det in blob_dets:
    x1, y1, x2, y2 = [int(c) for c in det.bbox]
    cv2.rectangle(frame_blob, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.putText(frame_blob, f"Blob: {len(blob_dets)}", (10, 30),
           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
combined[:, 0:w] = frame_blob

# SIFT detections
frame_sift = frame.copy()
for det in sift_dets:
    x1, y1, x2, y2 = [int(c) for c in det.bbox]
    cv2.rectangle(frame_sift, (x1, y1), (x2, y2), (255, 0, 0), 2)
cv2.putText(frame_sift, f"SIFT: {len(sift_dets)}", (10, 30),
           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
combined[:, w:w*2] = frame_sift

# YOLO detections
frame_yolo = frame.copy()
for det in yolo_dets:
    x1, y1, x2, y2 = [int(c) for c in det.bbox]
    cv2.rectangle(frame_yolo, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(frame_yolo, det.label, (x1, y1-5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
cv2.putText(frame_yolo, f"YOLO: {len(yolo_dets)}", (10, 30),
           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
combined[:, w*2:w*3] = frame_yolo

cv2.imshow('Detector Comparison', combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

═══════════════════════════════════════════════════════════════════════════
7. STATIONARY BEE TEST - Does SIFT Work?
═══════════════════════════════════════════════════════════════════════════

Test if SIFT detects stationary bees that FG/BG misses:

```python
# test_stationary_detection.py
from beemonitor.detection import BlobDetector, SIFTDetector
import cv2

# Load frame with stationary bee
frame = cv2.imread('test_data/stationary_bee.jpg')

# FG/BG won't detect it (no motion)
blob_det = BlobDetector(min_area=50)
blob_dets = blob_det.detect(frame)
print(f"FG/BG detections: {len(blob_dets)}")  # Should be 0 or low

# SIFT should detect it (texture-based)
sift_det = SIFTDetector(min_keypoints=3, cluster_eps=30.0)
sift_dets = sift_det.detect(frame)
print(f"SIFT detections: {len(sift_dets)}")  # Should find bee

# Visualize
vis = frame.copy()
for det in sift_dets:
    x1, y1, x2, y2 = [int(c) for c in det.bbox]
    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(vis, "SIFT", (x1, y1-5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

cv2.imshow('SIFT Stationary Detection', vis)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

═══════════════════════════════════════════════════════════════════════════
8. NOISE FILTER TEST
═══════════════════════════════════════════════════════════════════════════

Test if noise filter removes false positives:

```python
# test_noise_filter.py
from beemonitor.detection import BlobDetector, NoiseFilter
import cv2

frame = cv2.imread('test_frame.jpg')

# Get blob detections (may have false positives)
blob_det = BlobDetector(min_area=50)
blob_dets = blob_det.detect(frame)
print(f"Before filter: {len(blob_dets)} detections")

# Apply noise filter
noise_filter = NoiseFilter(classifier=cnn_model, threshold=0.7)
filtered_dets = noise_filter.filter_detections(frame, blob_dets)
print(f"After filter: {len(filtered_dets)} detections")

# Visualize removed detections
vis = frame.copy()
for det in blob_dets:
    if det in filtered_dets:
        # Kept - green
        color = (0, 255, 0)
        label = "KEPT"
    else:
        # Removed - red
        color = (0, 0, 255)
        label = "REMOVED"
    
    x1, y1, x2, y2 = [int(c) for c in det.bbox]
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
    cv2.putText(vis, label, (x1, y1-5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

cv2.imshow('Noise Filter', vis)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

═══════════════════════════════════════════════════════════════════════════
RECOMMENDED TESTING WORKFLOW
═══════════════════════════════════════════════════════════════════════════

1. Start with Visual Testing
   - Run test_visual.py to see what's happening
   - Try different detection modes
   - Verify detections look correct

2. Test Individual Detectors
   - Run test_detector_comparison.py
   - Compare Blob vs SIFT vs YOLO
   - Identify which works best for your data

3. Test Stationary Detection
   - Run test_stationary_detection.py
   - Verify SIFT finds stationary bees
   - Confirm FG/BG misses them

4. Performance Benchmarking
   - Run test_performance.py
   - Compare speed of different modes
   - Choose optimal mode for your needs

5. Integration Testing
   - Run test_bee_tracking.py
   - Test full pipeline
   - Verify results are correct

6. Compare MOT Algorithms
   - Run test_mot_comparison.py
   - Compare BeeTracker vs ByteTrack
   - Choose best for your scenario

═══════════════════════════════════════════════════════════════════════════
DEBUGGING TIPS
═══════════════════════════════════════════════════════════════════════════

If no detections:
- Check frame is not empty
- Verify detector parameters (min_area, conf_threshold)
- Try different detection modes
- Visualize to see what's happening

If too many false positives:
- Enable noise filter
- Increase confidence thresholds
- Use stricter detection parameters
- Switch to YOLO_ONLY mode

If missing bees:
- Try FGBG_SIFT mode (catches stationary)
- Decrease min_area threshold
- Decrease conf_threshold
- Check ROI is correct

If tracking is jumpy:
- Decrease iou_threshold
- Increase min_hits
- Try different MOT algorithm
- Check detection quality

If too slow:
- Use FGBG_ONLY mode
- Reduce YOLO frequency
- Process at lower resolution
- Use frame skipping

═══════════════════════════════════════════════════════════════════════════
