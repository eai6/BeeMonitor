# src/beemonitor Source Code Synthesis

## Package Structure

```
src/beemonitor/ (~25,000 LOC, 43 Python files)
├── __init__.py
├── core/
│   ├── video_analyzer.py     (347 lines)  — BeeMonitor orchestrator
│   ├── config.py             (1,116 lines) — Configuration with adaptive scaling
│   └── analysis_results.py   (224 lines)  — Results container
├── detection/
│   ├── base_detector.py      (358 lines)  — Abstract base + Detection dataclass
│   ├── yolo_detector.py      (160 lines)  — Ultralytics YOLO wrapper
│   ├── blob_detector.py      (1,276 lines) — MOG2 background subtraction
│   ├── sift_detector.py      (893 lines)  — SIFT keypoint detection
│   ├── nest_detector.py      (906 lines)  — Nest hole detection + clustering
│   └── noise_filter.py       (425 lines)  — PyTorch CNN noise filter
├── tracking/
│   ├── base_tracking.py      (90 lines)   — Base interface
│   ├── bee_tracking.py       (1,373 lines) — Tracking orchestrator
│   └── mot/
│       ├── base_mot.py       (92 lines)   — MOT base class
│       ├── bee_tracker.py    (673 lines)  — Kalman + Hungarian tracker
│       └── ultralytics_tracker.py (252 lines) — ByteTrack/BoT-SORT wrapper
├── processing/
│   ├── event_processor.py    (439 lines)  — ML-first event detection
│   ├── event_classifier.py   (396 lines)  — Random Forest classifier
│   ├── trajectory_analyzer.py (1,007 lines) — Movement analysis
│   └── interaction_analyzer.py (441 lines) — Bee interaction patterns
├── output/
│   ├── csv_generator.py      (266 lines)  — CSV export with timestamps
│   └── video_synthesizer.py  (1,436 lines) — Annotated video generation
├── visualization/
│   └── detection_source_visualizer.py
├── utils/
│   └── geometry.py           (706 lines)  — Bbox ops, spatial calculations
└── tests/                    (2,260 lines) — Test suite
```

---

## Core Classes & APIs

### BeeMonitor (Main Entry Point)
```python
from beemonitor import BeeMonitor, Config

config = Config.from_yaml("config.yaml")
monitor = BeeMonitor(config)
results = monitor.analyze_video(
    video_path="video.mp4",
    output_folder="output",
    visualize=True,
    detection_mode='yolo'  # or 'two_mode'
)
# results.events → DataFrame, results.tracks → DataFrame
```

**Key Methods:**
- `analyze_video()` — Full pipeline: nest detection → tracking → events → output
- `get_nest_detections()` — Just nest detection
- `get_motion_tracking()` — Detection + tracking only
- `process_motion_tracking()` — Events from tracks
- `synthesize_csv()` — Timestamp generation

### Config System (Dataclasses)
- `VideoConfig` — Resolution, FPS, auto-detection
- `ModelConfig` — Paths to .pt and .pkl models
- `TrackingConfig` — Adaptive params (seconds-based → frames)
- `NestConfig` — Reference-resolution scaling
- `DetectionConfig` — Blob detector thresholds
- `OutputConfig` — CSV columns, output format

### Detection (`Detection` dataclass)
```python
@dataclass
class Detection:
    bbox: Tuple[float, float, float, float]  # x1,y1,x2,y2
    centroid: Tuple[float, float]
    confidence: float
    label: str        # 'bee', 'wasp', etc.
    source: str       # 'yolo', 'blob', 'sift'
    metadata: Optional[Dict]
```

---

## Algorithm Details

### BeeTracker (Kalman + Hungarian MOT)
Per-frame loop:
1. **Predict** — Kalman filter predicts next position for each track
2. **Match** — Hungarian algorithm (scipy `linear_sum_assignment`) matches detections to predictions
3. **Update** — Matched tracks get Kalman update; unmatched detections create new tracks
4. **Manage** — Confirm tracks (hits ≥ min_hits), kill stale tracks, resurrect recently-lost tracks
5. **De-duplicate** — Remove near-identical simultaneous tracks

**Adaptive Parameters (scale with FPS + bee size):**
```python
max_age = max_age_seconds × fps              # e.g., 0.5s × 30fps = 15 frames
min_hits = min_hits_seconds × fps             # e.g., 0.1s × 30fps = 3 frames
match_distance = multiplier × bee_size        # e.g., 8.0 × 25px = 200px
resurrection_search = multiplier × bee_size   # e.g., 3.0 × 25px = 75px
```

**Bee Size Estimation (IQR):**
- Collect all detection sizes
- Filter outliers with IQR method (Q1-1.5×IQR to Q3+1.5×IQR)
- Use median of filtered sizes

### Event Classification (ML-First)
**20 Trajectory Features:**
- Shape: length, path_length, displacement, tortuosity
- Speed: avg, max, std, cv, start, middle, end, decel_ratio
- Nest proximity: start_to_nest, end_to_nest, approach_ratio
- Variance: x_var, y_var
- Direction: vertical_movement, horizontal_movement, is_entry

**Random Forest:** 100 trees, max_depth=10, balanced weights
- At threshold 0.6: 96.4% precision, 90.0% recall, F1=0.931

### Nest Detection
1. YOLO inference on reference frames
2. Cluster detections into rows/columns (k-means)
3. Interpolate missing holes
4. Assign unique IDs → Dict[nest_id, bbox]

---

## Module Dependencies (Import Graph)

```
video_analyzer.py
  ├── config.py
  ├── detection/ (YOLODetector, BlobDetector, NestDetector)
  ├── tracking/ (BeeTracking)
  ├── processing/ (EventProcessor)
  └── output/ (CSVGenerator, VideoSynthesizer)

BeeTracking
  ├── YOLODetector (primary detection)
  ├── BlobDetector (motion sensing)
  └── BeeTracker (MOT algorithm)
      └── scipy.optimize.linear_sum_assignment

EventProcessor
  ├── TrajectoryAnalyzer (feature extraction)
  └── EventClassifier (Random Forest via joblib)
```

---

## Key Configuration Defaults

```yaml
tracking:
  max_age_seconds: 0.5
  min_hits_seconds: 0.1
  max_resurrection_seconds: 0.3
  match_distance_multiplier: 8.0
  confidence_threshold: 0.25

detection:
  min_area: 120
  max_area: 4000
  min_solidity: 0.7

event:
  ml_threshold: 0.6
  window_size: 1
  padding: 40

nest:
  expected_total_nests: 60
  expected_rows: 6
  expected_nests_per_row: 10
```

---

## Models Required

| Model | File | Size | Purpose |
|-------|------|------|---------|
| Nest Detection | nest_detection.pt | 40.8 MB | YOLO nest hole detector |
| Bee Tracking | bee_tracking.pt | 5.4 MB | YOLO bee/wasp detector |
| Event Classifier | event_classifier_model.pkl | 617 KB | Random Forest classifier |
| Noise Filter | blob_noise_classifier.pth | — | CNN false positive filter (optional) |
