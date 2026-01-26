# BeeMonitor Source Code

**Core analysis engine for automated bee monitoring**

Version 1.0.0


## Overview

The BeeMonitor package implements a modular computer vision pipeline for detecting and tracking bee activity in bee hotel videos. The architecture separates detection (finding objects in frames), tracking (following objects over time), event processing (identifying biological behaviors), and output generation.

![alt text](beemonitor_software.png)

## Package Structure

```
beemonitor/
├── core/              # Pipeline orchestration & configuration
├── detection/         # Object detection algorithms
├── tracking/          # Multi-object tracking systems
│   └── mot/          # Low-level MOT algorithms
├── processing/        # Event detection & analysis
├── output/            # Results export & visualization
├── utils/             # Shared utilities
└── tests/             # Test suite
```

---

## Module Descriptions

### core/ - Pipeline Orchestration

**Main Files:**
- `video_analyzer.py` - Complete analysis pipeline (`BeeMonitor` class)
- `config.py` - Configuration management (`Config` class)
- `analysis_results.py` - Results data structures (`AnalysisResults`)

**Purpose:** High-level API for end-to-end video analysis. Coordinates all modules.

**Key Class:**
```python
from beemonitor import BeeMonitor, Config

# Initialize
config = Config.default()
analyzer = BeeMonitor(config)

# Process video
results = analyzer.analyze_video('bee_hotel.mp4')

# Access outputs
events_df = results.events
tracks_df = results.tracks
```

**Pipeline Flow:**
1. Initialize detectors (Blob & YOLO)
2. Initialize tracking system
3. Process frames → detections → tracks
4. Extract events from tracks
5. Generate outputs

### detection/ - Object Detection

**Main Files:**
- `base_detector.py` - Abstract detector interface (`BaseDetector`, `Detection`)
- `blob_detector.py` - Motion-based detection (`BlobDetector`)
- `yolo_detector.py` - Deep learning detection (`YOLODetector`)
- `nest_detector.py` - Nest tube localization (`NestDetector`)

**Purpose:** Multiple detection methods for different scenarios (moving/stationary bees).

**Example:**
```python
from beemonitor.detection import BlobDetector, YOLODetector

# Initialize detectors
blob = BlobDetector(min_area=50, max_area=5000)
yolo = YOLODetector('yolov8n.pt', conf_threshold=0.6)

# Detect in frame
blob_dets = blob.detect(frame)
yolo_dets = yolo.detect(frame)

# All return List[Detection]
# Detection: bbox, centroid, confidence, label, source
```

**Detection Object:**
```python
@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    centroid: Tuple[float, float]     # (x, y)
    confidence: float                 # 0.0-1.0
    label: str                        # 'bee', 'wasp', etc.
    source: str                       # 'blob', 'sift', 'yolo'
    metadata: Dict = field(default_factory=dict)
```

### tracking/ - Multi-Object Tracking

**Main Files:**
- `base_tracking.py` - Abstract tracking interface (`BaseTracking`)
- `bee_tracking.py` - Bee hotel tracking orchestrator (`BeeTracking`)
- `mot/base_mot.py` - Abstract MOT interface (`BaseMOT`)
- `mot/bee_tracker.py` - Custom Kalman tracker (`BeeTracker`)
- `mot/ultralytics_tracker.py` - ByteTrack/BoT-SORT wrapper (`UltralyticsTracker`)

**Purpose:** Two-layer architecture for flexible tracking.

**Architecture:**

```
Layer 1: High-Level Tracking (BeeTracking)
  ├─ Manages detection pipeline
  ├─ Coordinates multiple detectors
  ├─ Feeds detections to MOT
  └─ Domain-specific logic
           ↓
Layer 2: MOT Algorithm (BeeTracker / ByteTrack)
  ├─ Track-detection association
  ├─ State prediction (Kalman)
  ├─ Track lifecycle management
  └─ ID consistency
```

**Example:**
```python
from beemonitor.tracking import BeeTracking
from beemonitor.tracking.mot import BeeTracker

# Initialize MOT algorithm
mot = BeeTracker(
    max_age=30,
    min_hits=3,
    iou_threshold=0.3
)

# Create high-level tracking system
tracker = BeeTracking(
    mot_algorithm=mot,
    yolo_model='yolov8n.pt',
    detection_mode='YOLO_ONLY'
)

# Process video
results = tracker.process_video('video.mp4')
```

**Track Object:**
```python
@dataclass
class Track:
    track_id: int
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[float, float]
    label: str
    age: int
    frames_without_detection: int
    trajectory: List[Tuple[int, Tuple[float, float]]]  # [(frame, centroid), ...]
    velocity: Tuple[float, float]
```

### processing/ - Event Analysis

**Main Files:**
- `event_processor.py` - Entry/exit event extraction (`EventProcessor`)
- `event_classifier.py` - ML-based event classification (`EventClassifier`)
- `trajectory_analyzer.py` - Movement pattern analysis (`TrajectoryAnalyzer`)
- `interaction_analyzer.py` - Bee interaction detection (`InteractionAnalyzer`)

**Purpose:** Extract biological behaviors from tracking data.

**Example:**
```python
from beemonitor.processing import EventProcessor, TrajectoryAnalyzer

# Initialize
event_processor = EventProcessor(nest_positions)
traj_analyzer = TrajectoryAnalyzer()

# Analyze tracks
events = event_processor.process_tracks(tracks_df)
metrics = traj_analyzer.compute_metrics(tracks_df)

# Events DataFrame columns:
# - event_type: 'entry', 'exit', 'inspection'
# - track_id, nest_id, timestamp, duration
# - confidence: ML classifier score
```

**Event Types:**
- **Entry:** Bee entering nest tube
- **Exit:** Bee exiting nest tube
- **Inspection:** Bee hovering at entrance without entering

**Trajectory Metrics:**
- Speed distribution
- Acceleration patterns
- Directional consistency
- Flight smoothness

### output/ - Results Export

**Main Files:**
- `csv_generator.py` - CSV export functionality (`CSVGenerator`)
- `video_synthesizer.py` - Annotated video generation (`VideoSynthesizer`)

**Purpose:** Generate analysis outputs in various formats.

**Example:**
```python
from beemonitor.output import CSVGenerator, VideoSynthesizer

# CSV export
csv_gen = CSVGenerator()
csv_gen.save_events(events, 'events.csv')
csv_gen.save_tracks(tracks, 'tracks.csv')
csv_gen.save_metrics(metrics, 'metrics.json')

# Video visualization
video_syn = VideoSynthesizer()
video_syn.create_annotated_video(
    video_path='input.mp4',
    tracks=tracks,
    output_path='annotated.mp4',
    show_trajectories=True
)
```

**Output Files:**
- `*_events.csv` - Entry/exit events with timestamps
- `*_tracks.csv` - Complete trajectory data
- `*_metrics.json` - Performance statistics
- `*_annotated.mp4` - Visualization video

---

### utils/ - Shared Utilities

**Main Files:**
- `geometry.py` - Geometric computations

**Purpose:** Common utilities used across modules.

**Functions:**
- Bounding box operations (IoU, NMS, area)
- Coordinate transformations
- Distance calculations
- Polygon operations

## Data Flow

```
Input Video
    ↓
[DETECTION MODULE]
    ├─ BlobDetector → motion detection
    └─ YOLODetector → bee tracking
    ↓
List[Detection]
    ↓
[TRACKING MODULE]
    ├─ BeeTracking orchestrates detections
    └─ MOT algorithm (BeeTracker) associates & tracks
    ↓
Dict[track_id, Track]
    ↓
[PROCESSING MODULE]
    ├─ EventProcessor → entry/exit events
    └─EventClassifier → ML filtering
    ↓
Events + Metrics
    ↓
[OUTPUT MODULE]
    ├─ CSVGenerator → CSV files
    └─ VideoSynthesizer → annotated video
    ↓
Analysis Results
```

---

## Configuration

The system uses configuration py files inside core

## Quick Start

### Basic Usage

```python
from beemonitor import BeeMonitor, Config

# 1. Load configuration
config = Config.default()

# 2. Initialize analyzer
analyzer = BeeMonitor(config)

# 3. Process video
results = analyzer.analyze_video('bee_hotel.mp4')

# 4. Access results
print(f"Detected {len(results.events)} events")
print(f"Tracked {results.tracks['track_id'].nunique()} unique bees")

# 5. Save outputs
results.save_events('events.csv')
results.save_tracks('tracks.csv')
```

## Dependencies

**Core:**
- opencv-python >= 4.8.0
- numpy >= 1.24.0
- pandas >= 2.0.0
- ultralytics >= 8.0.0 (YOLOv8)
- scipy >= 1.11.0
- scikit-learn >= 1.3.0

**Optional:**
- torch >= 2.0.0 (GPU acceleration)
- tensorrt (NVIDIA GPU optimization)

## Architecture Principles

### Modularity
Each module has a clear responsibility and can be used independently.

### Extensibility
New detectors and trackers can be added by implementing base classes:
```python
class CustomDetector(BaseDetector):
    def detect(self, frame):
        # Your implementation
        return detections

class CustomMOT(BaseMOT):
    def update(self, detections, frame_num):
        # Your implementation
        return tracks
```

### Performance
- Optimized for laptop hardware (M3 Pro achieves near real-time)
- Adaptive algorithms reduce computation
- Parallel processing support

### Accuracy
- Multiple detection methods catch different scenarios
- ML-based noise filtering reduces false positives
- Custom tracking handles fast bee movements

