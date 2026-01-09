"""BeeMonitor Architecture Documentation

This document explains the modular detection and tracking architecture.

═══════════════════════════════════════════════════════════════════════════
ARCHITECTURE OVERVIEW
═══════════════════════════════════════════════════════════════════════════

The system is organized into two main processing layers:

1. DETECTION (Spatial) - Find objects in individual frames
2. TRACKING (Temporal) - Follow objects across frames

Each layer has abstract base classes that allow pluggable implementations.

═══════════════════════════════════════════════════════════════════════════
DETECTION MODULE (beemonitor.detection)
═══════════════════════════════════════════════════════════════════════════

Purpose: Find objects in static images (single frame analysis)

Base Interface:
    BaseDetector - Abstract class all detectors inherit from
    Detection - Data class for detection results

Available Detectors:
    BlobDetector     - FG/BG subtraction + morphology (fast, motion-based)
    SIFTDetector     - SIFT keypoints + clustering (finds stationary bees!)
    YOLODetector     - Deep learning detection (accurate, species ID)
    NestDetector     - Nest tube detection (YOLO-based)
    NoiseFilter      - CNN classifier (false positive filtering)

Detection Pipeline:
    ┌──────────────┐
    │    Frame     │
    └──────┬───────┘
           │
    ┌──────▼────────┐
    │   Detectors   │  ← BlobDetector / SIFTDetector / YOLODetector
    └──────┬────────┘
           │
    ┌──────▼────────┐
    │ Noise Filter  │  ← Optional CNN filtering
    └──────┬────────┘
           │
    ┌──────▼────────┐
    │  Detections   │  ← List[Detection]
    └───────────────┘

Usage Example:
    ```python
    from beemonitor.detection import BlobDetector, YOLODetector
    
    # Create detectors
    blob_det = BlobDetector(min_area=50, min_solidity=0.5)
    yolo_det = YOLODetector(model, conf_threshold=0.25)
    
    # Detect in frame
    blob_detections = blob_det.detect(frame)
    yolo_detections = yolo_det.detect(frame)
    
    # Each detection has:
    #   - bbox: (x1, y1, x2, y2)
    #   - centroid: (x, y)
    #   - confidence: 0.0-1.0
    #   - label: 'bee', 'wasp', etc.
    #   - source: 'blob', 'yolo', etc.
    ```

═══════════════════════════════════════════════════════════════════════════
TRACKING MODULE (beemonitor.tracking)
═══════════════════════════════════════════════════════════════════════════

Purpose: Follow objects across frames (temporal analysis)

Two-Level Architecture:
    1. High-Level: Tracking Systems (BeeTracking)
       - Orchestrates detection + MOT + domain logic
       - Bee hotel specific intelligence
    
    2. Low-Level: MOT Algorithms (BeeTracker, ByteTrack)
       - Pure track association logic
       - Prediction + matching + state management

Base Interfaces:
    BaseTracking - Abstract class for tracking systems
    BaseMOT - Abstract class for MOT algorithms

Tracking System Hierarchy:
    BaseTracking (abstract)
        │
        └── BeeTracking (bee hotel specific)
                - Detection pipeline orchestration
                - Mode switching (motion detection ↔ tracking)
                - Frame merging
                - ROI management

MOT Algorithm Hierarchy:
    BaseMOT (abstract)
        │
        ├── BeeTracker (custom Kalman + Hungarian)
        │       - Kalman filter prediction
        │       - Hungarian algorithm matching
        │       - Immature track handling
        │
        └── UltralyticsTracker (ByteTrack / BoT-SORT wrapper)
                - Wraps ultralytics tracking algorithms

Tracking Pipeline:
    ┌─────────────────────────────────────────────────────────┐
    │                    BeeTracking System                   │
    │                                                         │
    │  ┌─────────────┐      ┌─────────────┐                 │
    │  │   Detectors │      │     MOT     │                 │
    │  │             │      │  Algorithm  │                 │
    │  │ • Blob      │──┬──▶│             │                 │
    │  │ • SIFT      │  │   │ BeeTracker  │──▶ Tracks      │
    │  │ • YOLO      │  │   │    or       │                 │
    │  └─────────────┘  │   │ ByteTrack   │                 │
    │                   │   └─────────────┘                 │
    │  ┌─────────────┐  │                                   │
    │  │Noise Filter │──┘                                   │
    │  └─────────────┘                                      │
    └─────────────────────────────────────────────────────────┘

Detection Modes (configurable):
    FGBG_ONLY          - Fast motion detection only
    SIFT_ONLY          - Stationary bee detection only
    FGBG_SIFT          - Both moving and stationary bees
    FGBG_YOLO          - Motion + YOLO confirmation (default)
    SIFT_YOLO          - SIFT + YOLO confirmation
    FGBG_SIFT_YOLO     - All three methods (comprehensive)
    YOLO_ONLY          - Maximum accuracy (expensive)

Usage Example:
    ```python
    from beemonitor.tracking import BeeTracking, DetectionMode
    from beemonitor.tracking.mot import BeeTracker
    
    # Create MOT algorithm
    mot = BeeTracker(config, tracking_classes=['bee', 'wasp'])
    
    # Create bee tracking system
    tracker = BeeTracking(
        mot_algorithm=mot,
        yolo_model=yolo_model,
        detection_mode=DetectionMode.FGBG_SIFT_YOLO,  # Comprehensive
        use_noise_filter=True,
        noise_filter_model=cnn_model,
        config=config
    )
    
    # Process video
    results = tracker.process_video(
        video_path='video.mp4',
        roi=(100, 100, 800, 600)  # Hotel box region
    )
    
    # Results is a DataFrame with columns:
    #   frame, track_id, x1, y1, x2, y2, species, confidence
    ```

═══════════════════════════════════════════════════════════════════════════
KEY DESIGN PRINCIPLES
═══════════════════════════════════════════════════════════════════════════

1. Separation of Concerns
   - Detection = Find objects in images (spatial)
   - Tracking = Follow objects over time (temporal)
   - MOT = Low-level track association
   - BeeTracking = High-level orchestration

2. Abstraction & Pluggability
   - BaseDetector allows swapping detection methods
   - BaseMOT allows swapping tracking algorithms
   - Easy to test individual components

3. Configuration Over Code
   - DetectionMode enum for different pipelines
   - All parameters configurable
   - Easy experimentation

4. Hybrid Approaches
   - Combine multiple detectors (FG/BG + SIFT + YOLO)
   - Best of all worlds: speed + stationary detection + accuracy

═══════════════════════════════════════════════════════════════════════════
INTEGRATION WITH BeeMonitor
═══════════════════════════════════════════════════════════════════════════

The high-level BeeMonitor class uses this architecture:

```python
from beemonitor.core import BeeMonitor
from beemonitor.tracking import BeeTracking, DetectionMode
from beemonitor.tracking.mot import BeeTracker

# BeeMonitor.analyze_video() internally uses:
#   1. NestDetector to find nests
#   2. BeeTracking to track bees
#   3. EventProcessor to identify entry/exit events
```

═══════════════════════════════════════════════════════════════════════════
ADVANTAGES OF THIS ARCHITECTURE
═══════════════════════════════════════════════════════════════════════════

1. Modularity
   - Each component has single responsibility
   - Easy to understand and maintain

2. Extensibility
   - Add new detectors (ORB, AKAZE, template matching)
   - Add new MOT algorithms (SORT, DeepSORT, StrongSORT)
   - Add new tracking systems (multi-camera, different species)

3. Testability
   - Test detectors independently
   - Test MOT algorithms independently
   - Mock dependencies easily

4. Flexibility
   - Mix and match detection methods
   - Choose accuracy vs speed trade-offs
   - Adapt to different scenarios

5. Addresses Your Concerns
   - SIFT solves stationary bee detection
   - FG/BG+SIFT solves fast-moving blur issues
   - Configurable pipeline handles all cases

═══════════════════════════════════════════════════════════════════════════
FUTURE ENHANCEMENTS
═══════════════════════════════════════════════════════════════════════════

Potential additions:

Detectors:
- ORBDetector (faster than SIFT)
- TemplateMatchingDetector (for known bee patterns)
- OpticalFlowDetector (motion vectors)

MOT Algorithms:
- SORT (Simple Online Realtime Tracking)
- DeepSORT (SORT + appearance features)
- StrongSORT (DeepSORT improvements)

Tracking Systems:
- MultiCameraTracking (synchronized cameras)
- SpeciesSpecificTracking (different logic per species)
- IndoorOutdoorTracking (lighting adaptation)

"""
