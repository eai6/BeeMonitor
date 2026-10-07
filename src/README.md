# BeeMonitor analysis engine

The `beemonitor` Python package does the computer-vision work: detection, multi-object tracking, entry and
exit events at nest tubes, and species and marker identification by vote over each track's crops. The
platform's GPU workers run it, and it also runs on its own on a laptop or a cluster.

## Install

```bash
git clone https://github.com/eai6/BeeMonitor.git && cd BeeMonitor
# CPU build of PyTorch first: avoids several GB of CUDA packages, and the default
# wheels crash on a Raspberry Pi. On an NVIDIA machine, install the CUDA build instead.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

The released weights in [`models/`](../models/) are used by default. Set `BEEMONITOR_MODELS_DIR` to use
weights from somewhere else.

## Analyse a clip

```python
from beemonitor import BeeMonitor
from beemonitor.core.config import Config

monitor = BeeMonitor(config=Config.default())
results = monitor.analyze_video("examples/mendels_2024-05-08_15_57_03.mp4",
                                output_folder="output/")
print(results.events)   # one row per entry or exit: action, nest, frame, time
```

`output/` gets the events, tracking and detections as CSV files.

### Many clips

```python
import concurrent.futures
from pathlib import Path

def analyse(path):
    monitor = BeeMonitor(config=Config.default())   # one instance per clip
    return monitor.analyze_video(str(path), output_folder=f"output/{path.stem}")

clips = sorted(Path("videos/").glob("*.mp4"))
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(analyse, clips))
```

## Package layout

| Module | What it does |
|---|---|
| `core/` | `BeeMonitor` (the video analyser) and `Config` |
| `detection/` | YOLO and SAM 3 detectors, nest detection, motion blobs, tiling for large photos |
| `tracking/` | BeeTrack, plus ByteTrack, BoT-SORT, OC-SORT and SFSORT behind one interface; padded crops of every tracked frame |
| `processing/` | Event classifier (entry or exit), trajectory features, interactions |
| `identification/` | BeeMachine and BioCLIP species classifiers, colour-marker reading, voting over a track's crops |
| `output/`, `visualization/` | CSV writers and annotated video |

## Performance on bee hotels

Evaluated on 110 minutes of video containing 300 manually annotated foraging events:

| Mode | Precision | Recall | F1 Score | Processing Speed |
|------|-----------|--------|----------|------------------|
| **Full Tracking** | 93.9% | 87.7% | **0.907** | 2.3× real-time |
| **Two-Mode Adaptive** | 92.0% | 84.3% | 0.880 | **0.8× real-time** |

*Benchmarked on Apple M3 Pro (18GB) with 4 parallel workers and MPS acceleration.*

## How it works

### YOLO26 Object Detection

BeeMonitor uses fine-tuned YOLO26 models for bee and nest detection. Key advantages of YOLO26:

- **End-to-end NMS-free inference** — No post-processing required, simplifying deployment
- **Up to 43% faster CPU inference** — Critical for edge devices like Raspberry Pi
- **Improved small object detection** — Better accuracy for fast-moving bees
- **Simplified export** — DFL removal improves compatibility across platforms

We fine-tuned separate models for:
1. **Nest detection** — Identifies 60-tube grid layout (runs once per video)
2. **Bee detection** — Locates bees in each frame (confidence threshold 0.25)

### BeeTrack MOT

BeeTrack is a tracking-by-detection algorithm optimized for fast-moving insects:

1. **Adaptive Kalman Filter** — Position prediction with velocity smoothing
2. **Hungarian Assignment** — Optimal detection-to-track association
3. **Track Lifecycle Management** — Handles occlusions with resurrection capability

Key innovations:
- **Adaptive thresholds** scale with detected bee size and video FPS
- **Distance clamping** prevents wild predictions during rapid direction changes
- **Track resurrection** recovers temporarily lost tracks within 0.5s window

### Two-Mode Adaptive Processing

```
┌─────────────────────────────────────────────────────────┐
│                    Motion Detection Mode                │
│  • Lightweight blob detection on ROI                    │
│  • Skip YOLO inference when no motion                   │
│  • Maintain 0.5s lookback buffer                        │
└─────────────────────────┬───────────────────────────────┘
                          │ Motion detected
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    Full Tracking Mode                   │
│  • YOLO26 detection on full frame (NMS-free)            │
│  • BeeTrack MOT processing                              │
│  • Process lookback buffer first                        │
│  • 30-frame cooldown before returning to motion mode    │
└─────────────────────────────────────────────────────────┘
```

### ML Event Classification

The event classifier extracts 20 features from trajectory segments:

| Category | Features |
|----------|----------|
| **Trajectory Shape** | length, path_length, displacement, tortuosity |
| **Speed Profile** | avg, max, std, cv, start/middle/end speed, decel_ratio |
| **Nest Proximity** | start_to_nest, end_to_nest, approach_ratio |
| **Position Variance** | x_var, y_var |
| **Direction** | vertical_movement, horizontal_movement, is_entry |

## Tests

```bash
PYTHONPATH=src python -m pytest src/beemonitor/tests
```
