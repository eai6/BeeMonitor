# BeeMonitor

**An open-source machine learning system for automated monitoring of cavity-nesting solitary bees**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![YOLO26](https://img.shields.io/badge/YOLO-26-green.svg)](https://github.com/ultralytics/ultralytics)

BeeMonitor is an integrated hardware and software system for automated video surveillance and AI-powered analysis of cavity-nesting solitary bee activity at bee hotels. The system extracts entry/exit events from nesting tubes without requiring individual bee marking.

---

## Performance

Evaluated on 110 minutes of video containing 300 manually annotated foraging events:

| Mode | Precision | Recall | F1 Score | Processing Speed |
|------|-----------|--------|----------|------------------|
| **Full Tracking** | 93.9% | 87.7% | **0.907** | 2.3× real-time |
| **Two-Mode Adaptive** | 92.0% | 84.3% | 0.880 | **0.8× real-time** |

*Benchmarked on Apple M3 Pro (18GB) with 4 parallel workers and MPS acceleration.*

---

## Features

### Software Pipeline
- **YOLO26 Object Detection** — End-to-end NMS-free detection with up to 43% faster CPU inference
- **BeeTrack MOT Algorithm** — Custom multiple-object tracking optimized for fast-moving insects
- **ML Event Classifier** — Random Forest classifier distinguishes real events from noise
- **Two-Mode Adaptive Processing** — Motion detection skips idle periods for 2.9× speedup
- **Batch Processing** — Parallel video processing with configurable workers

### Hardware System (~$350 USD)
- Raspberry Pi 4 + Witty Pi 4 power management
- Raspberry Pi HQ Camera (30 fps, 1080p)
- Solar panel + LiFePO4 battery for off-grid deployment
- Weatherproof 3D-printed enclosure

---

## Installation

### Requirements
- Python 3.10+
- macOS, Linux, or Windows
- GPU recommended (CUDA, MPS, or ROCm) but not required

### Install from Source

```bash
git clone https://github.com/yourusername/beemonitor.git
cd beemonitor
pip install -e .
```

This installs all dependencies including PyTorch, Ultralytics, OpenCV, and scikit-learn.

### Verify Installation

```bash
python -c "from beemonitor import BeeMonitor; print('✓ BeeMonitor installed')"
```

---

## Quick Start

### Analyze a Single Video

```python
from beemonitor import BeeMonitor
from beemonitor.core.config import Config

# Load configuration
config = Config.from_yaml("config.yaml")

# Initialize monitor
monitor = BeeMonitor(config=config)

# Analyze video
results = monitor.analyze_video(
    video_path="path/to/video.mp4",
    output_folder="output/"
)

# Access detected events
print(f"Detected {len(results.events)} events")
for event in results.events:
    print(f"  {event.action} at nest {event.nest} (frame {event.frame_number})")
```

### Batch Processing

```python
from beemonitor import BeeMonitor
from beemonitor.core.config import Config
from pathlib import Path
import concurrent.futures

config = Config.from_yaml("config.yaml")
video_folder = Path("videos/")
output_folder = Path("output/")

def process_video(video_path):
    monitor = BeeMonitor(config=config)  # Fresh instance per video
    return monitor.analyze_video(str(video_path), str(output_folder / video_path.stem))

videos = list(video_folder.glob("*.mp4"))

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_video, videos))
```

---

## Configuration

### Minimal Config (config.yaml)

```yaml
video:
  fps: 30
  auto_detect_from_video: true

models:
  nest_detection: "models/nest_detection.pt"
  tracking: "models/bee_tracking.pt"
  event_classifier: "models/event_classifier_model.pkl"

tracking:
  confidence_threshold: 0.25
  max_age_seconds: 1.0
  min_hits_seconds: 0.1
  match_distance_multiplier: 8.0

processing:
  ml_threshold: 0.6  # Event classifier confidence threshold
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `confidence_threshold` | 0.25 | YOLO detection confidence |
| `ml_threshold` | 0.6 | Event classifier threshold (0.3-0.6) |
| `max_age_seconds` | 1.0 | Max time without detection before track dies |
| `match_distance_multiplier` | 8.0 | Association distance (× bee size) |

### ML Threshold Guide

| Threshold | Precision | Recall | F1 | Use Case |
|-----------|-----------|--------|-----|----------|
| 0.3 | 84.2% | 90.7% | 0.873 | Maximum recall |
| 0.4 | 88.0% | 90.3% | 0.891 | Balanced |
| 0.5 | 91.1% | 89.0% | 0.901 | High precision |
| **0.6** | **93.9%** | 87.7% | **0.907** | **Recommended** |

---

## Output Structure

```
output/
├── video_name/
│   ├── events.csv           # Detected entry/exit events
│   ├── tracks.csv           # Full tracking data (frame-by-frame)
│   ├── nests.json           # Detected nest tube locations
│   ├── summary.json         # Processing metadata
│   ├── background.png       # Background model visualization
│   └── crops/               # (Optional) Bee image crops per track
│       └── video_name/
│           ├── track_0001/
│           │   ├── frame_000123.jpg
│           │   └── frame_000456.jpg
│           └── track_0002/
```

### events.csv

| Column | Description |
|--------|-------------|
| `frame_number` | Frame where event occurred |
| `action` | "Entry" or "Exit" |
| `nest` | Nest tube ID (e.g., "R2C5" = Row 2, Column 5) |
| `track_id` | Associated track ID |
| `ml_confidence` | Event classifier confidence score |

---

## Project Structure

```
beemonitor/
├── __init__.py
├── core/
│   ├── config.py              # Configuration management
│   ├── video_analyzer.py      # Main BeeMonitor class
│   └── analysis_results.py    # Result containers
├── detection/
│   ├── yolo_detector.py       # YOLO26 wrapper (NMS-free end-to-end)
│   ├── blob_detector.py       # Motion detection (two-mode)
│   ├── nest_detector.py       # Nest grid detection
│   └── noise_filter.py        # Detection filtering
├── tracking/
│   ├── bee_tracking.py        # BeeTracking orchestrator
│   └── mot/
│       ├── bee_tracker.py     # BeeTrack MOT algorithm
│       └── base_mot.py        # Abstract tracker interface
├── processing/
│   ├── event_processor.py     # ML-first event detection
│   ├── event_classifier.py    # Random Forest classifier
│   ├── trajectory_analyzer.py # Trajectory feature extraction
│   └── interaction_analyzer.py
├── output/
│   ├── csv_generator.py       # CSV export
│   └── video_synthesizer.py   # Annotated video output
├── utils/
│   └── geometry.py            # Spatial utilities
└── visualization/
    └── detection_source_visualizer.py
```

---

## Algorithm Details

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

---

## Hardware Setup

See [HARDWARE.md](docs/HARDWARE.md) for detailed assembly instructions.

### Bill of Materials (~$350 USD)

| Component | Price |
|-----------|-------|
| Raspberry Pi 4 (4GB) | $55–65 |
| Witty Pi 4 | $70–75 |
| Raspberry Pi HQ Camera | $50–55 |
| 6mm CS-Mount Lens | $25–35 |
| 256GB MicroSD | $25–30 |
| 100W Solar Panel | $80 |
| 12V 30Ah LiFePO4 Battery | $80–100 |
| Enclosure & Misc | $40–50 |

---

## GPU Acceleration

### Apple Silicon (MPS)

```python
# Automatic — PyTorch detects MPS
import torch
print(torch.backends.mps.is_available())  # True
```

### NVIDIA (CUDA)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce `--workers` or process sequentially |
| Low recall | Lower `ml_threshold` to 0.4 or 0.3 |
| Too many false positives | Raise `ml_threshold` to 0.6 |
| Slow processing | Enable two-mode tracking, use GPU |
| Track fragmentation | Increase `max_age_seconds` to 1.5 |

---

## Citation

If you use BeeMonitor in your research, please cite:

```bibtex
@article{amoah2025beemonitor,
  title={BeeMonitor: An open-source machine learning system for automated 
         monitoring of cavity-nesting solitary bee activity},
  author={Amoah, Edward and Grozinger, Christina M.},
  journal={Methods in Ecology and Evolution},
  year={2026},
  doi={10.1111/2041-210X.XXXXX}
}
```

If you use YOLO26, please also cite:

```bibtex
@software{yolo26_ultralytics,
  author = {Glenn Jocher and Jing Qiu},
  title = {Ultralytics YOLO26},
  version = {26.0.0},
  year = {2026},
  url = {https://github.com/ultralytics/ultralytics},
  license = {AGPL-3.0}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- NSF Research Traineeship Program (INSECT NET, Grant 2243979)
- USDA NIFA Hatch and Smith-Lever Appropriations (Projects PEN04943, PEN08801)
- Penn State Joan Luerssen Faculty Enhancement Fund

---

## Contact

- **Author:** Edward Amoah
- **Email:** eai6@psu.edu
- **Lab:** [Grozinger Lab](https://ento.psu.edu/directory/cmg25), Penn State University