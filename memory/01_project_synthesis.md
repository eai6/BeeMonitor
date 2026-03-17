# BeeMonitor Project Synthesis

## Overview

**BeeMonitor** is an open-source integrated hardware and software system for automated video surveillance and AI-powered analysis of cavity-nesting solitary bee activity at bee hotels. It extracts entry/exit events from nesting tubes without requiring individual bee marking.

- **Author:** Edward Amoah (eai6@psu.edu)
- **Institution:** Grozinger Lab, INSECT-NET, Penn State University
- **License:** AGPL v3 | **Version:** 1.0.0 | **Python:** 3.10+

---

## Directory Structure

```
BeeMonitor_eai6/
├── src/beemonitor/          # Core Python package (~25,000 LOC)
│   ├── core/                # Config, orchestrator, results
│   ├── detection/           # YOLO, MOG2 blob, SIFT, nest detection
│   ├── tracking/            # Multi-object tracking (Kalman + Hungarian)
│   ├── processing/          # Event detection & ML classification
│   ├── output/              # CSV + annotated video generation
│   ├── visualization/       # Detection source overlays
│   ├── utils/               # Geometry helpers
│   ├── tests/               # Unit + integration tests
│   └── archives/            # Legacy code
│
├── desktop/                 # PyQt6 GUI application
│   ├── run_gui.py           # Entry point
│   └── beemonitor_gui/      # GUI modules (video panel, controls, dialogs)
│
├── hardware/                # Raspberry Pi acquisition system (~$595)
│   ├── main.py              # Recording service
│   ├── driver.py            # Device driver
│   └── enclosure/           # 3D-printable STL files
│
├── scripts/                 # Training, testing, validation scripts
├── cloud_scripts/           # HPC batch processing + SLURM
├── models/                  # Pre-trained models (~46 MB)
│   ├── nest_detection.pt    # YOLO nest model (40.8 MB)
│   ├── bee_tracking.pt      # YOLO bee model (5.4 MB)
│   └── event_classifier_model.pkl  # Random Forest (617 KB)
│
├── research/                # Jupyter notebooks, datasets, plots
├── data/                    # Sample videos + ground truth
├── setup.py / pyproject.toml / requirements.txt
└── README.md
```

---

## Major Components

### 1. Core Analysis Pipeline (`src/beemonitor/core/`)
- **BeeMonitor** — Main orchestrator class managing the full pipeline
- **Config** — Dynamic parameter scaling based on video resolution/FPS
- **AnalysisResults** — Results container with export methods
- Two modes: Full Tracking (YOLO-only) and Two-Mode Adaptive (motion + YOLO)

### 2. Detection System (`src/beemonitor/detection/`)
| Detector | Method | Purpose |
|----------|--------|---------|
| YOLODetector | Ultralytics YOLO | Primary bee/wasp detection |
| BlobDetector | MOG2 background subtraction | Motion detection for efficiency |
| SIFTDetector | SIFT keypoints | Stationary bee detection (legacy) |
| NestDetector | YOLO + clustering | Nest hole localization (60 tubes) |
| BeeNoiseFilter | PyTorch CNN | False positive filtering |

### 3. Tracking System (`src/beemonitor/tracking/`)
- **BeeTracker** — Custom MOT: Kalman filter + Hungarian assignment
- Adaptive thresholds scale with bee size, FPS, and resolution
- Track resurrection for occlusion handling (0.5s window)
- De-duplication of near-identical tracks

### 4. Event Processing (`src/beemonitor/processing/`)
- **EventProcessor** — ML-first event detection with 20 trajectory features
- **EventClassifier** — Random Forest (bee vs noise, entry vs exit)
- **TrajectoryAnalyzer** — Speed, acceleration, behavior classification
- Threshold: 0.6 confidence → 96.4% precision, 90.0% recall

### 5. Desktop GUI (`desktop/`)
- PyQt6-based with video playback, one-click analysis, nest editor
- Batch processing with parallel workers
- Real-time detection visualization

### 6. Hardware System (`hardware/`)
- Raspberry Pi 4 + HQ Camera + solar power (~$595)
- Autonomous field deployment with Witty Pi scheduler
- 3D-printable waterproof enclosure

---

## Key Dependencies

| Category | Libraries |
|----------|-----------|
| Deep Learning | PyTorch, Ultralytics YOLO, torchvision |
| ML | scikit-learn, filterpy (Kalman), scipy (Hungarian) |
| CV | OpenCV (MOG2, SIFT, video I/O) |
| GUI | PyQt6 |
| Data | pandas, numpy, polars |
| Web (existing) | FastAPI, Gradio |
| Config | PyYAML, Pydantic, dataclasses |

---

## Performance

| Mode | Precision | Recall | F1 | Speed |
|------|-----------|--------|----|-------|
| Full Tracking | 93.9% | 87.7% | 0.907 | 2.3× real-time |
| Two-Mode Adaptive | 92.0% | 84.3% | 0.880 | 0.8× real-time |

*Tested on M3 Pro with 110 min video, 300 annotated events*

---

## Data Pipeline

```
Video → Nest Detection (YOLO, frame 1)
      → Frame-by-Frame:
          Motion Detection (MOG2) → YOLO Detection
          → Multi-Object Tracking (Kalman + Hungarian)
      → Event Processing (trajectory features → ML classifier)
      → Output: CSV (timestamped events) + Annotated Video
```

---

## Deployment Options (Current)

1. **Desktop** — PyQt6 GUI (macOS/Linux/Windows)
2. **Raspberry Pi** — Autonomous field recording
3. **HPC** — SLURM batch processing
4. **API** — FastAPI + Gradio (partially configured)
