# Bee Monitor 🐝

A professional computer vision system for monitoring solitary bee activity in bee hotels using YOLO-based detection and tracking.

## Features

- **Nest Detection**: Automatically identifies and labels individual nest holes in bee hotels
- **Motion Detection**: Efficiently detects bee activity using frame differencing
- **Object Tracking**: Tracks individual bees across video frames using custom and YOLO-based algorithms
- **Event Processing**: Identifies entry/exit events for each nest
- **Data Export**: Generates CSV reports with timestamps and nest activity
- **Video Synthesis**: Creates annotated videos showing tracked bees and events
- **Web Interface**: Gradio-based UI for easy video analysis

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for real-time processing)

### Install from source

```bash
# Clone the repository
git clone https://github.com/yourusername/bee-monitor.git
cd bee-monitor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

### Command Line Interface

```python
from bee_monitor import BeeMonitor

# Initialize the monitor
monitor = BeeMonitor(
    nest_model_path="models/nest_detection_model.pt",
    tracking_model_path="models/bee_tracking_model.pt",
    config_path="config/default_config.yaml"
)

# Analyze a video
results = monitor.analyze_video("path/to/video.mp4")

# Export results
results.to_csv("output/results.csv")
results.save_video("output/annotated_video.mp4")
```

### Web Interface

```bash
# Launch Gradio interface
python scripts/gradio_app.py
```

Then open your browser to `http://localhost:7860`

## Project Structure

```
bee-monitor/
├── src/bee_monitor/       # Main package
│   ├── core/              # Core system components
│   ├── detection/         # Detection modules
│   ├── tracking/          # Tracking algorithms
│   ├── processing/        # Event processing
│   ├── output/            # Output generation
│   └── utils/             # Utility functions
├── tests/                 # Unit and integration tests
├── examples/              # Example scripts
├── docs/                  # Documentation
├── scripts/               # Utility scripts
├── config/                # Configuration files
└── models/                # Model weights (not included)

```

## Configuration

Create a `config.yaml` file to customize behavior:

```yaml
video:
  resolution:
    height: 720
    width: 1280
  fps: 30

models:
  nest_detection: "models/nest_detection_model.pt"
  tracking: "models/bee_tracking_model.pt"

tracking:
  max_age: 30
  distance_threshold: 100
  association_threshold: 200

detection:
  confidence_threshold: 0.25
  iou_threshold: 0.5
  motion_threshold: 5

output:
  base_folder: "output"
  save_visualizations: false
```

## Usage Examples

### Basic Video Analysis

```python
from bee_monitor import BeeMonitor

monitor = BeeMonitor.from_config("config/my_config.yaml")
results = monitor.analyze_video("video.mp4")
print(f"Found {len(results.events)} events")
```

### Batch Processing

```python
from bee_monitor import BeeMonitor
from pathlib import Path

monitor = BeeMonitor.from_config("config/my_config.yaml")

video_dir = Path("videos/")
for video_path in video_dir.glob("*.mp4"):
    print(f"Processing {video_path.name}")
    results = monitor.analyze_video(str(video_path))
    results.to_csv(f"output/{video_path.stem}_results.csv")
```

### Custom Tracking Parameters

```python
from bee_monitor import BeeMonitor
from bee_monitor.tracking import BeeTracker

# Use custom tracker
tracker = BeeTracker(
    max_age=50,
    distance_threshold=150,
    association_threshold=250
)

monitor = BeeMonitor(
    nest_model_path="models/nest_model.pt",
    tracking_model_path="models/tracking_model.pt",
    tracker=tracker
)
```

## API Documentation

### BeeMonitor Class

The main interface for video analysis.

**Methods:**
- `analyze_video(video_path: str) -> AnalysisResults`
- `get_nest_detection(video_path: str) -> pd.DataFrame`
- `process_nest_detection(video_path: str, nest_detection: pd.DataFrame) -> Dict`
- `get_motion_tracking(video_path: str, hotel_roi: Tuple, output_folder: str) -> pd.DataFrame`

### AnalysisResults Class

Container for analysis results.

**Attributes:**
- `events`: DataFrame with entry/exit events
- `tracks`: List of bee trajectories
- `nests`: Dictionary of nest locations

**Methods:**
- `to_csv(filename: str)`
- `save_video(filename: str)`
- `get_statistics() -> Dict`

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=bee_monitor tests/

# Run specific test file
pytest tests/test_detection.py
```

### Code Style

This project uses:
- **black** for code formatting
- **pylint** for linting
- **mypy** for type checking

```bash
# Format code
black src/

# Run linter
pylint src/bee_monitor/

# Type checking
mypy src/bee_monitor/
```

## Architecture

### Detection Pipeline

1. **Nest Detection**: YOLOv8 model identifies nest holes in first frame
2. **Nest Processing**: Clusters detections into rows and assigns IDs
3. **Motion Detection**: Frame differencing identifies areas with activity
4. **Object Detection**: YOLO confirms bee presence in motion areas
5. **Tracking**: Hungarian algorithm associates detections across frames

### Event Processing

1. **Trajectory Analysis**: Analyzes bee paths to determine behavior
2. **Entry/Exit Detection**: Identifies when bees enter or leave nests
3. **Speed Calculation**: Computes velocity to classify behavior
4. **Event Classification**: Labels events as entry, exit, or visit

## Performance

Typical processing times on NVIDIA RTX 3080:
- 720p video: ~0.5x real-time
- 1080p video: ~0.3x real-time

Memory usage: ~2-4GB GPU RAM

## Troubleshooting

### Common Issues

**Issue**: CUDA out of memory
**Solution**: Reduce batch size or video resolution in config

**Issue**: No nests detected
**Solution**: Check model path and confidence threshold

**Issue**: Too many false positives
**Solution**: Increase confidence threshold or adjust motion threshold

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{bee_monitor,
  title = {Bee Monitor: Automated Solitary Bee Activity Tracking},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/bee-monitor}
}
```

## Acknowledgments

- Uses YOLOv8 from Ultralytics
- ByteTrack tracking algorithm
- Built with OpenCV, NumPy, and Pandas

## Contact

For questions or support, please open an issue on GitHub or contact [your.email@example.com]

## Roadmap

- [ ] Multi-camera support
- [ ] Real-time processing
- [ ] Cloud deployment
- [ ] Mobile app
- [ ] Advanced behavior classification
- [ ] Integration with weather data
