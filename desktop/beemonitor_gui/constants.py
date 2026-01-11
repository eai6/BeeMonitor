"""
Constants and Configuration
============================

Shared constants, presets, and configuration values.
"""

# Version
VERSION = "3.0.0"
TITLE = "BeeMonitor - Video Analysis Tool"

# Detection parameter presets
DETECTION_PRESETS = {
    "sensitive": {
        "min_area": 30,
        "min_solidity": 30,
        "max_area": 10000,
        "description": "Detects small objects, may have noise"
    },
    "default": {
        "min_area": 50,
        "min_solidity": 50,
        "max_area": 8000,
        "description": "Balanced settings"
    },
    "conservative": {
        "min_area": 120,
        "min_solidity": 70,
        "max_area": 4000,
        "description": "Recommended for bee hotels ⭐"
    },
    "very_conservative": {
        "min_area": 200,
        "min_solidity": 80,
        "max_area": 3000,
        "description": "Very strict, minimal false positives"
    }
}

# Detection source colors (BGR format for OpenCV)
DETECTION_SOURCE_COLORS = {
    'blob': (0, 0, 255),      # RED - Blob/FG-BG motion detection
    'fgbg': (0, 0, 255),      # RED - Alias
    'sift': (0, 255, 0),      # GREEN - SIFT stationary detection
    'yolo': (255, 0, 0),      # BLUE - YOLO deep learning
    'unknown': (128, 128, 128) # GRAY - Unknown source
}

# Track visualization colors (BGR format)
TRACK_COLORS = [
    (255, 0, 0),      # Blue
    (0, 255, 255),    # Yellow
    (255, 0, 255),    # Magenta
    (255, 255, 0),    # Cyan
    (128, 0, 255),    # Purple
    (255, 128, 0),    # Orange
]

# UI Settings
DEFAULT_WINDOW_SIZE = (1400, 900)
CONTROL_PANEL_WIDTH = 380
MIN_VIDEO_SIZE = (640, 480)

# Detection defaults
DEFAULT_MIN_AREA = 120
DEFAULT_MIN_SOLIDITY = 0.70
DEFAULT_MAX_AREA = 4000

# Slider ranges
SLIDER_RANGES = {
    'min_area': (30, 300),
    'min_solidity': (30, 95),  # Stored as percentage
    'max_area': (1000, 10000),
    'playback_speed': (1, 10)
}

# File filters
VIDEO_FILTER = "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
CSV_FILTER = "CSV Files (*.csv);;All Files (*)"
JSON_FILTER = "JSON Files (*.json);;All Files (*)"

# Tracking parameters
TRAJECTORY_WINDOW = 30  # Frames to show in trajectory
TRAJECTORY_THICKNESS = 2
DETECTION_BOX_THICKNESS = 2
TRACK_CIRCLE_RADIUS = 5

# Common tracking CSV column names
FRAME_COLUMN_NAMES = ['frame', 'frame_number', 'frame_num', 'frame_id']
POSITION_COLUMN_SETS = [
    ['x1', 'y1', 'x2', 'y2'],          # Bounding box
    ['x', 'y'],                         # Centroid
    ['centroid_x', 'centroid_y']       # Alternative centroid
]
