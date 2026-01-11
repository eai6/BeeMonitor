# Complete the Modular GUI - Extraction Guide

## Overview

You have 5 completed core modules. Now you need to extract 5 more from your original file.

---

## ✅ Already Complete (Don't Touch!)

1. `__init__.py`
2. `constants.py`
3. `detection_visualizer.py`
4. `analysis_thread.py`
5. `video_canvas.py`
6. `dialogs.py`

---

## 🔄 Extract These from Your Original File

### Extract 1: control_panel.py

**Find in original:** Method `_create_control_panel()` (around line 416)

**Copy these sections:**
- Video section (lines ~427-450)
- Detection parameters (lines ~451-530)
- Presets (lines ~531-550)
- Actions (lines ~551-580)

**Convert to:**
```python
from PyQt6.QtWidgets import (
    QScrollArea, QWidget, QVBoxLayout, QGroupBox,
    QPushButton, QLabel, QSlider
)
from PyQt6.QtCore import Qt, pyqtSignal

from .constants import DETECTION_PRESETS


class ControlPanel(QScrollArea):
    # Signals for communication with main window
    test_detection_requested = pyqtSignal()
    initialize_background_requested = pyqtSignal()
    run_analysis_requested = pyqtSignal()
    parameters_changed = pyqtSignal(dict)  # Emits {min_area, min_solidity, max_area}
    preset_loaded = pyqtSignal(str)  # Emits preset name
    
    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        self.setMinimumWidth(380)
        
        container = QWidget()
        layout = QVBoxLayout()
        container.setLayout(layout)
        
        # Add groups
        layout.addWidget(self._create_video_group())
        layout.addWidget(self._create_parameters_group())
        layout.addWidget(self._create_presets_group())
        layout.addWidget(self._create_actions_group())
        layout.addStretch()
        
        self.setWidget(container)
    
    def _create_video_group(self):
        # Copy from lines ~427-450
        ...
    
    def _create_parameters_group(self):
        # Copy from lines ~451-530
        ...
    
    def _create_presets_group(self):
        # Copy from lines ~531-550
        ...
    
    def _create_actions_group(self):
        # Copy from lines ~551-580
        ...
    
    def get_parameters(self):
        """Get current parameter values."""
        return {
            'min_area': self.min_area_slider.value(),
            'min_solidity': self.min_solidity_slider.value() / 100.0,
            'max_area': self.max_area_slider.value()
        }
    
    def set_detection_count(self, count):
        """Update detection count label."""
        self.detection_count_label.setText(f"Detections: {count}")
    
    def set_video_info(self, text):
        """Update video info label."""
        self.video_info_label.setText(text)
    
    def set_output_folder_info(self, text):
        """Update output folder label."""
        self.output_folder_label.setText(text)
```

---

### Extract 2: video_panel.py

**Find in original:** Method `_create_video_panel()` (around line 666)

**Copy these sections:**
- VideoCanvas instantiation
- Playback controls (lines ~675-710)
- Frame slider (lines ~711-720)
- Info bar with checkboxes (lines ~721-750)

**Convert to:**
```python
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QSlider, QCheckBox
)
from PyQt6.QtCore import Qt, pyqtSignal

from .video_canvas import VideoCanvas


class VideoPanel(QWidget):
    # Signals
    play_pause_toggled = pyqtSignal()
    frame_step_requested = pyqtSignal(int)  # Emits delta (-1 or 1)
    frame_changed = pyqtSignal(int)  # Emits frame index
    speed_changed = pyqtSignal(int)  # Emits speed value
    show_detections_changed = pyqtSignal(bool)
    show_tracks_changed = pyqtSignal(bool)
    show_sources_changed = pyqtSignal(bool)  # NEW!
    
    def __init__(self):
        super().__init__()
        
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Video canvas
        self.video_canvas = VideoCanvas()
        layout.addWidget(self.video_canvas)
        
        # Controls
        layout.addLayout(self._create_controls())
        
        # Frame slider
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.valueChanged.connect(self.frame_changed.emit)
        layout.addWidget(self.frame_slider)
        
        # Info bar
        layout.addLayout(self._create_info_bar())
    
    def _create_controls(self):
        # Copy from lines ~675-710
        ...
    
    def _create_info_bar(self):
        # Copy from lines ~721-750
        # ADD: Show Sources checkbox
        ...
    
    def get_canvas(self):
        """Get VideoCanvas widget."""
        return self.video_canvas
    
    def set_playing(self, playing):
        """Update play/pause button state."""
        if playing:
            self.play_pause_btn.setText("⏸ Pause")
        else:
            self.play_pause_btn.setText("▶ Play")
    
    def set_frame_info(self, current, total):
        """Update frame label."""
        self.frame_label.setText(f"Frame: {current} / {total}")
    
    def set_data_status(self, text):
        """Update data status label."""
        self.data_status_label.setText(text)
```

---

### Extract 3: main_window.py

**This is the BIG one - orchestrates everything**

**Start with:**
```python
from pathlib import Path
import os
import json
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QSplitter, QMessageBox, QFileDialog
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction

try:
    from beemonitor import BeeMonitor
    from beemonitor.core.config import Config
    from beemonitor.detection import BlobDetector
except ImportError:
    raise ImportError("Cannot import beemonitor. Please install the package.")

from .constants import VERSION, TITLE, DEFAULT_WINDOW_SIZE
from .control_panel import ControlPanel
from .video_panel import VideoPanel
from .analysis_thread import AnalysisThread
from .dialogs import show_about_dialog, show_parameter_guide


class BeeMonitorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle(TITLE)
        self.setGeometry(100, 100, *DEFAULT_WINDOW_SIZE)
        
        # Video state (copy from original lines ~303-316)
        self.video_path = None
        self.video_cap = None
        # ... etc
        
        # Create panels
        self.control_panel = ControlPanel()
        self.video_panel = VideoPanel()
        self.video_canvas = self.video_panel.get_canvas()
        
        # Connect signals
        self._connect_signals()
        
        # Setup UI
        self._create_menu_bar()
        self._create_main_widget()
        
        self.statusBar().showMessage("Ready - Load a video to begin")
    
    def _connect_signals(self):
        """Connect all signals from panels."""
        # Control panel signals
        self.control_panel.test_detection_requested.connect(self.test_detection)
        self.control_panel.initialize_background_requested.connect(self.initialize_background)
        self.control_panel.run_analysis_requested.connect(self.run_analysis)
        self.control_panel.parameters_changed.connect(self.on_parameters_changed)
        
        # Video panel signals
        self.video_panel.play_pause_toggled.connect(self.toggle_play_pause)
        self.video_panel.frame_changed.connect(self.load_frame)
        self.video_panel.frame_step_requested.connect(self.jump_frame)
        self.video_panel.show_detections_changed.connect(self.on_show_detections_changed)
        self.video_panel.show_tracks_changed.connect(self.on_show_tracks_changed)
        self.video_panel.show_sources_changed.connect(self.on_show_sources_changed)  # NEW!
    
    def _create_menu_bar(self):
        # Copy from original lines ~325-391
        ...
    
    def _create_main_widget(self):
        """Create main layout with panels."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QHBoxLayout()
        central_widget.setLayout(layout)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)
        
        # Add panels
        splitter.addWidget(self.control_panel)
        splitter.addWidget(self.video_panel)
        splitter.setSizes([400, 1000])
    
    # Then copy ALL remaining methods from original
    # (load_video, load_frame, test_detection, etc.)
    
    # NEW: Add this method
    def on_show_sources_changed(self, enabled):
        """Handle detection source visualization toggle."""
        self.video_canvas.toggle_detection_sources(enabled)
        if enabled:
            self.statusBar().showMessage(
                "Detection sources: ENABLED (RED=Blob, GREEN=SIFT, BLUE=YOLO)"
            )
        else:
            self.statusBar().showMessage("Detection sources: DISABLED")
```

---

### Extract 4: utils.py

**Create utility functions:**

```python
"""
Utility Functions
=================

Helper functions for the GUI.
"""

from pathlib import Path
import pandas as pd
from typing import Optional, Dict, Tuple

from .constants import FRAME_COLUMN_NAMES, POSITION_COLUMN_SETS


def find_tracking_file(events_filepath: str) -> Optional[str]:
    """
    Try to find the tracking results file in the same directory.
    
    Args:
        events_filepath: Path to events CSV file
    
    Returns:
        Path to tracking file, or None if not found
    """
    directory = Path(events_filepath).parent
    
    # Common tracking file names
    possible_names = [
        'tracking_results.csv',
        'tracks.csv',
        'tracking.csv',
        'trajectories.csv'
    ]
    
    for name in possible_names:
        path = directory / name
        if path.exists():
            return str(path)
    
    # Try with video name prefix
    for file in directory.glob('*_tracks.csv'):
        return str(file)
    for file in directory.glob('*_tracking.csv'):
        return str(file)
    
    return None


def validate_tracking_csv(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate tracking CSV has required columns.
    
    Args:
        df: DataFrame to validate
    
    Returns:
        (is_valid, error_message)
    """
    # Check for track_id
    if 'track_id' not in df.columns:
        return False, "Missing 'track_id' column"
    
    # Check for frame column
    frame_col = None
    for col in FRAME_COLUMN_NAMES:
        if col in df.columns:
            frame_col = col
            break
    
    if frame_col is None:
        return False, f"Missing frame column (tried: {', '.join(FRAME_COLUMN_NAMES)})"
    
    # Check for position columns
    has_positions = False
    for col_set in POSITION_COLUMN_SETS:
        if all(col in df.columns for col in col_set):
            has_positions = True
            break
    
    if not has_positions:
        return False, "Missing position columns (need x1,y1,x2,y2 or x,y or centroid_x,centroid_y)"
    
    return True, ""


def get_position_from_row(row: pd.Series) -> Optional[Tuple[int, int]]:
    """
    Extract position from a DataFrame row.
    
    Handles different column formats:
    - x1, y1, x2, y2 → centroid
    - x, y → direct
    - centroid_x, centroid_y → direct
    
    Args:
        row: DataFrame row
    
    Returns:
        (x, y) position, or None if no position columns
    """
    # Format 1: Bounding box
    if all(col in row.index for col in ['x1', 'y1', 'x2', 'y2']):
        cx = int((row['x1'] + row['x2']) / 2)
        cy = int((row['y1'] + row['y2']) / 2)
        return (cx, cy)
    
    # Format 2: Direct x, y
    if all(col in row.index for col in ['x', 'y']):
        return (int(row['x']), int(row['y']))
    
    # Format 3: Centroid
    if all(col in row.index for col in ['centroid_x', 'centroid_y']):
        return (int(row['centroid_x']), int(row['centroid_y']))
    
    return None


def format_time(seconds: float) -> str:
    """Format seconds as MM:SS."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def format_file_size(bytes: int) -> str:
    """Format bytes as KB/MB/GB."""
    if bytes < 1024:
        return f"{bytes} B"
    elif bytes < 1024 * 1024:
        return f"{bytes / 1024:.1f} KB"
    elif bytes < 1024 * 1024 * 1024:
        return f"{bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{bytes / (1024 * 1024 * 1024):.1f} GB"
```

---

### Extract 5: run_gui.py

**Create entry point:**

```python
#!/usr/bin/env python3
"""
BeeMonitor GUI - Entry Point
=============================

Run the BeeMonitor video analysis GUI.

Usage:
    python run_gui.py
"""

import sys
from PyQt6.QtWidgets import QApplication

from beemonitor_gui import BeeMonitorGUI


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = BeeMonitorGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
```

---

## ✅ Verification Checklist

After extraction, verify:

- [ ] All imports work
- [ ] Signals connect properly
- [ ] GUI launches without errors
- [ ] Video loads
- [ ] Detection works
- [ ] Checkboxes toggle overlays
- [ ] NEW: "Show Detection Sources" checkbox works
- [ ] Play/pause works
- [ ] Results load
- [ ] Tracks display
- [ ] Analysis runs

---

## 🚀 Run Instructions

```bash
# After completing all extractions:

cd /path/to/project
python run_gui.py

# Or as module:
python -m beemonitor_gui.run_gui
```

---

## 🐛 Common Issues

### Issue: Import errors
**Solution:** Check all relative imports use `.` prefix

### Issue: Signal not firing
**Solution:** Check signal connection in `_connect_signals()`

### Issue: Missing attributes
**Solution:** Check all instance variables initialized in `__init__()`

### Issue: Checkbox doesn't work
**Solution:** Check signal connected AND VideoCanvas method called

---

## 💡 Pro Tips

1. **Test Each Module**: Test after extracting each module
2. **Keep Original**: Don't delete original until verified
3. **Use Git**: Commit after each successful extraction
4. **Add Docstrings**: Document as you extract
5. **Run Often**: Run GUI frequently to catch issues early

---

## 🎯 Summary

**You need to:**
1. Extract control_panel.py (~300 lines)
2. Extract video_panel.py (~200 lines)
3. Extract main_window.py (~600 lines)
4. Create utils.py (~100 lines)
5. Create run_gui.py (~20 lines)

**Then test everything works!**

Good luck! 🚀
