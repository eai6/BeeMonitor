"""
BeeMonitor GUI Package v3.0
============================

Modular PyQt6 GUI for bee video analysis and visualization.

Features:
- Video playback controls
- Real-time detection preview
- Track visualization with trajectories
- Detection source color-coding (Blob/SIFT/YOLO)
- Parameter tuning with presets
- Full video analysis
- Results loading and visualization
"""

from .main_window import BeeMonitorGUI

__version__ = "3.0.0"
__all__ = ['BeeMonitorGUI']
