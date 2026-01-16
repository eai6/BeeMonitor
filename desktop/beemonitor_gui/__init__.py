"""
BeeMonitor GUI Package v3.1
============================

Modular PyQt6 GUI for bee video analysis and visualization.

v2.2 Features:
- YOLO-only tracking (100% accuracy)
- Two-mode optimization (5-7x faster)
- Real-time detection preview
- Track visualization with trajectories
- Detection source color-coding (Blob/YOLO)
- Batch folder analysis
"""

from .main_window import BeeMonitorGUI

__version__ = "3.1.0"
__all__ = ['BeeMonitorGUI']