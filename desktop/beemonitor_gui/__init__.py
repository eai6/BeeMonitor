"""
BeeMonitor GUI Package v3.2
============================

Modular PyQt6 GUI for bee video analysis and visualization.

v2.3 Features:
- Reference configuration (nest rows/cols)
- Interaction metrics analysis
- Manual nest editing
- Bee crop saving for ID training

v2.2 Features:
- YOLO-only tracking (100% accuracy)
- Two-mode optimization (5-7x faster)
- Real-time detection preview
- Track visualization with trajectories
- Detection source color-coding (Blob/YOLO)
- Batch folder analysis
"""

from .main_window import BeeMonitorGUI

__version__ = "3.2.0"
__all__ = ['BeeMonitorGUI']