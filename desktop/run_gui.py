#!/usr/bin/env python3
"""
BeeMonitor GUI v3.0 - Video Player Edition
===========================================

Entry point for the BeeMonitor video analysis GUI.

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
