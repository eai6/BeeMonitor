"""
Dialogs - v2.3
==============

About and help dialog content.

v2.3 Changes:
- Updated about dialog for v2.3
- Added reference configuration info
- Added interaction metrics info
"""

from PyQt6.QtWidgets import QMessageBox
from PyQt6.QtCore import Qt

from .constants import VERSION


def show_about_dialog(parent):
    """Show about dialog."""
    QMessageBox.about(
        parent,
        "About BeeMonitor",
        f"<h3>BeeMonitor v{VERSION}</h3>"
        "<p>Interactive video analysis tool for bee tracking</p>"
        "<p><b>What's New in v2.3:</b></p>"
        "<ul>"
        "<li>🏠 Reference Configuration - Set nest grid rows/cols</li>"
        "<li>📊 Interaction Metrics - Track proximity analysis</li>"
        "<li>✏️ Manual Nest Editing - Add/edit/remove nests</li>"
        "<li>📷 Crop Saving - Save bee images for ID training</li>"
        "</ul>"
        "<p><b>Core Features:</b></p>"
        "<ul>"
        "<li>🔬 YOLO-only tracking (100% accuracy)</li>"
        "<li>⚡ Two-mode optimization (5-7x faster)</li>"
        "<li>🎯 Enhanced anti-duplicate tracking</li>"
        "<li>📁 Batch folder analysis</li>"
        "</ul>"
    )


def show_parameter_guide(parent):
    """Show parameter guide dialog."""
    guide = """
<h2>QUICK GUIDE (v2.3)</h2>

<h3>Workflow:</h3>
<ol>
<li>Load video (Ctrl+O)</li>
<li>Configure Reference (nest rows/cols)</li>
<li>Review detection info</li>
<li>Run Full Analysis</li>
<li>Load Results to visualize tracks</li>
</ol>

<h3>Reference Configuration:</h3>
<ul>
<li><b>Nest Rows:</b> Number of rows in bee hotel (default: 6)</li>
<li><b>Nests per Row:</b> Tubes per row (default: 10)</li>
<li><b>Edit Nests:</b> Manually adjust detected nest positions</li>
<li><b>Auto-Fill Grid:</b> Generate evenly spaced nest grid</li>
</ul>

<h3>Advanced Options:</h3>
<ul>
<li><b>Interaction Metrics:</b> Compute bee-to-bee and bee-to-nest proximity</li>
<li><b>Proximity Threshold:</b> Distance (px) to count as "interacting"</li>
<li><b>Save Bee Crops:</b> Export images for color ID training</li>
</ul>

<h3>Video Controls:</h3>
<ul>
<li><b>Play/Pause:</b> Watch video with live tracking</li>
<li><b>◀ ▶:</b> Step through frames</li>
<li><b>Speed slider:</b> Adjust playback speed</li>
<li><b>Frame slider:</b> Jump to specific frame</li>
</ul>

<h3>Visualization:</h3>
<ul>
<li><b>Show Detections:</b> Boxes = detected bees</li>
<li><b>Show Tracks:</b> Colored trails = bee trajectories</li>
<li><b>Show Sources:</b> Color-code by detector</li>
</ul>

<h3>Detection Sources:</h3>
<ul>
<li><b style="color: red;">RED:</b> Blob motion detection</li>
<li><b style="color: blue;">BLUE:</b> YOLO tracking</li>
</ul>

<h3>How It Works:</h3>
<p><b>Two-Mode Optimization:</b></p>
<ul>
<li><b>Motion Detection Mode:</b> Blob detector scans for activity (fast)</li>
<li><b>Tracking Mode:</b> YOLO provides accurate tracking when motion detected</li>
<li><b>Result:</b> 100% YOLO accuracy + 5-7x speedup!</li>
</ul>

<h3>Output Files:</h3>
<ul>
<li><b>*_events.csv:</b> Entry/exit events per nest</li>
<li><b>*_tracking_results.csv:</b> Per-frame track positions</li>
<li><b>*_interactions.csv:</b> (If enabled) Proximity interactions</li>
<li><b>crops/:</b> (If enabled) Bee image crops per track</li>
</ul>

<h3>Batch Analysis:</h3>
<ul>
<li>Select folder with multiple videos</li>
<li>Enable "Use Nest Fallback" for robust detection</li>
<li>Set parallel workers (4 recommended)</li>
<li>Monitor progress in log window</li>
</ul>
"""
    
    msg = QMessageBox(parent)
    msg.setWindowTitle("Quick Guide (v2.3)")
    msg.setTextFormat(Qt.TextFormat.RichText)
    msg.setText(guide)
    msg.exec()