"""
Dialogs
=======

About and help dialog content.
"""

from PyQt6.QtWidgets import QMessageBox
from PyQt6.QtCore import Qt

from .constants import VERSION


def show_about_dialog(parent):
    """Show about dialog."""
    QMessageBox.about(
        parent,
        "About BeeMonitor",
        f"<h3>BeeMonitor v{VERSION} - Video Player Edition</h3>"
        "<p>Interactive video analysis tool for bee tracking</p>"
        "<p><b>Features:</b></p>"
        "<ul>"
        "<li>Play/Pause video controls</li>"
        "<li>Real-time detection preview</li>"
        "<li>Color-coded detection sources</li>"
        "<li>Track visualization overlay</li>"
        "<li>Parameter tuning</li>"
        "</ul>"
    )


def show_parameter_guide(parent):
    """Show parameter guide dialog."""
    guide = """
<h2>QUICK GUIDE</h2>

<h3>Workflow:</h3>
<ol>
<li>Load video (Ctrl+O)</li>
<li>Initialize background (uses first 100 frames)</li>
<li>Load "Conservative" preset ⭐</li>
<li>Test detection (Space) - navigate frames to check</li>
<li>Adjust sliders if needed</li>
<li>Run Full Analysis</li>
<li>Load Results to visualize tracks</li>
</ol>

<h3>Video Controls:</h3>
<ul>
<li><b>Play/Pause:</b> Watch video with current settings</li>
<li><b>◀ ▶:</b> Step through frames</li>
<li><b>Speed slider:</b> Adjust playback speed</li>
<li><b>Frame slider:</b> Jump to specific frame</li>
</ul>

<h3>Visualization:</h3>
<ul>
<li><b>Show Detections:</b> Green boxes = detected blobs</li>
<li><b>Show Tracks:</b> Colored trails = bee trajectories</li>
<li><b>Show Sources:</b> Color-code by detector (RED/GREEN/BLUE)</li>
</ul>

<h3>Detection Sources (Color-Coded):</h3>
<ul>
<li><b style="color: red;">RED:</b> Blob/FG-BG (motion detection)</li>
<li><b style="color: green;">GREEN:</b> SIFT (stationary detection)</li>
<li><b style="color: blue;">BLUE:</b> YOLO (deep learning)</li>
</ul>

<h3>Parameters:</h3>
<ul>
<li><b>Min Area:</b> 120 (increase to reduce noise)</li>
<li><b>Min Solidity:</b> 0.7 (shape filtering)</li>
<li><b>Max Area:</b> 4000 (filter large objects)</li>
</ul>

<h3>Diagnostic Tips:</h3>
<ul>
<li>Many RED boxes on empty frame? → Increase blob thresholds or check CNN filter</li>
<li>No GREEN boxes on standing bee? → SIFT not initialized</li>
<li>No BLUE boxes? → YOLO confirmation disabled</li>
</ul>
"""
    
    msg = QMessageBox(parent)
    msg.setWindowTitle("Quick Guide")
    msg.setTextFormat(Qt.TextFormat.RichText)
    msg.setText(guide)
    msg.exec()
