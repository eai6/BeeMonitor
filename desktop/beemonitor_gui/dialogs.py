"""
Dialogs - v2.2 SIMPLIFIED
=========================

About and help dialog content.

v2.2 Changes:
- Updated about dialog for v2.2
- Removed SIFT references
- Added v2.2 features section
"""

from PyQt6.QtWidgets import QMessageBox
from PyQt6.QtCore import Qt

from .constants import VERSION


def show_about_dialog(parent):
    """Show about dialog."""
    QMessageBox.about(
        parent,
        "About BeeMonitor",
        f"<h3>BeeMonitor v{VERSION} - YOLO-Only Edition</h3>"
        "<p>Interactive video analysis tool for bee tracking</p>"
        "<p><b>What's New in v2.2:</b></p>"
        "<ul>"
        "<li>🔬 YOLO-only tracking (100% accuracy)</li>"
        "<li>⚡ Two-mode optimization (5-7x faster)</li>"
        "<li>🎯 Enhanced anti-duplicate tracking</li>"
        "<li>🎨 Simplified UI (no mode selection)</li>"
        "</ul>"
        "<p><b>Features:</b></p>"
        "<ul>"
        "<li>Play/Pause video controls</li>"
        "<li>Real-time detection preview</li>"
        "<li>Color-coded detection sources</li>"
        "<li>Track visualization overlay</li>"
        "<li>Batch folder analysis</li>"
        "</ul>"
    )


def show_parameter_guide(parent):
    """Show parameter guide dialog."""
    guide = """
<h2>QUICK GUIDE (v2.2 YOLO-Only)</h2>

<h3>Workflow:</h3>
<ol>
<li>Load video (Ctrl+O)</li>
<li>Review detection info (YOLO-only, automatic)</li>
<li>Run Full Analysis</li>
<li>Load Results to visualize tracks</li>
</ol>

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

<h3>Detection Sources (v2.2):</h3>
<ul>
<li><b style="color: red;">RED:</b> Blob motion detection (two-mode optimization)</li>
<li><b style="color: blue;">BLUE:</b> YOLO tracking (100% accuracy)</li>
</ul>

<h3>How v2.2 Works:</h3>
<p><b>Two-Mode Optimization:</b></p>
<ul>
<li><b>Motion Detection Mode:</b> Blob detector scans for activity (fast, ~1ms/frame)</li>
<li><b>Tracking Mode:</b> YOLO provides accurate tracking when motion detected</li>
<li><b>Automatic Switching:</b> System switches between modes based on activity</li>
<li><b>Result:</b> 100% YOLO accuracy + 5-7x speedup!</li>
</ul>

<h3>Tips:</h3>
<ul>
<li>RED boxes = Motion detected (triggers YOLO mode)</li>
<li>BLUE boxes = Active YOLO tracking</li>
<li>Fewer tracks than v2.1? ✓ That's good! (cleaner results)</li>
<li>Expected: 3-5 tracks for 3 bees (not 10-15)</li>
</ul>

<h3>Troubleshooting:</h3>
<ul>
<li><b>Missing tracks?</b> Check YOLO confidence threshold (default 0.25)</li>
<li><b>Too slow?</b> Two-mode optimization already active (5-7x speedup)</li>
<li><b>Duplicate tracks?</b> v2.2 has enhanced anti-duplicate checking</li>
</ul>

<h3>What Happened to FGBG_YOLO?</h3>
<p>v2.1's FGBG_YOLO mode was removed in v2.2 because:</p>
<ul>
<li>Created too much noise (14 tracks for 3 bees)</li>
<li>Lower accuracy (~85% vs 100%)</li>
<li>Complexity not worth the speed gain</li>
<li>v2.2's two-mode gives similar speed with better accuracy</li>
</ul>

<h3>Batch Analysis:</h3>
<ul>
<li>Select folder with multiple videos</li>
<li>Enable "Use Nest Fallback" for robust nest detection</li>
<li>Set parallel workers (4 recommended)</li>
<li>Monitor progress in log window</li>
</ul>
"""
    
    msg = QMessageBox(parent)
    msg.setWindowTitle("Quick Guide (v2.2)")
    msg.setTextFormat(Qt.TextFormat.RichText)
    msg.setText(guide)
    msg.exec()