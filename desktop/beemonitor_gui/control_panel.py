"""
Control Panel - v2.2 ULTRA SIMPLIFIED
======================================

Left sidebar control panel with single video analysis and batch folder analysis.

v2.2 CHANGES:
- Removed detection section entirely (not needed)
- Detection mode always 'yolo' internally
- Cleaner, simpler UI
"""

from PyQt6.QtWidgets import (
    QScrollArea, QWidget, QVBoxLayout, QGroupBox, QHBoxLayout,
    QPushButton, QLabel, QSpinBox, QCheckBox,
    QProgressBar, QTextEdit, QFileDialog
)
from PyQt6.QtCore import Qt, pyqtSignal
from pathlib import Path


class ControlPanel(QScrollArea):
    """Control panel with action buttons and folder analysis."""
    
    # Signals for single video
    test_detection_requested = pyqtSignal()
    run_analysis_requested = pyqtSignal()
    stop_analysis_requested = pyqtSignal()
    load_video_requested = pyqtSignal()
    parameters_changed = pyqtSignal(dict)
    
    # Signals for folder analysis
    folder_selected = pyqtSignal(str)
    analyze_folder_requested = pyqtSignal()
    
    def __init__(self):
        """Initialize control panel."""
        super().__init__()
        self.setWidgetResizable(True)
        self.setMinimumWidth(400)
        
        container = QWidget()
        layout = QVBoxLayout()
        container.setLayout(layout)
        
        # Add sections (removed detection section)
        layout.addWidget(self._create_video_group())
        layout.addWidget(self._create_single_analysis_group())
        layout.addWidget(self._create_folder_analysis_group())
        layout.addStretch()
        
        self.setWidget(container)
    
    def _create_video_group(self):
        """Create video info section."""
        video_group = QGroupBox("Video")
        video_layout = QVBoxLayout()
        
        load_btn = QPushButton("📁 Load Video")
        load_btn.clicked.connect(self.load_video_requested.emit)
        video_layout.addWidget(load_btn)
        
        self.video_info_label = QLabel("No video loaded")
        self.video_info_label.setStyleSheet("color: gray; font-size: 10pt;")
        self.video_info_label.setWordWrap(True)
        video_layout.addWidget(self.video_info_label)
        
        self.output_folder_label = QLabel("Output: -")
        self.output_folder_label.setStyleSheet("color: gray; font-size: 9pt;")
        self.output_folder_label.setWordWrap(True)
        video_layout.addWidget(self.output_folder_label)
        
        video_group.setLayout(video_layout)
        return video_group
    
    def _create_single_analysis_group(self):
        """Create single video analysis section."""
        analysis_group = QGroupBox("Single Video Analysis")
        analysis_layout = QVBoxLayout()
        
        self.analyze_btn = QPushButton("▶ Run Analysis")
        self.analyze_btn.setStyleSheet(
            "background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 10px;"
        )
        self.analyze_btn.clicked.connect(self.run_analysis_requested.emit)
        self.analyze_btn.setEnabled(False)
        analysis_layout.addWidget(self.analyze_btn)
        
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setStyleSheet(
            "background-color: #f44336; color: white; "
            "font-weight: bold; padding: 8px;"
        )
        self.stop_btn.clicked.connect(self.stop_analysis_requested.emit)
        self.stop_btn.setEnabled(False)
        analysis_layout.addWidget(self.stop_btn)
        
        analysis_group.setLayout(analysis_layout)
        return analysis_group
    
    def _create_folder_analysis_group(self):
        """Create batch folder analysis section."""
        folder_group = QGroupBox("Batch Video Analysis")
        folder_layout = QVBoxLayout()
        
        folder_btn = QPushButton("📂 Select Video Folder")
        folder_btn.clicked.connect(self._select_folder)
        folder_layout.addWidget(folder_btn)
        
        self.folder_path_label = QLabel("No folder selected")
        self.folder_path_label.setStyleSheet("color: gray; font-size: 9pt;")
        self.folder_path_label.setWordWrap(True)
        folder_layout.addWidget(self.folder_path_label)
        
        self.use_fallback_checkbox = QCheckBox("Use Nest Fallback (Recommended)")
        self.use_fallback_checkbox.setChecked(True)
        self.use_fallback_checkbox.setToolTip(
            "If nest detection fails, try previous/next video for nests"
        )
        folder_layout.addWidget(self.use_fallback_checkbox)
        
        workers_layout = QHBoxLayout()
        workers_layout.addWidget(QLabel("Parallel Workers:"))
        self.workers_spinner = QSpinBox()
        self.workers_spinner.setRange(1, 8)
        self.workers_spinner.setValue(4)
        workers_layout.addWidget(self.workers_spinner)
        folder_layout.addLayout(workers_layout)
        
        self.analyze_folder_btn = QPushButton("▶ Analyze Folder")
        self.analyze_folder_btn.setStyleSheet(
            "background-color: #2196F3; color: white; "
            "font-weight: bold; padding: 10px;"
        )
        self.analyze_folder_btn.clicked.connect(self.analyze_folder_requested.emit)
        self.analyze_folder_btn.setEnabled(False)
        folder_layout.addWidget(self.analyze_folder_btn)
        
        self.folder_progress = QProgressBar()
        self.folder_progress.setMinimum(0)
        self.folder_progress.setMaximum(100)
        self.folder_progress.setValue(0)
        self.folder_progress.setFormat("Processing: %v/%m videos")
        folder_layout.addWidget(self.folder_progress)
        
        log_label = QLabel("Analysis Log:")
        folder_layout.addWidget(log_label)
        
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(150)
        self.log_output.setStyleSheet(
            "background-color: #1e1e1e; color: #d4d4d4; "
            "font-family: 'Courier New', monospace; font-size: 9pt;"
        )
        folder_layout.addWidget(self.log_output)
        
        folder_group.setLayout(folder_layout)
        return folder_group
    
    def _select_folder(self):
        """Handle folder selection."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Video Folder",
            str(Path.home()),
            QFileDialog.Option.ShowDirsOnly
        )
        
        if folder:
            self.folder_path_label.setText(f"Selected: {Path(folder).name}")
            self.analyze_folder_btn.setEnabled(True)
            self.folder_selected.emit(folder)
    
    # === Control Methods ===
    
    def get_parameters(self):
        """Get current parameters (v2.2: detection_mode always 'yolo')."""
        return {
            'detection_mode': 'yolo',  # v2.2: Always YOLO
            'use_fallback': self.use_fallback_checkbox.isChecked(),
            'max_workers': self.workers_spinner.value()
        }
    
    def set_detection_mode(self, mode):
        """Set detection mode (v2.2: no-op, always YOLO)."""
        pass  # Kept for backwards compatibility
    
    def set_video_info(self, text):
        """Update video info label."""
        self.video_info_label.setText(text)
    
    def set_output_folder_info(self, text):
        """Update output folder label."""
        self.output_folder_label.setText(f"Output: {text}")
    
    def set_video_loaded(self, loaded: bool):
        """Enable/disable analysis based on video loaded state."""
        self.analyze_btn.setEnabled(loaded)
    
    def set_analysis_running(self, running: bool):
        """Update UI state based on whether analysis is running."""
        self.analyze_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
    
    def set_folder_path(self, path: str):
        """Set folder path display."""
        self.folder_path_label.setText(f"Selected: {Path(path).name}")
        self.analyze_folder_btn.setEnabled(True)
    
    def set_folder_analyzing(self, analyzing: bool):
        """Update UI state for folder analysis."""
        self.analyze_folder_btn.setEnabled(not analyzing)
        if not analyzing:
            self.folder_progress.setValue(0)
    
    def set_folder_progress(self, current: int, total: int):
        """Update folder analysis progress bar."""
        self.folder_progress.setMaximum(total)
        self.folder_progress.setValue(current)
        percent = int(100 * current / total) if total > 0 else 0
        self.folder_progress.setFormat(f"Processing: {current}/{total} videos ({percent}%)")
    
    def append_log(self, message: str):
        """Append message to analysis log."""
        self.log_output.append(message)
        self.log_output.verticalScrollBar().setValue(
            self.log_output.verticalScrollBar().maximum()
        )
    
    def clear_log(self):
        """Clear analysis log."""
        self.log_output.clear()