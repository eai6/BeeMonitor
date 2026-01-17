"""
Control Panel - v2.3 with Reference Configuration
==================================================

Left sidebar control panel with:
- Single video analysis
- Batch folder analysis
- Reference configuration (nest grid)
- Advanced options (interaction metrics)

v2.3 CHANGES:
- Added Reference Configuration section (nest rows/cols)
- Added Advanced Options section
- Added interaction metrics toggle
- Added manual nest editing button
"""

from PyQt6.QtWidgets import (
    QScrollArea, QWidget, QVBoxLayout, QGroupBox, QHBoxLayout,
    QPushButton, QLabel, QSpinBox, QCheckBox, QComboBox,
    QProgressBar, QTextEdit, QFileDialog, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal
from pathlib import Path


class ControlPanel(QScrollArea):
    """Control panel with action buttons, folder analysis, and reference configuration."""
    
    # Signals for single video
    test_detection_requested = pyqtSignal()
    run_analysis_requested = pyqtSignal()
    stop_analysis_requested = pyqtSignal()
    load_video_requested = pyqtSignal()
    parameters_changed = pyqtSignal(dict)
    
    # Signals for folder analysis
    folder_selected = pyqtSignal(str)
    analyze_folder_requested = pyqtSignal()
    
    # NEW: Signals for reference configuration
    reference_config_changed = pyqtSignal(dict)  # {'rows': int, 'cols': int, 'total': int}
    edit_nests_requested = pyqtSignal()  # Request to open nest editor
    
    def __init__(self):
        """Initialize control panel."""
        super().__init__()
        self.setWidgetResizable(True)
        self.setMinimumWidth(400)
        
        container = QWidget()
        layout = QVBoxLayout()
        container.setLayout(layout)
        
        # Add sections
        layout.addWidget(self._create_video_group())
        layout.addWidget(self._create_single_analysis_group())
        layout.addWidget(self._create_reference_config_group())
        layout.addWidget(self._create_folder_analysis_group())
        layout.addStretch()
        
        self.setWidget(container)
    
    def _create_video_group(self):
        """Create video info section."""
        video_group = QGroupBox("Video")
        video_layout = QVBoxLayout()
        
        load_btn = QPushButton("📂 Load Video")
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
    
    def _create_reference_config_group(self):
        """Create reference configuration section for nest grid."""
        ref_group = QGroupBox("🏠 Reference Configuration")
        ref_layout = QVBoxLayout()
        
        # Description
        desc_label = QLabel("Configure bee hotel nest grid layout:")
        desc_label.setStyleSheet("color: gray; font-size: 9pt;")
        ref_layout.addWidget(desc_label)
        
        # Rows spinner
        rows_layout = QHBoxLayout()
        rows_layout.addWidget(QLabel("Nest Rows:"))
        self.rows_spinner = QSpinBox()
        self.rows_spinner.setRange(1, 20)
        self.rows_spinner.setValue(6)
        self.rows_spinner.setToolTip("Number of rows in the bee hotel grid")
        self.rows_spinner.valueChanged.connect(self._on_grid_changed)
        rows_layout.addWidget(self.rows_spinner)
        ref_layout.addLayout(rows_layout)
        
        # Columns spinner
        cols_layout = QHBoxLayout()
        cols_layout.addWidget(QLabel("Nests per Row:"))
        self.cols_spinner = QSpinBox()
        self.cols_spinner.setRange(1, 30)
        self.cols_spinner.setValue(10)
        self.cols_spinner.setToolTip("Number of nest tubes per row")
        self.cols_spinner.valueChanged.connect(self._on_grid_changed)
        cols_layout.addWidget(self.cols_spinner)
        ref_layout.addLayout(cols_layout)
        
        # Total nests display
        self.total_nests_label = QLabel("Total Nests: 60")
        self.total_nests_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
        ref_layout.addWidget(self.total_nests_label)
        
        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #444;")
        ref_layout.addWidget(line)
        
        # Nest editing section
        nest_edit_label = QLabel("Nest Detection:")
        nest_edit_label.setStyleSheet("font-weight: bold;")
        ref_layout.addWidget(nest_edit_label)
        
        # Detected nests info
        self.detected_nests_label = QLabel("Detected: - nests")
        self.detected_nests_label.setStyleSheet("color: gray;")
        ref_layout.addWidget(self.detected_nests_label)
        
        # Edit nests button
        edit_nests_btn = QPushButton("✏️ Edit Nests")
        edit_nests_btn.setToolTip("Visually edit nest positions on video frame")
        edit_nests_btn.clicked.connect(self.edit_nests_requested.emit)
        ref_layout.addWidget(edit_nests_btn)
        
        ref_group.setLayout(ref_layout)
        return ref_group
    
    def _create_folder_analysis_group(self):
        """Create batch folder analysis section."""
        folder_group = QGroupBox("Batch Video Analysis")
        folder_layout = QVBoxLayout()
        
        folder_btn = QPushButton("📁 Select Video Folder")
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
    
    def _on_grid_changed(self):
        """Handle grid configuration change."""
        rows = self.rows_spinner.value()
        cols = self.cols_spinner.value()
        total = rows * cols
        self.total_nests_label.setText(f"Total Nests: {total}")
    
    # === Control Methods ===
    
    def get_parameters(self):
        """Get current parameters."""
        return {
            'detection_mode': 'yolo',  # Always YOLO
            'use_fallback': self.use_fallback_checkbox.isChecked(),
            'max_workers': self.workers_spinner.value(),
            # Reference config
            'nest_rows': self.rows_spinner.value(),
            'nests_per_row': self.cols_spinner.value(),
            # Hardcoded defaults (advanced options removed from UI)
            'enable_interaction_metrics': True,
            'proximity_threshold': 50,
            'save_crops': False,
            'crops_per_track': 5
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
    
    # === NEW: Reference Config Methods ===
    
    def set_reference_config(self, rows: int, cols: int):
        """Set reference configuration values.
        
        Args:
            rows: Number of nest rows
            cols: Number of nests per row
        """
        self.rows_spinner.setValue(rows)
        self.cols_spinner.setValue(cols)
        self._on_grid_changed()
    
    def set_detected_nests_count(self, count: int):
        """Update detected nests count display.
        
        Args:
            count: Number of detected nests
        """
        if count > 0:
            expected = self.rows_spinner.value() * self.cols_spinner.value()
            color = "#4CAF50" if count == expected else "#FF9800"
            self.detected_nests_label.setText(f"Detected: {count} nests")
            self.detected_nests_label.setStyleSheet(f"color: {color};")
        else:
            self.detected_nests_label.setText("Detected: - nests")
            self.detected_nests_label.setStyleSheet("color: gray;")
    
    def get_reference_config(self):
        """Get current reference configuration.
        
        Returns:
            Dictionary with rows, cols, total
        """
        return {
            'rows': self.rows_spinner.value(),
            'cols': self.cols_spinner.value(),
            'total': self.rows_spinner.value() * self.cols_spinner.value()
        }
    
    def get_advanced_options(self):
        """Get advanced options settings (hardcoded defaults).
        
        Returns:
            Dictionary with advanced options
        """
        return {
            'enable_interaction_metrics': True,  # Always enabled
            'proximity_threshold': 50,  # Default from config
            'save_crops': False,
            'crops_per_track': 5
        }