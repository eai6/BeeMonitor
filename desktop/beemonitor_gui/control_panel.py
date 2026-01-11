"""Control Panel with Batch Folder Analysis

Left sidebar control panel with detection parameters and batch analysis support.
"""

from PyQt6.QtWidgets import (
    QScrollArea, QWidget, QVBoxLayout, QGroupBox,
    QPushButton, QLabel, QComboBox, QSpinBox,
    QProgressBar, QTextEdit, QHBoxLayout, QCheckBox
)
from PyQt6.QtCore import Qt, pyqtSignal


class ControlPanel(QScrollArea):
    """Control panel with detection parameters and action buttons."""
    
    # Signals
    load_video_requested = pyqtSignal()
    initialize_background_requested = pyqtSignal()
    run_analysis_requested = pyqtSignal()
    stop_analysis_requested = pyqtSignal()
    parameters_changed = pyqtSignal(dict)
    
    # Folder analysis signals
    folder_selected = pyqtSignal(str)
    analyze_folder_requested = pyqtSignal(dict)
    
    def __init__(self):
        """Initialize control panel."""
        super().__init__()
        self.setWidgetResizable(True)
        self.setMinimumWidth(400)
        
        container = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)
        container.setLayout(layout)
        
        # Add sections
        layout.addWidget(self._create_video_group())
        layout.addWidget(self._create_detection_group())
        layout.addWidget(self._create_background_group())
        layout.addWidget(self._create_analysis_group())
        layout.addWidget(self._create_folder_group())  # NEW: Folder analysis
        layout.addWidget(self._create_log_group())
        layout.addStretch()
        
        self.setWidget(container)
    
    def _create_video_group(self):
        """Create video info section."""
        group = QGroupBox("Video")
        layout = QVBoxLayout()
        
        # Load video button
        self.load_video_btn = QPushButton("Load Video")
        self.load_video_btn.clicked.connect(
            lambda: self.load_video_requested.emit()
        )
        layout.addWidget(self.load_video_btn)
        
        # Video info label
        self.video_info_label = QLabel("No video loaded")
        self.video_info_label.setWordWrap(True)
        layout.addWidget(self.video_info_label)
        
        # Output folder info
        self.output_folder_label = QLabel("")
        self.output_folder_label.setWordWrap(True)
        layout.addWidget(self.output_folder_label)
        
        group.setLayout(layout)
        return group
    
    def _create_detection_group(self):
        """Create detection mode section."""
        group = QGroupBox("Detection Mode")
        layout = QVBoxLayout()
        
        # Detection mode dropdown
        layout.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Motion + YOLO (Recommended) ⭐", "fgbg_yolo")
        self.mode_combo.addItem("YOLO Only", "yolo_only")
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        layout.addWidget(self.mode_combo)
        
        # Mode description
        self.mode_description = QLabel(
            "<i>Fast motion detection + YOLO confirmation<br>"
            "Automatic CNN filtering + learned thresholds</i>"
        )
        self.mode_description.setWordWrap(True)
        layout.addWidget(self.mode_description)
        
        group.setLayout(layout)
        return group
    
    def _create_background_group(self):
        """Create background initialization section."""
        group = QGroupBox("Background Initialization")
        layout = QVBoxLayout()
        
        # Frames selector
        frames_layout = QHBoxLayout()
        frames_layout.addWidget(QLabel("Frames:"))
        self.bg_frames_spin = QSpinBox()
        self.bg_frames_spin.setMinimum(50)
        self.bg_frames_spin.setMaximum(500)
        self.bg_frames_spin.setValue(100)
        self.bg_frames_spin.setSingleStep(50)
        frames_layout.addWidget(self.bg_frames_spin)
        frames_layout.addStretch()
        layout.addLayout(frames_layout)
        
        # Initialize button
        self.init_bg_btn = QPushButton("Initialize Background")
        self.init_bg_btn.clicked.connect(
            lambda: self.initialize_background_requested.emit()
        )
        self.init_bg_btn.setEnabled(False)
        layout.addWidget(self.init_bg_btn)
        
        # Status label
        self.bg_status_label = QLabel("")
        layout.addWidget(self.bg_status_label)
        
        group.setLayout(layout)
        return group
    
    def _create_analysis_group(self):
        """Create analysis controls section."""
        group = QGroupBox("Single Video Analysis")
        layout = QVBoxLayout()
        
        # Run analysis button
        self.run_analysis_btn = QPushButton("Run Analysis")
        self.run_analysis_btn.clicked.connect(
            lambda: self.run_analysis_requested.emit()
        )
        self.run_analysis_btn.setEnabled(False)
        layout.addWidget(self.run_analysis_btn)
        
        # Stop button
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(
            lambda: self.stop_analysis_requested.emit()
        )
        self.stop_btn.setEnabled(False)
        layout.addWidget(self.stop_btn)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)
        
        group.setLayout(layout)
        return group
    
    def _create_folder_group(self):
        """Create batch folder analysis section."""
        group = QGroupBox("Batch Video Analysis")
        layout = QVBoxLayout()
        
        # Folder selection
        layout.addWidget(QLabel("<b>Folder:</b>"))
        self.select_folder_btn = QPushButton("Select Video Folder")
        self.select_folder_btn.clicked.connect(self._on_select_folder)
        layout.addWidget(self.select_folder_btn)
        
        self.folder_path_label = QLabel("No folder selected")
        self.folder_path_label.setWordWrap(True)
        layout.addWidget(self.folder_path_label)
        
        # Fallback option
        self.use_fallback_checkbox = QCheckBox("Use Nest Fallback (Recommended)")
        self.use_fallback_checkbox.setChecked(True)
        self.use_fallback_checkbox.setToolTip(
            "If nest detection fails on one video,\n"
            "try using nests from adjacent videos"
        )
        layout.addWidget(self.use_fallback_checkbox)
        
        # Max workers
        workers_layout = QHBoxLayout()
        workers_layout.addWidget(QLabel("Parallel Workers:"))
        self.max_workers_spin = QSpinBox()
        self.max_workers_spin.setMinimum(1)
        self.max_workers_spin.setMaximum(8)
        self.max_workers_spin.setValue(4)
        self.max_workers_spin.setToolTip("Number of videos to process in parallel")
        workers_layout.addWidget(self.max_workers_spin)
        workers_layout.addStretch()
        layout.addLayout(workers_layout)
        
        # Analyze folder button
        self.analyze_folder_btn = QPushButton("Analyze Folder")
        self.analyze_folder_btn.clicked.connect(self._on_analyze_folder)
        self.analyze_folder_btn.setEnabled(False)
        layout.addWidget(self.analyze_folder_btn)
        
        # Folder progress
        self.folder_progress = QProgressBar()
        self.folder_progress.setVisible(False)
        layout.addWidget(self.folder_progress)
        
        # Folder status
        self.folder_status_label = QLabel("")
        layout.addWidget(self.folder_status_label)
        
        group.setLayout(layout)
        return group
    
    def _create_log_group(self):
        """Create log output section."""
        group = QGroupBox("Analysis Log")
        layout = QVBoxLayout()
        
        # Log text area
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        layout.addWidget(self.log_text)
        
        # Clear log button
        clear_log_btn = QPushButton("Clear Log")
        clear_log_btn.clicked.connect(lambda: self.log_text.clear())
        layout.addWidget(clear_log_btn)
        
        group.setLayout(layout)
        return group
    
    def _on_mode_changed(self):
        """Handle detection mode change."""
        mode = self.mode_combo.currentData()
        
        descriptions = {
            'fgbg_yolo': (
                "<i>Fast motion detection + YOLO confirmation<br>"
                "Automatic CNN filtering + learned thresholds</i>"
            ),
            'yolo_only': (
                "<i>YOLO detection every frame<br>"
                "Highest accuracy, slower processing</i>"
            )
        }
        
        self.mode_description.setText(descriptions.get(mode, ""))
        self.parameters_changed.emit(self.get_parameters())
    
    def _on_select_folder(self):
        """Handle folder selection button click."""
        self.folder_selected.emit("")
    
    def _on_analyze_folder(self):
        """Handle analyze folder button click."""
        params = {
            'detection_mode': self.mode_combo.currentData(),
            'use_fallback': self.use_fallback_checkbox.isChecked(),
            'max_workers': self.max_workers_spin.value(),
            'visualize': True
        }
        self.analyze_folder_requested.emit(params)
    
    def get_parameters(self):
        """Get current parameter values."""
        return {
            'detection_mode': self.mode_combo.currentData(),
            'bg_frames': self.bg_frames_spin.value(),
        }
    
    def set_detection_mode(self, mode: str):
        """Set detection mode."""
        for i in range(self.mode_combo.count()):
            if self.mode_combo.itemData(i) == mode:
                self.mode_combo.setCurrentIndex(i)
                break
    
    def set_video_info(self, info: str):
        """Update video info display."""
        self.video_info_label.setText(info)
    
    def set_output_folder_info(self, info: str):
        """Update output folder display."""
        self.output_folder_label.setText(info)
    
    def set_video_loaded(self, loaded: bool):
        """Update UI when video is loaded."""
        self.init_bg_btn.setEnabled(loaded)
        self.run_analysis_btn.setEnabled(loaded)
    
    def set_background_initialized(self, initialized: bool):
        """Update UI when background is initialized."""
        if initialized:
            self.bg_status_label.setText("✓ Background ready")
            self.run_analysis_btn.setEnabled(True)
        else:
            self.bg_status_label.setText("")
    
    def set_analysis_running(self, running: bool):
        """Update UI during analysis."""
        self.run_analysis_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.load_video_btn.setEnabled(not running)
        
        if running:
            self.progress_bar.setVisible(True)
            self.status_label.setText("Running...")
        else:
            self.progress_bar.setVisible(False)
            self.status_label.setText("")
    
    def set_progress(self, value: int, maximum: int):
        """Update progress bar."""
        self.progress_bar.setMaximum(maximum)
        self.progress_bar.setValue(value)
    
    def set_status(self, message: str, color: str = "black"):
        """Update status label."""
        self.status_label.setText(f'<span style="color:{color}">{message}</span>')
    
    def append_log(self, message: str):
        """Append message to log."""
        self.log_text.append(message)
        # Auto-scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def set_folder_path(self, folder_path: str):
        """Update folder path display."""
        if folder_path:
            from pathlib import Path
            folder_name = Path(folder_path).name
            self.folder_path_label.setText(f"<b>{folder_name}</b>")
            self.analyze_folder_btn.setEnabled(True)
        else:
            self.folder_path_label.setText("No folder selected")
            self.analyze_folder_btn.setEnabled(False)
    
    def set_folder_progress(self, current: int, total: int):
        """Update folder analysis progress."""
        if total > 0:
            self.folder_progress.setVisible(True)
            self.folder_progress.setMaximum(total)
            self.folder_progress.setValue(current)
            self.folder_status_label.setText(f"Processing: {current}/{total} videos")
        else:
            self.folder_progress.setVisible(False)
            self.folder_status_label.setText("")
    
    def set_folder_analyzing(self, is_analyzing: bool):
        """Update UI during folder analysis."""
        self.analyze_folder_btn.setEnabled(not is_analyzing)
        self.select_folder_btn.setEnabled(not is_analyzing)
        self.run_analysis_btn.setEnabled(not is_analyzing)
        self.load_video_btn.setEnabled(not is_analyzing)
        
        if not is_analyzing:
            self.folder_progress.setVisible(False)