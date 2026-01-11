"""
Control Panel
=============

Left sidebar control panel with detection parameters and actions.
SIMPLIFIED - No SIFT, only Motion + CNN + YOLO options.
"""

from PyQt6.QtWidgets import (
    QScrollArea, QWidget, QVBoxLayout, QGroupBox,
    QPushButton, QLabel, QComboBox
)
from PyQt6.QtCore import Qt, pyqtSignal


class ControlPanel(QScrollArea):
    """Control panel with detection parameters and action buttons."""
    
    # Signals
    test_detection_requested = pyqtSignal()
    initialize_background_requested = pyqtSignal()
    run_analysis_requested = pyqtSignal()
    load_video_requested = pyqtSignal()
    parameters_changed = pyqtSignal(dict)
    
    def __init__(self):
        """Initialize control panel."""
        super().__init__()
        self.setWidgetResizable(True)
        self.setMinimumWidth(380)
        
        container = QWidget()
        layout = QVBoxLayout()
        container.setLayout(layout)
        
        # Add sections
        layout.addWidget(self._create_video_group())
        layout.addWidget(self._create_detection_group())
        layout.addWidget(self._create_actions_group())
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
        self.video_info_label.setWordWrap(True)
        video_layout.addWidget(self.video_info_label)
        
        self.output_folder_label = QLabel("<i>Output: (load video first)</i>")
        self.output_folder_label.setWordWrap(True)
        self.output_folder_label.setStyleSheet("color: #666; font-size: 9pt;")
        video_layout.addWidget(self.output_folder_label)
        
        video_group.setLayout(video_layout)
        return video_group
    
    def _create_detection_group(self):
        """Create detection mode selector (SIMPLIFIED - Auto CNN+Solidity)."""
        params_group = QGroupBox("Detection Settings")
        params_layout = QVBoxLayout()
        
        # Detection Mode Dropdown (3 modes only!)
        params_layout.addWidget(QLabel("Detection Mode:"))
        self.detection_mode_combo = QComboBox()
        self.detection_mode_combo.addItem("Motion Detection (Recommended) ⭐", "fgbg")
        self.detection_mode_combo.addItem("Motion + YOLO (High Accuracy)", "fgbg_yolo")
        self.detection_mode_combo.addItem("YOLO Only (Highest Accuracy)", "yolo_only")
        self.detection_mode_combo.setCurrentIndex(0)  # Default to "Motion Detection"
        self.detection_mode_combo.currentIndexChanged.connect(self._on_mode_change)
        params_layout.addWidget(self.detection_mode_combo)
        
        # Info label
        self.mode_info_label = QLabel()
        self.mode_info_label.setWordWrap(True)
        self.mode_info_label.setStyleSheet("color: #666; font-size: 9pt; padding: 10px;")
        self._update_mode_info()
        params_layout.addWidget(self.mode_info_label)
        
        # Pipeline details
        pipeline_label = QLabel(
            "<b>Motion modes automatically include:</b><br>"
            "• CNN noise filter (removes 66% of false positives)<br>"
            "• Learned solidity filter (shape-based fallback)<br>"
            "<br>"
            "<i>ℹ️ All thresholds are learned from your video during analysis</i>"
        )
        pipeline_label.setWordWrap(True)
        pipeline_label.setStyleSheet("color: #666; font-size: 8pt; padding: 5px;")
        params_layout.addWidget(pipeline_label)
        
        params_group.setLayout(params_layout)
        return params_group
    
    def _update_mode_info(self):
        """Update mode info label based on selected mode."""
        mode = self.detection_mode_combo.currentData()
        
        info_text = {
            'fgbg': '⭐ Fast & accurate - Motion + CNN filter + learned shape filter',
            'fgbg_yolo': '🔬 High accuracy - Motion + filters + YOLO confirmation + species ID',
            'yolo_only': '🎯 Highest accuracy - Deep learning every frame (slowest)'
        }
        
        self.mode_info_label.setText(info_text.get(mode, ''))
    
    def _on_mode_change(self):
        """Handle detection mode changes."""
        self._update_mode_info()
        self.parameters_changed.emit(self.get_parameters())
    
    def _create_actions_group(self):
        """Create action buttons."""
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout()
        
        init_bg_btn = QPushButton("Initialize Background")
        init_bg_btn.clicked.connect(self.initialize_background_requested.emit)
        actions_layout.addWidget(init_bg_btn)
        
        test_btn = QPushButton("Test Detection (Space)")
        test_btn.clicked.connect(self.test_detection_requested.emit)
        actions_layout.addWidget(test_btn)
        
        analyze_btn = QPushButton("▶ Run Full Analysis")
        analyze_btn.setStyleSheet(
            "background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 10px;"
        )
        analyze_btn.clicked.connect(self.run_analysis_requested.emit)
        actions_layout.addWidget(analyze_btn)
        
        self.detection_count_label = QLabel("Detections: 0")
        self.detection_count_label.setStyleSheet(
            "font-size: 12pt; font-weight: bold;"
        )
        self.detection_count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        actions_layout.addWidget(self.detection_count_label)
        
        actions_group.setLayout(actions_layout)
        return actions_group
    
    def get_parameters(self):
        """Get current detection mode."""
        return {
            'detection_mode': self.detection_mode_combo.currentData()
        }
    
    def set_detection_mode(self, mode):
        """Set detection mode."""
        for i in range(self.detection_mode_combo.count()):
            if self.detection_mode_combo.itemData(i) == mode:
                self.detection_mode_combo.setCurrentIndex(i)
                break
    
    def set_detection_count(self, count):
        """Update detection count label."""
        self.detection_count_label.setText(f"Detections: {count}")
    
    def set_video_info(self, text):
        """Update video info label."""
        self.video_info_label.setText(text)
    
    def set_output_folder_info(self, text):
        """Update output folder label."""
        self.output_folder_label.setText(text)