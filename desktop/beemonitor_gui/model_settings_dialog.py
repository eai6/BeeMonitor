"""
Model Settings Dialog
=====================

Dialog for selecting custom YOLO models for nest detection and bee tracking.
"""

import os
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QLineEdit, QGroupBox, QFileDialog, QMessageBox
)
from PyQt6.QtCore import pyqtSignal


class ModelSettingsDialog(QDialog):
    """Dialog for configuring custom detection models."""
    
    models_updated = pyqtSignal(dict)
    
    def __init__(self, parent=None, current_models: dict = None):
        super().__init__(parent)
        
        self.setWindowTitle("Model Settings")
        self.setMinimumWidth(500)
        
        self.current_models = current_models or {
            'nest_detection': '',
            'bee_tracking': ''
        }
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup dialog UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Info label
        info_label = QLabel(
            "Select custom YOLO models for detection.\n"
            "Leave empty to use default models from config."
        )
        info_label.setStyleSheet("color: gray; margin-bottom: 10px;")
        layout.addWidget(info_label)
        
        # Nest Detection Model
        nest_group = QGroupBox("Nest Detection Model")
        nest_layout = QVBoxLayout()
        
        nest_path_layout = QHBoxLayout()
        self.nest_model_edit = QLineEdit()
        self.nest_model_edit.setPlaceholderText("Path to nest detection model (.pt)")
        self.nest_model_edit.setText(self.current_models.get('nest_detection', ''))
        nest_path_layout.addWidget(self.nest_model_edit)
        
        nest_browse_btn = QPushButton("Browse...")
        nest_browse_btn.clicked.connect(lambda: self._browse_model('nest'))
        nest_path_layout.addWidget(nest_browse_btn)
        
        nest_layout.addLayout(nest_path_layout)
        
        nest_clear_btn = QPushButton("Clear (Use Default)")
        nest_clear_btn.clicked.connect(lambda: self.nest_model_edit.clear())
        nest_layout.addWidget(nest_clear_btn)
        
        nest_group.setLayout(nest_layout)
        layout.addWidget(nest_group)
        
        # Bee Tracking Model
        bee_group = QGroupBox("Bee Tracking Model")
        bee_layout = QVBoxLayout()
        
        bee_path_layout = QHBoxLayout()
        self.bee_model_edit = QLineEdit()
        self.bee_model_edit.setPlaceholderText("Path to bee tracking model (.pt)")
        self.bee_model_edit.setText(self.current_models.get('bee_tracking', ''))
        bee_path_layout.addWidget(self.bee_model_edit)
        
        bee_browse_btn = QPushButton("Browse...")
        bee_browse_btn.clicked.connect(lambda: self._browse_model('bee'))
        bee_path_layout.addWidget(bee_browse_btn)
        
        bee_layout.addLayout(bee_path_layout)
        
        bee_clear_btn = QPushButton("Clear (Use Default)")
        bee_clear_btn.clicked.connect(lambda: self.bee_model_edit.clear())
        bee_layout.addWidget(bee_clear_btn)
        
        bee_group.setLayout(bee_layout)
        layout.addWidget(bee_group)
        
        # Status
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #4CAF50;")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        
        apply_btn = QPushButton("Apply")
        apply_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        apply_btn.clicked.connect(self._apply)
        btn_layout.addWidget(apply_btn)
        
        layout.addLayout(btn_layout)
    
    def _browse_model(self, model_type: str):
        """Browse for model file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select {model_type.title()} Model",
            "",
            "YOLO Models (*.pt);;All Files (*)"
        )
        
        if file_path:
            if model_type == 'nest':
                self.nest_model_edit.setText(file_path)
            else:
                self.bee_model_edit.setText(file_path)
    
    def _validate_model(self, path: str) -> bool:
        """Validate model file exists and is readable."""
        if not path:
            return True  # Empty is OK (use default)
        
        if not os.path.exists(path):
            return False
        
        if not path.endswith('.pt'):
            return False
        
        return True
    
    def _apply(self):
        """Validate and apply model settings."""
        nest_path = self.nest_model_edit.text().strip()
        bee_path = self.bee_model_edit.text().strip()
        
        # Validate paths
        if nest_path and not self._validate_model(nest_path):
            QMessageBox.warning(
                self,
                "Invalid Model",
                f"Nest detection model not found or invalid:\n{nest_path}"
            )
            return
        
        if bee_path and not self._validate_model(bee_path):
            QMessageBox.warning(
                self,
                "Invalid Model",
                f"Bee tracking model not found or invalid:\n{bee_path}"
            )
            return
        
        # Emit updated models
        models = {
            'nest_detection': nest_path,
            'bee_tracking': bee_path
        }
        
        self.models_updated.emit(models)
        self.accept()
    
    def get_models(self) -> dict:
        """Get current model paths."""
        return {
            'nest_detection': self.nest_model_edit.text().strip(),
            'bee_tracking': self.bee_model_edit.text().strip()
        }


def show_model_settings(parent, current_models: dict = None) -> dict:
    """Show model settings dialog.
    
    Args:
        parent: Parent widget
        current_models: Current model paths dict
    
    Returns:
        Updated model paths dict or None if cancelled
    """
    dialog = ModelSettingsDialog(parent, current_models)
    
    if dialog.exec() == QDialog.DialogCode.Accepted:
        return dialog.get_models()
    return None