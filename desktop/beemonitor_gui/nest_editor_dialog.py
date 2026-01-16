"""
Visual Nest Editor Dialog
=========================

Dialog for visually editing nest positions on the video frame.
- Click to add nests
- Drag to move nests
- Right-click or Delete key to remove
- Auto-fill grid option
"""

import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSpinBox, QGroupBox, QMessageBox, QWidget,
    QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt6.QtCore import Qt, pyqtSignal, QPoint
from PyQt6.QtGui import QImage, QPixmap
from typing import List, Dict, Optional, Tuple


class NestCanvas(QLabel):
    """Canvas for visual nest editing."""
    
    nest_added = pyqtSignal(int, int)  # x, y clicked
    nest_selected = pyqtSignal(int)  # nest index
    nest_moved = pyqtSignal(int, int, int)  # index, new_x, new_y
    nest_deleted = pyqtSignal(int)  # nest index
    
    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 480)
        self.setStyleSheet("background-color: #1e1e1e; border: 1px solid #444;")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
        self.frame = None
        self.display_pixmap = None
        self.nests = []  # List of {'id': int, 'x': float, 'y': float, 'w': int, 'h': int}
        
        # Interaction state
        self.selected_nest = -1
        self.hovered_nest = -1
        self.dragging = False
        self.drag_offset = (0, 0)
        
        # Display settings
        self.nest_width = 24
        self.nest_height = 14
        self.scale = 1.0
        self.offset = (0, 0)
    
    def set_frame(self, frame: np.ndarray):
        """Set the background frame."""
        self.frame = frame.copy()
        self._update_display()
    
    def set_nests(self, nests: List[Dict]):
        """Set nest positions."""
        self.nests = [n.copy() for n in nests]
        self.selected_nest = -1
        self._update_display()
    
    def get_nests(self) -> List[Dict]:
        """Get current nest positions."""
        return [n.copy() for n in self.nests]
    
    def set_nest_size(self, width: int, height: int):
        """Set default nest size."""
        self.nest_width = width
        self.nest_height = height
        self._update_display()
    
    def _update_display(self):
        """Redraw the canvas with frame and nests."""
        if self.frame is None:
            return
        
        # Create display image
        display = self.frame.copy()
        
        # Draw all nests
        for i, nest in enumerate(self.nests):
            x, y = int(nest['x']), int(nest['y'])
            w = nest.get('w', self.nest_width)
            h = nest.get('h', self.nest_height)
            
            x1, y1 = x - w // 2, y - h // 2
            x2, y2 = x + w // 2, y + h // 2
            
            # Color based on state
            if i == self.selected_nest:
                color = (0, 255, 0)  # Green for selected
                thickness = 2
            elif i == self.hovered_nest:
                color = (0, 255, 255)  # Yellow for hovered
                thickness = 2
            else:
                color = (255, 0, 0)  # Blue for normal
                thickness = 1
            
            cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)
            
            # Draw nest ID
            label = str(nest.get('id', i + 1))
            cv2.putText(display, label, (x - 5, y1 - 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        
        # Draw instructions
        cv2.putText(display, "Click: Add | Drag: Move | Right-click/Del: Remove", 
                   (10, display.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Convert to QPixmap
        frame_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qt_image)
        
        # Scale to fit widget
        scaled = pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        # Calculate scale and offset for coordinate conversion
        self.scale = scaled.width() / pixmap.width()
        self.offset = (
            (self.width() - scaled.width()) // 2,
            (self.height() - scaled.height()) // 2
        )
        
        self.display_pixmap = pixmap
        self.setPixmap(scaled)
    
    def _widget_to_image(self, pos: QPoint) -> Optional[Tuple[int, int]]:
        """Convert widget coordinates to image coordinates."""
        if self.frame is None:
            return None
        
        x = (pos.x() - self.offset[0]) / self.scale
        y = (pos.y() - self.offset[1]) / self.scale
        
        h, w = self.frame.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            return (int(x), int(y))
        return None
    
    def _find_nest_at(self, img_x: int, img_y: int) -> int:
        """Find nest index at given image coordinates."""
        for i, nest in enumerate(self.nests):
            x, y = nest['x'], nest['y']
            w = nest.get('w', self.nest_width) // 2 + 5  # Add margin
            h = nest.get('h', self.nest_height) // 2 + 5
            
            if abs(img_x - x) <= w and abs(img_y - y) <= h:
                return i
        return -1
    
    def mousePressEvent(self, event):
        """Handle mouse press."""
        pos = self._widget_to_image(event.pos())
        if pos is None:
            return
        
        img_x, img_y = pos
        nest_idx = self._find_nest_at(img_x, img_y)
        
        if event.button() == Qt.MouseButton.LeftButton:
            if nest_idx >= 0:
                # Select and start dragging
                self.selected_nest = nest_idx
                self.dragging = True
                nest = self.nests[nest_idx]
                self.drag_offset = (img_x - nest['x'], img_y - nest['y'])
                self.nest_selected.emit(nest_idx)
            else:
                # Add new nest
                new_id = max([n.get('id', 0) for n in self.nests], default=0) + 1
                self.nests.append({
                    'id': new_id,
                    'x': img_x,
                    'y': img_y,
                    'w': self.nest_width,
                    'h': self.nest_height
                })
                self.selected_nest = len(self.nests) - 1
                self.nest_added.emit(img_x, img_y)
            
            self._update_display()
        
        elif event.button() == Qt.MouseButton.RightButton:
            if nest_idx >= 0:
                # Delete nest
                del self.nests[nest_idx]
                self.selected_nest = -1
                self.nest_deleted.emit(nest_idx)
                self._update_display()
    
    def mouseMoveEvent(self, event):
        """Handle mouse move."""
        pos = self._widget_to_image(event.pos())
        if pos is None:
            self.hovered_nest = -1
            return
        
        img_x, img_y = pos
        
        if self.dragging and self.selected_nest >= 0:
            # Move selected nest
            new_x = img_x - self.drag_offset[0]
            new_y = img_y - self.drag_offset[1]
            
            # Clamp to frame bounds
            h, w = self.frame.shape[:2]
            new_x = max(self.nest_width // 2, min(w - self.nest_width // 2, new_x))
            new_y = max(self.nest_height // 2, min(h - self.nest_height // 2, new_y))
            
            self.nests[self.selected_nest]['x'] = new_x
            self.nests[self.selected_nest]['y'] = new_y
            self._update_display()
        else:
            # Update hover state
            old_hover = self.hovered_nest
            self.hovered_nest = self._find_nest_at(img_x, img_y)
            if self.hovered_nest != old_hover:
                self._update_display()
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        if self.dragging and self.selected_nest >= 0:
            nest = self.nests[self.selected_nest]
            self.nest_moved.emit(self.selected_nest, int(nest['x']), int(nest['y']))
        
        self.dragging = False
    
    def keyPressEvent(self, event):
        """Handle key press."""
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            if self.selected_nest >= 0:
                del self.nests[self.selected_nest]
                self.nest_deleted.emit(self.selected_nest)
                self.selected_nest = -1
                self._update_display()
    
    def resizeEvent(self, event):
        """Handle resize."""
        super().resizeEvent(event)
        self._update_display()


class VisualNestEditorDialog(QDialog):
    """Visual dialog for editing nest positions on video frame."""
    
    nests_updated = pyqtSignal(list)
    
    def __init__(self, parent=None, frame: np.ndarray = None, 
                 nests: List[Dict] = None,
                 grid_rows: int = 6, grid_cols: int = 10):
        super().__init__(parent)
        
        self.setWindowTitle("Visual Nest Editor")
        self.setMinimumSize(900, 600)
        self.resize(1100, 700)
        
        self.frame = frame
        self.nests = nests or []
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        
        self._setup_ui()
        
        # Initialize canvas
        if self.frame is not None:
            self.canvas.set_frame(self.frame)
        if self.nests:
            self.canvas.set_nests(self.nests)
            self._update_table()
        
        self._update_count_label()
    
    def _setup_ui(self):
        """Setup dialog UI."""
        layout = QHBoxLayout()
        self.setLayout(layout)
        
        # Left side: Canvas
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_widget.setLayout(left_layout)
        
        # Canvas
        self.canvas = NestCanvas()
        self.canvas.nest_added.connect(self._on_nest_added)
        self.canvas.nest_selected.connect(self._on_nest_selected)
        self.canvas.nest_moved.connect(self._on_nest_moved)
        self.canvas.nest_deleted.connect(self._on_nest_deleted)
        left_layout.addWidget(self.canvas)
        
        layout.addWidget(left_widget, stretch=3)
        
        # Right side: Controls
        right_widget = QWidget()
        right_widget.setMaximumWidth(300)
        right_layout = QVBoxLayout()
        right_widget.setLayout(right_layout)
        
        # Info section
        info_group = QGroupBox("Info")
        info_layout = QVBoxLayout()
        
        self.count_label = QLabel("Nests: 0")
        self.count_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        info_layout.addWidget(self.count_label)
        
        expected_label = QLabel(f"Expected: {self.grid_rows} × {self.grid_cols} = {self.grid_rows * self.grid_cols}")
        expected_label.setStyleSheet("color: gray;")
        info_layout.addWidget(expected_label)
        
        info_group.setLayout(info_layout)
        right_layout.addWidget(info_group)
        
        # Nest size settings
        size_group = QGroupBox("Nest Size")
        size_layout = QHBoxLayout()
        
        size_layout.addWidget(QLabel("W:"))
        self.width_spin = QSpinBox()
        self.width_spin.setRange(10, 100)
        self.width_spin.setValue(24)
        self.width_spin.valueChanged.connect(self._on_size_changed)
        size_layout.addWidget(self.width_spin)
        
        size_layout.addWidget(QLabel("H:"))
        self.height_spin = QSpinBox()
        self.height_spin.setRange(10, 100)
        self.height_spin.setValue(14)
        self.height_spin.valueChanged.connect(self._on_size_changed)
        size_layout.addWidget(self.height_spin)
        
        size_group.setLayout(size_layout)
        right_layout.addWidget(size_group)
        
        # Grid generation
        grid_group = QGroupBox("Auto-Generate Grid")
        grid_layout = QVBoxLayout()
        
        rows_layout = QHBoxLayout()
        rows_layout.addWidget(QLabel("Rows:"))
        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(1, 20)
        self.rows_spin.setValue(self.grid_rows)
        rows_layout.addWidget(self.rows_spin)
        grid_layout.addLayout(rows_layout)
        
        cols_layout = QHBoxLayout()
        cols_layout.addWidget(QLabel("Cols:"))
        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(1, 30)
        self.cols_spin.setValue(self.grid_cols)
        cols_layout.addWidget(self.cols_spin)
        grid_layout.addLayout(cols_layout)
        
        padding_layout = QHBoxLayout()
        padding_layout.addWidget(QLabel("Padding:"))
        self.padding_spin = QSpinBox()
        self.padding_spin.setRange(0, 200)
        self.padding_spin.setValue(50)
        padding_layout.addWidget(self.padding_spin)
        grid_layout.addLayout(padding_layout)
        
        gen_btn = QPushButton("🔲 Generate Grid")
        gen_btn.clicked.connect(self._generate_grid)
        grid_layout.addWidget(gen_btn)
        
        grid_group.setLayout(grid_layout)
        right_layout.addWidget(grid_group)
        
        # Actions
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout()
        
        clear_btn = QPushButton("🗑️ Clear All")
        clear_btn.clicked.connect(self._clear_all)
        actions_layout.addWidget(clear_btn)
        
        actions_group.setLayout(actions_layout)
        right_layout.addWidget(actions_group)
        
        # Nest table (compact)
        table_group = QGroupBox("Nest List")
        table_layout = QVBoxLayout()
        
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(['ID', 'X', 'Y'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setMaximumHeight(150)
        self.table.itemSelectionChanged.connect(self._on_table_selection)
        table_layout.addWidget(self.table)
        
        table_group.setLayout(table_layout)
        right_layout.addWidget(table_group)
        
        right_layout.addStretch()
        
        # Dialog buttons
        btn_layout = QHBoxLayout()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        
        apply_btn = QPushButton("Apply")
        apply_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        apply_btn.clicked.connect(self._apply)
        btn_layout.addWidget(apply_btn)
        
        right_layout.addLayout(btn_layout)
        
        layout.addWidget(right_widget, stretch=1)
    
    def _update_count_label(self):
        """Update nest count label."""
        count = len(self.canvas.get_nests())
        expected = self.grid_rows * self.grid_cols
        
        if count == expected:
            color = "#4CAF50"  # Green
        elif count > 0:
            color = "#FF9800"  # Orange
        else:
            color = "#f44336"  # Red
        
        self.count_label.setText(f"Nests: {count}")
        self.count_label.setStyleSheet(f"font-weight: bold; font-size: 12pt; color: {color};")
    
    def _update_table(self):
        """Update nest table."""
        nests = self.canvas.get_nests()
        self.table.setRowCount(len(nests))
        
        for i, nest in enumerate(nests):
            self.table.setItem(i, 0, QTableWidgetItem(str(nest.get('id', i + 1))))
            self.table.setItem(i, 1, QTableWidgetItem(f"{nest['x']:.0f}"))
            self.table.setItem(i, 2, QTableWidgetItem(f"{nest['y']:.0f}"))
    
    def _on_nest_added(self, x, y):
        """Handle nest added."""
        self._update_count_label()
        self._update_table()
    
    def _on_nest_selected(self, idx):
        """Handle nest selected on canvas."""
        self.table.selectRow(idx)
    
    def _on_nest_moved(self, idx, x, y):
        """Handle nest moved."""
        self._update_table()
    
    def _on_nest_deleted(self, idx):
        """Handle nest deleted."""
        self._update_count_label()
        self._update_table()
    
    def _on_table_selection(self):
        """Handle table selection change."""
        rows = self.table.selectionModel().selectedRows()
        if rows:
            self.canvas.selected_nest = rows[0].row()
            self.canvas._update_display()
    
    def _on_size_changed(self):
        """Handle nest size change."""
        self.canvas.set_nest_size(
            self.width_spin.value(),
            self.height_spin.value()
        )
    
    def _generate_grid(self):
        """Generate evenly spaced nest grid."""
        if self.frame is None:
            QMessageBox.warning(self, "No Frame", "No video frame available")
            return
        
        reply = QMessageBox.question(
            self,
            "Generate Grid",
            f"Replace current nests with {self.rows_spin.value()}×{self.cols_spin.value()} grid?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        rows = self.rows_spin.value()
        cols = self.cols_spin.value()
        padding = self.padding_spin.value()
        nest_w = self.width_spin.value()
        nest_h = self.height_spin.value()
        
        h, w = self.frame.shape[:2]
        
        # Calculate spacing
        avail_w = w - 2 * padding
        avail_h = h - 2 * padding
        
        spacing_x = avail_w / cols
        spacing_y = avail_h / rows
        
        # Generate nests
        nests = []
        nest_id = 1
        
        for row in range(rows):
            for col in range(cols):
                cx = padding + (col + 0.5) * spacing_x
                cy = padding + (row + 0.5) * spacing_y
                
                nests.append({
                    'id': nest_id,
                    'x': cx,
                    'y': cy,
                    'w': nest_w,
                    'h': nest_h,
                    'row': row + 1,
                    'col': col + 1
                })
                nest_id += 1
        
        self.canvas.set_nests(nests)
        self._update_count_label()
        self._update_table()
    
    def _clear_all(self):
        """Clear all nests."""
        reply = QMessageBox.question(
            self,
            "Clear All",
            "Remove all nests?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.canvas.set_nests([])
            self._update_count_label()
            self._update_table()
    
    def _apply(self):
        """Apply changes and close."""
        nests = self.canvas.get_nests()
        self.nests_updated.emit(nests)
        self.accept()
    
    def get_nests(self) -> List[Dict]:
        """Get current nests."""
        return self.canvas.get_nests()


def show_visual_nest_editor(
    parent,
    frame: np.ndarray,
    nests: List[Dict] = None,
    grid_rows: int = 6,
    grid_cols: int = 10
) -> Optional[List[Dict]]:
    """Show visual nest editor dialog.
    
    Args:
        parent: Parent widget
        frame: Video frame to display
        nests: Current nests
        grid_rows: Expected rows
        grid_cols: Expected cols
    
    Returns:
        Updated nests or None if cancelled
    """
    dialog = VisualNestEditorDialog(
        parent,
        frame=frame,
        nests=nests,
        grid_rows=grid_rows,
        grid_cols=grid_cols
    )
    
    if dialog.exec() == QDialog.DialogCode.Accepted:
        return dialog.get_nests()
    return None