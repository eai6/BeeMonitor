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
    """Canvas for visual nest editing with resize support."""
    
    nest_added = pyqtSignal(int, int)  # x, y clicked
    nest_selected = pyqtSignal(int)  # nest index
    nest_moved = pyqtSignal(int, int, int)  # index, new_x, new_y
    nest_resized = pyqtSignal(int)  # nest index
    nest_deleted = pyqtSignal(int)  # nest index
    hotel_changed = pyqtSignal()  # hotel ROI changed
    
    # Resize handle positions
    HANDLE_NONE = 0
    HANDLE_TL = 1  # Top-left
    HANDLE_TR = 2  # Top-right
    HANDLE_BL = 3  # Bottom-left
    HANDLE_BR = 4  # Bottom-right
    HANDLE_T = 5   # Top
    HANDLE_B = 6   # Bottom
    HANDLE_L = 7   # Left
    HANDLE_R = 8   # Right
    
    # Edit modes
    MODE_NESTS = 0
    MODE_HOTEL = 1
    
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
        self.hotel_roi = None  # (x1, y1, x2, y2) or None
        
        # Edit mode
        self.edit_mode = self.MODE_NESTS
        
        # Interaction state
        self.selected_nest = -1
        self.hovered_nest = -1
        self.dragging = False
        self.resizing = False
        self.resize_handle = self.HANDLE_NONE
        self.drag_offset = (0, 0)
        self.resize_start = None  # (x, y, w, h) at resize start
        
        # Hotel interaction state
        self.hotel_selected = False
        self.hotel_dragging = False
        self.hotel_resizing = False
        self.hotel_resize_handle = self.HANDLE_NONE
        self.hotel_resize_start = None
        
        # Display settings
        self.nest_width = 24
        self.nest_height = 14
        self.handle_size = 6  # Size of resize handles
        self.scale = 1.0
        self.offset = (0, 0)
    
    def set_edit_mode(self, mode: int):
        """Set edit mode (MODE_NESTS or MODE_HOTEL)."""
        self.edit_mode = mode
        self.selected_nest = -1
        self.hotel_selected = (mode == self.MODE_HOTEL)
        self._update_display()
    
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
    
    def set_hotel_roi(self, roi):
        """Set hotel ROI (x1, y1, x2, y2) or None."""
        self.hotel_roi = roi
        self._update_display()
    
    def get_hotel_roi(self):
        """Get hotel ROI."""
        return self.hotel_roi
    
    def set_nest_size(self, width: int, height: int):
        """Set default nest size."""
        self.nest_width = width
        self.nest_height = height
        self._update_display()
    
    def _get_nest_rect(self, nest) -> Tuple[int, int, int, int]:
        """Get nest bounding box (x1, y1, x2, y2)."""
        x, y = int(nest['x']), int(nest['y'])
        w = nest.get('w', self.nest_width)
        h = nest.get('h', self.nest_height)
        return (x - w // 2, y - h // 2, x + w // 2, y + h // 2)
    
    def _update_display(self):
        """Redraw the canvas with frame and nests."""
        if self.frame is None:
            return
        
        # Create display image
        display = self.frame.copy()
        
        # Draw hotel ROI first (behind nests)
        if self.hotel_roi is not None:
            x1, y1, x2, y2 = [int(v) for v in self.hotel_roi]
            
            if self.edit_mode == self.MODE_HOTEL:
                # Highlight when in hotel edit mode
                color = (0, 255, 0) if self.hotel_selected else (0, 200, 200)
                thickness = 2
                # Draw semi-transparent overlay outside ROI
                overlay = display.copy()
                cv2.rectangle(overlay, (0, 0), (display.shape[1], y1), (0, 0, 0), -1)
                cv2.rectangle(overlay, (0, y2), (display.shape[1], display.shape[0]), (0, 0, 0), -1)
                cv2.rectangle(overlay, (0, y1), (x1, y2), (0, 0, 0), -1)
                cv2.rectangle(overlay, (x2, y1), (display.shape[1], y2), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)
            else:
                color = (0, 200, 200)  # Cyan when not editing
                thickness = 1
            
            cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(display, "Hotel", (x1 + 5, y1 + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw resize handles if in hotel mode
            if self.edit_mode == self.MODE_HOTEL and self.hotel_selected:
                self._draw_handles(display, x1, y1, x2, y2, color=(0, 255, 255))
        
        # Draw all nests
        for i, nest in enumerate(self.nests):
            x1, y1, x2, y2 = self._get_nest_rect(nest)
            
            # Color based on state
            if self.edit_mode == self.MODE_NESTS and i == self.selected_nest:
                color = (0, 255, 0)  # Green for selected
                thickness = 2
            elif self.edit_mode == self.MODE_NESTS and i == self.hovered_nest:
                color = (0, 255, 255)  # Yellow for hovered
                thickness = 2
            else:
                color = (255, 0, 0)  # Blue for normal
                thickness = 1
            
            cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)
            
            # Draw nest ID
            label = str(nest.get('id', i + 1))
            cv2.putText(display, label, (x1, y1 - 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            
            # Draw resize handles for selected nest
            if self.edit_mode == self.MODE_NESTS and i == self.selected_nest:
                self._draw_handles(display, x1, y1, x2, y2)
        
        # Draw instructions based on mode
        if self.edit_mode == self.MODE_HOTEL:
            instructions = "Hotel Mode: Drag to move | Corners/Edges to resize | Click outside to create"
        else:
            instructions = "Nest Mode: Click: Add | Drag: Move | Corners: Resize | Del: Remove"
        cv2.putText(display, instructions, (10, display.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
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
    
    def _draw_handles(self, img, x1, y1, x2, y2, color=(0, 255, 0)):
        """Draw resize handles on selected item."""
        hs = self.handle_size
        
        # Corner handles
        cv2.rectangle(img, (x1-hs//2, y1-hs//2), (x1+hs//2, y1+hs//2), color, -1)  # TL
        cv2.rectangle(img, (x2-hs//2, y1-hs//2), (x2+hs//2, y1+hs//2), color, -1)  # TR
        cv2.rectangle(img, (x1-hs//2, y2-hs//2), (x1+hs//2, y2+hs//2), color, -1)  # BL
        cv2.rectangle(img, (x2-hs//2, y2-hs//2), (x2+hs//2, y2+hs//2), color, -1)  # BR
        
        # Edge handles
        mx, my = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.rectangle(img, (mx-hs//2, y1-hs//2), (mx+hs//2, y1+hs//2), color, -1)  # T
        cv2.rectangle(img, (mx-hs//2, y2-hs//2), (mx+hs//2, y2+hs//2), color, -1)  # B
        cv2.rectangle(img, (x1-hs//2, my-hs//2), (x1+hs//2, my+hs//2), color, -1)  # L
        cv2.rectangle(img, (x2-hs//2, my-hs//2), (x2+hs//2, my+hs//2), color, -1)  # R
    
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
            x1, y1, x2, y2 = self._get_nest_rect(nest)
            margin = 5
            if x1 - margin <= img_x <= x2 + margin and y1 - margin <= img_y <= y2 + margin:
                return i
        return -1
    
    def _is_inside_hotel(self, img_x: int, img_y: int) -> bool:
        """Check if point is inside hotel ROI."""
        if self.hotel_roi is None:
            return False
        x1, y1, x2, y2 = self.hotel_roi
        margin = 10
        return x1 - margin <= img_x <= x2 + margin and y1 - margin <= img_y <= y2 + margin
    
    def _find_handle_at(self, img_x: int, img_y: int, rect: Tuple[int, int, int, int]) -> int:
        """Find which resize handle is at the given position."""
        x1, y1, x2, y2 = [int(v) for v in rect]
        mx, my = (x1 + x2) // 2, (y1 + y2) // 2
        
        hs = self.handle_size + 5  # Hit area slightly larger than visual
        
        # Check corners first (higher priority)
        if abs(img_x - x1) <= hs and abs(img_y - y1) <= hs:
            return self.HANDLE_TL
        if abs(img_x - x2) <= hs and abs(img_y - y1) <= hs:
            return self.HANDLE_TR
        if abs(img_x - x1) <= hs and abs(img_y - y2) <= hs:
            return self.HANDLE_BL
        if abs(img_x - x2) <= hs and abs(img_y - y2) <= hs:
            return self.HANDLE_BR
        
        # Check edges
        if abs(img_x - mx) <= hs and abs(img_y - y1) <= hs:
            return self.HANDLE_T
        if abs(img_x - mx) <= hs and abs(img_y - y2) <= hs:
            return self.HANDLE_B
        if abs(img_x - x1) <= hs and abs(img_y - my) <= hs:
            return self.HANDLE_L
        if abs(img_x - x2) <= hs and abs(img_y - my) <= hs:
            return self.HANDLE_R
        
        return self.HANDLE_NONE
    
    def mousePressEvent(self, event):
        """Handle mouse press."""
        pos = self._widget_to_image(event.pos())
        if pos is None:
            return
        
        img_x, img_y = pos
        
        if event.button() == Qt.MouseButton.LeftButton:
            if self.edit_mode == self.MODE_HOTEL:
                self._handle_hotel_press(img_x, img_y)
            else:
                self._handle_nest_press(img_x, img_y)
        
        elif event.button() == Qt.MouseButton.RightButton:
            if self.edit_mode == self.MODE_NESTS:
                nest_idx = self._find_nest_at(img_x, img_y)
                if nest_idx >= 0:
                    del self.nests[nest_idx]
                    self.selected_nest = -1
                    self.nest_deleted.emit(nest_idx)
                    self._update_display()
    
    def _handle_hotel_press(self, img_x: int, img_y: int):
        """Handle mouse press in hotel edit mode."""
        if self.hotel_roi is not None:
            # Check for resize handle
            handle = self._find_handle_at(img_x, img_y, self.hotel_roi)
            if handle != self.HANDLE_NONE:
                self.hotel_resizing = True
                self.hotel_resize_handle = handle
                self.hotel_resize_start = tuple(self.hotel_roi)
                self.drag_offset = (img_x, img_y)
                return
            
            # Check if inside hotel for dragging
            if self._is_inside_hotel(img_x, img_y):
                self.hotel_dragging = True
                x1, y1, x2, y2 = self.hotel_roi
                self.drag_offset = (img_x - x1, img_y - y1)
                return
        
        # Create new hotel ROI
        h, w = self.frame.shape[:2]
        # Default to 80% of frame centered
        margin_x = int(w * 0.1)
        margin_y = int(h * 0.1)
        self.hotel_roi = (margin_x, margin_y, w - margin_x, h - margin_y)
        self.hotel_selected = True
        self.hotel_changed.emit()
        self._update_display()
    
    def _handle_nest_press(self, img_x: int, img_y: int):
        """Handle mouse press in nest edit mode."""
        # Check if clicking on resize handle of selected nest
        if self.selected_nest >= 0:
            nest = self.nests[self.selected_nest]
            rect = self._get_nest_rect(nest)
            handle = self._find_handle_at(img_x, img_y, rect)
            if handle != self.HANDLE_NONE:
                self.resizing = True
                self.resize_handle = handle
                self.resize_start = (nest['x'], nest['y'], 
                                    nest.get('w', self.nest_width),
                                    nest.get('h', self.nest_height))
                self.drag_offset = (img_x, img_y)
                return
        
        # Check if clicking on a nest
        nest_idx = self._find_nest_at(img_x, img_y)
        
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
    
    def mouseMoveEvent(self, event):
        """Handle mouse move."""
        pos = self._widget_to_image(event.pos())
        if pos is None:
            self.hovered_nest = -1
            return
        
        img_x, img_y = pos
        
        if self.edit_mode == self.MODE_HOTEL:
            self._handle_hotel_move(img_x, img_y)
        else:
            self._handle_nest_move(img_x, img_y)
    
    def _handle_hotel_move(self, img_x: int, img_y: int):
        """Handle mouse move in hotel edit mode."""
        if self.hotel_resizing and self.hotel_roi is not None:
            self._do_hotel_resize(img_x, img_y)
            self._update_display()
        elif self.hotel_dragging and self.hotel_roi is not None:
            # Move hotel
            x1, y1, x2, y2 = self.hotel_roi
            w, h_roi = x2 - x1, y2 - y1
            
            new_x1 = img_x - self.drag_offset[0]
            new_y1 = img_y - self.drag_offset[1]
            
            # Clamp to frame bounds
            h, w_frame = self.frame.shape[:2]
            new_x1 = max(0, min(w_frame - w, new_x1))
            new_y1 = max(0, min(h - h_roi, new_y1))
            
            self.hotel_roi = (new_x1, new_y1, new_x1 + w, new_y1 + h_roi)
            self._update_display()
        else:
            # Update cursor for resize handles
            if self.hotel_roi is not None:
                handle = self._find_handle_at(img_x, img_y, self.hotel_roi)
                self._set_cursor_for_handle(handle)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
    
    def _handle_nest_move(self, img_x: int, img_y: int):
        """Handle mouse move in nest edit mode."""
        if self.resizing and self.selected_nest >= 0:
            # Resize selected nest
            self._do_resize(img_x, img_y)
            self._update_display()
        elif self.dragging and self.selected_nest >= 0:
            # Move selected nest
            new_x = img_x - self.drag_offset[0]
            new_y = img_y - self.drag_offset[1]
            
            # Clamp to frame bounds
            h, w = self.frame.shape[:2]
            nest = self.nests[self.selected_nest]
            nw = nest.get('w', self.nest_width) // 2
            nh = nest.get('h', self.nest_height) // 2
            new_x = max(nw, min(w - nw, new_x))
            new_y = max(nh, min(h - nh, new_y))
            
            self.nests[self.selected_nest]['x'] = new_x
            self.nests[self.selected_nest]['y'] = new_y
            self._update_display()
        else:
            # Update hover state and cursor
            old_hover = self.hovered_nest
            self.hovered_nest = self._find_nest_at(img_x, img_y)
            
            # Check for resize handle hover on selected nest
            if self.selected_nest >= 0:
                nest = self.nests[self.selected_nest]
                rect = self._get_nest_rect(nest)
                handle = self._find_handle_at(img_x, img_y, rect)
                self._set_cursor_for_handle(handle)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
            
            if self.hovered_nest != old_hover:
                self._update_display()
    
    def _set_cursor_for_handle(self, handle: int):
        """Set cursor based on resize handle."""
        if handle in (self.HANDLE_TL, self.HANDLE_BR):
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        elif handle in (self.HANDLE_TR, self.HANDLE_BL):
            self.setCursor(Qt.CursorShape.SizeBDiagCursor)
        elif handle in (self.HANDLE_T, self.HANDLE_B):
            self.setCursor(Qt.CursorShape.SizeVerCursor)
        elif handle in (self.HANDLE_L, self.HANDLE_R):
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
    
    def _do_hotel_resize(self, img_x: int, img_y: int):
        """Apply hotel resize operation."""
        if self.hotel_resize_start is None:
            return
        
        ox1, oy1, ox2, oy2 = self.hotel_resize_start
        start_x, start_y = self.drag_offset
        dx = img_x - start_x
        dy = img_y - start_y
        
        min_size = 50
        h, w = self.frame.shape[:2]
        
        x1, y1, x2, y2 = ox1, oy1, ox2, oy2
        
        if self.hotel_resize_handle == self.HANDLE_TL:
            x1 = max(0, min(ox2 - min_size, ox1 + dx))
            y1 = max(0, min(oy2 - min_size, oy1 + dy))
        elif self.hotel_resize_handle == self.HANDLE_TR:
            x2 = min(w, max(ox1 + min_size, ox2 + dx))
            y1 = max(0, min(oy2 - min_size, oy1 + dy))
        elif self.hotel_resize_handle == self.HANDLE_BL:
            x1 = max(0, min(ox2 - min_size, ox1 + dx))
            y2 = min(h, max(oy1 + min_size, oy2 + dy))
        elif self.hotel_resize_handle == self.HANDLE_BR:
            x2 = min(w, max(ox1 + min_size, ox2 + dx))
            y2 = min(h, max(oy1 + min_size, oy2 + dy))
        elif self.hotel_resize_handle == self.HANDLE_T:
            y1 = max(0, min(oy2 - min_size, oy1 + dy))
        elif self.hotel_resize_handle == self.HANDLE_B:
            y2 = min(h, max(oy1 + min_size, oy2 + dy))
        elif self.hotel_resize_handle == self.HANDLE_L:
            x1 = max(0, min(ox2 - min_size, ox1 + dx))
        elif self.hotel_resize_handle == self.HANDLE_R:
            x2 = min(w, max(ox1 + min_size, ox2 + dx))
        
        self.hotel_roi = (x1, y1, x2, y2)
    
    def _do_resize(self, img_x: int, img_y: int):
        """Apply nest resize operation."""
        if self.resize_start is None:
            return
        
        ox, oy, ow, oh = self.resize_start
        start_x, start_y = self.drag_offset
        dx = img_x - start_x
        dy = img_y - start_y
        
        nest = self.nests[self.selected_nest]
        min_size = 8  # Minimum nest size
        
        # Calculate new dimensions based on handle
        new_x, new_y, new_w, new_h = ox, oy, ow, oh
        
        if self.resize_handle == self.HANDLE_TL:
            new_w = max(min_size, ow - dx)
            new_h = max(min_size, oh - dy)
            new_x = ox + (ow - new_w) / 2
            new_y = oy + (oh - new_h) / 2
        elif self.resize_handle == self.HANDLE_TR:
            new_w = max(min_size, ow + dx)
            new_h = max(min_size, oh - dy)
            new_x = ox + (new_w - ow) / 2
            new_y = oy + (oh - new_h) / 2
        elif self.resize_handle == self.HANDLE_BL:
            new_w = max(min_size, ow - dx)
            new_h = max(min_size, oh + dy)
            new_x = ox + (ow - new_w) / 2
            new_y = oy + (new_h - oh) / 2
        elif self.resize_handle == self.HANDLE_BR:
            new_w = max(min_size, ow + dx)
            new_h = max(min_size, oh + dy)
            new_x = ox + (new_w - ow) / 2
            new_y = oy + (new_h - oh) / 2
        elif self.resize_handle == self.HANDLE_T:
            new_h = max(min_size, oh - dy)
            new_y = oy + (oh - new_h) / 2
        elif self.resize_handle == self.HANDLE_B:
            new_h = max(min_size, oh + dy)
            new_y = oy + (new_h - oh) / 2
        elif self.resize_handle == self.HANDLE_L:
            new_w = max(min_size, ow - dx)
            new_x = ox + (ow - new_w) / 2
        elif self.resize_handle == self.HANDLE_R:
            new_w = max(min_size, ow + dx)
            new_x = ox + (new_w - ow) / 2
        
        nest['x'] = new_x
        nest['y'] = new_y
        nest['w'] = int(new_w)
        nest['h'] = int(new_h)
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        if self.edit_mode == self.MODE_HOTEL:
            if self.hotel_resizing or self.hotel_dragging:
                self.hotel_changed.emit()
            self.hotel_dragging = False
            self.hotel_resizing = False
            self.hotel_resize_handle = self.HANDLE_NONE
            self.hotel_resize_start = None
        else:
            if self.resizing and self.selected_nest >= 0:
                self.nest_resized.emit(self.selected_nest)
            elif self.dragging and self.selected_nest >= 0:
                nest = self.nests[self.selected_nest]
                self.nest_moved.emit(self.selected_nest, int(nest['x']), int(nest['y']))
            
            self.dragging = False
            self.resizing = False
            self.resize_handle = self.HANDLE_NONE
            self.resize_start = None
    
    def keyPressEvent(self, event):
        """Handle key press."""
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            if self.edit_mode == self.MODE_NESTS and self.selected_nest >= 0:
                del self.nests[self.selected_nest]
                self.nest_deleted.emit(self.selected_nest)
                self.selected_nest = -1
                self._update_display()
            elif self.edit_mode == self.MODE_HOTEL and self.hotel_roi is not None:
                self.hotel_roi = None
                self.hotel_changed.emit()
                self._update_display()
    
    def resizeEvent(self, event):
        """Handle resize."""
        super().resizeEvent(event)
        self._update_display()


class VisualNestEditorDialog(QDialog):
    """Visual dialog for editing nest positions and hotel ROI on video frame."""
    
    nests_updated = pyqtSignal(list)
    
    def __init__(self, parent=None, frame: np.ndarray = None, 
                 nests: List[Dict] = None, hotel_roi = None,
                 grid_rows: int = 6, grid_cols: int = 10):
        super().__init__(parent)
        
        self.setWindowTitle("Visual Nest & Hotel Editor")
        self.setMinimumSize(900, 600)
        self.resize(1100, 700)
        
        self.frame = frame
        self.nests = nests or []
        self.hotel_roi = hotel_roi
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        
        self._setup_ui()
        
        # Initialize canvas
        if self.frame is not None:
            self.canvas.set_frame(self.frame)
        if self.nests:
            self.canvas.set_nests(self.nests)
            self._update_table()
        if self.hotel_roi:
            self.canvas.set_hotel_roi(self.hotel_roi)
            self._update_hotel_label()
        
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
        self.canvas.nest_resized.connect(self._on_nest_resized)
        self.canvas.nest_deleted.connect(self._on_nest_deleted)
        self.canvas.hotel_changed.connect(self._on_hotel_changed)
        left_layout.addWidget(self.canvas)
        
        layout.addWidget(left_widget, stretch=3)
        
        # Right side: Controls
        right_widget = QWidget()
        right_widget.setMaximumWidth(300)
        right_layout = QVBoxLayout()
        right_widget.setLayout(right_layout)
        
        # Edit Mode Toggle
        mode_group = QGroupBox("Edit Mode")
        mode_layout = QVBoxLayout()
        
        self.nest_mode_btn = QPushButton("🔲 Edit Nests")
        self.nest_mode_btn.setCheckable(True)
        self.nest_mode_btn.setChecked(True)
        self.nest_mode_btn.clicked.connect(lambda: self._set_mode('nests'))
        mode_layout.addWidget(self.nest_mode_btn)
        
        self.hotel_mode_btn = QPushButton("🏨 Edit Hotel ROI")
        self.hotel_mode_btn.setCheckable(True)
        self.hotel_mode_btn.clicked.connect(lambda: self._set_mode('hotel'))
        mode_layout.addWidget(self.hotel_mode_btn)
        
        mode_group.setLayout(mode_layout)
        right_layout.addWidget(mode_group)
        
        # Info section
        info_group = QGroupBox("Info")
        info_layout = QVBoxLayout()
        
        self.count_label = QLabel("Nests: 0")
        self.count_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        info_layout.addWidget(self.count_label)
        
        expected_label = QLabel(f"Expected: {self.grid_rows} × {self.grid_cols} = {self.grid_rows * self.grid_cols}")
        expected_label.setStyleSheet("color: gray;")
        info_layout.addWidget(expected_label)
        
        self.hotel_label = QLabel("Hotel: Not set")
        self.hotel_label.setStyleSheet("color: #00CED1;")
        info_layout.addWidget(self.hotel_label)
        
        info_group.setLayout(info_layout)
        right_layout.addWidget(info_group)
        
        # Nest size settings
        size_group = QGroupBox("Default Nest Size")
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
        
        clear_nests_btn = QPushButton("🗑️ Clear Nests")
        clear_nests_btn.clicked.connect(self._clear_nests)
        actions_layout.addWidget(clear_nests_btn)
        
        clear_hotel_btn = QPushButton("🗑️ Clear Hotel")
        clear_hotel_btn.clicked.connect(self._clear_hotel)
        actions_layout.addWidget(clear_hotel_btn)
        
        actions_group.setLayout(actions_layout)
        right_layout.addWidget(actions_group)
        
        # Nest table (compact)
        table_group = QGroupBox("Nest List")
        table_layout = QVBoxLayout()
        
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(['ID', 'X', 'Y', 'W', 'H'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setMaximumHeight(120)
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
    
    def _set_mode(self, mode: str):
        """Set edit mode."""
        if mode == 'nests':
            self.nest_mode_btn.setChecked(True)
            self.hotel_mode_btn.setChecked(False)
            self.canvas.set_edit_mode(NestCanvas.MODE_NESTS)
        else:
            self.nest_mode_btn.setChecked(False)
            self.hotel_mode_btn.setChecked(True)
            self.canvas.set_edit_mode(NestCanvas.MODE_HOTEL)
    
    def _update_hotel_label(self):
        """Update hotel ROI label."""
        roi = self.canvas.get_hotel_roi()
        if roi:
            x1, y1, x2, y2 = [int(v) for v in roi]
            w, h = x2 - x1, y2 - y1
            self.hotel_label.setText(f"Hotel: {w}×{h} @ ({x1},{y1})")
            self.hotel_label.setStyleSheet("color: #00CED1; font-weight: bold;")
        else:
            self.hotel_label.setText("Hotel: Not set")
            self.hotel_label.setStyleSheet("color: gray;")
    
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
            self.table.setItem(i, 3, QTableWidgetItem(f"{nest.get('w', 24)}"))
            self.table.setItem(i, 4, QTableWidgetItem(f"{nest.get('h', 14)}"))
    
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
    
    def _on_nest_resized(self, idx):
        """Handle nest resized."""
        self._update_table()
    
    def _on_nest_deleted(self, idx):
        """Handle nest deleted."""
        self._update_count_label()
        self._update_table()
    
    def _on_hotel_changed(self):
        """Handle hotel ROI changed."""
        self._update_hotel_label()
    
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
    
    def _clear_nests(self):
        """Clear all nests."""
        reply = QMessageBox.question(
            self,
            "Clear Nests",
            "Remove all nests?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.canvas.set_nests([])
            self._update_count_label()
            self._update_table()
    
    def _clear_hotel(self):
        """Clear hotel ROI."""
        reply = QMessageBox.question(
            self,
            "Clear Hotel",
            "Remove hotel ROI?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.canvas.set_hotel_roi(None)
            self._update_hotel_label()
    
    def _apply(self):
        """Apply changes and close."""
        nests = self.canvas.get_nests()
        self.nests_updated.emit(nests)
        self.accept()
    
    def get_nests(self) -> List[Dict]:
        """Get current nests."""
        return self.canvas.get_nests()
    
    def get_hotel_roi(self):
        """Get current hotel ROI."""
        return self.canvas.get_hotel_roi()


def show_visual_nest_editor(
    parent,
    frame: np.ndarray,
    nests: List[Dict] = None,
    hotel_roi = None,
    grid_rows: int = 6,
    grid_cols: int = 10
) -> Optional[Tuple[List[Dict], any]]:
    """Show visual nest editor dialog.
    
    Args:
        parent: Parent widget
        frame: Video frame to display
        nests: Current nests
        hotel_roi: Current hotel ROI (x1, y1, x2, y2) or None
        grid_rows: Expected rows
        grid_cols: Expected cols
    
    Returns:
        Tuple of (updated_nests, updated_hotel_roi) or None if cancelled
    """
    dialog = VisualNestEditorDialog(
        parent,
        frame=frame,
        nests=nests,
        hotel_roi=hotel_roi,
        grid_rows=grid_rows,
        grid_cols=grid_cols
    )
    
    if dialog.exec() == QDialog.DialogCode.Accepted:
        return (dialog.get_nests(), dialog.get_hotel_roi())
    return None