"""YOLO-based object detector.

Wraps Ultralytics YOLO for bee/wasp detection and classification.
"""

import logging
from typing import List, Optional, Set
import numpy as np

from .base_detector import BaseDetector, Detection

logger = logging.getLogger(__name__)


class YOLODetector(BaseDetector):
    """YOLO-based deep learning detector.
    
    Uses Ultralytics YOLO for accurate bee/wasp detection and classification.
    
    Attributes:
        model: YOLO model instance
        conf_threshold: Confidence threshold (0-1)
        iou_threshold: IOU threshold for NMS
        tracking_classes: Set of class names to detect
        label_map: Mapping of YOLO classes to output labels
    """
    
    def __init__(
        self,
        model,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        tracking_classes: Optional[List[str]] = None,
        label_map: Optional[dict] = None,
        imgsz: int = 640  # Add this parameter
    ):
        """Initialize YOLO detector.
        
        Args:
            model: Ultralytics YOLO model
            conf_threshold: Confidence threshold
            iou_threshold: IOU threshold for NMS
            tracking_classes: List of class names to detect
            label_map: Mapping {yolo_class: output_label}
            imgsz: Image size for inference (640=default, 320=fast)
        """
        self.model = model
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.tracking_classes = set(tracking_classes) if tracking_classes else None
        self.label_map = label_map or {}
        self.imgsz = imgsz  # Store image size
        
        logger.info(f"YOLODetector initialized: conf={conf_threshold}, imgsz={imgsz}, classes={tracking_classes}")
    
    # def detect(self, frame: np.ndarray, **kwargs) -> List[Detection]:
    #     """Detect objects using YOLO.
        
    #     Args:
    #         frame: Input frame (BGR)
    #         **kwargs: Optional overrides (conf, iou)
            
    #     Returns:
    #         List of YOLO detections
    #     """
    #     # Override thresholds if provided
    #     conf = kwargs.get('conf', self.conf_threshold)
    #     iou = kwargs.get('iou', self.iou_threshold)
        
    #     # Run YOLO inference
    #     results = self.model(frame, conf=conf, iou=iou, verbose=False)
    def detect(self, frame: np.ndarray, **kwargs) -> List[Detection]:
        """Detect objects using YOLO.
        
        Args:
            frame: Input frame (BGR)
            **kwargs: Optional overrides (conf, iou, imgsz)
            
        Returns:
            List of YOLO detections
        """
        # Override thresholds if provided
        conf = kwargs.get('conf', self.conf_threshold)
        iou = kwargs.get('iou', self.iou_threshold)
        imgsz = kwargs.get('imgsz', self.imgsz)  # Add this
        
        # Run YOLO inference with image size
        results = self.model(frame, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
        
        # ... rest stays the same
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            if boxes is None or len(boxes) == 0:
                continue
            
            for i in range(len(boxes)):
                # Get class
                cls = int(boxes.cls[i])
                label = result.names[cls]
                
                # Filter by tracking classes
                if self.tracking_classes and label not in self.tracking_classes:
                    continue
                
                # Get bbox
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                bbox = (float(x1), float(y1), float(x2), float(y2))
                centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
                
                # Get confidence
                confidence = float(boxes.conf[i])
                
                # Map label if needed
                output_label = self.label_map.get(label, label)
                
                detections.append(Detection(
                    bbox=bbox,
                    centroid=centroid,
                    confidence=confidence,
                    label=output_label,
                    source='yolo',
                    metadata={
                        'class_id': cls,
                        'original_label': label
                    }
                ))
        
        return detections
    
   
    def configure(self, **kwargs) -> None:
        """Configure YOLO detector parameters.
        
        Args:
            **kwargs: conf_threshold, iou_threshold, tracking_classes, label_map, imgsz
        """
        if 'conf_threshold' in kwargs:
            self.conf_threshold = kwargs['conf_threshold']
        if 'iou_threshold' in kwargs:
            self.iou_threshold = kwargs['iou_threshold']
        if 'tracking_classes' in kwargs:
            self.tracking_classes = set(kwargs['tracking_classes'])
        if 'label_map' in kwargs:
            self.label_map = kwargs['label_map']
        if 'imgsz' in kwargs:  # Add this
            self.imgsz = kwargs['imgsz']
        
        logger.debug(f"YOLODetector configured: {kwargs}")
    
    def reset(self) -> None:
        """Reset YOLO detector (no state to reset)."""
        pass
    
    def get_source_name(self) -> str:
        """Get detector source name."""
        return 'yolo'
