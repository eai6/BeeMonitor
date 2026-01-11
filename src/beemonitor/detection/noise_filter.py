"""CNN-based noise filter for blob detections.

Filters false positive blobs using a trained CNN classifier.
"""

import logging
from typing import List, Tuple, Optional
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from .base_detector import Detection

logger = logging.getLogger(__name__)


class SimpleBeeClassifier(nn.Module):
    """Enhanced CNN for bee vs noise classification."""
    
    def __init__(self, num_classes: int = 2):
        super(SimpleBeeClassifier, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class BeeNoiseFilter:
    """Fast noise filter using CNN classifier.
    
    Filters out clear noise cases, passes uncertain/bee cases to YOLO.
    Conservative approach - won't miss bees.
    """
    
    def __init__(
        self,
        model_path: str,
        noise_threshold: float = 0.9,
        image_size: int = 64,
        device: Optional[str] = None,
        batch_size: int = 32
    ):
        """Initialize noise filter.
        
        Args:
            model_path: Path to trained model
            noise_threshold: Threshold for noise filtering (higher = more conservative)
                           - 0.9: Filter if noise_prob > 0.9 (bee_prob < 0.1)
                           - 0.8: Filter if noise_prob > 0.8 (bee_prob < 0.2)
            image_size: Input size
            device: 'cuda', 'mps', 'cpu', or None (auto-detect)
            batch_size: Batch size for inference
        """
        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        self.device = torch.device(device)
        self.noise_threshold = noise_threshold
        self.bee_threshold = 1.0 - noise_threshold
        self.image_size = image_size
        self.batch_size = batch_size
        
        # Load model
        self.model = SimpleBeeClassifier(num_classes=2).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(f"BeeNoiseFilter initialized on {self.device}")
        logger.info(f"Noise threshold: {noise_threshold} (bee_prob < {self.bee_threshold})")
    
    def is_noise(self, crop: np.ndarray) -> Tuple[bool, float]:
        """Check if crop is noise.
        
        Args:
            crop: Image crop (BGR)
            
        Returns:
            Tuple of (is_noise, bee_probability)
        """
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(img_tensor)
            probabilities = torch.softmax(output, dim=1)
            bee_prob = probabilities[0, 1].item()
        
        is_noise = bee_prob < self.bee_threshold
        return is_noise, bee_prob
    
    def is_bee(self, crop: np.ndarray) -> bool:
        """Check if crop is a bee (legacy interface).
        
        Args:
            crop: Image crop (BGR)
            
        Returns:
            True if bee, False if noise
        """
        is_noise, _ = self.is_noise(crop)
        return not is_noise
    
    def filter_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection]
    ) -> List[Detection]:
        """Filter detections using CNN classifier.
        
        Args:
            frame: Original frame
            detections: List of detections to filter
            
        Returns:
            Filtered list of detections
        """
        if not detections:
            return []
        
        filtered = []
        
        # Process in batches
        for i in range(0, len(detections), self.batch_size):
            batch_dets = detections[i:i + self.batch_size]
            
            # Extract crops
            crops = []
            valid_dets = []
            
            for det in batch_dets:
                x1, y1, x2, y2 = [int(c) for c in det.bbox]
                
                # Validate bbox
                if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                    continue
                if x2 <= x1 or y2 <= y1:
                    continue
                
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                    continue
                
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crops.append(self.transform(crop_rgb))
                valid_dets.append(det)
            
            if not crops:
                continue
            
            # Batch inference
            batch = torch.stack(crops).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(batch)
                probabilities = torch.softmax(outputs, dim=1)
                bee_probs = probabilities[:, 1].cpu().numpy()
            
            # Filter based on bee probability
            for det, bee_prob in zip(valid_dets, bee_probs):
                if bee_prob >= self.bee_threshold:
                    # Update detection confidence with CNN score
                    det.metadata['cnn_score'] = float(bee_prob)
                    filtered.append(det)
        
        logger.debug(f"NoiseFilter: {len(detections)} → {len(filtered)} detections")
        return filtered
    
    def filter_blobs(
        self,
        frame: np.ndarray,
        blobs: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[int, int, int, int, float]]:
        """Filter blob list, removing clear noise.
        
        Args:
            frame: Full frame (BGR)
            blobs: List of (x, y, w, h) bounding boxes
            
        Returns:
            List of (x, y, w, h, bee_probability) for non-noise blobs
        """
        if not blobs:
            return []
        
        non_noise_blobs = []
        
        for i in range(0, len(blobs), self.batch_size):
            batch_blobs = blobs[i:i + self.batch_size]
            
            crops = []
            valid_indices = []
            
            for idx, (x, y, w, h) in enumerate(batch_blobs):
                crop = frame[y:y+h, x:x+w]
                
                if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                    continue
                
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crops.append(self.transform(crop_rgb))
                valid_indices.append(idx)
            
            if not crops:
                continue
            
            # Batch inference
            batch = torch.stack(crops).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(batch)
                probabilities = torch.softmax(outputs, dim=1)
                bee_probs = probabilities[:, 1].cpu().numpy()
            
            for idx, bee_prob in zip(valid_indices, bee_probs):
                x, y, w, h = batch_blobs[idx]
                if bee_prob >= self.bee_threshold:
                    non_noise_blobs.append((x, y, w, h, float(bee_prob)))
        
        return non_noise_blobs
    
    def configure(self, threshold: float) -> None:
        """Configure noise filter threshold.
        
        Args:
            threshold: New noise threshold (0-1)
        """
        self.noise_threshold = threshold
        self.bee_threshold = 1.0 - threshold
        logger.debug(f"NoiseFilter threshold set to {threshold}")
    
    def get_stats(self, blobs_in: int, blobs_out: int) -> str:
        """Get filtering statistics string."""
        if blobs_in == 0:
            return "No blobs detected"
        
        filtered = blobs_in - blobs_out
        filter_rate = (filtered / blobs_in) * 100
        
        return (f"Filtered {filtered}/{blobs_in} blobs ({filter_rate:.1f}% noise), "
                f"{blobs_out} passed to YOLO")


# Legacy wrapper for backwards compatibility
class NoiseFilter:
    """Wrapper class for backwards compatibility.
    
    Supports multiple classifier interfaces:
    - BeeNoiseFilter instances (full CNN)
    - Mock classifiers with predict() method (for testing)
    - Legacy classifiers with is_bee() method
    - Model path strings
    """
    
    def __init__(self, classifier, threshold: float = 0.7):
        """Initialize from existing classifier or model path.
        
        Args:
            classifier: Can be:
                - BeeNoiseFilter instance
                - Model path (str/Path)
                - Mock/legacy classifier with predict() or is_bee() method
            threshold: Confidence threshold (0-1)
        """
        self.classifier = classifier
        self.threshold = threshold
        
        # If it's a BeeNoiseFilter or model path, use full implementation
        if isinstance(classifier, BeeNoiseFilter):
            self._use_full_cnn = True
            self._cnn_filter = classifier
        elif isinstance(classifier, (str, Path)):
            self._use_full_cnn = True
            self._cnn_filter = BeeNoiseFilter(
                model_path=str(classifier),
                noise_threshold=1.0 - threshold
            )
        else:
            # Mock or legacy classifier
            self._use_full_cnn = False
            self._cnn_filter = None
        
        logger.info(f"NoiseFilter initialized: threshold={threshold}")
    
    def filter_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection]
    ) -> List[Detection]:
        """Filter detections using classifier.
        
        Args:
            frame: Original frame
            detections: List of detections to filter
            
        Returns:
            Filtered list of detections
        """
        if not detections:
            return []
        
        # Use full CNN implementation if available
        if self._use_full_cnn:
            return self._cnn_filter.filter_detections(frame, detections)
        
        # Legacy/mock classifier path
        filtered = []
        
        # Extract all crops first
        crops = []
        valid_detections = []
        
        for det in detections:
            x1, y1, x2, y2 = [int(c) for c in det.bbox]
            
            # Validate bbox
            if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            
            crops.append(crop)
            valid_detections.append(det)
        
        if not crops:
            return []
        
        # Get predictions from classifier
        if hasattr(self.classifier, 'is_bee'):
            # Legacy interface - process one at a time
            for crop, det in zip(crops, valid_detections):
                if self.classifier.is_bee(crop):
                    filtered.append(det)
        elif hasattr(self.classifier, 'predict'):
            # Mock/batch interface - process all at once
            probs = self.classifier.predict(crops)
            for prob, det in zip(probs, valid_detections):
                if prob >= self.threshold:
                    filtered.append(det)
        else:
            logger.warning(f"Classifier has no is_bee() or predict() method")
        
        logger.debug(f"NoiseFilter: {len(detections)} → {len(filtered)} detections")
        return filtered
    
    def configure(self, threshold: float) -> None:
        """Configure noise filter threshold.
        
        Args:
            threshold: New threshold (0-1)
        """
        self.threshold = threshold
        
        if self._use_full_cnn:
            # Update CNN filter's threshold (inverse)
            self._cnn_filter.configure(1.0 - threshold)
        
        logger.debug(f"NoiseFilter threshold set to {threshold}")