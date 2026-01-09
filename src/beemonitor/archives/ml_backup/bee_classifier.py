"""Inference script for bee classifier.

Fast classification of image crops to filter out noise before YOLO.
"""

import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


class SimpleBeeClassifier(nn.Module):
    """Enhanced CNN for bee vs noise classification."""
    
    def __init__(self, num_classes: int = 2):
        super(SimpleBeeClassifier, self).__init__()
        
        self.features = nn.Sequential(
            # Conv block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Conv block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Conv block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Conv block 4
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


class BeeClassifierInference:
    """Fast bee vs noise classifier for filtering blobs."""
    
    def __init__(
        self,
        model_path: str,
        image_size: int = 64,
        threshold: float = 0.5,
        device: str = 'cuda'
    ):
        """Initialize classifier.
        
        Args:
            model_path: Path to trained model checkpoint
            image_size: Input image size (must match training)
            threshold: Classification threshold
            device: Device ('cuda' or 'cpu')
        """
        self.image_size = image_size
        self.threshold = threshold
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = SimpleBeeClassifier(num_classes=2).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        logger.info(f"Loaded classifier from {model_path}")
        logger.info(f"Using device: {self.device}")
        
        # Transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def classify(self, image: np.ndarray) -> Tuple[bool, float]:
        """Classify a single image crop.
        
        Args:
            image: Image crop (BGR format)
            
        Returns:
            Tuple of (is_bee, confidence)
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        img_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get bee probability (class 1)
            bee_prob = probabilities[0][1].item()
        
        is_bee = bee_prob > self.threshold
        return is_bee, bee_prob
    
    def classify_batch(self, images: List[np.ndarray]) -> List[Tuple[bool, float]]:
        """Classify a batch of images.
        
        Args:
            images: List of image crops (BGR format)
            
        Returns:
            List of (is_bee, confidence) tuples
        """
        if not images:
            return []
        
        # Preprocess batch
        img_tensors = []
        for image in images:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_tensor = self.transform(image_rgb)
            img_tensors.append(img_tensor)
        
        # Stack into batch
        batch = torch.stack(img_tensors).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(batch)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get bee probabilities (class 1)
            bee_probs = probabilities[:, 1].cpu().numpy()
        
        # Return results
        results = []
        for prob in bee_probs:
            is_bee = prob > self.threshold
            results.append((is_bee, float(prob)))
        
        return results
    
    def filter_blobs(
        self,
        frame: np.ndarray,
        blobs: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[int, int, int, int, float]]:
        """Filter blob candidates using classifier.
        
        Args:
            frame: Full video frame
            blobs: List of (x, y, w, h) bounding boxes
            
        Returns:
            List of (x, y, w, h, confidence) for blobs classified as bees
        """
        if not blobs:
            return []
        
        # Extract crops
        crops = []
        for x, y, w, h in blobs:
            crop = frame[y:y+h, x:x+w]
            if crop.size > 0:
                crops.append(crop)
            else:
                crops.append(None)
        
        # Classify
        results = []
        for i, (x, y, w, h) in enumerate(blobs):
            if crops[i] is None:
                continue
            
            is_bee, confidence = self.classify(crops[i])
            
            if is_bee:
                results.append((x, y, w, h, confidence))
        
        return results


if __name__ == '__main__':
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Test bee classifier')
    parser.add_argument('model_path', type=str, help='Path to model checkpoint')
    parser.add_argument('image_path', type=str, help='Path to test image')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold')
    
    args = parser.parse_args()
    
    # Load classifier
    classifier = BeeClassifierInference(
        model_path=args.model_path,
        threshold=args.threshold
    )
    
    # Load test image
    image = cv2.imread(args.image_path)
    
    # Classify
    is_bee, confidence = classifier.classify(image)
    
    print(f"Is bee: {is_bee}")
    print(f"Confidence: {confidence:.4f}")