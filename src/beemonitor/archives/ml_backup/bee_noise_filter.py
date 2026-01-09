"""Noise filtering inference for bee classification.

Instead of detecting bees, this filters out noise with high confidence.
Uncertain cases are passed through to YOLO for final decision.
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
    """Fast noise filter for blob pre-filtering.
    
    Filters out clear noise cases, passes uncertain/bee cases to YOLO.
    More conservative approach - won't miss bees.
    """
    
    def __init__(
        self,
        model_path: str,
        noise_threshold: float = 0.9,
        image_size: int = 64,
        device: str = 'cuda',
        batch_size: int = 32
    ):
        """Initialize noise filter.
        
        Args:
            model_path: Path to trained model
            noise_threshold: Threshold for noise filtering (higher = more conservative)
                           - 0.9: Filter if noise_prob > 0.9 (bee_prob < 0.1) - Aggressive
                           - 0.8: Filter if noise_prob > 0.8 (bee_prob < 0.2) - Conservative
            image_size: Input size
            device: 'cuda' or 'cpu'
            batch_size: Batch size for inference
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.noise_threshold = noise_threshold
        self.bee_threshold = 1.0 - noise_threshold  # Inverse threshold
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
        logger.info(f"Strategy: Filter clear noise, pass uncertain/bees to YOLO")
    
    def is_noise(self, crop: np.ndarray) -> Tuple[bool, float]:
        """Check if crop is noise.
        
        Args:
            crop: Image crop (BGR)
            
        Returns:
            Tuple of (is_noise, bee_probability)
        """
        # Convert BGR to RGB
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        # Transform
        img_tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(img_tensor)
            probabilities = torch.softmax(output, dim=1)
            bee_prob = probabilities[0, 1].item()
        
        # Filter if bee probability is LOW (i.e., noise probability is HIGH)
        is_noise = bee_prob < self.bee_threshold
        
        return is_noise, bee_prob
    
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
        
        # Process in batches
        for i in range(0, len(blobs), self.batch_size):
            batch_blobs = blobs[i:i + self.batch_size]
            
            # Extract crops
            crops = []
            valid_indices = []
            
            for idx, (x, y, w, h) in enumerate(batch_blobs):
                # Extract crop
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
            
            # Filter based on bee probability
            for idx, bee_prob in zip(valid_indices, bee_probs):
                x, y, w, h = batch_blobs[idx]
                
                # Keep if bee_prob >= threshold (not clear noise)
                if bee_prob >= self.bee_threshold:
                    non_noise_blobs.append((x, y, w, h, float(bee_prob)))
        
        return non_noise_blobs
    
    def get_stats(self, blobs_in: int, blobs_out: int) -> str:
        """Get filtering statistics string.
        
        Args:
            blobs_in: Number of input blobs
            blobs_out: Number of output blobs (after filtering)
            
        Returns:
            Statistics string
        """
        if blobs_in == 0:
            return "No blobs detected"
        
        filtered = blobs_in - blobs_out
        filter_rate = (filtered / blobs_in) * 100
        
        return (f"Filtered {filtered}/{blobs_in} blobs ({filter_rate:.1f}% noise), "
                f"{blobs_out} passed to YOLO")


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test noise filter on single image')
    parser.add_argument('model_path', type=str, help='Path to model')
    parser.add_argument('image_path', type=str, help='Path to image')
    parser.add_argument('--noise-threshold', type=float, default=0.9,
                        help='Noise probability threshold (0.8-0.95)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Initialize filter
    noise_filter = BeeNoiseFilter(
        model_path=args.model_path,
        noise_threshold=args.noise_threshold,
        device=args.device
    )
    
    # Load image
    img = cv2.imread(args.image_path)
    if img is None:
        print(f"Failed to load image: {args.image_path}")
        return
    
    # Check if noise
    is_noise, bee_prob = noise_filter.is_noise(img)
    
    print("\n" + "="*60)
    print("NOISE FILTER RESULT")
    print("="*60)
    print(f"Image: {args.image_path}")
    print(f"Noise threshold: {args.noise_threshold}")
    print(f"Bee probability: {bee_prob:.4f}")
    print(f"Noise probability: {1-bee_prob:.4f}")
    print(f"\nDecision: {'FILTER (Noise)' if is_noise else 'PASS TO YOLO'}")
    print("="*60)


if __name__ == '__main__':
    main()