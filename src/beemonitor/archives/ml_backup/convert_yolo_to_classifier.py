"""Convert YOLO detection dataset to classification dataset.

Extracts bounding box crops from YOLO annotations and organizes them
into a classification-ready structure.

Dataset structure expected:
    yolo_dataset/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import logging
from tqdm import tqdm
import shutil
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YOLOToClassifierConverter:
    """Converts YOLO detection dataset to classification crops."""
    
    def __init__(
        self,
        yolo_dataset_path: Path,
        output_path: Path,
        class_names: Dict[int, str] = None,
        min_crop_size: int = 20,
        padding: int = 5
    ):
        """Initialize converter.
        
        Args:
            yolo_dataset_path: Path to YOLO dataset root
            output_path: Output path for classification dataset
            class_names: Mapping of class IDs to names (e.g., {0: 'bee'})
            min_crop_size: Minimum crop dimension
            padding: Padding around bbox in pixels
        """
        self.yolo_path = Path(yolo_dataset_path)
        self.output_path = Path(output_path)
        self.class_names = class_names or {0: 'bee'}
        self.min_crop_size = min_crop_size
        self.padding = padding
        
        # Stats
        self.stats = {
            'total_images': 0,
            'total_crops': 0,
            'crops_per_class': {},
            'skipped_too_small': 0,
            'failed_crops': 0
        }
    
    def convert(self, split: str = 'train') -> None:
        """Convert YOLO dataset to classification crops.
        
        Args:
            split: Dataset split ('train', 'valid', 'test')
        """
        # Path structure: dataset/train/images/ and dataset/train/labels/
        split_path = self.yolo_path / split
        images_path = split_path / 'images'
        labels_path = split_path / 'labels'
        
        if not images_path.exists():
            logger.error(f"Images path not found: {images_path}")
            return
        
        if not labels_path.exists():
            logger.error(f"Labels path not found: {labels_path}")
            return
        
        # Create output structure
        for class_id, class_name in self.class_names.items():
            class_dir = self.output_path / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            self.stats['crops_per_class'][class_name] = 0
        
        # Process all images
        image_files = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png'))
        logger.info(f"Found {len(image_files)} images in {split} split")
        
        for img_path in tqdm(image_files, desc=f"Converting {split}"):
            self._process_image(img_path, labels_path, split)
        
        self._print_stats(split)
    
    def _process_image(self, img_path: Path, labels_path: Path, split: str) -> None:
        """Process single image and extract crops."""
        self.stats['total_images'] += 1
        
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"Could not read image: {img_path}")
            return
        
        h, w = img.shape[:2]
        
        # Read annotation
        label_path = labels_path / f"{img_path.stem}.txt"
        if not label_path.exists():
            # No annotations for this image (empty frame)
            return
        
        # Parse YOLO annotations
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for idx, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            class_id = int(parts[0])
            x_center, y_center, bbox_w, bbox_h = map(float, parts[1:5])
            
            # Convert YOLO format to pixel coordinates
            x_center_px = x_center * w
            y_center_px = y_center * h
            bbox_w_px = bbox_w * w
            bbox_h_px = bbox_h * h
            
            # Calculate crop coordinates with padding
            x1 = int(max(0, x_center_px - bbox_w_px/2 - self.padding))
            y1 = int(max(0, y_center_px - bbox_h_px/2 - self.padding))
            x2 = int(min(w, x_center_px + bbox_w_px/2 + self.padding))
            y2 = int(min(h, y_center_px + bbox_h_px/2 + self.padding))
            
            # Check minimum size
            crop_w = x2 - x1
            crop_h = y2 - y1
            
            if crop_w < self.min_crop_size or crop_h < self.min_crop_size:
                self.stats['skipped_too_small'] += 1
                continue
            
            # Extract crop
            crop = img[y1:y2, x1:x2]
            
            if crop.size == 0:
                self.stats['failed_crops'] += 1
                continue
            
            # Save crop
            class_name = self.class_names.get(class_id, f'class_{class_id}')
            output_dir = self.output_path / split / class_name
            
            # Create unique filename
            crop_filename = f"{img_path.stem}_crop{idx:03d}.jpg"
            crop_path = output_dir / crop_filename
            
            cv2.imwrite(str(crop_path), crop)
            
            self.stats['total_crops'] += 1
            self.stats['crops_per_class'][class_name] += 1
    
    def _print_stats(self, split: str) -> None:
        """Print conversion statistics."""
        logger.info("="*60)
        logger.info(f"Conversion complete for {split} split!")
        logger.info(f"  Images processed: {self.stats['total_images']}")
        logger.info(f"  Total crops: {self.stats['total_crops']}")
        logger.info(f"  Crops per class:")
        for class_name, count in self.stats['crops_per_class'].items():
            logger.info(f"    {class_name}: {count}")
        logger.info(f"  Skipped (too small): {self.stats['skipped_too_small']}")
        logger.info(f"  Failed crops: {self.stats['failed_crops']}")
        logger.info(f"  Output: {self.output_path / split}")
        logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(description='Convert YOLO detection to classification')
    parser.add_argument('yolo_path', type=str, help='Path to YOLO dataset root')
    parser.add_argument('output_path', type=str, help='Output path for classification dataset')
    parser.add_argument('--splits', nargs='+', default=['train', 'valid', 'test'],
                        help='Dataset splits to convert')
    parser.add_argument('--padding', type=int, default=5,
                        help='Padding around bboxes')
    parser.add_argument('--min-size', type=int, default=20,
                        help='Minimum crop size')
    
    args = parser.parse_args()
    
    converter = YOLOToClassifierConverter(
        yolo_dataset_path=args.yolo_path,
        output_path=args.output_path,
        padding=args.padding,
        min_crop_size=args.min_size
    )
    
    for split in args.splits:
        converter.convert(split)


if __name__ == '__main__':
    main()