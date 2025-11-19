# """Auto-generate classification training data from videos.

# Uses foreground masks to detect motion blobs, then validates with YOLO
# on full frame to create positive (bee) and negative (noise) samples.
# """

# import cv2
# import numpy as np
# from pathlib import Path
# from typing import List, Tuple, Optional, Dict
# import logging
# from tqdm import tqdm
# import argparse
# from ultralytics import YOLO
# from dataclasses import dataclass

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# @dataclass
# class BlobSample:
#     """A blob sample with metadata."""
#     crop: np.ndarray
#     is_bee: bool
#     confidence: float
#     area: int
#     frame_number: int
#     video_name: str
#     bbox: Tuple[int, int, int, int]  # Original blob bbox in frame


# class ClassifierDataGenerator:
#     """Generates classification training data from videos using FG + YOLO."""
    
#     def __init__(
#         self,
#         yolo_model_path: str,
#         output_path: Path,
#         min_blob_area: int = 100,
#         max_blob_area: int = 5000,
#         yolo_confidence: float = 0.5,
#         crop_padding: int = 10,
#         sample_rate: int = 5,
#         max_samples_per_video: int = 1000,
#         iou_threshold: float = 0.3
#     ):
#         """Initialize generator.
        
#         Args:
#             yolo_model_path: Path to YOLO model for verification
#             output_path: Output directory
#             min_blob_area: Minimum blob area (pixels)
#             max_blob_area: Maximum blob area
#             yolo_confidence: YOLO confidence threshold
#             crop_padding: Padding around blobs
#             sample_rate: Process every Nth frame
#             max_samples_per_video: Max samples per video
#             iou_threshold: IoU threshold for blob-detection matching
#         """
#         self.yolo_model = YOLO(yolo_model_path)
#         self.output_path = Path(output_path)
#         self.min_blob_area = min_blob_area
#         self.max_blob_area = max_blob_area
#         self.yolo_confidence = yolo_confidence
#         self.crop_padding = crop_padding
#         self.sample_rate = sample_rate
#         self.max_samples_per_video = max_samples_per_video
#         self.iou_threshold = iou_threshold
        
#         # Create output dirs
#         (self.output_path / 'bee').mkdir(parents=True, exist_ok=True)
#         (self.output_path / 'noise').mkdir(parents=True, exist_ok=True)
        
#         # Stats
#         self.stats = {
#             'videos_processed': 0,
#             'frames_analyzed': 0,
#             'blobs_detected': 0,
#             'bee_samples': 0,
#             'noise_samples': 0,
#             'yolo_confirms': 0,
#             'yolo_rejects': 0
#         }
    
#     def _initialize_background(self, cap, max_init_frames: int = 100) -> cv2.BackgroundSubtractor:
#         """Initialize background subtractor with first N frames.
        
#         Args:
#             cap: Video capture object
#             max_init_frames: Maximum frames to use for initialization
            
#         Returns:
#             Initialized background subtractor
#         """
#         bg_subtractor = cv2.createBackgroundSubtractorMOG2(
#             history=500,
#             varThreshold=16,
#             detectShadows=False
#         )
        
#         logger.info("Initializing background model...")
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         init_frames = min(max_init_frames, total_frames // 10)
        
#         cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#         for _ in range(init_frames):
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             bg_subtractor.apply(frame, learningRate=0.01)
        
#         # Reset to start
#         cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
#         return bg_subtractor
    
#     def _compute_iou(self, box1: Tuple[int, int, int, int], 
#                      box2: Tuple[int, int, int, int]) -> float:
#         """Compute IoU between two boxes.
        
#         Args:
#             box1: (x, y, w, h)
#             box2: (x, y, w, h)
            
#         Returns:
#             IoU value
#         """
#         x1, y1, w1, h1 = box1
#         x2, y2, w2, h2 = box2
        
#         # Convert to corners
#         x1_max = x1 + w1
#         y1_max = y1 + h1
#         x2_max = x2 + w2
#         y2_max = y2 + h2
        
#         # Intersection
#         xi1 = max(x1, x2)
#         yi1 = max(y1, y2)
#         xi2 = min(x1_max, x2_max)
#         yi2 = min(y1_max, y2_max)
        
#         inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
#         # Union
#         box1_area = w1 * h1
#         box2_area = w2 * h2
#         union_area = box1_area + box2_area - inter_area
        
#         return inter_area / union_area if union_area > 0 else 0.0
    
#     def process_video(self, video_path: Path) -> None:
#         """Process a single video to extract samples.
        
#         Args:
#             video_path: Path to video file
#         """
#         cap = cv2.VideoCapture(str(video_path))
#         if not cap.isOpened():
#             logger.error(f"Cannot open video: {video_path}")
#             return
        
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         video_name = video_path.stem
        
#         logger.info(f"Processing video: {video_name} ({total_frames} frames)")
        
#         # Initialize background subtractor
#         bg_subtractor = self._initialize_background(cap)
        
#         samples_this_video = 0
#         frame_number = 0
        
#         pbar = tqdm(total=total_frames, desc=video_name)
        
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             frame_number += 1
#             pbar.update(1)
            
#             # Sample rate
#             if frame_number % self.sample_rate != 0:
#                 continue
            
#             # Check max samples
#             if samples_this_video >= self.max_samples_per_video:
#                 logger.info(f"Reached max samples for {video_name}")
#                 break
            
#             self.stats['frames_analyzed'] += 1
            
#             # Get foreground mask
#             fg_mask = bg_subtractor.apply(frame, learningRate=0)
            
#             # Clean mask
#             kernel = np.ones((3, 3), np.uint8)
#             fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
#             fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            
#             # Find contours (blobs)
#             contours, _ = cv2.findContours(
#                 fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
#             )
            
#             # Collect valid blobs
#             blobs = []
#             for contour in contours:
#                 area = cv2.contourArea(contour)
                
#                 if area < self.min_blob_area or area > self.max_blob_area:
#                     continue
                
#                 x, y, w, h = cv2.boundingRect(contour)
#                 blobs.append((x, y, w, h, area))
#                 self.stats['blobs_detected'] += 1
            
#             if not blobs:
#                 continue
            
#             # Run YOLO on full frame (not crops!)
#             yolo_detections = self._run_yolo_on_frame(frame)
            
#             # Match blobs with YOLO detections
#             for x, y, w, h, area in blobs:
#                 # Check max samples
#                 if samples_this_video >= self.max_samples_per_video:
#                     break
                
#                 # Extract crop
#                 x1 = max(0, x - self.crop_padding)
#                 y1 = max(0, y - self.crop_padding)
#                 x2 = min(frame.shape[1], x + w + self.crop_padding)
#                 y2 = min(frame.shape[0], y + h + self.crop_padding)
                
#                 crop = frame[y1:y2, x1:x2]
                
#                 if crop.size == 0:
#                     continue
                
#                 # Check if this blob overlaps with any YOLO detection
#                 is_bee, confidence = self._check_blob_overlap(
#                     (x, y, w, h), yolo_detections
#                 )
                
#                 # Save sample
#                 sample = BlobSample(
#                     crop=crop,
#                     is_bee=is_bee,
#                     confidence=confidence,
#                     area=int(area),
#                     frame_number=frame_number,
#                     video_name=video_name,
#                     bbox=(x, y, w, h)
#                 )
                
#                 self._save_sample(sample)
#                 samples_this_video += 1
        
#         pbar.close()
#         cap.release()
        
#         self.stats['videos_processed'] += 1
#         logger.info(f"Collected {samples_this_video} samples from {video_name}")
    
#     def _run_yolo_on_frame(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
#         """Run YOLO on full frame and return detections.
        
#         Args:
#             frame: Full video frame
            
#         Returns:
#             List of (x, y, w, h, confidence) detections
#         """
#         results = self.yolo_model(frame, verbose=False)
        
#         detections = []
#         if len(results) > 0 and len(results[0].boxes) > 0:
#             boxes = results[0].boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
#             confidences = results[0].boxes.conf.cpu().numpy()
            
#             for box, conf in zip(boxes, confidences):
#                 if conf >= self.yolo_confidence:
#                     x1, y1, x2, y2 = box
#                     x = int(x1)
#                     y = int(y1)
#                     w = int(x2 - x1)
#                     h = int(y2 - y1)
#                     detections.append((x, y, w, h, float(conf)))
        
#         return detections
    
#     def _check_blob_overlap(
#         self,
#         blob_bbox: Tuple[int, int, int, int],
#         yolo_detections: List[Tuple[int, int, int, int, float]]
#     ) -> Tuple[bool, float]:
#         """Check if blob overlaps with any YOLO detection.
        
#         Args:
#             blob_bbox: (x, y, w, h) of the blob
#             yolo_detections: List of YOLO detections
            
#         Returns:
#             Tuple of (is_bee, max_confidence)
#         """
#         if not yolo_detections:
#             self.stats['yolo_rejects'] += 1
#             return False, 0.0
        
#         # Check IoU with each detection
#         max_iou = 0.0
#         max_conf = 0.0
        
#         for det_x, det_y, det_w, det_h, conf in yolo_detections:
#             iou = self._compute_iou(blob_bbox, (det_x, det_y, det_w, det_h))
            
#             if iou > max_iou:
#                 max_iou = iou
#                 max_conf = conf
        
#         # Consider it a bee if IoU exceeds threshold
#         if max_iou >= self.iou_threshold:
#             self.stats['yolo_confirms'] += 1
#             return True, max_conf
#         else:
#             self.stats['yolo_rejects'] += 1
#             return False, 0.0
    
#     def _save_sample(self, sample: BlobSample) -> None:
#         """Save sample to appropriate directory.
        
#         Args:
#             sample: BlobSample to save
#         """
#         # Determine directory
#         if sample.is_bee:
#             output_dir = self.output_path / 'bee'
#             self.stats['bee_samples'] += 1
#         else:
#             output_dir = self.output_path / 'noise'
#             self.stats['noise_samples'] += 1
        
#         # Create filename
#         filename = (
#             f"{sample.video_name}_"
#             f"f{sample.frame_number:06d}_"
#             f"a{sample.area}_"
#             f"c{sample.confidence:.2f}.jpg"
#         )
        
#         output_path = output_dir / filename
#         cv2.imwrite(str(output_path), sample.crop)
    
#     def process_directory(self, video_dir: Path, pattern: str = "*.mp4") -> None:
#         """Process all videos in a directory.
        
#         Args:
#             video_dir: Directory containing videos
#             pattern: File pattern to match
#         """
#         video_files = list(Path(video_dir).glob(pattern))
#         logger.info(f"Found {len(video_files)} videos to process")
        
#         for video_path in video_files:
#             self.process_video(video_path)
        
#         self._print_stats()
    
#     def _print_stats(self) -> None:
#         """Print generation statistics."""
#         logger.info("="*60)
#         logger.info("Data generation complete!")
#         logger.info(f"  Videos processed: {self.stats['videos_processed']}")
#         logger.info(f"  Frames analyzed: {self.stats['frames_analyzed']}")
#         logger.info(f"  Blobs detected: {self.stats['blobs_detected']}")
#         logger.info(f"  YOLO confirmations: {self.stats['yolo_confirms']}")
#         logger.info(f"  YOLO rejections: {self.stats['yolo_rejects']}")
#         logger.info(f"  Bee samples: {self.stats['bee_samples']}")
#         logger.info(f"  Noise samples: {self.stats['noise_samples']}")
#         logger.info(f"  Output directory: {self.output_path}")
#         logger.info("="*60)


# def main():
#     parser = argparse.ArgumentParser(
#         description='Generate classifier training data from videos'
#     )
#     parser.add_argument('--video-dir', type=str, help='Directory containing videos',
#                         default="/Users/edwardamoah/Documents/GitHub/BeeVision_1/monitoring_data/sample_data/mendels/2024-04-30",
#                         required=False)
#     parser.add_argument('--yolo-model', type=str, help='Path to YOLO model',
#                         default="/Users/edwardamoah/Documents/GitHub/BeeMonitor/models/bee_tracking_back_up_Full_Mode.pt",
#                         required=False)
#     parser.add_argument('--output-dir', type=str, help='Output directory',
#                         default="/Users/edwardamoah/Documents/GitHub/BeeMonitor/output/classifier_training/test_2",
#                         required=False)
#     parser.add_argument('--min-area', type=int, default=100,
#                         help='Minimum blob area', required=False)
#     parser.add_argument('--max-area', type=int, default=1500,
#                         help='Maximum blob area', required=False)
#     parser.add_argument('--yolo-conf', type=float, default=0.5,
#                         help='YOLO confidence threshold', required=False)
#     parser.add_argument('--iou-threshold', type=float, default=0.3,
#                         help='IoU threshold for blob-detection matching', required=False)
#     parser.add_argument('--padding', type=int, default=10,
#                         help='Crop padding', required=False)
#     parser.add_argument('--sample-rate', type=int, default=2,
#                         help='Process every Nth frame', required=False)
#     parser.add_argument('--max-samples', type=int, default=100000000,
#                         help='Max samples per video', required=False)
#     parser.add_argument('--pattern', type=str, default='*.mp4',
#                         help='Video file pattern', required=False)
    
#     args = parser.parse_args()
    
#     generator = ClassifierDataGenerator(
#         yolo_model_path=args.yolo_model,
#         output_path=Path(args.output_dir),
#         min_blob_area=args.min_area,
#         max_blob_area=args.max_area,
#         yolo_confidence=args.yolo_conf,
#         iou_threshold=args.iou_threshold,
#         crop_padding=args.padding,
#         sample_rate=args.sample_rate,
#         max_samples_per_video=args.max_samples
#     )
    
#     generator.process_directory(args.video_dir, args.pattern)


# if __name__ == '__main__':
#     main()













"""Auto-generate classification training data from videos.

Uses foreground masks to detect motion blobs, then validates with YOLO
on full frame to create positive (bee) and negative (noise) samples.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging
from tqdm import tqdm
import argparse
from ultralytics import YOLO
from dataclasses import dataclass
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BlobSample:
    """A blob sample with metadata."""
    crop: np.ndarray
    is_bee: bool
    confidence: float
    area: int
    frame_number: int
    video_name: str
    bbox: Tuple[int, int, int, int]  # Original blob bbox in frame


class ClassifierDataGenerator:
    """Generates classification training data from videos using FG + YOLO."""
    
    def __init__(
        self,
        yolo_model_path: str,
        output_path: Path,
        min_blob_area: int = 100,
        max_blob_area: int = 5000,
        yolo_confidence: float = 0.5,
        crop_padding: int = 10,
        sample_rate: int = 5,
        max_bee_samples_per_video: int = 1000,
        max_noise_samples_per_video: int = 1000,
        iou_threshold: float = 0.3
    ):
        """Initialize generator.
        
        Args:
            yolo_model_path: Path to YOLO model for verification
            output_path: Output directory
            min_blob_area: Minimum blob area (pixels)
            max_blob_area: Maximum blob area
            yolo_confidence: YOLO confidence threshold
            crop_padding: Padding around blobs
            sample_rate: Process every Nth frame
            max_bee_samples_per_video: Max bee samples per video
            max_noise_samples_per_video: Max noise samples per video
            iou_threshold: IoU threshold for blob-detection matching
        """
        self.yolo_model = YOLO(yolo_model_path)
        self.output_path = Path(output_path)
        self.min_blob_area = min_blob_area
        self.max_blob_area = max_blob_area
        self.yolo_confidence = yolo_confidence
        self.crop_padding = crop_padding
        self.sample_rate = sample_rate
        self.max_bee_samples_per_video = max_bee_samples_per_video
        self.max_noise_samples_per_video = max_noise_samples_per_video
        self.iou_threshold = iou_threshold
        
        # Create output dirs
        (self.output_path / 'bee').mkdir(parents=True, exist_ok=True)
        (self.output_path / 'noise').mkdir(parents=True, exist_ok=True)
        
        # Stats
        self.stats = {
            'videos_processed': 0,
            'frames_analyzed': 0,
            'blobs_detected': 0,
            'bee_samples': 0,
            'noise_samples': 0,
            'yolo_confirms': 0,
            'yolo_rejects': 0,
            'bee_samples_seen': 0,
            'noise_samples_seen': 0
        }
    
    def _initialize_background(self, cap, max_init_frames: int = 100) -> cv2.BackgroundSubtractor:
        """Initialize background subtractor with first N frames.
        
        Args:
            cap: Video capture object
            max_init_frames: Maximum frames to use for initialization
            
        Returns:
            Initialized background subtractor
        """
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=False
        )
        
        logger.info("Initializing background model...")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        init_frames = min(max_init_frames, total_frames // 10)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for _ in range(init_frames):
            ret, frame = cap.read()
            if not ret:
                break
            bg_subtractor.apply(frame, learningRate=0.01)
        
        # Reset to start
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        return bg_subtractor
    
    def _compute_iou(self, box1: Tuple[int, int, int, int], 
                     box2: Tuple[int, int, int, int]) -> float:
        """Compute IoU between two boxes.
        
        Args:
            box1: (x, y, w, h)
            box2: (x, y, w, h)
            
        Returns:
            IoU value
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Convert to corners
        x1_max = x1 + w1
        y1_max = y1 + h1
        x2_max = x2 + w2
        y2_max = y2 + h2
        
        # Intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1_max, x2_max)
        yi2 = min(y1_max, y2_max)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Union
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def process_video(self, video_path: Path) -> None:
        """Process a single video to extract samples using reservoir sampling.
        
        Uses reservoir sampling to ensure uniform distribution across video duration.
        
        Args:
            video_path: Path to video file
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_name = video_path.stem
        
        logger.info(f"Processing video: {video_name} ({total_frames} frames)")
        
        # Initialize background subtractor
        bg_subtractor = self._initialize_background(cap)
        
        # Reservoir sampling: keep samples in memory
        bee_reservoir: List[BlobSample] = []
        noise_reservoir: List[BlobSample] = []
        
        # Track total samples seen per class
        bee_count = 0
        noise_count = 0
        
        frame_number = 0
        
        pbar = tqdm(total=total_frames, desc=video_name)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_number += 1
            pbar.update(1)
            
            # Sample rate
            if frame_number % self.sample_rate != 0:
                continue
            
            self.stats['frames_analyzed'] += 1
            
            # Get foreground mask
            fg_mask = bg_subtractor.apply(frame, learningRate=0)
            
            # Clean mask
            kernel = np.ones((3, 3), np.uint8)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours (blobs)
            contours, _ = cv2.findContours(
                fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Collect valid blobs
            blobs = []
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if area < self.min_blob_area or area > self.max_blob_area:
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                blobs.append((x, y, w, h, area))
                self.stats['blobs_detected'] += 1
            
            if not blobs:
                continue
            
            # Run YOLO on full frame (not crops!)
            yolo_detections = self._run_yolo_on_frame(frame)
            
            # Match blobs with YOLO detections
            for x, y, w, h, area in blobs:
                # Extract crop
                x1 = max(0, x - self.crop_padding)
                y1 = max(0, y - self.crop_padding)
                x2 = min(frame.shape[1], x + w + self.crop_padding)
                y2 = min(frame.shape[0], y + h + self.crop_padding)
                
                crop = frame[y1:y2, x1:x2]
                
                if crop.size == 0:
                    continue
                
                # Check if this blob overlaps with any YOLO detection
                is_bee, confidence = self._check_blob_overlap(
                    (x, y, w, h), yolo_detections
                )
                
                # Create sample
                sample = BlobSample(
                    crop=crop.copy(),  # Important: copy to avoid reference issues
                    is_bee=is_bee,
                    confidence=confidence,
                    area=int(area),
                    frame_number=frame_number,
                    video_name=video_name,
                    bbox=(x, y, w, h)
                )
                
                # Reservoir sampling based on class
                if is_bee:
                    bee_count += 1
                    self.stats['bee_samples_seen'] += 1
                    
                    if len(bee_reservoir) < self.max_bee_samples_per_video:
                        # Reservoir not full, add sample
                        bee_reservoir.append(sample)
                    else:
                        # Reservoir full, randomly replace with probability k/n
                        # where k = reservoir size, n = total samples seen
                        j = random.randint(0, bee_count - 1)
                        if j < self.max_bee_samples_per_video:
                            bee_reservoir[j] = sample
                else:
                    noise_count += 1
                    self.stats['noise_samples_seen'] += 1
                    
                    if len(noise_reservoir) < self.max_noise_samples_per_video:
                        # Reservoir not full, add sample
                        noise_reservoir.append(sample)
                    else:
                        # Reservoir full, randomly replace
                        j = random.randint(0, noise_count - 1)
                        if j < self.max_noise_samples_per_video:
                            noise_reservoir[j] = sample
        
        pbar.close()
        cap.release()
        
        # Save all samples from reservoirs
        logger.info(f"Saving samples from {video_name}...")
        logger.info(f"  Bee samples: {len(bee_reservoir)} (from {bee_count} total)")
        logger.info(f"  Noise samples: {len(noise_reservoir)} (from {noise_count} total)")
        
        for sample in bee_reservoir:
            self._save_sample(sample)
        
        for sample in noise_reservoir:
            self._save_sample(sample)
        
        self.stats['videos_processed'] += 1
        logger.info(f"Completed processing {video_name}")
    
    def _run_yolo_on_frame(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Run YOLO on full frame and return detections.
        
        Args:
            frame: Full video frame
            
        Returns:
            List of (x, y, w, h, confidence) detections
        """
        results = self.yolo_model(frame, verbose=False)
        
        detections = []
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confidences = results[0].boxes.conf.cpu().numpy()
            
            for box, conf in zip(boxes, confidences):
                if conf >= self.yolo_confidence:
                    x1, y1, x2, y2 = box
                    x = int(x1)
                    y = int(y1)
                    w = int(x2 - x1)
                    h = int(y2 - y1)
                    detections.append((x, y, w, h, float(conf)))
        
        return detections
    
    def _check_blob_overlap(
        self,
        blob_bbox: Tuple[int, int, int, int],
        yolo_detections: List[Tuple[int, int, int, int, float]]
    ) -> Tuple[bool, float]:
        """Check if blob overlaps with any YOLO detection.
        
        Args:
            blob_bbox: (x, y, w, h) of the blob
            yolo_detections: List of YOLO detections
            
        Returns:
            Tuple of (is_bee, max_confidence)
        """
        if not yolo_detections:
            self.stats['yolo_rejects'] += 1
            return False, 0.0
        
        # Check IoU with each detection
        max_iou = 0.0
        max_conf = 0.0
        
        for det_x, det_y, det_w, det_h, conf in yolo_detections:
            iou = self._compute_iou(blob_bbox, (det_x, det_y, det_w, det_h))
            
            if iou > max_iou:
                max_iou = iou
                max_conf = conf
        
        # Consider it a bee if IoU exceeds threshold
        if max_iou >= self.iou_threshold:
            self.stats['yolo_confirms'] += 1
            return True, max_conf
        else:
            self.stats['yolo_rejects'] += 1
            return False, 0.0
    
    def _save_sample(self, sample: BlobSample) -> None:
        """Save sample to appropriate directory.
        
        Args:
            sample: BlobSample to save
        """
        # Determine directory
        if sample.is_bee:
            output_dir = self.output_path / 'bee'
            self.stats['bee_samples'] += 1
        else:
            output_dir = self.output_path / 'noise'
            self.stats['noise_samples'] += 1
        
        # Create filename
        filename = (
            f"{sample.video_name}_"
            f"f{sample.frame_number:06d}_"
            f"a{sample.area}_"
            f"c{sample.confidence:.2f}.jpg"
        )
        
        output_path = output_dir / filename
        cv2.imwrite(str(output_path), sample.crop)
    
    def process_directory(self, video_dir: Path, pattern: str = "*.mp4") -> None:
        """Process all videos in a directory.
        
        Args:
            video_dir: Directory containing videos
            pattern: File pattern to match
        """
        video_files = list(Path(video_dir).glob(pattern))
        logger.info(f"Found {len(video_files)} videos to process")
        
        for video_path in video_files:
            self.process_video(video_path)
        
        self._print_stats()
    
    def _print_stats(self) -> None:
        """Print generation statistics."""
        logger.info("="*60)
        logger.info("Data generation complete!")
        logger.info(f"  Videos processed: {self.stats['videos_processed']}")
        logger.info(f"  Frames analyzed: {self.stats['frames_analyzed']}")
        logger.info(f"  Blobs detected: {self.stats['blobs_detected']}")
        logger.info(f"  YOLO confirmations: {self.stats['yolo_confirms']}")
        logger.info(f"  YOLO rejections: {self.stats['yolo_rejects']}")
        logger.info(f"\nReservoir Sampling Results:")
        logger.info(f"  Bee samples seen: {self.stats['bee_samples_seen']}")
        logger.info(f"  Bee samples saved: {self.stats['bee_samples']}")
        if self.stats['bee_samples_seen'] > 0:
            logger.info(f"  Bee sampling rate: {self.stats['bee_samples'] / self.stats['bee_samples_seen'] * 100:.1f}%")
        logger.info(f"  Noise samples seen: {self.stats['noise_samples_seen']}")
        logger.info(f"  Noise samples saved: {self.stats['noise_samples']}")
        if self.stats['noise_samples_seen'] > 0:
            logger.info(f"  Noise sampling rate: {self.stats['noise_samples'] / self.stats['noise_samples_seen'] * 100:.1f}%")
        logger.info(f"\n  Output directory: {self.output_path}")
        logger.info("="*60)


# def main():
#     parser = argparse.ArgumentParser(
#         description='Generate classifier training data from videos with reservoir sampling'
#     )
#     parser.add_argument('video_dir', type=str, help='Directory containing videos')
#     parser.add_argument('yolo_model', type=str, help='Path to YOLO model')
#     parser.add_argument('output_dir', type=str, help='Output directory')
#     parser.add_argument('--min-area', type=int, default=100,
#                         help='Minimum blob area')
#     parser.add_argument('--max-area', type=int, default=5000,
#                         help='Maximum blob area')
#     parser.add_argument('--yolo-conf', type=float, default=0.5,
#                         help='YOLO confidence threshold')
#     parser.add_argument('--iou-threshold', type=float, default=0.3,
#                         help='IoU threshold for blob-detection matching')
#     parser.add_argument('--padding', type=int, default=10,
#                         help='Crop padding')
#     parser.add_argument('--sample-rate', type=int, default=5,
#                         help='Process every Nth frame')
#     parser.add_argument('--max-bee-samples', type=int, default=1000,
#                         help='Max bee samples per video (reservoir sampling)')
#     parser.add_argument('--max-noise-samples', type=int, default=1000,
#                         help='Max noise samples per video (reservoir sampling)')
#     parser.add_argument('--pattern', type=str, default='*.mp4',
#                         help='Video file pattern')
    
#     args = parser.parse_args()
    
#     generator = ClassifierDataGenerator(
#         yolo_model_path=args.yolo_model,
#         output_path=Path(args.output_dir),
#         min_blob_area=args.min_area,
#         max_blob_area=args.max_area,
#         yolo_confidence=args.yolo_conf,
#         iou_threshold=args.iou_threshold,
#         crop_padding=args.padding,
#         sample_rate=args.sample_rate,
#         max_bee_samples_per_video=args.max_bee_samples,
#         max_noise_samples_per_video=args.max_noise_samples
#     )
    
#     generator.process_directory(args.video_dir, args.pattern)



def main():
    parser = argparse.ArgumentParser(
        description='Generate classifier training data from videos'
    )
    parser.add_argument('--video_dir', type=str, help='Directory containing videos',
                        default="/Users/edwardamoah/Documents/GitHub/BeeVision_1/monitoring_data/sample_data/mendels/2024-04-30",
                        required=False)
    parser.add_argument('--yolo_model', type=str, help='Path to YOLO model',
                        default="/Users/edwardamoah/Documents/GitHub/BeeMonitor/models/bee_tracking_back_up_Full_Mode.pt",
                        required=False)
    parser.add_argument('--output-dir', type=str, help='Output directory',
                        default="/Users/edwardamoah/Documents/GitHub/BeeMonitor/output/classifier_training/test_2",
                        required=False)
    parser.add_argument('--min-area', type=int, default=100,
                        help='Minimum blob area', required=False)
    parser.add_argument('--max-area', type=int, default=1500,
                        help='Maximum blob area', required=False)
    parser.add_argument('--yolo-conf', type=float, default=0.5,
                        help='YOLO confidence threshold', required=False)
    parser.add_argument('--iou-threshold', type=float, default=0.3,
                        help='IoU threshold for blob-detection matching', required=False)
    parser.add_argument('--padding', type=int, default=10,
                        help='Crop padding', required=False)
    parser.add_argument('--sample-rate', type=int, default=2,
                        help='Process every Nth frame', required=False)
    parser.add_argument('--pattern', type=str, default='*.mp4',
                        help='Video file pattern', required=False)
    parser.add_argument('--max-bee-samples', type=int, default=1500,
                        help='Max bee samples per video (reservoir sampling)')
    parser.add_argument('--max-noise-samples', type=int, default=1500,
                        help='Max noise samples per video (reservoir sampling)')
    
    
    args = parser.parse_args()
    
    generator = ClassifierDataGenerator(
        yolo_model_path=args.yolo_model,
        output_path=Path(args.output_dir),
        min_blob_area=args.min_area,
        max_blob_area=args.max_area,
        yolo_confidence=args.yolo_conf,
        iou_threshold=args.iou_threshold,
        crop_padding=args.padding,
        sample_rate=args.sample_rate,
        max_bee_samples_per_video=args.max_bee_samples,
        max_noise_samples_per_video=args.max_noise_samples
    )
    
    generator.process_directory(args.video_dir, args.pattern)


if __name__ == '__main__':
    main()




