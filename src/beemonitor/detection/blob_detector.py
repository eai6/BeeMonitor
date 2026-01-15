"""Blob detector using foreground/background subtraction.

Detects motion blobs using background subtraction and morphological operations.
"""

import logging
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from .base_detector import BaseDetector, Detection, BBox, Point

logger = logging.getLogger(__name__)


class BlobDetector(BaseDetector):
    """Foreground/background subtraction-based blob detector.
    
    Uses MOG2 background subtraction to detect moving objects.
    Applies morphological operations and filtering for noise reduction.
    
    **IMPORTANT for Bee Hotels:**
    Initialize the background model with clean frames (no bee activity)
    before starting detection. This establishes a proper baseline.
    
    **RECOMMENDED: Use DL-Verified Initialization**
    Instead of assuming frame ranges are bee-free, use YOLO to automatically
    verify and select only frames with no bee activity.
    
    Workflow:
        1. Create detector
        2. Initialize background with DL-verified clean frames (RECOMMENDED)
           OR manually initialize with known clean frames
        3. Start detecting bees
    
    Example (DL-Verified - RECOMMENDED):
        >>> from ultralytics import YOLO
        >>> from beemonitor.detection import BlobDetector, YOLODetector
        >>> 
        >>> detector = BlobDetector()
        >>> yolo = YOLODetector(YOLO('yolo11n.pt'), tracking_classes=['bee'])
        >>> 
        >>> # Automatically find and use bee-free frames
        >>> detector.initialize_from_video_with_verification(
        ...     video_path='hotel.mp4',
        ...     yolo_detector=yolo,
        ...     num_frames=100,
        ...     max_detections=0  # Strictly no bees
        ... )
        >>> 
        >>> # Now ready to detect
        >>> detections = detector.detect(frame)
    
    Example (Manual - assumes frames 0-100 are clean):
        >>> detector = BlobDetector()
        >>> detector.initialize_from_video('hotel.mp4', num_frames=100)
        >>> detections = detector.detect(frame)
    
    Attributes:
        bg_subtractor: Background subtractor (MOG2)
        min_area: Minimum blob area (pixels)
        min_solidity: Minimum solidity (blob_area / convex_hull_area)
        morph_kernel_size: Morphology kernel size
        morph_iterations: Number of morphology iterations
    """
    
    def __init__(
        self,
        min_area: float = 50.0,
        max_area: float = 5000.0,
        min_solidity: float = 0.5,
        min_circularity: float = 0.0,
        min_aspect_ratio: float = 0.0,
        max_aspect_ratio: float = 10.0,
        min_extent: float = 0.0,
        morph_kernel_size: int = 5,
        morph_iterations: int = 2,
        history: int = 500,
        var_threshold: int = 16
    ):
        """Initialize blob detector.
        
        Args:
            min_area: Minimum blob area in pixels
            max_area: Maximum blob area in pixels (filters out large background noise)
            min_solidity: Minimum solidity ratio (0-1, convex hull fill)
            min_circularity: Minimum circularity (0-1, how round the blob is)
            min_aspect_ratio: Minimum aspect ratio (width/height)
            max_aspect_ratio: Maximum aspect ratio (width/height)
            min_extent: Minimum extent (ratio of contour area to bounding box area)
            morph_kernel_size: Morphology kernel size (odd number)
            morph_iterations: Morphology iterations
            history: Background subtractor history frames
            var_threshold: Background subtractor variance threshold
        """
        # Tracking mode thresholds (precise - can be learned from YOLO)
        self.min_area = min_area
        self.max_area = max_area
        self.min_solidity = min_solidity
        self.min_circularity = min_circularity
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_extent = min_extent
        
        # Motion mode thresholds (sensitive - for two-mode tracking)
        # More relaxed to catch ANY movement and trigger switch to tracking mode
        self.motion_min_area = 50.0
        self.motion_max_area = 5000.0
        self.motion_min_solidity = 0.3
        self.motion_min_circularity = 0.0
        self.motion_min_aspect_ratio = 0.2
        self.motion_max_aspect_ratio = 5.0
        self.motion_min_extent = 0.0
        
        self.morph_kernel_size = morph_kernel_size
        self.morph_iterations = morph_iterations
        
        # Initialize background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=False
        )
        
        logger.info(f"BlobDetector initialized: min_area={min_area}, max_area={max_area}, "
                   f"min_solidity={min_solidity}, min_circularity={min_circularity}")
        logger.debug(f"Motion mode thresholds: min_area={self.motion_min_area}, "
                    f"max_area={self.motion_max_area}, min_solidity={self.motion_min_solidity}")
    
    def detect(self, frame: np.ndarray, mode: str = 'tracking', **kwargs) -> List[Detection]:
        """Detect motion blobs in frame.
        
        Args:
            frame: Input frame (BGR)
            mode: 'tracking' (precise) or 'motion' (sensitive)
            **kwargs: Optional overrides (min_area, min_solidity, etc.)
            
        Returns:
            List of blob detections
        """
        # Select thresholds based on mode
        if mode == 'motion':
            # Motion mode: Use relaxed thresholds (catch any movement)
            default_min_area = self.motion_min_area
            default_max_area = self.motion_max_area
            default_min_solidity = self.motion_min_solidity
            default_min_circularity = self.motion_min_circularity
            default_min_aspect_ratio = self.motion_min_aspect_ratio
            default_max_aspect_ratio = self.motion_max_aspect_ratio
            default_min_extent = self.motion_min_extent
        else:
            # Tracking mode: Use precise thresholds (learned or default)
            default_min_area = self.min_area
            default_max_area = self.max_area
            default_min_solidity = self.min_solidity
            default_min_circularity = self.min_circularity
            default_min_aspect_ratio = self.min_aspect_ratio
            default_max_aspect_ratio = self.max_aspect_ratio
            default_min_extent = self.min_extent
        
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Morphological operations to reduce noise
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.morph_kernel_size, self.morph_kernel_size)
        )
        fg_mask = cv2.morphologyEx(
            fg_mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=self.morph_iterations
        )
        
        # Find contours
        contours, _ = cv2.findContours(
            fg_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter and convert to detections
        detections = []
        min_area = kwargs.get('min_area', default_min_area)
        max_area = kwargs.get('max_area', default_max_area)
        min_solidity = kwargs.get('min_solidity', default_min_solidity)
        min_circularity = kwargs.get('min_circularity', default_min_circularity)
        min_aspect_ratio = kwargs.get('min_aspect_ratio', default_min_aspect_ratio)
        max_aspect_ratio = kwargs.get('max_aspect_ratio', default_max_aspect_ratio)
        min_extent = kwargs.get('min_extent', default_min_extent)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Area filters
            if area < min_area:
                continue
            if area > max_area:
                continue
            
            # Bounding box for aspect ratio and extent
            x, y, w, h = cv2.boundingRect(contour)
            
            # Aspect ratio filter (width/height)
            if h > 0:
                aspect_ratio = w / h
                if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
                    continue
            else:
                continue
            
            # Solidity filter (area / convex hull area)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = area / hull_area
                if solidity < min_solidity:
                    continue
            else:
                solidity = 0.0
            
            # Circularity filter (4π * area / perimeter²)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = (4 * np.pi * area) / (perimeter * perimeter)
                if circularity < min_circularity:
                    continue
            else:
                circularity = 0.0
            
            # Extent filter (contour area / bounding box area)
            bbox_area = w * h
            if bbox_area > 0:
                extent = area / bbox_area
                if extent < min_extent:
                    continue
            else:
                extent = 0.0
            
            # Create detection
            bbox = (float(x), float(y), float(x + w), float(y + h))
            centroid = (float(x + w / 2), float(y + h / 2))
            
            detections.append(Detection(
                bbox=bbox,
                centroid=centroid,
                confidence=1.0,  # Blob detection doesn't have confidence
                label='blob',
                source='fgbg',
                metadata={
                    'area': float(area),
                    'solidity': float(solidity),
                    'circularity': float(circularity),
                    'aspect_ratio': float(aspect_ratio),
                    'extent': float(extent),
                    'perimeter': float(perimeter),
                    'contour': contour
                }
            ))
        
        return detections
    
    def configure(self, **kwargs) -> None:
        """Configure blob detector parameters.
        
        Args:
            **kwargs: min_area, min_solidity, morph_kernel_size, morph_iterations
        """
        if 'min_area' in kwargs:
            self.min_area = kwargs['min_area']
        if 'min_solidity' in kwargs:
            self.min_solidity = kwargs['min_solidity']
        if 'morph_kernel_size' in kwargs:
            self.morph_kernel_size = kwargs['morph_kernel_size']
        if 'morph_iterations' in kwargs:
            self.morph_iterations = kwargs['morph_iterations']
        
        logger.debug(f"BlobDetector configured: {kwargs}")
    
    def reset(self) -> None:
        """Reset background subtractor."""
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=False
        )
        logger.debug("BlobDetector reset")
    
    def initialize_background(
        self,
        frames: List[np.ndarray],
        learning_rate: float = -1
    ) -> None:
        """Initialize background model with clean frames (no bees).
        
        Critical for bee hotels: Train the background model on frames
        without bee activity to establish a clean baseline.
        
        Args:
            frames: List of clean frames (no bees moving)
            learning_rate: Learning rate (-1 for automatic, 0-1 for manual)
                          Higher values = faster learning
        
        Example:
            >>> # Capture 100 frames before bees arrive
            >>> clean_frames = [cap.read()[1] for _ in range(100)]
            >>> detector.initialize_background(clean_frames)
            >>> # Now ready to detect bees
        """
        logger.info(f"Initializing background with {len(frames)} clean frames")
        
        for i, frame in enumerate(frames):
            self.bg_subtractor.apply(frame, learningRate=learning_rate)
            
            if (i + 1) % 50 == 0:
                logger.debug(f"  Processed {i + 1}/{len(frames)} frames")
        
        logger.info("Background initialization complete")
    
    def initialize_from_video(
        self,
        video_path: str,
        num_frames: int = 100,
        start_frame: int = 0,
        learning_rate: float = -1
    ) -> None:
        """Initialize background from video file.
        
        Args:
            video_path: Path to video with clean background
            num_frames: Number of frames to use for initialization
            start_frame: Starting frame index
            learning_rate: Learning rate for background model
            
        Example:
            >>> # Use first 100 frames from video
            >>> detector.initialize_from_video('hotel.mp4', num_frames=100)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        logger.info(f"Initializing background from {video_path}")
        logger.info(f"  Frames {start_frame} to {start_frame + num_frames}")
        
        frames_processed = 0
        
        while frames_processed < num_frames:
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Only {frames_processed} frames available")
                break
            
            self.bg_subtractor.apply(frame, learningRate=learning_rate)
            frames_processed += 1
            
            if frames_processed % 50 == 0:
                logger.debug(f"  Processed {frames_processed}/{num_frames} frames")
        
        cap.release()
        logger.info(f"Background initialization complete ({frames_processed} frames)")
    
    def initialize_from_roi_video(
        self,
        video_path: str,
        roi: Tuple[int, int, int, int],
        num_frames: int = 100,
        start_frame: int = 0,
        learning_rate: float = -1
    ) -> None:
        """Initialize background from ROI in video.
        
        Useful when bee hotel is only in part of frame.
        
        Args:
            video_path: Path to video
            roi: Region of interest (x1, y1, x2, y2)
            num_frames: Number of frames to use
            start_frame: Starting frame index
            learning_rate: Learning rate
            
        Example:
            >>> # Initialize using hotel region only
            >>> roi = (100, 100, 800, 600)
            >>> detector.initialize_from_roi_video('hotel.mp4', roi, num_frames=100)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        x1, y1, x2, y2 = roi
        
        logger.info(f"Initializing background from ROI {roi}")
        logger.info(f"  Frames {start_frame} to {start_frame + num_frames}")
        
        frames_processed = 0
        
        while frames_processed < num_frames:
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Only {frames_processed} frames available")
                break
            
            # Apply ROI mask
            masked_frame = self.apply_roi_mask(frame, roi)
            
            self.bg_subtractor.apply(masked_frame, learningRate=learning_rate)
            frames_processed += 1
            
            if frames_processed % 50 == 0:
                logger.debug(f"  Processed {frames_processed}/{num_frames} frames")
        
        cap.release()
        logger.info(f"Background initialization complete ({frames_processed} frames)")
    
    def initialize_from_video_with_verification(
        self,
        video_path: str,
        yolo_detector,
        num_frames: int = 100,
        start_frame: int = 0,
        max_detections: int = 0,
        search_limit: int = 500,
        learning_rate: float = -1
    ) -> int:
        """Initialize background with DL verification of bee-free frames.
        
        **RECOMMENDED for bee hotels** - Automatically finds and uses only
        frames verified to be bee-free by YOLO, instead of blindly assuming
        a frame range is clean.
        
        Args:
            video_path: Path to video
            yolo_detector: YOLODetector instance to verify frames
            num_frames: Target number of clean frames needed
            start_frame: Frame to start searching from
            max_detections: Maximum allowed bee detections per frame (0 = must be empty)
            search_limit: Maximum frames to search before giving up
            learning_rate: Learning rate for background model
            
        Returns:
            Number of clean frames actually used
            
        Raises:
            ValueError: If insufficient clean frames found
            
        Example:
            >>> from beemonitor.detection import YOLODetector
            >>> from ultralytics import YOLO
            >>> 
            >>> yolo = YOLODetector(YOLO('yolo11n.pt'), tracking_classes=['bee'])
            >>> blob = BlobDetector()
            >>> 
            >>> # Automatically find 100 bee-free frames
            >>> num_clean = blob.initialize_from_video_with_verification(
            ...     video_path='hotel.mp4',
            ...     yolo_detector=yolo,
            ...     num_frames=100,
            ...     max_detections=0  # Strictly no bees
            ... )
            >>> print(f"Initialized with {num_clean} verified clean frames")
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        logger.info(f"DL-Verified background initialization:")
        logger.info(f"  Searching for {num_frames} bee-free frames")
        logger.info(f"  Starting from frame {start_frame}")
        logger.info(f"  Max detections per frame: {max_detections}")
        
        clean_frames = []
        frames_searched = 0
        frames_rejected = 0
        current_frame_num = start_frame
        
        while len(clean_frames) < num_frames and frames_searched < search_limit:
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Video ended after {frames_searched} frames")
                break
            
            frames_searched += 1
            
            # Verify frame is bee-free with YOLO
            detections = yolo_detector.detect(frame)
            num_bees = len(detections)
            
            if num_bees <= max_detections:
                # Frame is clean - use it
                clean_frames.append(frame.copy())
                
                if len(clean_frames) % 20 == 0:
                    logger.debug(f"  Found {len(clean_frames)}/{num_frames} clean frames "
                               f"(searched {frames_searched}, rejected {frames_rejected})")
            else:
                # Frame has bees - skip it
                frames_rejected += 1
                logger.debug(f"  Frame {current_frame_num}: Rejected ({num_bees} bees detected)")
            
            current_frame_num += 1
        
        cap.release()
        
        # Check if we found enough clean frames
        if len(clean_frames) < num_frames:
            logger.warning(
                f"Only found {len(clean_frames)}/{num_frames} clean frames "
                f"after searching {frames_searched} frames"
            )
            
            if len(clean_frames) < num_frames * 0.5:
                raise ValueError(
                    f"Insufficient clean frames: found {len(clean_frames)}, "
                    f"needed {num_frames}. Try increasing search_limit or "
                    f"max_detections, or start from an earlier frame."
                )
        
        # Initialize background with verified clean frames
        logger.info(f"Initializing background with {len(clean_frames)} verified clean frames...")
        self.initialize_background(clean_frames, learning_rate=learning_rate)
        
        logger.info(f"✓ DL-verified initialization complete:")
        logger.info(f"  Clean frames used: {len(clean_frames)}")
        logger.info(f"  Frames rejected: {frames_rejected}")
        logger.info(f"  Success rate: {len(clean_frames)/(frames_searched)*100:.1f}%")
        
        return len(clean_frames)
    
    def initialize_from_roi_video_with_verification(
        self,
        video_path: str,
        roi: Tuple[int, int, int, int],
        yolo_detector,
        num_frames: int = 100,
        start_frame: int = 0,
        max_detections: int = 0,
        search_limit: int = 500,
        learning_rate: float = -1
    ) -> int:
        """Initialize background from ROI with DL verification.
        
        Combines ROI masking with YOLO verification for optimal initialization.
        
        Args:
            video_path: Path to video
            roi: Region of interest (x1, y1, x2, y2)
            yolo_detector: YOLODetector instance
            num_frames: Target number of clean frames
            start_frame: Starting frame
            max_detections: Max allowed detections
            search_limit: Max frames to search
            learning_rate: Learning rate
            
        Returns:
            Number of clean frames used
            
        Example:
            >>> # Verify bee-free frames only in hotel region
            >>> roi = (100, 100, 800, 600)
            >>> num_clean = blob.initialize_from_roi_video_with_verification(
            ...     video_path='hotel.mp4',
            ...     roi=roi,
            ...     yolo_detector=yolo,
            ...     num_frames=100
            ... )
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        x1, y1, x2, y2 = roi
        
        logger.info(f"DL-Verified ROI background initialization:")
        logger.info(f"  ROI: {roi}")
        logger.info(f"  Searching for {num_frames} bee-free frames")
        
        clean_frames = []
        frames_searched = 0
        frames_rejected = 0
        current_frame_num = start_frame
        
        while len(clean_frames) < num_frames and frames_searched < search_limit:
            ret, frame = cap.read()
            if not ret:
                break
            
            frames_searched += 1
            
            # Verify only ROI is bee-free
            roi_crop = frame[y1:y2, x1:x2]
            detections = yolo_detector.detect(roi_crop)
            num_bees = len(detections)
            
            if num_bees <= max_detections:
                # ROI is clean - use masked frame
                masked_frame = self.apply_roi_mask(frame, roi)
                clean_frames.append(masked_frame.copy())
                
                if len(clean_frames) % 20 == 0:
                    logger.debug(f"  Found {len(clean_frames)}/{num_frames} clean frames")
            else:
                frames_rejected += 1
            
            current_frame_num += 1
        
        cap.release()
        
        if len(clean_frames) < num_frames * 0.5:
            raise ValueError(
                f"Insufficient clean frames in ROI: found {len(clean_frames)}, "
                f"needed {num_frames}"
            )
        
        # Initialize with verified clean frames
        logger.info(f"Initializing background with {len(clean_frames)} verified clean ROI frames...")
        self.initialize_background(clean_frames, learning_rate=learning_rate)
        
        logger.info(f"✓ ROI DL-verified initialization complete:")
        logger.info(f"  Clean frames used: {len(clean_frames)}")
        logger.info(f"  Frames rejected: {frames_rejected}")
        
        return len(clean_frames)
    
    def set_learning_rate(self, learning_rate: float) -> None:
        """Set background subtractor learning rate.
        
        Args:
            learning_rate: Learning rate
                -1: Automatic (default)
                0: No learning (static background)
                0.001-0.01: Slow adaptation (typical for bee hotels)
                0.1-1.0: Fast adaptation (for changing conditions)
        """
        self.bg_subtractor.setVarThreshold(learning_rate)
        logger.debug(f"Learning rate set to {learning_rate}")
    
    def is_background_ready(self, frame: np.ndarray, threshold: float = 0.05) -> bool:
        """Check if background model is properly initialized.
        
        Tests by checking if foreground detection is minimal on a static frame.
        
        Args:
            frame: Test frame (should be clean/static)
            threshold: Maximum foreground ratio (0-1) to consider ready
            
        Returns:
            True if background appears initialized
        """
        fg_mask = self.bg_subtractor.apply(frame, learningRate=0)
        
        # Calculate foreground ratio
        fg_pixels = np.sum(fg_mask > 0)
        total_pixels = fg_mask.shape[0] * fg_mask.shape[1]
        fg_ratio = fg_pixels / total_pixels
        
        is_ready = fg_ratio < threshold
        
        if not is_ready:
            logger.warning(f"Background not ready: {fg_ratio:.2%} foreground (threshold: {threshold:.2%})")
        else:
            logger.info(f"Background ready: {fg_ratio:.2%} foreground")
        
        return is_ready
    
    def get_background_image(self) -> np.ndarray:
        """Get current background model as image.
        
        Returns:
            Background image (grayscale)
        """
        return self.bg_subtractor.getBackgroundImage()
    
    def save_background(self, path: str) -> None:
        """Save current background model to file.
        
        Args:
            path: Output path for background image
        """
        bg_image = self.get_background_image()
        cv2.imwrite(path, bg_image)
        logger.info(f"Background saved to {path}")
    
    def visualize_background_difference(
        self,
        frame: np.ndarray,
        apply_morphology: bool = True
    ) -> np.ndarray:
        """Visualize difference between frame and background.
        
        Useful for debugging background initialization.
        
        Args:
            frame: Current frame
            apply_morphology: Apply morphological operations
            
        Returns:
            Visualization image (BGR)
        """
        # Get foreground mask
        if apply_morphology:
            fg_mask = self.get_foreground_mask(frame)
        else:
            fg_mask = self.bg_subtractor.apply(frame, learningRate=0)
        
        # Create colored visualization
        vis = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
        
        # Overlay on original frame
        overlay = cv2.addWeighted(frame, 0.7, vis, 0.3, 0)
        
        # Add stats
        fg_pixels = np.sum(fg_mask > 0)
        total_pixels = fg_mask.shape[0] * fg_mask.shape[1]
        fg_ratio = fg_pixels / total_pixels
        
        cv2.putText(
            overlay,
            f"Foreground: {fg_ratio:.2%}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )
        
        return overlay
    
    def get_source_name(self) -> str:
        """Get detector source name."""
        return 'fgbg'
    
    # Advanced blob filtering methods
    
    def filter_by_hsv(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        hue_range: Optional[Tuple[int, int]] = None,
        sat_range: Optional[Tuple[int, int]] = None,
        val_range: Optional[Tuple[int, int]] = None
    ) -> List[Detection]:
        """Filter blobs by HSV color criteria.
        
        Args:
            frame: Original frame (BGR)
            detections: Blob detections to filter
            hue_range: (min_hue, max_hue) in [0, 180]
            sat_range: (min_sat, max_sat) in [0, 255]
            val_range: (min_val, max_val) in [0, 255]
            
        Returns:
            Filtered detections
        """
        if not detections:
            return []
        
        # Convert frame to HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        filtered = []
        
        for det in detections:
            x1, y1, x2, y2 = [int(c) for c in det.bbox]
            
            # Extract blob region
            if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                continue
            
            blob_hsv = hsv_frame[y1:y2, x1:x2]
            
            if blob_hsv.size == 0:
                continue
            
            # Calculate mean HSV
            mean_hue = np.mean(blob_hsv[:, :, 0])
            mean_sat = np.mean(blob_hsv[:, :, 1])
            mean_val = np.mean(blob_hsv[:, :, 2])
            
            # Apply filters
            if hue_range and not (hue_range[0] <= mean_hue <= hue_range[1]):
                continue
            if sat_range and not (sat_range[0] <= mean_sat <= sat_range[1]):
                continue
            if val_range and not (val_range[0] <= mean_val <= val_range[1]):
                continue
            
            # Add HSV info to metadata
            det.metadata['mean_hue'] = float(mean_hue)
            det.metadata['mean_sat'] = float(mean_sat)
            det.metadata['mean_val'] = float(mean_val)
            
            filtered.append(det)
        
        logger.debug(f"HSV filter: {len(detections)} → {len(filtered)} detections")
        return filtered
    
    def filter_by_aspect_ratio(
        self,
        detections: List[Detection],
        min_ratio: float = 0.5,
        max_ratio: float = 2.0
    ) -> List[Detection]:
        """Filter blobs by aspect ratio.
        
        Args:
            detections: Blob detections to filter
            min_ratio: Minimum width/height ratio
            max_ratio: Maximum width/height ratio
            
        Returns:
            Filtered detections
        """
        filtered = []
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            w = x2 - x1
            h = y2 - y1
            
            if h == 0:
                continue
            
            ratio = w / h
            
            if min_ratio <= ratio <= max_ratio:
                det.metadata['aspect_ratio'] = float(ratio)
                filtered.append(det)
        
        logger.debug(f"Aspect ratio filter: {len(detections)} → {len(filtered)} detections")
        return filtered
    
    def filter_by_circularity(
        self,
        detections: List[Detection],
        min_circularity: float = 0.5
    ) -> List[Detection]:
        """Filter blobs by circularity.
        
        Circularity = 4π * area / perimeter²
        Circle = 1.0, Square ≈ 0.785
        
        Args:
            detections: Blob detections to filter
            min_circularity: Minimum circularity (0-1)
            
        Returns:
            Filtered detections
        """
        filtered = []
        
        for det in detections:
            if 'contour' not in det.metadata:
                continue
            
            contour = det.metadata['contour']
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity >= min_circularity:
                det.metadata['circularity'] = float(circularity)
                filtered.append(det)
        
        logger.debug(f"Circularity filter: {len(detections)} → {len(filtered)} detections")
        return filtered
    
    def get_blob_features(self, detection: Detection) -> dict:
        """Extract comprehensive features from blob detection.
        
        Args:
            detection: Blob detection
            
        Returns:
            Dictionary of features
        """
        features = {
            'bbox': detection.bbox,
            'centroid': detection.centroid,
            'area': detection.metadata.get('area', 0.0),
            'solidity': detection.metadata.get('solidity', 0.0)
        }
        
        # Bounding box features
        x1, y1, x2, y2 = detection.bbox
        features['width'] = x2 - x1
        features['height'] = y2 - y1
        features['aspect_ratio'] = features['width'] / features['height'] if features['height'] > 0 else 0
        
        # Contour-based features
        if 'contour' in detection.metadata:
            contour = detection.metadata['contour']
            
            # Perimeter
            features['perimeter'] = cv2.arcLength(contour, True)
            
            # Circularity
            if features['perimeter'] > 0:
                features['circularity'] = 4 * np.pi * features['area'] / (features['perimeter'] ** 2)
            else:
                features['circularity'] = 0.0
            
            # Moments
            moments = cv2.moments(contour)
            if moments['m00'] != 0:
                features['hu_moments'] = cv2.HuMoments(moments).flatten()
        
        # HSV features
        if 'mean_hue' in detection.metadata:
            features['mean_hue'] = detection.metadata['mean_hue']
            features['mean_sat'] = detection.metadata['mean_sat']
            features['mean_val'] = detection.metadata['mean_val']
        
        return features
    
    def apply_roi_mask(
        self,
        frame: np.ndarray,
        roi: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """Apply ROI mask to frame (zero out pixels outside ROI).
        
        Args:
            frame: Input frame
            roi: Region of interest (x1, y1, x2, y2)
            
        Returns:
            Masked frame
        """
        x1, y1, x2, y2 = roi
        
        # Create mask
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        
        # Apply mask
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        
        return masked_frame
    
    def detect_in_roi(
        self,
        frame: np.ndarray,
        roi: Tuple[int, int, int, int],
        **kwargs
    ) -> List[Detection]:
        """Detect blobs only within ROI.
        
        Args:
            frame: Input frame
            roi: Region of interest (x1, y1, x2, y2)
            **kwargs: Detection parameters
            
        Returns:
            Detections within ROI
        """
        # Apply ROI mask
        masked_frame = self.apply_roi_mask(frame, roi)
        
        # Detect in masked frame
        detections = self.detect(masked_frame, **kwargs)
        
        # Filter to only ROI
        x1, y1, x2, y2 = roi
        filtered = []
        
        for det in detections:
            cx, cy = det.centroid
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                filtered.append(det)
        
        return filtered
    
    def visualize_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        show_bbox: bool = True,
        show_centroid: bool = True,
        show_label: bool = True,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """Draw detections on frame.
        
        Args:
            frame: Input frame
            detections: Detections to visualize
            show_bbox: Draw bounding boxes
            show_centroid: Draw centroids
            show_label: Draw labels
            color: BGR color
            thickness: Line thickness
            
        Returns:
            Frame with visualizations
        """
        vis_frame = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = [int(c) for c in det.bbox]
            cx, cy = [int(c) for c in det.centroid]
            
            # Draw bbox
            if show_bbox:
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw centroid
            if show_centroid:
                cv2.circle(vis_frame, (cx, cy), 3, color, -1)
            
            # Draw label
            if show_label:
                label_text = f"{det.label}"
                if 'area' in det.metadata:
                    label_text += f" {det.metadata['area']:.0f}px"
                cv2.putText(
                    vis_frame,
                    label_text,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1
                )
        
        return vis_frame
    
    def learn_geometric_thresholds_from_video(
        self,
        video_path: str,
        yolo_detector,
        num_frames: int = 100,
        start_frame: int = 0,
        min_detections: int = 20,
        percentile_low: float = 5.0,
        percentile_high: float = 95.0
    ) -> Dict[str, float]:
        """Learn geometric filter thresholds from YOLO-detected bees.
        
        Similar to background initialization, this analyzes frames with
        YOLO-detected bees and computes statistics on their geometric
        properties to set adaptive thresholds.
        
        Args:
            video_path: Path to video file
            yolo_detector: YOLODetector instance for finding bees
            num_frames: Number of frames to analyze
            start_frame: Starting frame number
            min_detections: Minimum number of bee detections needed
            percentile_low: Percentile for minimum thresholds (e.g., 5th)
            percentile_high: Percentile for maximum thresholds (e.g., 95th)
            
        Returns:
            Dictionary of learned thresholds
        """
        import cv2
        
        logger.info(f"Learning geometric thresholds from video: {video_path}")
        logger.info(f"  Analyzing {num_frames} frames starting at {start_frame}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Collect geometric metrics from YOLO-detected bees
        areas = []
        circularities = []
        solidities = []
        aspect_ratios = []
        extents = []
        
        frames_processed = 0
        
        for i in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect bees with YOLO
            yolo_detections = yolo_detector.detect(frame, conf=0.5)
            
            if not yolo_detections:
                continue
            
            # For each YOLO detection, extract the region and compute metrics
            for det in yolo_detections:
                x1, y1, x2, y2 = [int(c) for c in det.bbox]
                
                # Validate bounds
                if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                    continue
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Create a binary mask for this region
                # We'll use the bbox as a proxy for the bee shape
                w = x2 - x1
                h = y2 - y1
                area = w * h
                
                # Extract crop and threshold to get bee contour
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                
                # Convert to grayscale and threshold
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Find contours in the thresholded crop
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if not contours:
                    continue
                
                # Use the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                contour_area = cv2.contourArea(largest_contour)
                
                if contour_area < 10:  # Too small
                    continue
                
                # Calculate metrics
                # Area
                areas.append(contour_area)
                
                # Aspect ratio
                if h > 0:
                    aspect_ratio = w / h
                    aspect_ratios.append(aspect_ratio)
                
                # Solidity
                hull = cv2.convexHull(largest_contour)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0:
                    solidity = contour_area / hull_area
                    solidities.append(solidity)
                
                # Circularity
                perimeter = cv2.arcLength(largest_contour, True)
                if perimeter > 0:
                    circularity = (4 * np.pi * contour_area) / (perimeter * perimeter)
                    circularities.append(circularity)
                
                # Extent
                bbox_area = w * h
                if bbox_area > 0:
                    extent = contour_area / bbox_area
                    extents.append(extent)
            
            frames_processed += 1
        
        cap.release()
        
        # Check if we have enough data
        total_detections = len(areas)
        logger.info(f"Processed {frames_processed} frames, found {total_detections} bee detections")
        
        if total_detections < min_detections:
            logger.warning(f"Only found {total_detections} detections (minimum {min_detections} needed)")
            logger.warning("Using default thresholds")
            return {
                'min_area': 100.0,
                'max_area': 2000.0,
                'min_solidity': 0.75,
                'min_circularity': 0.3,
                'min_aspect_ratio': 0.4,
                'max_aspect_ratio': 2.5,
                'min_extent': 0.3
            }
        
        # Compute percentile-based thresholds
        thresholds = {
            'min_area': float(np.percentile(areas, percentile_low)) if areas else 100.0,
            'max_area': float(np.percentile(areas, percentile_high)) if areas else 2000.0,
            'min_solidity': float(np.percentile(solidities, percentile_low)) if solidities else 0.75,
            'min_circularity': float(np.percentile(circularities, percentile_low)) if circularities else 0.3,
            'min_aspect_ratio': float(np.percentile(aspect_ratios, percentile_low)) if aspect_ratios else 0.4,
            'max_aspect_ratio': float(np.percentile(aspect_ratios, percentile_high)) if aspect_ratios else 2.5,
            'min_extent': float(np.percentile(extents, percentile_low)) if extents else 0.3
        }
        
        # Log statistics
        logger.info("Learned geometric thresholds from bee detections:")
        logger.info(f"  Area: {thresholds['min_area']:.1f} - {thresholds['max_area']:.1f} px")
        logger.info(f"  Solidity: ≥{thresholds['min_solidity']:.2f}")
        logger.info(f"  Circularity: ≥{thresholds['min_circularity']:.2f}")
        logger.info(f"  Aspect ratio: {thresholds['min_aspect_ratio']:.2f} - {thresholds['max_aspect_ratio']:.2f}")
        logger.info(f"  Extent: ≥{thresholds['min_extent']:.2f}")
        
        if areas:
            logger.info(f"  [Statistics from {len(areas)} detections]")
            logger.info(f"    Area: mean={np.mean(areas):.1f}, std={np.std(areas):.1f}")
            logger.info(f"    Circularity: mean={np.mean(circularities):.2f}, std={np.std(circularities):.2f}")
            logger.info(f"    Solidity: mean={np.mean(solidities):.2f}, std={np.std(solidities):.2f}")
        
        # Apply learned thresholds to this detector
        self.min_area = thresholds['min_area']
        self.max_area = thresholds['max_area']
        self.min_solidity = thresholds['min_solidity']
        self.min_circularity = thresholds['min_circularity']
        self.min_aspect_ratio = thresholds['min_aspect_ratio']
        self.max_aspect_ratio = thresholds['max_aspect_ratio']
        self.min_extent = thresholds['min_extent']
        
        logger.info("✓ Geometric thresholds learned and applied")
        
        return thresholds
    
    def get_foreground_mask(self, frame: np.ndarray) -> np.ndarray:
        """Get foreground mask without creating detections.
        
        Useful for visualization or custom processing.
        
        Args:
            frame: Input frame
            
        Returns:
            Foreground mask (binary)
        """
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.morph_kernel_size, self.morph_kernel_size)
        )
        fg_mask = cv2.morphologyEx(
            fg_mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=self.morph_iterations
        )
        
        return fg_mask
    
    def update_background_only(self, frame: np.ndarray) -> None:
        """Update background model without detecting.
        
        Useful for initial learning period.
        
        Args:
            frame: Input frame
        """
        self.bg_subtractor.apply(frame, learningRate=-1)
    
    def get_info(self) -> dict:
        """Get detector information and statistics."""
        info = super().get_info()
        info.update({
            'min_area': self.min_area,
            'min_solidity': self.min_solidity,
            'morph_kernel_size': self.morph_kernel_size,
            'morph_iterations': self.morph_iterations,
            'bg_subtractor': 'MOG2'
        })
        return info