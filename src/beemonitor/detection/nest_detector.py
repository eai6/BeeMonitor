"""Nest detection and processing module.

This module handles detecting nest holes in bee hotel videos and processing
them to identify individual nest locations with IDs.
"""

import logging
import traceback
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
import pandas as pd
import math

from beemonitor.core.config import Config
from beemonitor.utils.geometry import remove_overlapping_points


logger = logging.getLogger(__name__)

# Type aliases
Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]


class NestDetector:
    """Detector for bee hotel nests.
    
    This class handles nest detection and processing, including:
    - Running YOLO detection on video frames
    - Clustering detections into rows and columns
    - Fixing missing nest holes
    - Assigning unique IDs to each nest
    
    Attributes:
        model: YOLO model for nest detection
        config: Configuration object
    
    Example:
        >>> from ultralytics import YOLO
        >>> model = YOLO("models/nest_model.pt")
        >>> detector = NestDetector(model, config)
        >>> detections = detector.detect_nests("video.mp4", 720, 1280)
        >>> nests = detector.process_detections("video.mp4", detections, 720, 1280)
    """
    
    def __init__(self, model, config: Optional[Config] = None):
        """Initialize NestDetector.
        
        Args:
            model: YOLO model for nest detection
            config: Configuration object (optional)
        """
        self.model = model
        self.config = config if config is not None else Config.default()
    
    def detect_nests(
        self,
        reference_frame: int,
        video_path: str,
        res_height: int,
        res_width: int
    ) -> pd.DataFrame:
        """Detect nests in video frames.
        
        Processes frames from the video to find nest holes using YOLO detection.
        Continues until sufficient detections are found (typically ~60).
        
        Args:
            reference_frame: Frame number to start detection from
            video_path: Path to video file
            res_height: Target frame height
            res_width: Target frame width
            
        Returns:
            DataFrame with columns: frame, coordinates, state, confidence
            
        Example:
            >>> detections = detector.detect_nests("video.mp4", 720, 1280)
            >>> print(f"Found {len(detections.iloc[0]['coordinates'])} nests")
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        frame_counter = reference_frame
        nest_detections = [[]]
        nest_state = []
        frames = []
        confs = []
        
        confidence_threshold = self.config.nest.confidence_threshold
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_counter)
        
        # Read frame
        success, frame = cap.read()

        if not success:
            return pd.DataFrame({
                'frame': [reference_frame],
                'coordinates': [[]],
                'state': [[]],
                'confidence': [[]]
            })
        
        # Resize frame
        frame = cv2.resize(frame, (res_width, res_height))
        
        logger.debug(f"Processing frame {frame_counter}")
        
        # Run YOLO inference
        results = self.model.predict(frame, conf=confidence_threshold, verbose=False)
        
        # Extract detections
        boxes = results[0].boxes.xyxy.tolist()
        boxes = [(x, y, x1, y1) for (x, y, x1, y1) in boxes]
        nest_detections = [boxes]
        
        # Extract labels
        labels = results[0].boxes.cls.tolist()
        nest_state = [labels]
        
        # Extract confidence scores
        conf = results[0].boxes.conf.tolist()
        confs = [conf]
        
        frames = [frame_counter]
        
        cap.release()
        
        
        nest_df = pd.DataFrame({
            'frame': frames,
            'coordinates': nest_detections,
            'state': nest_state,
            'confidence': confs
        })
        
        return nest_df
    
    def process_detections(
        self,
        video_path: str,
        nest_detection: pd.DataFrame,
        res_height: int,
        res_width: int
    ) -> Dict:
        """Process nest detections to identify individual nest holes.
        
        Takes raw nest detections and processes them to:
        1. Extract nest hole coordinates
        2. Cluster into rows
        3. Fix missing holes
        4. Assign unique IDs
        5. Calculate hotel ROI
        
        Args:
            video_path: Path to video file
            nest_detection: DataFrame from detect_nests
            res_height: Frame height
            res_width: Frame width
            
        Returns:
            Dictionary with keys:
                - 'hotel': Tuple (x1, y1, x2, y2) for hotel ROI
                - 'nests': Dict mapping nest IDs to bounding boxes
                
        Example:
            >>> nests = detector.process_detections("video.mp4", detections, 720, 1280)
            >>> print(f"Hotel ROI: {nests['hotel']}")
            >>> print(f"Found {len(nests['nests'])} individual nests")
        """
        logger.info("Processing nest detections...")
        
        # Extract nest coordinates
        nest_coords = self._get_nest_coordinates(nest_detection)
        logger.debug(f"Extracted {len(nest_coords)} nest coordinates")

        # Get scaled parameters for this resolution
        row_threshold = self.config.nest.row_threshold(res_width, res_height)
        col_threshold = self.config.nest.col_threshold(res_width, res_height)
        
        # Cluster into rows
        dl_rows = self._cluster_points(
            nest_coords,
            row_threshold=row_threshold,
            col_threshold=col_threshold,
        )
        logger.debug(f"Clustered into {len(dl_rows)} rows")
        
        # Validate we have detections
        if not dl_rows:
            logger.warning("No nest rows detected")
            return {
                'hotel': (0, 0, res_width, res_height),
                'nests': [],
                'num_rows': 0,
                'nests_per_row': []
            }
        
        # Filter out empty rows
        dl_rows = [row for row in dl_rows if len(row) > 0]
        if not dl_rows:
            logger.warning("All nest rows are empty")
            return {
                'hotel': (0, 0, res_width, res_height),
                'nests': [],
                'num_rows': 0,
                'nests_per_row': []
            }
        
        # Get reference points for hole fixing
        rows_10 = [row for row in dl_rows if len(row) == 10]
        if rows_10:
            hole_first = self._get_average_x([x[0][0] for x in rows_10])
            hole_last = self._get_average_x([x[-1][0] for x in rows_10])
        else:
            # Use all rows if no perfect rows found
            try:
                hole_first = min(min(row, key=lambda p: p[0])[0] for row in dl_rows if len(row) > 0)
                hole_last = max(max(row, key=lambda p: p[0])[0] for row in dl_rows if len(row) > 0)
            except ValueError:
                # If still failing, use frame boundaries
                logger.warning("Could not determine hole boundaries, using frame dimensions")
                hole_first = res_width * 0.1
                hole_last = res_width * 0.9
        
        # Calculate average x distance
        x_distances = []
        for row in dl_rows:
            if len(row) > 1:
                x_distances.append(self._get_row_average_nest_distance(row))
        x_distance = self._get_average_x(x_distances) if x_distances else 50
        
        # Fix missing holes in rows
        fixed_dl_rows = []
        for row in dl_rows:
            fixed_row = self._fix_row_coords(
                row,
                hole_first,
                hole_last,
                x_average_width=x_distance
            )
            fixed_dl_rows.append(fixed_row)
        
        logger.debug(f"Fixed missing holes, now have {sum(len(r) for r in fixed_dl_rows)} nests")
        
        # Calculate hotel boundaries
        hole_top = self._get_average_x([x[1] for x in fixed_dl_rows[0]])
        hole_bottom = self._get_average_x([x[1] for x in fixed_dl_rows[-1]])
        
        # Hotel ROI with padding
        hx = max(0, hole_first - self.config.nest.hotel_padding_x(res_width, res_height))
        hy = max(0, hole_top - self.config.nest.hotel_padding_y(res_width, res_height))
        hx1 = min(res_width, hole_last + self.config.nest.hotel_padding_x(res_width, res_height))
        hy2 = min(res_height, hole_bottom + self.config.nest.hotel_padding_y(res_width, res_height))
        
        # Generate nest hole coordinates with IDs
        nest_ids = self._assign_nest_ids(fixed_dl_rows)
        
        result = {
            "hotel": (hx, hy, hx1, hy2),
            "nests": nest_ids
        }
        
        logger.info(f"Processed {len(nest_ids)} nests in hotel ROI")

        # Save visualization for debugging
        self._save_visualization(
            video_path,
            result,
            res_height,
            res_width
        )
           
        
        
        return result
    
    def get_nests_and_hotel_detections(
        self,
        video_path: str,
    ) -> Optional[Dict]:
        """Combined method to get nest detection and processing.

        Takes the video path and resolution, runs detection and processing,
        with quality checks and retries if needed.
        
        Process:
        0. Updates config.video.res* values with video resolution
        1. Detects nests and hotel in the video, starting from the first frame
        2. Processes the detections to identify individual nest holes and assign IDs
        3. If the processed nests meet quality criteria, returns the result
        4. If quality check fails, runs detection again on subsequent frames
        5. If maximum attempts reached without success, returns None
        
        Args:
            video_path: Path to video file

        Returns:
            Dictionary with nest information if successful, None otherwise
            
        Example:
            >>> nests = detector.get_nest_detection("video.mp4")
            >>> if nests is not None:
            >>>     print(f"Successfully detected {len(nests['nests'])} nests")
        """
        max_attempts = self.config.nest.max_detection_attempts

        cap = cv2.VideoCapture(video_path)
        res_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        res_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        self.config.video.res_height = res_height
        self.config.video.res_width = res_width
        self.config.video.fps = fps

        min_detections = self.config.nest.min_detections 
        
        logger.info(f"Starting nest detection pipeline (max attempts: {max_attempts})")

        reference_frame = 0
        
        for attempt in range(max_attempts):
            try:
                logger.info(f"Detection attempt {attempt + 1}/{max_attempts}")
                
                # Run detection
                nest_detection = self.detect_nests(reference_frame, video_path, res_height, res_width)

                if not nest_detection.empty:
                    num_detections = len(nest_detection.iloc[0]['coordinates'])
                    reference_frame = nest_detection.iloc[0]['frame']
                else:
                    num_detections = 0

                if num_detections < min_detections:
                    logger.warning(
                        f"Attempt {attempt + 1}: Insufficient detections "
                        f"({num_detections} found, {min_detections} required)"
                    )
                    # Skip ahead more frames if too few detections
                    reference_frame += self.config.nest.frame_skip 
                    continue

                # Process detections
                processed_nests = self.process_detections(
                    video_path,
                    nest_detection,
                    res_height,
                    res_width
                )

                # assing hotel box to config for motion tracking use later
                self.config.hotel_box.hotel_box_cords = processed_nests['hotel']
                
                # Check quality
                if self._check_nest_quality(processed_nests):
                    logger.info(f"Successful nest detection on attempt {attempt + 1}")

                    # save reference frame with nests
                    self._save_visualization(
                        video_path=video_path,
                        nest_ids=processed_nests,
                        res_height=res_height,
                        res_width=res_width, 
                    )

                    return processed_nests
                else:
                    logger.warning(f"Attempt {attempt + 1}: Quality check failed")
                    # Move to next frame for next attempt
                    reference_frame += self.config.nest.frame_skip 
                    continue
                    
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed with error: {e}")
                # print traceback for debugging
                logger.warning(traceback.format_exc())
                continue
        
        logger.error(f"Failed to detect nests after {max_attempts} attempts")
        return None
    
    def _check_nest_quality(
        self,
        processed_nests: Dict
    ) -> bool:
        """Check if nest detection meets quality criteria.

        Validates the processed nest detection against multiple quality criteria:
        - Correct total number of nests (within tolerance)
        - Correct number of rows
        - Correct number of nests per row
        - Consistent horizontal spacing between nests in rows
        - Consistent vertical position within each row
        - Sequential nest IDs without gaps (1, 2, 3, ..., N)
        - Consistent vertical alignment of nests in same column
        
        Args:
            processed_nests: Dictionary with 'hotel' and 'nests' keys
            
        Returns:
            True if all quality criteria are met, False otherwise
        """

        if not processed_nests or 'nests' not in processed_nests:
            logger.warning("Quality check failed: Invalid nest structure")
            return False
        
        nests = processed_nests['nests']
        
        if not nests:
            logger.warning("Quality check failed: No nests found")
            return False
        
        # 1. Check total number of nests
        total_nests = len(nests)
        expected_total = self.config.nest.expected_total_nests
        nest_count_tolerance = self.config.nest.nest_count_tolerance
        
        if abs(total_nests - expected_total) > nest_count_tolerance:
            logger.warning(
                f"Quality check failed: Expected {expected_total} nests "
                f"(±{nest_count_tolerance}), got {total_nests}"
            )
            return False
        
        # 2. Check nest IDs are sequential and unique
        nest_ids = sorted([int(nid) for nid in nests.keys()])
        expected_ids = list(range(1, total_nests + 1))
        
        if nest_ids != expected_ids:
            missing_ids = set(expected_ids) - set(nest_ids)
            extra_ids = set(nest_ids) - set(expected_ids)
            logger.warning(
                f"Quality check failed: Non-sequential IDs. "
                f"Missing: {missing_ids}, Extra: {extra_ids}"
            )
            return False
        
        # 3. Organize nests by rows
        nests_per_row = self.config.nest.expected_nests_per_row
        expected_rows = self.config.nest.expected_rows
        
        rows = {}
        for nest_id, bbox in nests.items():
            row_num = (int(nest_id) - 1) // nests_per_row
            if row_num not in rows:
                rows[row_num] = []
            rows[row_num].append((int(nest_id), bbox))
        
        # Check number of rows
        if len(rows) != expected_rows:
            logger.warning(
                f"Quality check failed: Expected {expected_rows} rows, "
                f"got {len(rows)}"
            )
            return False
        
        # 4. Check each row has correct number of nests
        for row_num, row_nests in rows.items():
            if len(row_nests) != nests_per_row:
                logger.warning(
                    f"Quality check failed: Row {row_num} has {len(row_nests)} nests, "
                    f"expected {nests_per_row}"
                )
                return False
        
        # 5. Check horizontal spacing consistency within rows
        spacing_tolerance = self.config.nest.spacing_tolerance(self.config.video.res_width, self.config.video.res_height)
        #spacing_tolerance(self.config.video.res_width, self.config.video.res_height)
        
        for row_num, row_nests in rows.items():
            # Sort by nest ID
            row_nests = sorted(row_nests, key=lambda x: x[0])
            
            # Calculate x-positions (center of bboxes)
            x_positions = []
            for _, bbox in row_nests:
                x_center = (bbox[0] + bbox[2]) / 2
                x_positions.append(x_center)
            
            # Calculate spacing between consecutive nests
            spacings = [x_positions[i+1] - x_positions[i] for i in range(len(x_positions)-1)]
            
            if len(spacings) > 0:
                avg_spacing = np.mean(spacings)
                
                # Check all spacings are within tolerance
                for i, spacing in enumerate(spacings):
                    if abs(spacing - avg_spacing) > spacing_tolerance:
                        logger.warning(
                            f"Quality check failed: Row {row_num} has inconsistent spacing. "
                            f"Gap {i}: {spacing:.1f}, Average: {avg_spacing:.1f}"
                        )
                        return False
        
        # 6. Check vertical position consistency within rows - FIXED
        y_tolerance = self.config.nest.y_position_tolerance(self.config.video.res_width, self.config.video.res_height)
        
        for row_num, row_nests in rows.items():
            # Calculate y-positions (center of bboxes)
            y_positions = []
            for _, bbox in row_nests:
                y_center = (bbox[1] + bbox[3]) / 2
                y_positions.append(y_center)
            
            if len(y_positions) > 0:
                avg_y = np.mean(y_positions)
                
                # Check all y-positions are within tolerance
                for i, y_pos in enumerate(y_positions):
                    if abs(y_pos - avg_y) > y_tolerance:
                        logger.warning(
                            f"Quality check failed: Row {row_num} has inconsistent y-positions. "
                            f"Nest {i}: y={y_pos:.1f}, Average: {avg_y:.1f}"
                        )
                        return False
        
        # 7. Check vertical alignment of columns - FIXED
        x_tolerance = self.config.nest.x_position_tolerance(self.config.video.res_width, self.config.video.res_height)
        
        for col_num in range(nests_per_row):
            # Get all nests in this column (1, 11, 21, ...)
            column_nests = []
            for row_num in range(expected_rows):
                nest_id = str(col_num + 1 + row_num * nests_per_row)
                if nest_id in nests:
                    bbox = nests[nest_id]
                    x_center = (bbox[0] + bbox[2]) / 2
                    column_nests.append(x_center)
            
            if len(column_nests) > 1:
                avg_x = np.mean(column_nests)
                
                # Check all x-positions are within tolerance
                for i, x_pos in enumerate(column_nests):
                    if abs(x_pos - avg_x) > x_tolerance:
                        logger.warning(
                            f"Quality check failed: Column {col_num + 1} has inconsistent alignment. "
                            f"Row {i}: x={x_pos:.1f}, Average: {avg_x:.1f}"
                        )
                        return False
        
        logger.info("All quality checks passed")
        return True
    
    def _get_nest_coordinates(
        self,
        nest: pd.DataFrame,
        index: int = 0
    ) -> List[Point]:
        """Extract nest hole midpoints from detection DataFrame.
        
        Args:
            nest: DataFrame with nest detections
            index: Row index to use (default: 0)
            
        Returns:
            List of (x, y) midpoints
        """

        nest_tube_class = self.config.nest.nest_tube_class
        def get_midpoint(nest_coords):
            x1, y1, x2, y2 = nest_coords
            midpoint_x = (x1 + x2) / 2
            midpoint_y = (y1 + y2) / 2
            return (int(midpoint_x), int(midpoint_y))
        
        states = nest.iloc[index]['state']
        coordinates = nest.iloc[index]['coordinates']
        
        # Filter for nest_tube class 
        nest_coords = []
        for i in range(len(states)):
            if states[i] == nest_tube_class:
                nest_coords.append(coordinates[i])
        
        return [get_midpoint(nest_hole) for nest_hole in nest_coords]
    
    def _cluster_points(
        self,
        points: List[Point],
        row_threshold: int = 10,
        col_threshold: int = 10
    ) -> List[List[Point]]:
        """Cluster points into rows and columns.
        
        Args:
            points: List of (x, y) points
            row_threshold: Y-distance threshold for row clustering
            col_threshold: X-distance threshold for column clustering
            
        Returns:
            List of rows, where each row is a list of points
        """
        if not points:
            return []
        
        # Sort points by y-coordinate to cluster into rows
        points_sorted_y = sorted(points, key=lambda x: x[1])
        rows = []
        current_row = [points_sorted_y[0]]
        
        for point in points_sorted_y[1:]:
            if abs(point[1] - current_row[-1][1]) < row_threshold:
                current_row.append(point)
            else:
                rows.append(current_row)
                current_row = [point]
        rows.append(current_row)
        
        # Sort each row by x-coordinate
        for i in range(len(rows)):
            rows[i] = sorted(rows[i], key=lambda x: x[0])
        
        # Remove rows with too few points
        min_row_size = self.config.nest.min_row_size
        rows = [row for row in rows if len(row) >= min_row_size]
        
        return rows
    
    def _get_row_average_nest_distance(self, row: List[Point]) -> int:
        """Calculate average horizontal distance between nests in a row.
        
        Args:
            row: List of points in the row
            
        Returns:
            Average distance between consecutive nests
        """
        if len(row) < 2:
            return 0
        
        distances = np.diff([x[0] for x in row]).tolist()
        distances = sorted(distances)
        
        # Filter out outliers
        filtered = []
        if distances:
            current = distances[0]
            for dist in distances:
                if abs(current - dist) < 10:
                    filtered.append(dist)
                current = dist
        
        return int(np.mean(filtered)) if filtered else int(np.mean(distances))
    
    def _get_average_x(self, nums: List[float]) -> int:
        """Calculate average of x-coordinates, filtering outliers.
        
        Args:
            nums: List of x-coordinates
            
        Returns:
            Average x-coordinate
        """
        if not nums:
            return 0
        
        nums = sorted(nums)
        
        filtered = []
        current = nums[0]
        for num in nums:
            if abs(current - num) < 10:
                filtered.append(num)
            current = num
        
        return int(np.mean(filtered)) if filtered else int(np.mean(nums))
    
    def _fix_row_coords(
        self,
        row: List[Point],
        first_hole_x: float,
        last_hole_x: float,
        pixel_threshold: int = 10,
        x_average_width: int = 0
    ) -> List[Point]:
        """Fix missing holes in a row by interpolating positions.
        
        Args:
            row: List of detected points in the row
            first_hole_x: X-coordinate of first hole
            last_hole_x: X-coordinate of last hole
            pixel_threshold: Threshold for position matching
            x_average_width: Average distance between holes
            
        Returns:
            List of points with missing holes filled in
        """
        if len(row) < 2:
            return row
        
        new_row_coords = []
        y_average = int(np.mean([x[1] for x in row]))
        
        # Check if first hole is missing
        if abs(first_hole_x - row[0][0]) > pixel_threshold:
            new_row_coords.append((first_hole_x, y_average))
        
        for i in range(len(row) - 1):
            current_hole = row[i]
            next_hole = row[i + 1]
            x_diff = next_hole[0] - current_hole[0]
            
            if abs(x_diff - x_average_width) < pixel_threshold:
                new_row_coords.append(current_hole)
                
                if i == len(row) - 2:  # Last pair
                    new_row_coords.append(next_hole)
            else:
                # Add current hole
                new_row_coords.append(current_hole)
                y_average = int((current_hole[1] + next_hole[1]) / 2)
                
                # Interpolate missing holes
                num_missing = math.ceil(x_diff / x_average_width)
                for j in range(num_missing - 1):
                    new_x = current_hole[0] + int(x_average_width * (j + 1))
                    new_row_coords.append((new_x, y_average))
        
        # Remove overlapping points
        new_row_coords = remove_overlapping_points(new_row_coords, threshold=30)
        
        # Check if last hole is missing
        if len(new_row_coords) < 10:
            new_row_coords.append((last_hole_x, y_average))
        
        return new_row_coords
    
    def _assign_nest_ids(self, rows: List[List[Point]]) -> Dict[str, BBox]:
        """Assign unique IDs to each nest and calculate bounding boxes.
        
        Args:
            rows: List of rows, each containing nest points
            
        Returns:
            Dictionary mapping nest IDs (str) to bounding boxes
        """

        width = self.config.nest.nest_width(self.config.video.res_width, self.config.video.res_height)
        height = self.config.nest.nest_height(self.config.video.res_width, self.config.video.res_height)
        pad_x = self.config.nest.padding_x(self.config.video.res_width, self.config.video.res_height)
        pad_y = self.config.nest.padding_y(self.config.video.res_width, self.config.video.res_height)
        
        nest_ids = {}
        
        for row_idx, row in enumerate(rows):
            sorted_row = sorted(row, key=lambda x: x[0])
            for col_idx, hole in enumerate(sorted_row):
                nest_id = str((col_idx + 1) + row_idx * 10)
                x, y = hole
                
                bbox = (
                    x - width // 2 - pad_x,
                    y - height // 2 - pad_y,
                    x + width // 2 + pad_x,
                    y + height // 2 + pad_y
                )

                
                nest_ids[nest_id] = bbox
        
        return nest_ids
    
    def _save_frame(self, video_path: str, frame, frame_number: int) -> None:
        """Save a frame as an image file with proper error handling.
        
        Args:
            video_path: Path to video file
            frame: Frame to save (numpy array)
            frame_number: Frame number for filename
        """
        # CRITICAL: Validate frame before attempting to save
        if frame is None:
            logger.warning(f"Cannot save frame {frame_number}: frame is None")
            return
        
        if not hasattr(frame, 'shape'):
            logger.warning(f"Cannot save frame {frame_number}: frame has no shape attribute")
            return
        
        if frame.size == 0:
            logger.warning(f"Cannot save frame {frame_number}: frame is empty")
            return
        
        try:
            output_folder = video_path.replace('.mp4', '_frames')
            filename = f"{output_folder}_frame_{frame_number:05d}.png"
            
            # Ensure parent directory exists
            from pathlib import Path
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            
            # Attempt to save
            success = cv2.imwrite(filename, frame)
            
            if success:
                logger.debug(f"Saved frame to {filename}")
            else:
                logger.warning(f"cv2.imwrite returned False for {filename}")
        
        except Exception as e:
            logger.error(f"Error saving frame {frame_number}: {e}")
            # Don't raise exception - just log and continue
    
    def _save_visualization(
        self,
        video_path: str,
        nest_ids: Dict,
        res_height: int,
        res_width: int
    ) -> None:
        """Save visualization of detected nests with IDs overlaid on frame.
        
        Args:
            video_path: Path to video file
            nest_ids: Dictionary with 'hotel' and 'nests' keys
            res_height: Frame height
            res_width: Frame width
        """


        # load a frame from the video. Try the first frame if that doesn't work loop for 10 
        
        cap = cv2.VideoCapture(video_path)
        frame = None
        for i in range(100):
            success, frame = cap.read()
            if success:
                frame = cv2.resize(frame, (res_width, res_height))
                break
        cap.release()
        
        try:
            if frame is None or frame.size == 0:
                logger.warning("Cannot save visualization: frame is empty or None")
                return
            
            # Draw hotel ROI
            if 'hotel' in nest_ids:
                hx, hy, hx1, hy2 = nest_ids['hotel']
                cv2.rectangle(
                    frame,
                    (int(hx), int(hy)),
                    (int(hx1), int(hy2)),
                    (255, 0, 0),  # Blue for hotel boundary
                    3
                )
                cv2.putText(
                    frame,
                    "Hotel ROI",
                    (int(hx), int(hy) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2
                )
            
            # Draw nest IDs and bounding boxes
            if 'nests' in nest_ids:
                for nest_id, bbox in nest_ids["nests"].items():
                    x1, y1, x2, y2 = bbox
                    
                    # Draw rectangle
                    cv2.rectangle(
                        frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 255, 0),  # Green for nests
                        2
                    )
                    
                    # Draw nest ID
                    cv2.putText(
                        frame,
                        nest_id,
                        (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )
            
            # Save annotated frame
            output_folder = self.config.output.base_folder
            file = video_path.split("/")[-1].replace('.mp4', '.png')
            filename = f"{output_folder}/nests_visualization_{file}"
            
            # Ensure parent directory exists
            from pathlib import Path
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            
            success = cv2.imwrite(filename, frame)
            
            if success:
                logger.info(f"Saved nest visualization to {filename}")
            else:
                logger.warning(f"Failed to save visualization to {filename}")
        
        except Exception as e:
            logger.warning(f"Could not save visualization: {e}")