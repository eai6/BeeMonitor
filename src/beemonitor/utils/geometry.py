# """Geometry utilities for bounding box operations and spatial calculations."""

# import numpy as np
# from typing import Tuple, List

# # Type aliases for better readability
# BBox = Tuple[float, float, float, float]  # (x1, y1, x2, y2)
# Point = Tuple[float, float]  # (x, y)


# def compute_centroid(bbox: BBox) -> Point:
#     """Compute the centroid of a bounding box.
    
#     Args:
#         bbox: Bounding box in format (x1, y1, x2, y2)
        
#     Returns:
#         Centroid coordinates (x, y)
        
#     Example:
#         >>> bbox = (10, 20, 30, 40)
#         >>> compute_centroid(bbox)
#         (20.0, 30.0)
#     """
#     x1, y1, x2, y2 = bbox
#     return ((x1 + x2) / 2, (y1 + y2) / 2)


# def compute_iou(box1: BBox, box2: BBox) -> float:
#     """Compute Intersection over Union (IoU) between two bounding boxes.
    
#     Args:
#         box1: First bounding box (x1, y1, x2, y2)
#         box2: Second bounding box (x1, y1, x2, y2)
        
#     Returns:
#         IoU value between 0 and 1
        
#     Example:
#         >>> box1 = (0, 0, 10, 10)
#         >>> box2 = (5, 5, 15, 15)
#         >>> iou = compute_iou(box1, box2)
#         >>> 0.0 < iou < 1.0
#         True
#     """
#     # Calculate intersection coordinates
#     x1 = max(box1[0], box2[0])
#     y1 = max(box1[1], box2[1])
#     x2 = min(box1[2], box2[2])
#     y2 = min(box1[3], box2[3])

#     # Calculate intersection area
#     intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
#     # Calculate union area
#     box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
#     box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
#     union_area = box1_area + box2_area - intersection_area
    
#     # Calculate IoU
#     iou = intersection_area / union_area if union_area != 0 else 0
#     return iou


# def euclidean_distance(point1: Point, point2: Point) -> float:
#     """Calculate Euclidean distance between two points.
    
#     Args:
#         point1: First point (x, y)
#         point2: Second point (x, y)
        
#     Returns:
#         Euclidean distance
        
#     Example:
#         >>> p1 = (0, 0)
#         >>> p2 = (3, 4)
#         >>> euclidean_distance(p1, p2)
#         5.0
#     """
#     return np.linalg.norm(np.array(point1) - np.array(point2))


# def is_inside_bbox(point: Point, bbox: BBox, padding: int = 0) -> bool:
#     """Check if a point is inside a bounding box with optional padding.
    
#     Args:
#         point: Point coordinates (x, y)
#         bbox: Bounding box (x_min, y_min, x_max, y_max)
#         padding: Padding to add around bbox (default: 0)
        
#     Returns:
#         True if point is inside (padded) bbox, False otherwise
        
#     Example:
#         >>> point = (15, 15)
#         >>> bbox = (10, 10, 20, 20)
#         >>> is_inside_bbox(point, bbox)
#         True
#         >>> is_inside_bbox(point, bbox, padding=-10)
#         False
#     """
#     x, y = point
#     x_min, y_min, x_max, y_max = bbox
    
#     # Apply padding
#     x_min -= padding
#     y_min -= padding
#     x_max += padding
#     y_max += padding
    
#     return x_min <= x <= x_max and y_min <= y <= y_max


# def expand_bbox(bbox: BBox, padding_x: int = 0, padding_y: int = 0) -> BBox:
#     """Expand a bounding box by adding padding.
    
#     Args:
#         bbox: Original bounding box (x1, y1, x2, y2)
#         padding_x: Horizontal padding to add
#         padding_y: Vertical padding to add
        
#     Returns:
#         Expanded bounding box
        
#     Example:
#         >>> bbox = (10, 10, 20, 20)
#         >>> expand_bbox(bbox, padding_x=5, padding_y=5)
#         (5, 5, 25, 25)
#     """
#     x1, y1, x2, y2 = bbox
#     return (x1 - padding_x, y1 - padding_y, x2 + padding_x, y2 + padding_y)


# def clip_bbox(bbox: BBox, max_width: int, max_height: int) -> BBox:
#     """Clip bounding box coordinates to image boundaries.
    
#     Args:
#         bbox: Bounding box (x1, y1, x2, y2)
#         max_width: Maximum width (image width)
#         max_height: Maximum height (image height)
        
#     Returns:
#         Clipped bounding box
        
#     Example:
#         >>> bbox = (-5, -5, 1300, 800)
#         >>> clip_bbox(bbox, max_width=1280, max_height=720)
#         (0, 0, 1280, 720)
#     """
#     x1, y1, x2, y2 = bbox
#     x1 = max(0, min(x1, max_width))
#     y1 = max(0, min(y1, max_height))
#     x2 = max(0, min(x2, max_width))
#     y2 = max(0, min(y2, max_height))
#     return (x1, y1, x2, y2)


# def bbox_area(bbox: BBox) -> float:
#     """Calculate the area of a bounding box.
    
#     Args:
#         bbox: Bounding box (x1, y1, x2, y2)
        
#     Returns:
#         Area of the bounding box
        
#     Example:
#         >>> bbox = (0, 0, 10, 20)
#         >>> bbox_area(bbox)
#         200.0
#     """
#     x1, y1, x2, y2 = bbox
#     return (x2 - x1) * (y2 - y1)


# def aspect_ratio(bbox: BBox) -> float:
#     """Calculate aspect ratio of a bounding box.
    
#     Args:
#         bbox: Bounding box (x1, y1, x2, y2)
        
#     Returns:
#         Aspect ratio (width / height)
        
#     Example:
#         >>> bbox = (0, 0, 20, 10)
#         >>> aspect_ratio(bbox)
#         2.0
#     """
#     x1, y1, x2, y2 = bbox
#     width = x2 - x1
#     height = y2 - y1
#     return width / height if height != 0 else 0


# def xywh_to_xyxy(bbox_xywh: Tuple[float, float, float, float]) -> BBox:
#     """Convert bounding box from (x_center, y_center, width, height) to (x1, y1, x2, y2).
    
#     Args:
#         bbox_xywh: Bounding box in xywh format
        
#     Returns:
#         Bounding box in xyxy format
        
#     Example:
#         >>> bbox_xywh = (10, 10, 20, 20)
#         >>> xywh_to_xyxy(bbox_xywh)
#         (0.0, 0.0, 20.0, 20.0)
#     """
#     x, y, w, h = bbox_xywh
#     x1 = x - w / 2
#     y1 = y - h / 2
#     x2 = x + w / 2
#     y2 = y + h / 2
#     return (x1, y1, x2, y2)


# def xyxy_to_xywh(bbox_xyxy: BBox) -> Tuple[float, float, float, float]:
#     """Convert bounding box from (x1, y1, x2, y2) to (x_center, y_center, width, height).
    
#     Args:
#         bbox_xyxy: Bounding box in xyxy format
        
#     Returns:
#         Bounding box in xywh format
        
#     Example:
#         >>> bbox_xyxy = (0, 0, 20, 20)
#         >>> xyxy_to_xywh(bbox_xyxy)
#         (10.0, 10.0, 20.0, 20.0)
#     """
#     x1, y1, x2, y2 = bbox_xyxy
#     x = (x1 + x2) / 2
#     y = (y1 + y2) / 2
#     w = x2 - x1
#     h = y2 - y1
#     return (x, y, w, h)


# def remove_overlapping_points(points: List[Point], threshold: float = 20) -> List[Point]:
#     """Remove points that are too close to each other.
    
#     Keeps the first point from each group of overlapping points.
    
#     Args:
#         points: List of points (x, y)
#         threshold: Minimum distance between points
        
#     Returns:
#         Filtered list of points
        
#     Example:
#         >>> points = [(0, 0), (5, 5), (100, 100)]
#         >>> filtered = remove_overlapping_points(points, threshold=10)
#         >>> len(filtered)
#         2
#     """
#     if not points:
#         return []
    
#     points_array = np.array(points)
#     keep_indices = []
    
#     for i in range(len(points_array)):
#         keep = True
#         for j in keep_indices:
#             distance = np.linalg.norm(points_array[i] - points_array[j])
#             if distance < threshold:
#                 keep = False
#                 break
#         if keep:
#             keep_indices.append(i)
    
#     return points_array[keep_indices].tolist()


# def calculate_distance_matrix(points1: List[Point], points2: List[Point]) -> np.ndarray:
#     """Calculate pairwise distance matrix between two sets of points.
    
#     Args:
#         points1: First set of points
#         points2: Second set of points
        
#     Returns:
#         Distance matrix of shape (len(points1), len(points2))
        
#     Example:
#         >>> p1 = [(0, 0), (1, 1)]
#         >>> p2 = [(2, 2), (3, 3)]
#         >>> dist_matrix = calculate_distance_matrix(p1, p2)
#         >>> dist_matrix.shape
#         (2, 2)
#     """
#     if not points1 or not points2:
#         return np.array([])
    
#     arr1 = np.array(points1)
#     arr2 = np.array(points2)
    
#     # Calculate pairwise distances using broadcasting
#     diff = arr1[:, np.newaxis, :] - arr2[np.newaxis, :, :]
#     distances = np.sqrt(np.sum(diff ** 2, axis=2))
    
#     return distances





"""Geometry utility functions for BeeMonitor.

This module provides geometric operations used throughout the system,
including point manipulation, distance calculations, and bounding box operations.
"""

import numpy as np
from typing import List, Tuple


# Type aliases
Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]


def remove_overlapping_points(
    points: List[Point],
    threshold: float = 20
) -> List[Point]:
    """Remove points that are too close to each other.
    
    Keeps the first occurrence of each point and removes subsequent
    points within the threshold distance.
    
    Args:
        points: List of (x, y) points
        threshold: Minimum distance between points (pixels)
        
    Returns:
        Filtered list of points without overlaps
        
    Example:
        >>> points = [(10, 10), (12, 11), (50, 50)]
        >>> filtered = remove_overlapping_points(points, threshold=5)
        >>> # Returns [(10, 10), (50, 50)] since (12, 11) is too close to (10, 10)
    """
    if not points:
        return []
    
    # Convert to numpy array for easier manipulation
    points_array = np.array(points)
    
    # List to store indices of points to keep
    keep_indices = []
    
    # Iterate through each point
    for i in range(len(points_array)):
        keep = True
        
        # Compare with all previously kept points
        for j in keep_indices:
            # Calculate Euclidean distance
            distance = np.linalg.norm(points_array[i] - points_array[j])
            
            if distance < threshold:
                keep = False
                break
        
        if keep:
            keep_indices.append(i)
    
    # Return the filtered points
    return points_array[keep_indices].tolist()


def compute_centroid(bbox: BBox) -> Point:
    """Compute the center point of a bounding box.
    
    Args:
        bbox: Bounding box in format (x1, y1, x2, y2)
        
    Returns:
        Center point as (x, y) tuple
        
    Example:
        >>> bbox = (10, 20, 30, 40)
        >>> centroid = compute_centroid(bbox)
        >>> # Returns (20.0, 30.0)
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def compute_iou(box1: BBox, box2: BBox) -> float:
    """Compute Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        box1: First bounding box (x1, y1, x2, y2)
        box2: Second bounding box (x1, y1, x2, y2)
        
    Returns:
        IoU value between 0.0 and 1.0
        
    Example:
        >>> box1 = (0, 0, 10, 10)
        >>> box2 = (5, 5, 15, 15)
        >>> iou = compute_iou(box1, box2)
        >>> # Returns 0.14... (intersection area / union area)
    """
    # Calculate intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate intersection area
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate areas of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate union area
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area if union_area != 0 else 0
    
    return iou


def euclidean_distance(point1: Point, point2: Point) -> float:
    """Calculate Euclidean distance between two points.
    
    Args:
        point1: First point as (x, y)
        point2: Second point as (x, y)
        
    Returns:
        Euclidean distance
        
    Example:
        >>> p1 = (0, 0)
        >>> p2 = (3, 4)
        >>> distance = euclidean_distance(p1, p2)
        >>> # Returns 5.0
    """
    return np.linalg.norm(np.array(point1) - np.array(point2))


def bbox_from_center(
    center: Point,
    width: float,
    height: float
) -> BBox:
    """Create a bounding box from center point and dimensions.
    
    Args:
        center: Center point as (x, y)
        width: Box width
        height: Box height
        
    Returns:
        Bounding box as (x1, y1, x2, y2)
        
    Example:
        >>> center = (50, 50)
        >>> bbox = bbox_from_center(center, 20, 10)
        >>> # Returns (40, 45, 60, 55)
    """
    x, y = center
    x1 = x - width / 2
    y1 = y - height / 2
    x2 = x + width / 2
    y2 = y + height / 2
    
    return (x1, y1, x2, y2)


def bbox_area(bbox: BBox) -> float:
    """Calculate the area of a bounding box.
    
    Args:
        bbox: Bounding box as (x1, y1, x2, y2)
        
    Returns:
        Area in square pixels
        
    Example:
        >>> bbox = (0, 0, 10, 20)
        >>> area = bbox_area(bbox)
        >>> # Returns 200.0
    """
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)


def expand_bbox(
    bbox: BBox,
    padding: float
) -> BBox:
    """Expand a bounding box by adding padding on all sides.
    
    Args:
        bbox: Bounding box as (x1, y1, x2, y2)
        padding: Pixels to add on each side
        
    Returns:
        Expanded bounding box
        
    Example:
        >>> bbox = (10, 10, 20, 20)
        >>> expanded = expand_bbox(bbox, 5)
        >>> # Returns (5, 5, 25, 25)
    """
    x1, y1, x2, y2 = bbox
    return (
        x1 - padding,
        y1 - padding,
        x2 + padding,
        y2 + padding
    )


def is_point_in_bbox(
    point: Point,
    bbox: BBox,
    padding: float = 0
) -> bool:
    """Check if a point is inside a bounding box.
    
    Args:
        point: Point as (x, y)
        bbox: Bounding box as (x1, y1, x2, y2)
        padding: Optional padding to add to bbox before checking
        
    Returns:
        True if point is inside (possibly expanded) bbox
        
    Example:
        >>> point = (15, 15)
        >>> bbox = (10, 10, 20, 20)
        >>> is_inside = is_point_in_bbox(point, bbox)
        >>> # Returns True
    """
    x, y = point
    x1, y1, x2, y2 = bbox
    
    # Apply padding
    x1 -= padding
    y1 -= padding
    x2 += padding
    y2 += padding
    
    return x1 <= x <= x2 and y1 <= y <= y2


def clip_bbox_to_frame(
    bbox: BBox,
    frame_width: int,
    frame_height: int
) -> BBox:
    """Clip a bounding box to fit within frame boundaries.
    
    Args:
        bbox: Bounding box as (x1, y1, x2, y2)
        frame_width: Width of frame in pixels
        frame_height: Height of frame in pixels
        
    Returns:
        Clipped bounding box
        
    Example:
        >>> bbox = (-5, -5, 1300, 800)
        >>> clipped = clip_bbox_to_frame(bbox, 1280, 720)
        >>> # Returns (0, 0, 1280, 720)
    """
    x1, y1, x2, y2 = bbox
    
    x1 = max(0, min(x1, frame_width))
    y1 = max(0, min(y1, frame_height))
    x2 = max(0, min(x2, frame_width))
    y2 = max(0, min(y2, frame_height))
    
    return (x1, y1, x2, y2)


def bbox_overlap_area(box1: BBox, box2: BBox) -> float:
    """Calculate the overlapping area between two bounding boxes.
    
    Args:
        box1: First bounding box (x1, y1, x2, y2)
        box2: Second bounding box (x1, y1, x2, y2)
        
    Returns:
        Overlapping area in square pixels
        
    Example:
        >>> box1 = (0, 0, 10, 10)
        >>> box2 = (5, 5, 15, 15)
        >>> overlap = bbox_overlap_area(box1, box2)
        >>> # Returns 25.0
    """
    # Calculate intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate and return intersection area
    return max(0, x2 - x1) * max(0, y2 - y1)


def average_point(points: List[Point]) -> Point:
    """Calculate the average (mean) point from a list of points.
    
    Args:
        points: List of (x, y) points
        
    Returns:
        Average point as (x, y)
        
    Example:
        >>> points = [(0, 0), (10, 10), (20, 20)]
        >>> avg = average_point(points)
        >>> # Returns (10.0, 10.0)
    """
    if not points:
        return (0, 0)
    
    points_array = np.array(points)
    mean_point = np.mean(points_array, axis=0)
    
    return tuple(mean_point.tolist())


def filter_points_by_distance(
    points: List[Point],
    reference_point: Point,
    max_distance: float
) -> List[Point]:
    """Filter points within a maximum distance from a reference point.
    
    Args:
        points: List of (x, y) points to filter
        reference_point: Reference point as (x, y)
        max_distance: Maximum allowed distance
        
    Returns:
        Filtered list of points within max_distance
        
    Example:
        >>> points = [(0, 0), (5, 5), (100, 100)]
        >>> ref = (0, 0)
        >>> filtered = filter_points_by_distance(points, ref, 10)
        >>> # Returns [(0, 0), (5, 5)]
    """
    filtered = []
    
    for point in points:
        if euclidean_distance(point, reference_point) <= max_distance:
            filtered.append(point)
    
    return filtered


def rotate_point(
    point: Point,
    angle: float,
    center: Point = (0, 0)
) -> Point:
    """Rotate a point around a center by a given angle.
    
    Args:
        point: Point to rotate as (x, y)
        angle: Rotation angle in radians (positive = counter-clockwise)
        center: Center of rotation as (x, y), default (0, 0)
        
    Returns:
        Rotated point as (x, y)
        
    Example:
        >>> import math
        >>> point = (1, 0)
        >>> rotated = rotate_point(point, math.pi / 2)  # Rotate 90 degrees
        >>> # Returns approximately (0, 1)
    """
    x, y = point
    cx, cy = center
    
    # Translate to origin
    x -= cx
    y -= cy
    
    # Rotate
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    
    x_new = x * cos_angle - y * sin_angle
    y_new = x * sin_angle + y * cos_angle
    
    # Translate back
    x_new += cx
    y_new += cy
    
    return (x_new, y_new)
