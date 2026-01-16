"""Tracking module for bee monitoring.

This module provides tracking systems for following objects over time.
Organized into two layers:

1. High-level tracking systems (BaseTracking)
   - BeeTracking: Bee hotel-specific orchestrator

2. Low-level MOT algorithms (BaseMOT)
   - BeeTracker: Custom Kalman-based tracker
   - UltralyticsTracker: ByteTrack/BoT-SORT wrapper

Example:
    >>> from beemonitor.tracking import BeeTracking
    >>> from beemonitor.tracking.mot import BeeTracker
    >>> 
    >>> # Setup MOT
    >>> mot = BeeTracker(config, tracking_classes=['bee'])
    >>> 
    >>> # Create bee hotel tracking system
    >>> tracker = BeeTracking(
    ...     mot_algorithm=mot,
    ...     yolo_model=yolo_model,
    ... )
    >>> 
    >>> # Process video
    >>> results = tracker.process_video('video.mp4')
"""

from beemonitor.tracking.base_tracking import BaseTracking
from beemonitor.tracking.bee_tracking import BeeTracking

__all__ = [
    'BaseTracking',
    'BeeTracking'
]
