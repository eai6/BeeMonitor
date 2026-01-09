"""Multiple Object Tracking (MOT) algorithms.

This module provides low-level MOT algorithms for track association.
All MOT algorithms implement the BaseMOT interface.

Available MOT Algorithms:
- BeeTracker: Custom Kalman filter + Hungarian matching
- UltralyticsTracker: ByteTrack/BoT-SORT wrapper

Example:
    >>> from beemonitor.tracking.mot import BeeTracker, UltralyticsTracker
    >>> 
    >>> # Custom Kalman-based tracker
    >>> bee_tracker = BeeTracker(config, tracking_classes=['bee', 'wasp'])
    >>> 
    >>> # Ultralytics tracker
    >>> byte_tracker = UltralyticsTracker(tracker_type='bytetrack')
"""

from beemonitor.tracking.mot.base_mot import BaseMOT, Detection, Track
from beemonitor.tracking.mot.bee_tracker import BeeTracker
from beemonitor.tracking.mot.ultralytics_tracker import UltralyticsTracker

__all__ = [
    'BaseMOT',
    'Detection',
    'Track',
    'BeeTracker',
    'UltralyticsTracker',
]
