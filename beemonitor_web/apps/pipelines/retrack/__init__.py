"""Re-track on the web from saved detections (memory/69).

The GPU worker detects and tracks in one pass. Only detection needs the GPU:
the detections it saved are re-tracked here with the run's current MOT
settings, using the same BeeTracker (``bee_tracker``, vendored), then species
votes, review votes and every count follow the new tracks.
"""
