"""Unit tests for the MOT layer — Detection, Track, and BeeTracker.

The BeeTracking-level tests that used to live here exercised the v2.1 detection
modes (FGBG_ONLY / SIFT_ONLY / FGBG_SIFT) and a ``process_video(roi=...)``
signature, all removed in v2.2 "YOLO-only". They imported a ``DetectionMode``
that no longer exists, so this whole module failed to COLLECT — taking every
other test in the package down with it, unnoticed, because CI ran neither suite.
Deleted rather than ported: there is nothing left for them to test.
"""

import unittest
import numpy as np

from beemonitor.tracking.mot import BeeTracker, BaseMOT, Detection, Track
from beemonitor.core.config import Config


class TestMOTDetection(unittest.TestCase):
    """Test MOT Detection data class."""
    
    def test_detection_creation(self):
        """Test creating Detection for MOT."""
        det = Detection(
            bbox=(100, 100, 200, 200),
            centroid=(150, 150),
            label='bee',
            confidence=0.9,
            source='test'
        )
        
        self.assertEqual(det.bbox, (100, 100, 200, 200))
        self.assertEqual(det.centroid, (150, 150))
        self.assertEqual(det.label, 'bee')


class TestTrack(unittest.TestCase):
    """Test Track data class."""
    
    def test_track_creation(self):
        """Test creating Track object."""
        track = Track(
            track_id=1,
            bbox=(100, 100, 200, 200),
            centroid=(150, 150),
            label='bee',
            age=5,
            frames_without_detection=0,
            last_confirmation_frame=10,
            trajectory=[(10, (150, 150))]
        )
        
        self.assertEqual(track.track_id, 1)
        self.assertEqual(track.label, 'bee')
        self.assertEqual(track.age, 5)


class TestBeeTracker(unittest.TestCase):
    """BeeTracker against its CURRENT API.

    What was here tested ``predict()``, ``reset()``, ``get_tracks()`` and
    ``BeeTracker(config=..., tracking_classes=[...])`` — none of which exist:
    the class takes adaptive fps/size parameters and exposes ``update`` +
    ``get_active_tracks``. It also fed ``Detection`` objects, where the tracker
    actually consumes positional rows ``[x1, y1, x2, y2, conf, source, taxon]``
    (the untyped detector->tracker seam).
    """

    # The one format the tracker consumes. Kept here as a named helper so the
    # positional layout has at least one authoritative reference in tests.
    @staticmethod
    def _row(x1, y1, x2, y2, conf=0.9, source="yolo", taxon="bee"):
        return [x1, y1, x2, y2, conf, source, taxon]

    def setUp(self):
        self.tracker = BeeTracker(fps=30.0, bee_size=50.0)

    def test_a_persistent_detection_becomes_a_confirmed_track(self):
        # min_hits_seconds defaults to 0.1s => ~3 frames at 30 fps.
        for frame_num in range(10):
            drift = frame_num  # a slow, trackable walk
            self.tracker.update(
                [self._row(100 + drift, 100, 150 + drift, 150)], frame_num=frame_num)

        tracks = self.tracker.get_active_tracks()
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0]["taxon"], "bee")
        self.assertGreater(tracks[0]["track_id"], 0)

    def test_two_separated_detections_track_separately(self):
        for frame_num in range(10):
            self.tracker.update(
                [self._row(100, 100, 150, 150), self._row(400, 400, 450, 450)],
                frame_num=frame_num)

        self.assertEqual(len({t["track_id"] for t in self.tracker.get_active_tracks()}), 2)

    def test_a_single_frame_blip_is_not_confirmed(self):
        self.tracker.update([self._row(100, 100, 150, 150)], frame_num=0)

        self.assertEqual(self.tracker.get_active_tracks(), [])


if __name__ == "__main__":
    unittest.main()
