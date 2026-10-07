"""Standard trackers behind BeeTracker's surface (memory/43).

Two insects cross the frame with one empty frame in between; every tracker
must keep two stable ids, carry each detection's class, and present the
attributes crops / species / the CSV read.
"""

import unittest

import numpy as np

from beemonitor.tracking.mot.external import TRACKERS, ExternalTracker


def _rows(f):
    return [[10 + f * 5, 100, 50 + f * 5, 140, 0.9, "yolo", "bee"],
            [400 - f * 5, 300, 440 - f * 5, 340, 0.85, "yolo", "wasp"]]


class ExternalTrackerTests(unittest.TestCase):
    def _run(self, kind, frames=30, gap=(15,)):
        t = ExternalTracker(kind, {}, fps=20, frame_size=(640, 480))
        t.frame = np.zeros((480, 640, 3), np.uint8)
        ids = set()
        for f in range(frames):
            out = t.update([] if f in gap else _rows(f), f)
            ids |= {o["track_id"] for o in out}
        return t, ids

    def test_every_tracker_keeps_two_ids_through_a_gap(self):
        for kind in TRACKERS:
            with self.subTest(kind):
                t, ids = self._run(kind)
                self.assertEqual(len(ids), 2, ids)
                self.assertEqual({v.taxon for v in t.tracks}, {"bee", "wasp"})

    def test_tracks_look_like_beetracker_tracks(self):
        t, _ = self._run("bytetrack", frames=5, gap=())
        view = t.tracks[0]
        for attr in ("id", "is_confirmed", "time_since_update", "last_bbox",
                     "history", "taxon", "last_confidence", "last_source", "set_bee_id"):
            self.assertTrue(hasattr(view, attr), attr)
        self.assertEqual(view.time_since_update, 0)
        row = t.get_active_tracks()[0]
        for key in ("track_id", "x1", "y1", "x2", "y2", "cx", "cy", "taxon", "history"):
            self.assertIn(key, row)

    def test_settings_override_defaults_and_unknown_keys_are_ignored(self):
        t = ExternalTracker("ocsort", {"max_age": 5, "nonsense": 1}, fps=20)
        self.assertEqual(t.params["max_age"], 5)
        self.assertNotIn("nonsense", t.params)

    def test_unknown_tracker_is_refused(self):
        with self.assertRaises(ValueError):
            ExternalTracker("deepsort")

    def test_beetracking_builds_the_chosen_tracker(self):
        from beemonitor.tracking.bee_tracking import BeeTracking
        bt = object.__new__(BeeTracking)   # skip __init__ (loads YOLO weights)
        bt.tracker_kind, bt.tracker_options = "ocsort", {"min_hits": 1}
        bt.video_width, bt.tracker_params, bt.lookback_seconds = 640, {}, 0.5
        bt._initialize_tracker(20.0, 480)
        self.assertIsInstance(bt.tracker, ExternalTracker)
        self.assertEqual(bt.tracker.params["min_hits"], 1)


if __name__ == "__main__":
    unittest.main()


class SecondsSettingsTests(unittest.TestCase):
    def test_seconds_become_this_clips_frames(self):
        from beemonitor.tracking.mot.external import _frames_from_seconds
        self.assertEqual(_frames_from_seconds("bytetrack", {"track_buffer_seconds": 1.2}, 25),
                         {"track_buffer": 30})
        self.assertEqual(_frames_from_seconds("ocsort", {"max_age_seconds": 2, "min_hits_seconds": 0.12,
                                                         "delta_t_seconds": 0.12}, 30),
                         {"max_age": 60, "min_hits": 4, "delta_t": 4})
        # Never zero frames: 0 would drop a track the first frame it is missed.
        self.assertEqual(_frames_from_seconds("ocsort", {"min_hits_seconds": 0}, 25), {"min_hits": 1})
