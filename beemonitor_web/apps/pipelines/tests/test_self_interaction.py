"""A track is not in proximity to itself.

An exported interactions table carried rows like

    start_frame  a  a_kind    b  b_kind    relation   min_distance
    64           1  organism  1  organism  proximity  0.0002

which is not an interaction at all. The proximity pass skips i >= j so a row
cannot pair with itself, but that only means "different ROWS": the worker's
tracking CSV can carry two rows for the same (frame, track) — two overlapping
detections handed the same id — and those two rows, a few ten-thousandths
apart, paired with each other.

Fixed at both ends: the tidy table now holds one position per track per frame,
and the pairing refuses equal ids whatever the table says.
"""

from django.test import SimpleTestCase

from apps.pipelines import ops

try:
    import pandas as pd
except ImportError:  # pragma: no cover - pandas is in base.txt
    pd = None


def _df(rows):
    return pd.DataFrame(rows)


SUMMARY = {"frame_width": 1920, "frame_height": 1080}


class TidyDedupeTests(SimpleTestCase):
    def setUp(self):
        if pd is None:
            self.skipTest("pandas not installed")

    def test_two_rows_for_one_track_in_one_frame_collapse(self):
        tidy = ops.normalized_tracks(_df([
            {"track_id": 1, "frame": 64, "cx": 0.500, "cy": 0.500},
            {"track_id": 1, "frame": 64, "cx": 0.5002, "cy": 0.5001},
        ]), SUMMARY)

        self.assertEqual(len(tidy), 1)

    def test_the_same_track_in_different_frames_is_kept(self):
        tidy = ops.normalized_tracks(_df([
            {"track_id": 1, "frame": 64, "cx": 0.5, "cy": 0.5},
            {"track_id": 1, "frame": 65, "cx": 0.5, "cy": 0.5},
        ]), SUMMARY)

        self.assertEqual(len(tidy), 2)

    def test_different_tracks_in_one_frame_are_kept(self):
        tidy = ops.normalized_tracks(_df([
            {"track_id": 1, "frame": 64, "cx": 0.5, "cy": 0.5},
            {"track_id": 2, "frame": 64, "cx": 0.5, "cy": 0.5},
        ]), SUMMARY)

        self.assertEqual(len(tidy), 2)


class ProximitySelfPairTests(SimpleTestCase):
    def setUp(self):
        if pd is None:
            self.skipTest("pandas not installed")

    def _episodes(self, rows):
        tidy = ops.normalized_tracks(_df(rows), SUMMARY)
        return ops.compute_proximity_episodes(tidy, radius=0.05, gap_frames=15,
                                              aspect=16 / 9)

    def test_a_duplicated_track_produces_no_episode(self):
        """The exact shape that exported 'track 1 interacting with track 1'."""
        rows = []
        for frame in range(60, 70):
            rows.append({"track_id": 1, "frame": frame, "cx": 0.5, "cy": 0.5})
            rows.append({"track_id": 1, "frame": frame, "cx": 0.5002, "cy": 0.5001})

        self.assertEqual(self._episodes(rows), [])

    def test_two_real_tracks_close_together_still_pair(self):
        """The fix must not cost a genuine organism-to-organism episode."""
        rows = []
        for frame in range(60, 70):
            rows.append({"track_id": 1, "frame": frame, "cx": 0.50, "cy": 0.50})
            rows.append({"track_id": 2, "frame": frame, "cx": 0.51, "cy": 0.50})

        episodes = self._episodes(rows)

        self.assertEqual(len(episodes), 1)
        self.assertEqual({str(episodes[0]["track"]), str(episodes[0]["partner"])},
                         {"1", "2"})

    def test_tracks_far_apart_still_do_not_pair(self):
        rows = []
        for frame in range(60, 70):
            rows.append({"track_id": 1, "frame": frame, "cx": 0.10, "cy": 0.10})
            rows.append({"track_id": 2, "frame": frame, "cx": 0.90, "cy": 0.90})

        self.assertEqual(self._episodes(rows), [])

    def test_equal_ids_are_refused_even_if_the_dedupe_is_bypassed(self):
        """Belt and braces: the pairing must not rely on tidy being clean."""
        tidy = pd.DataFrame([
            {"tid": 1, "frame": 64, "x": 0.5, "y": 0.5},
            {"tid": 1, "frame": 64, "x": 0.5002, "y": 0.5001},
        ])

        self.assertEqual(
            ops.compute_proximity_episodes(tidy, radius=0.05, gap_frames=15), [])

    def test_ids_that_differ_only_by_type_are_still_the_same_track(self):
        """1 and "1" are one track; comparing natives would let them pair."""
        tidy = pd.DataFrame([
            {"tid": 1, "frame": 64, "x": 0.5, "y": 0.5},
            {"tid": "1", "frame": 64, "x": 0.5002, "y": 0.5001},
        ])

        self.assertEqual(
            ops.compute_proximity_episodes(tidy, radius=0.05, gap_frames=15), [])


class ContainmentDoubleCountTests(SimpleTestCase):
    """Duplicated rows also inflated containment, which counts frames."""

    def setUp(self):
        if pd is None:
            self.skipTest("pandas not installed")

    def test_a_duplicated_frame_is_not_counted_twice(self):
        refs = [{"id": "1", "label": "Nest 1",
                 "box": (0.4, 0.4, 0.6, 0.6), "points": None}]
        rows, dupes = [], []
        for frame in range(10):
            rows.append({"track_id": 1, "frame": frame, "cx": 0.5, "cy": 0.5})
            dupes.append({"track_id": 1, "frame": frame, "cx": 0.5, "cy": 0.5})
            dupes.append({"track_id": 1, "frame": frame, "cx": 0.5001, "cy": 0.5})

        clean = ops.compute_episodes(ops.normalized_tracks(_df(rows), SUMMARY),
                                     refs, gap_frames=15)
        doubled = ops.compute_episodes(ops.normalized_tracks(_df(dupes), SUMMARY),
                                       refs, gap_frames=15)

        self.assertEqual([e["frames"] for e in clean],
                         [e["frames"] for e in doubled])
