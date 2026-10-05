"""Species / marker identity by vote over every crop of a track."""

import os
import tempfile
import unittest

import cv2
import numpy as np

from beemonitor.identification.track_vote import track_crop_files, vote_tracks


def _write_crops(root, track_id, n, value=0):
    d = os.path.join(root, "clip", f"track_{track_id:04d}")
    os.makedirs(d, exist_ok=True)
    for f in range(n):
        img = np.full((40, 40, 3), value + f, dtype=np.uint8)
        cv2.imwrite(os.path.join(d, f"frame_{f:06d}.jpg"), img)


class _Species:
    """Labels crops by a script, in order, ignoring pixels."""

    def __init__(self, script):
        self.script = list(script)

    def classify_images(self, images):
        out = []
        for _ in images:
            out.append(self.script.pop(0))
        return out


class _Marker:
    method = "color"

    def __init__(self, label):
        self.label = label

    def identify(self, image):
        return (self.label, "color", 0.8)


class VoteTests(unittest.TestCase):
    def test_finds_every_crop_in_frame_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_crops(tmp, 3, 4)
            files = track_crop_files(tmp)
            self.assertEqual([f for f, _ in files[3]], [0, 1, 2, 3])

    def test_most_votes_wins_and_every_crop_votes(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_crops(tmp, 1, 5)
            species = _Species([("Osmia lignaria", 0.3), ("Osmia lignaria", 0.2),
                                ("Apis mellifera", 0.9), ("Osmia lignaria", 0.25), None])
            tracks, rows = vote_tracks(tmp, species=species)
            t = tracks[1]
            self.assertEqual(t["taxon"], "Osmia lignaria")   # 3 low-confidence votes beat 1 high
            self.assertEqual(t["taxon_votes"], 3)
            self.assertEqual(t["taxon_crops"], 5)
            self.assertEqual(t["taxon_vote_share"], 0.75)    # of the 4 crops that read
            self.assertEqual(len(rows), 5)

    def test_ties_break_on_mean_confidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_crops(tmp, 2, 2)
            species = _Species([("A b", 0.2), ("C d", 0.6)])
            tracks, _ = vote_tracks(tmp, species=species)
            self.assertEqual(tracks[2]["taxon"], "C d")

    def test_markers_vote_in_the_same_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_crops(tmp, 7, 3)
            tracks, rows = vote_tracks(tmp, marker=_Marker("red-blue"))
            self.assertEqual(tracks[7]["marker"], "red-blue")
            self.assertEqual(tracks[7]["marker_votes"], 3)
            self.assertNotIn("taxon", tracks[7])
            self.assertEqual(rows[0]["marker"], "red-blue")

    def test_no_crops_no_tracks(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(vote_tracks(tmp, species=_Species([])), ({}, []))


if __name__ == "__main__":
    unittest.main()
