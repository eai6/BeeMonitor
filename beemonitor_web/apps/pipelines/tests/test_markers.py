"""Marker identification from the per-track crops already stored in S3.

The behaviour that earns its keep here is **voting**: the tracker's own hook
keeps the first confident reading forever, so a single misread would brand a bee
for its whole trajectory. Decoding every crop and taking the majority is what
makes one bad frame survivable.
"""

from unittest.mock import patch

import cv2
import numpy as np
from django.test import SimpleTestCase

from apps.pipelines import markers

PAINT = {"red": (0, 0, 220), "green": (0, 190, 0), "blue": (220, 60, 0)}


def crop_bytes(color=None, size=60):
    """JPEG bytes of a bee crop, optionally wearing a paint dot."""
    img = np.zeros((size, size, 3), np.uint8)
    img[:] = (20, 20, 20)
    cv2.ellipse(img, (size // 2, size // 2), (size // 3, size // 4),
                0, 0, 360, (35, 45, 70), -1)
    if color is not None:
        cv2.circle(img, (size // 2, size // 2), 9, PAINT[color], -1)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()


class FakeStorage:
    """Serves crop bytes by key; records what was fetched."""

    def __init__(self, blobs):
        self.blobs = blobs
        self.fetched = []

    def download_to_stream(self, container, key, buf):
        self.fetched.append(key)
        if key not in self.blobs:
            raise FileNotFoundError(key)
        buf.write(self.blobs[key])


class CropIndexTests(SimpleTestCase):
    def test_manifest_in_summary_stats_is_preferred(self):
        result = {"summary_stats": {"crops_manifest": {"1": ["a.jpg", "b.jpg"]}}}
        self.assertEqual(markers._load_crop_index(result), {"1": ["a.jpg", "b.jpg"]})

    def test_falls_back_to_the_crops_csv(self):
        rows = "track_id,frame,crop_key\n1,0,a.jpg\n1,5,b.jpg\n2,0,c.jpg\n"
        import pandas as pd

        with patch("apps.pipelines.ops._read_csv",
                   return_value=pd.read_csv(__import__("io").StringIO(rows))):
            index = markers._load_crop_index({"crops_csv_path": "x/track_crops.csv"})

        self.assertEqual(index, {"1": ["a.jpg", "b.jpg"], "2": ["c.jpg"]})

    def test_no_crops_gives_an_empty_index(self):
        self.assertEqual(markers._load_crop_index({}), {})
        self.assertEqual(markers._load_crop_index(None), {})


class IdentifyFromCropsTests(SimpleTestCase):
    def _run(self, manifest, blobs, **kwargs):
        storage = FakeStorage(blobs)
        result = {"summary_stats": {"crops_manifest": manifest}}
        with patch("config.storage.get_s3_client", return_value=storage):
            return markers.identify_from_crops(result, **kwargs), storage

    def test_reads_a_marker_per_track(self):
        manifest = {"1": ["t1a.jpg", "t1b.jpg"], "2": ["t2a.jpg"]}
        blobs = {"t1a.jpg": crop_bytes("green"), "t1b.jpg": crop_bytes("green"),
                 "t2a.jpg": crop_bytes("blue")}

        out, _ = self._run(manifest, blobs)

        self.assertEqual(out["identified_tracks"], 2)
        self.assertEqual(out["unique_markers"], 2)
        by_track = {r["track"]: r for r in out["rows"]}
        self.assertEqual(by_track[1]["marker"], "green")
        self.assertEqual(by_track[2]["marker"], "blue")
        self.assertEqual(by_track[1]["method"], "color")
        self.assertEqual(out["source"], "crops")

    def test_majority_vote_survives_a_single_misread(self):
        """Two greens and one red must resolve to green, not to whichever came
        first — the failure mode the tracker's own hook has."""
        manifest = {"1": ["a.jpg", "b.jpg", "c.jpg"]}
        blobs = {"a.jpg": crop_bytes("red"),      # the odd one out, read first
                 "b.jpg": crop_bytes("green"),
                 "c.jpg": crop_bytes("green")}

        out, _ = self._run(manifest, blobs)

        row = out["rows"][0]
        self.assertEqual(row["marker"], "green")
        self.assertEqual(row["votes"], 2)
        self.assertEqual(row["crops_read"], 3)

    def test_unmarked_tracks_are_omitted_not_guessed(self):
        manifest = {"1": ["a.jpg"], "2": ["b.jpg"]}
        blobs = {"a.jpg": crop_bytes("red"), "b.jpg": crop_bytes(None)}

        out, _ = self._run(manifest, blobs)

        self.assertEqual([r["track"] for r in out["rows"]], [1])

    def test_unreadable_crops_are_skipped(self):
        manifest = {"1": ["missing.jpg", "good.jpg"]}
        blobs = {"good.jpg": crop_bytes("blue")}

        out, _ = self._run(manifest, blobs)

        self.assertEqual(out["rows"][0]["marker"], "blue")
        self.assertEqual(out["rows"][0]["crops_read"], 1)

    def test_crops_per_track_are_capped(self):
        manifest = {"1": [f"c{i}.jpg" for i in range(50)]}
        blobs = {f"c{i}.jpg": crop_bytes("red") for i in range(50)}

        _out, storage = self._run(manifest, blobs, max_crops=5)

        self.assertEqual(len(storage.fetched), 5)

    def test_unsupported_marker_type_returns_none(self):
        """A marker type with no decoder must not silently fall back to colour."""
        out, storage = self._run({"1": ["a.jpg"]}, {"a.jpg": crop_bytes("red")},
                                 marker_type="qr")

        self.assertIsNone(out)
        self.assertEqual(storage.fetched, [])  # no pointless S3 reads

    def test_no_crops_returns_none(self):
        with patch("config.storage.get_s3_client", return_value=FakeStorage({})):
            self.assertIsNone(markers.identify_from_crops({}))
