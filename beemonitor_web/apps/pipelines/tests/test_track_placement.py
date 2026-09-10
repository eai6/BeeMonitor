"""Tracks are placed against the FRAME, never against themselves.

`normalized_tracks` fell back to the observed maximum when no frame size was
known, which rescales every clip by its own activity. A bee at (700, 620) in a
1920x1080 frame is at (0.365, 0.574) — the top-left flower — but divided by the
extent of its own track it lands at (0.974, 0.970), the bottom-right one.

References are stored in true frame fractions, so every episode was attributed
to whichever box the distortion happened to reach. That is why a clip whose
activity was plainly in flower 1 reported every interaction against nest_4.
"""

import pandas as pd
from django.test import SimpleTestCase

from apps.pipelines import ops

# Four references, one per quadrant.
REFS = [
    {"id": "nest_1", "label": "Nest 1", "box": (0.30, 0.50, 0.45, 0.70), "points": None},
    {"id": "nest_4", "label": "Nest 4", "box": (0.85, 0.85, 1.00, 1.00), "points": None},
]

# A track sitting in flower 1 of a 1920x1080 frame, in pixels.
PIXELS = pd.DataFrame([{"frame": f, "track_id": 1, "cx": 700 + f, "cy": 620}
                       for f in range(20)])


class PlacementTests(SimpleTestCase):
    def test_a_track_lands_in_the_reference_it_is_actually_in(self):
        tidy = ops.normalized_tracks(PIXELS, {"frame_width": 1920, "frame_height": 1080})

        episodes = ops.compute_episodes(tidy, REFS)

        self.assertEqual([e["reference"] for e in episodes], ["nest_1"])

    def test_without_a_frame_size_it_refuses_rather_than_misplacing(self):
        """Scaling by the track's own extent used to send this to nest_4."""
        self.assertIsNone(ops.normalized_tracks(PIXELS, {}))

    def test_the_old_fallback_really_did_reach_the_wrong_box(self):
        # Reproduce it explicitly so the regression is unmistakable.
        scaled = PIXELS.copy()
        scaled["cx"] = scaled["cx"] / scaled["cx"].abs().max()
        scaled["cy"] = scaled["cy"] / scaled["cy"].abs().max()

        tidy = ops.normalized_tracks(scaled, {})
        episodes = ops.compute_episodes(tidy, REFS)

        self.assertEqual([e["reference"] for e in episodes], ["nest_4"])

    def test_coordinates_already_fractional_are_left_alone(self):
        frac = pd.DataFrame([{"frame": 0, "track_id": 1, "cx": 0.36, "cy": 0.57}])

        tidy = ops.normalized_tracks(frac, {})

        self.assertAlmostEqual(float(tidy["x"].iloc[0]), 0.36)

    def test_the_alternative_dimension_spellings_are_honoured(self):
        for keys in ({"width": 1920, "height": 1080},
                     {"res_width": 1920, "res_height": 1080},
                     {"video_width": 1920, "video_height": 1080}):
            tidy = ops.normalized_tracks(PIXELS, keys)
            self.assertIsNotNone(tidy, keys)
            self.assertAlmostEqual(float(tidy["x"].iloc[0]), 700 / 1920, places=4)

    def test_one_axis_fractional_and_one_in_pixels_still_needs_its_dimension(self):
        mixed = pd.DataFrame([{"frame": 0, "track_id": 1, "cx": 0.36, "cy": 620}])

        self.assertIsNone(ops.normalized_tracks(mixed, {}))
        self.assertIsNotNone(ops.normalized_tracks(mixed, {"frame_height": 1080}))
