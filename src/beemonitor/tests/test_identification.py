"""Colour-mark identification.

Synthetic bees: a dull brown blob (roughly bee cuticle) with a saturated paint
dot on it. The decoder has to find the dot, name its colour, and — just as
importantly — say nothing when there is no dot, because a confident wrong ID is
worse than no ID (the tracker's set_bee_id keeps the first confident reading).
"""

import unittest

import cv2
import numpy as np

from beemonitor.identification import (
    QUEEN_MARKING_PALETTE, BeeIdentifierManager, ColorIdentifier,
    build_identifier, crop_bbox, hue_distance,
)

# BGR paint colours, matching the queen-marking palette names.
PAINT = {
    "red": (0, 0, 220),
    "yellow": (0, 220, 230),
    "green": (0, 190, 0),
    "blue": (220, 60, 0),
    "white": (250, 250, 250),
}
BEE_BROWN = (35, 45, 70)   # dull, low-saturation cuticle


def bee(color=None, size=60, dot_radius=8, bg=BEE_BROWN):
    """A crop of a bee, optionally wearing a paint dot."""
    img = np.zeros((size, size, 3), np.uint8)
    img[:] = (20, 20, 20)                      # dark background
    cv2.ellipse(img, (size // 2, size // 2), (size // 3, size // 4),
                0, 0, 360, bg, -1)             # the bee
    if color is not None:
        cv2.circle(img, (size // 2, size // 2), dot_radius, PAINT[color], -1)
    return img


class CropBboxTests(unittest.TestCase):
    def test_none_bbox_returns_whole_frame(self):
        frame = bee("red")
        self.assertIs(crop_bbox(frame, None), frame)

    def test_out_of_bounds_bbox_is_clamped(self):
        frame = bee("red", size=40)
        region = crop_bbox(frame, (-50, -50, 500, 500))
        self.assertEqual(region.shape[:2], (40, 40))

    def test_inverted_bbox_is_normalised(self):
        frame = bee("red", size=40)
        region = crop_bbox(frame, (30, 30, 10, 10))
        self.assertEqual(region.shape[:2], (20, 20))

    def test_empty_region_returns_none(self):
        self.assertIsNone(crop_bbox(bee("red"), (10, 10, 10, 10)))
        self.assertIsNone(crop_bbox(np.zeros((0, 0, 3), np.uint8), None))


class ColorIdentifierTests(unittest.TestCase):
    def setUp(self):
        self.identifier = ColorIdentifier()

    def test_reads_each_queen_marking_colour(self):
        for name in PAINT:
            with self.subTest(colour=name):
                result = self.identifier.identify(bee(name))
                self.assertIsNotNone(result, f"{name} not detected")
                read, method, confidence = result
                self.assertEqual(read, name)
                self.assertEqual(method, "color")
                self.assertGreater(confidence, 0.35)

    def test_unmarked_bee_reads_as_nothing(self):
        """The important negative: no dot must mean no ID, not a guess."""
        self.assertIsNone(self.identifier.identify(bee(None)))

    def test_speck_below_min_area_is_ignored(self):
        self.assertIsNone(self.identifier.identify(bee("red", dot_radius=1)))

    def test_bbox_is_honoured_on_a_full_frame(self):
        """The tracker path: one frame, several bees, identify one by bbox."""
        frame = np.zeros((80, 160, 3), np.uint8)
        frame[0:80, 0:80] = bee("green", size=80)
        frame[0:80, 80:160] = bee("blue", size=80)

        left = self.identifier.identify(frame, (0, 0, 80, 80))
        right = self.identifier.identify(frame, (80, 0, 160, 80))

        self.assertEqual(left[0], "green")
        self.assertEqual(right[0], "blue")

    def test_bigger_mark_reads_more_confidently(self):
        small = self.identifier.identify(bee("green", dot_radius=4))
        large = self.identifier.identify(bee("green", dot_radius=12))
        self.assertGreater(large[2], small[2])

    def test_never_raises_on_degenerate_input(self):
        for frame in (None,
                      np.zeros((0, 0, 3), np.uint8),
                      np.zeros((10, 10), np.uint8),        # greyscale
                      np.zeros((10, 10, 4), np.uint8)):    # BGRA
            self.assertIsNone(self.identifier.identify(frame))

    def test_min_confidence_gates_the_reading(self):
        strict = ColorIdentifier(min_confidence=0.99)
        self.assertIsNone(strict.identify(bee("red")))

    def test_off_palette_hue_is_refused(self):
        """A colour nothing in the palette is near must not snap to a neighbour."""
        narrow = ColorIdentifier(palette={"blue": 110.0}, max_hue_distance=10.0)
        self.assertIsNone(narrow.identify(bee("yellow")))

    def test_white_can_be_disabled(self):
        no_white = ColorIdentifier(detect_white=False)
        self.assertIsNone(no_white.identify(bee("white")))


class HueDistanceTests(unittest.TestCase):
    def test_wraps_around_the_circle(self):
        self.assertEqual(hue_distance(0, 179), 1.0)
        self.assertEqual(hue_distance(175, 5), 10.0)
        self.assertEqual(hue_distance(30, 30), 0.0)

    def test_red_either_side_of_the_wrap_matches_red(self):
        identifier = ColorIdentifier()
        self.assertEqual(identifier.identify(bee("red"))[0], "red")
        self.assertLess(hue_distance(178, QUEEN_MARKING_PALETTE["red"]), 5)


class ManagerTests(unittest.TestCase):
    def test_picks_the_most_confident_reading(self):
        class Stub:
            def __init__(self, result):
                self.result = result

            def identify(self, frame, bbox=None):
                return self.result

        manager = BeeIdentifierManager([
            Stub(("a", "color", 0.4)), Stub(("b", "number", 0.9)),
        ])
        self.assertEqual(manager.identify(bee("red")), ("b", "number", 0.9))

    def test_a_raising_decoder_does_not_sink_the_others(self):
        class Boom:
            def identify(self, frame, bbox=None):
                raise RuntimeError("bad decoder")

        class Fine:
            def identify(self, frame, bbox=None):
                return ("ok", "color", 0.8)

        manager = BeeIdentifierManager([Boom(), Fine()])
        self.assertEqual(manager.identify(bee("red"))[0], "ok")

    def test_min_confidence_filters(self):
        class Stub:
            def identify(self, frame, bbox=None):
                return ("a", "color", 0.2)

        self.assertIsNone(BeeIdentifierManager([Stub()], min_confidence=0.5)
                          .identify(bee("red")))


class BuildIdentifierTests(unittest.TestCase):
    def test_auto_and_color_are_supported(self):
        self.assertIsInstance(build_identifier("auto"), BeeIdentifierManager)
        self.assertIsInstance(build_identifier("color"), ColorIdentifier)

    def test_unimplemented_marker_types_return_none(self):
        """Tag/QR decoding isn't written yet — say so by returning nothing
        rather than silently handing back a colour decoder."""
        self.assertIsNone(build_identifier("number"))
        self.assertIsNone(build_identifier("qr"))

    def test_satisfies_the_trackers_hook_contract(self):
        """(bee_id, method, confidence), positionally unpackable."""
        result = build_identifier("auto").identify(bee("yellow"))
        bee_id, method, confidence = result
        self.assertIsInstance(bee_id, str)
        self.assertIsInstance(method, str)
        self.assertIsInstance(confidence, float)


if __name__ == "__main__":
    unittest.main()
