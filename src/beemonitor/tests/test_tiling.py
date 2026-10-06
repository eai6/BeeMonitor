"""Tiled detection for large photos (memory/45)."""

import unittest

import numpy as np

from beemonitor.detection.tiling import detect_tiled, merge_boxes, tile_grid


class TileGridTests(unittest.TestCase):
    def test_small_photos_are_one_tile(self):
        self.assertEqual(tile_grid(1600, 1200), [(0, 0, 1600, 1200)])

    def test_64mp_is_covered_edge_to_edge(self):
        grid = tile_grid(9152, 6944)
        self.assertTrue(40 <= len(grid) <= 80, len(grid))
        self.assertEqual(max(x + w for x, _y, w, _h in grid), 9152)
        self.assertEqual(max(y + h for _x, y, _w, h in grid), 6944)
        self.assertTrue(all(w <= 1280 and h <= 1280 for _x, _y, w, h in grid))


class MergeTests(unittest.TestCase):
    def box(self, x, y, w, h, cls="bee", conf=0.9):
        return {"x": x, "y": y, "w": w, "h": h, "class": cls, "confidence": conf}

    def test_a_box_cut_by_a_tile_edge_yields_to_the_whole_one(self):
        whole, half = self.box(1000, 100, 80, 60), self.box(1000, 100, 30, 60)
        self.assertEqual(merge_boxes([half, whole]), [whole])

    def test_different_classes_and_separate_boxes_are_kept(self):
        a, b = self.box(0, 0, 50, 50), self.box(0, 0, 50, 50, cls="wasp")
        c = self.box(500, 500, 50, 50)
        self.assertEqual(len(merge_boxes([a, b, c])), 3)


class DetectTiledTests(unittest.TestCase):
    def test_boxes_come_back_in_full_image_pixels_and_dedupe(self):
        image = np.zeros((3000, 3000, 3), np.uint8)
        image[1500:1560, 1500:1560] = 255           # one bright "insect"

        def detect(tile):
            ys, xs = np.nonzero(tile[:, :, 0])
            if not len(xs):
                return []
            return [{"x": int(xs.min()), "y": int(ys.min()), "w": int(xs.max() - xs.min() + 1),
                     "h": int(ys.max() - ys.min() + 1), "class": "bee", "confidence": 0.9}]

        boxes, tiles = detect_tiled(image, detect)
        self.assertGreater(tiles, 1)
        self.assertEqual(len(boxes), 1)
        self.assertEqual((boxes[0]["x"], boxes[0]["y"], boxes[0]["w"]), (1500, 1500, 60))


if __name__ == "__main__":
    unittest.main()
