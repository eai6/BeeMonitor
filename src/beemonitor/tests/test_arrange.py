"""Numbering a model's reference boxes as a grid (memory/83)."""
import math
import unittest

from beemonitor.references.arrange import grid, reading_order


def box(cx, cy, s=14):
    return [cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2]


def centre(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def hotel(rows=6, cols=10, x0=200, y0=80, dx=42, dy=60, tilt=0.0):
    """Centres of a regular hotel, row by row, turned by ``tilt`` radians."""
    out = []
    for r in range(rows):
        for c in range(cols):
            x, y = x0 + c * dx, y0 + r * dy + (60 if r >= 3 else 0)   # a board between the boxes
            out.append((x * math.cos(tilt) - y * math.sin(tilt), x * math.sin(tilt) + y * math.cos(tilt)))
    return out


# The tubes of one natalies clip (the player, 2026-11-16): six rows of ten,
# and three false boxes on the bottom board, between the bottom row's.
NATALIES = [
    (209, 80), (252, 80), (295, 80), (339, 80), (381, 80), (425, 80), (465, 80), (507, 80), (543, 80), (584, 80),
    (206, 140), (250, 140), (293, 140), (338, 140), (379, 140), (424, 140), (465, 140), (503, 140), (546, 140), (583, 140),
    (203, 205), (248, 205), (290, 205), (335, 205), (379, 205), (424, 205), (465, 205), (505, 205), (546, 205), (583, 205),
    (203, 322), (244, 322), (287, 322), (330, 322), (375, 322), (417, 322), (458, 322), (500, 322), (540, 322), (577, 322),
    (201, 385), (243, 385), (285, 385), (329, 385), (373, 385), (416, 385), (457, 385), (495, 385), (540, 385), (573, 385),
    (205, 445), (247, 445), (288, 445), (327, 445), (370, 445), (412, 445), (452, 445), (490, 445), (530, 445), (567, 445),
]
STRAYS = [(226, 449), (268, 449), (309, 449)]


class GridTests(unittest.TestCase):
    def assertAt(self, out, number, xy, within=4):
        cx, cy = centre(out["boxes"][number - 1])
        self.assertLess(abs(cx - xy[0]), within, f"tube {number} x")
        self.assertLess(abs(cy - xy[1]), within, f"tube {number} y")

    def test_numbered_row_by_row_from_top_left(self):
        pts = hotel()
        out = grid([box(*p) for p in reversed(pts)], 6, 10)
        self.assertTrue(out["ok"], out["reason"])
        self.assertEqual(len(out["boxes"]), 60)
        for n, p in enumerate(pts, 1):
            self.assertAt(out, n, p, 0.01)
        self.assertEqual((out["filled"], out["dropped"]), (0, 0))

    def test_natalies_with_three_false_boxes(self):
        out = grid([box(*p) for p in NATALIES + STRAYS], 6, 10)
        self.assertTrue(out["ok"], out["reason"])
        self.assertEqual(len(out["boxes"]), 60)
        self.assertEqual(out["dropped"], 3)
        for n, p in enumerate(NATALIES, 1):
            self.assertAt(out, n, p)

    def test_missed_tubes_placed_in_their_cells(self):
        pts = hotel()
        missed = {0, 9, 14, 15, 33, 59}          # corners, a pair, the last
        out = grid([box(*p) for i, p in enumerate(pts) if i not in missed], 6, 10)
        self.assertTrue(out["ok"], out["reason"])
        self.assertEqual(out["filled"], len(missed))
        for i in missed:
            self.assertFalse(out["found"][i])
            self.assertAt(out, i + 1, pts[i], 1)

    def test_a_tilted_camera(self):
        pts = hotel(tilt=math.radians(4))
        out = grid([box(*p) for p in pts if p != pts[23]], 6, 10)
        self.assertTrue(out["ok"], out["reason"])
        for n, p in enumerate(pts, 1):
            self.assertAt(out, n, p, 2)

    def test_a_row_of_false_boxes_left_out(self):
        pts = hotel()
        noise = [(250, 700), (400, 700)]
        out = grid([box(*p) for p in pts + noise], 6, 10)
        self.assertTrue(out["ok"], out["reason"])
        self.assertEqual(out["dropped"], 2)
        self.assertAt(out, 60, pts[59], 0.01)

    def test_a_row_with_few_boxes_follows_the_others(self):
        pts = hotel()
        keep = [p for i, p in enumerate(pts) if not (50 <= i < 60) or i in (52, 57)]
        out = grid([box(*p) for p in keep], 6, 10)
        self.assertTrue(out["ok"], out["reason"])
        for n in range(51, 61):
            self.assertAt(out, n, pts[n - 1], 1)

    def test_no_guess_at_a_missing_edge_column(self):
        pts = [p for i, p in enumerate(hotel()) if i % 10 != 9]
        out = grid([box(*p) for p in pts], 6, 10)
        self.assertFalse(out["ok"])
        self.assertIn("9 columns of 10", out["reason"])
        self.assertEqual(out["boxes"], [])

    def test_fewer_than_half_fails(self):
        out = grid([box(*p) for p in hotel()[:20]], 6, 10)
        self.assertFalse(out["ok"])
        self.assertIn("fewer than half", out["reason"])

    def test_a_missing_row_fails(self):
        out = grid([box(*p) for p in hotel()[:50]], 6, 10)
        self.assertFalse(out["ok"])
        self.assertIn("5 rows of 6", out["reason"])

    def test_one_row_and_one_column(self):
        self.assertTrue(grid([box(100, 50), box(140, 50), box(180, 50)], 1, 3)["ok"])
        out = grid([box(100, 50), box(100, 120)], 2, 1)
        self.assertTrue(out["ok"], out["reason"])
        self.assertLess(centre(out["boxes"][0])[1], centre(out["boxes"][1])[1])


class ReadingOrderTests(unittest.TestCase):
    def test_rows_then_left_to_right(self):
        a, b, c = box(300, 50), box(100, 52), box(200, 150)
        self.assertEqual(reading_order([c, a, b]), [b, a, c])
        self.assertEqual(reading_order([]), [])


if __name__ == "__main__":
    unittest.main()
