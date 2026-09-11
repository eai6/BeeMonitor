"""Polygon reference geometry: shapes carry their outline, and containment uses it.

The point of a polygon ROI is that the bounding box is NOT the region — these
tests pin the difference, since every visitation/activity metric depends on it.
"""

from django.test import SimpleTestCase

from apps.pipelines import ops


# A diamond inscribed in the unit square: its bounding box is the whole square,
# but the four corners of that box are outside the shape.
DIAMOND = [[0.5, 0.0], [1.0, 0.5], [0.5, 1.0], [0.0, 0.5]]


class RoiShapesTests(SimpleTestCase):
    def test_hotel_polygon_rides_with_the_box(self):
        shapes = ops.roi_shapes({
            "hotel_roi": [0.0, 0.0, 1.0, 1.0],
            "hotel_polygon": DIAMOND,
        })
        self.assertEqual(len(shapes), 1)
        box, points = shapes[0]
        self.assertEqual(box, (0.0, 0.0, 1.0, 1.0))
        self.assertEqual(points, [(0.5, 0.0), (1.0, 0.5), (0.5, 1.0), (0.0, 0.5)])

    def test_rectangles_have_no_points(self):
        shapes = ops.roi_shapes({"hotel_roi": [0.1, 0.1, 0.4, 0.4]})
        self.assertEqual(shapes, [((0.1, 0.1, 0.4, 0.4), None)])

    def test_nest_and_region_outlines(self):
        shapes = ops.roi_shapes({
            "nest_layout": [{"id": 1, "box": [0.0, 0.0, 1.0, 1.0], "points": DIAMOND}],
            "regions": [{"box": [0.0, 0.0, 0.2, 0.2]}, [0.5, 0.5, 0.6, 0.6]],
        })
        self.assertEqual(len(shapes), 3)
        self.assertIsNotNone(shapes[0][1])
        self.assertIsNone(shapes[1][1])
        self.assertIsNone(shapes[2][1])

    def test_degenerate_outline_falls_back_to_the_box(self):
        shapes = ops.roi_shapes({"hotel_roi": [0, 0, 1, 1], "hotel_polygon": [[0, 0], [1, 1]]})
        self.assertEqual(shapes, [((0.0, 0.0, 1.0, 1.0), None)])

    def test_roi_boxes_still_returns_plain_boxes(self):
        boxes = ops.roi_boxes({"hotel_roi": [0, 0, 1, 1], "hotel_polygon": DIAMOND})
        self.assertEqual(boxes, [(0.0, 0.0, 1.0, 1.0)])


class ContainmentTests(SimpleTestCase):
    def test_polygon_excludes_the_corners_of_its_box(self):
        shapes = ops.roi_shapes({"hotel_roi": [0, 0, 1, 1], "hotel_polygon": DIAMOND})
        self.assertTrue(ops.in_any_box(0.5, 0.5, shapes))    # centre — inside
        self.assertFalse(ops.in_any_box(0.02, 0.02, shapes))  # corner — background
        self.assertFalse(ops.in_any_box(0.97, 0.97, shapes))

    def test_box_alone_would_have_counted_those_corners(self):
        boxes = ops.roi_boxes({"hotel_roi": [0, 0, 1, 1], "hotel_polygon": DIAMOND})
        self.assertTrue(ops.in_any_box(0.02, 0.02, boxes))

    def test_bare_boxes_still_accepted(self):
        self.assertTrue(ops.in_any_box(0.2, 0.2, [(0.0, 0.0, 0.5, 0.5)]))
        self.assertFalse(ops.in_any_box(0.8, 0.2, [(0.0, 0.0, 0.5, 0.5)]))

    def test_concave_outline(self):
        # An L: the missing quadrant is inside the bounding box, outside the shape.
        el = [[0, 0], [1, 0], [1, 0.5], [0.5, 0.5], [0.5, 1], [0, 1]]
        shapes = ops.roi_shapes({"hotel_roi": [0, 0, 1, 1], "hotel_polygon": el})
        self.assertTrue(ops.in_any_box(0.25, 0.25, shapes))
        self.assertFalse(ops.in_any_box(0.75, 0.75, shapes))
