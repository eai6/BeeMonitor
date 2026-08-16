"""Saving hotel ROI / nest shapes from the ROI editor.

A shape may be a dragged rectangle or a traced polygon. Both are stored with a
box — the polygon's box is its bounding box — so every box-only consumer (the
device gate's crop, the analyzer's manual layout, the overlays) keeps working,
while polygon-aware consumers get the outline that excludes the background.
"""

import json

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from .models import Device

User = get_user_model()

# A diamond: its bounding box is the whole frame, the shape is half of it.
DIAMOND = [[0.5, 0.1], [0.9, 0.5], [0.5, 0.9], [0.1, 0.5]]


class RoiEditorSaveTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("alice", password="x")
        self.device = Device.objects.create(
            owner=self.user, name="Trap 1", key_hash="h1", prefix="bmk_1")
        self.client.force_login(self.user)
        self.url = reverse("devices:roi_editor", args=[self.device.pk])

    def _post(self, payload):
        return self.client.post(self.url, data=json.dumps(payload),
                                content_type="application/json")

    def test_polygon_roi_saves_points_and_bounding_box(self):
        r = self._post({"roi": {"points": DIAMOND}, "nests": []})
        self.assertEqual(r.status_code, 200)
        self.device.refresh_from_db()
        self.assertEqual(self.device.roi_polygon, DIAMOND)
        self.assertEqual(self.device.roi_override, [0.1, 0.1, 0.9, 0.9])

    def test_rectangle_roi_clears_any_previous_polygon(self):
        self._post({"roi": {"points": DIAMOND}, "nests": []})
        self._post({"roi": {"box": [0.2, 0.2, 0.8, 0.8]}, "nests": []})
        self.device.refresh_from_db()
        self.assertIsNone(self.device.roi_polygon)
        self.assertEqual(self.device.roi_override, [0.2, 0.2, 0.8, 0.8])

    def test_legacy_bare_box_payload_still_accepted(self):
        r = self._post({"roi": [0.1, 0.1, 0.5, 0.5], "nests": []})
        self.assertEqual(r.status_code, 200)
        self.device.refresh_from_db()
        self.assertEqual(self.device.roi_override, [0.1, 0.1, 0.5, 0.5])
        self.assertIsNone(self.device.roi_polygon)

    def test_nest_polygon_saves_points_and_box(self):
        self._post({"roi": None, "nests": [{"id": 3, "points": DIAMOND}]})
        self.device.refresh_from_db()
        self.assertEqual(self.device.nest_layout,
                         [{"id": 3, "box": [0.1, 0.1, 0.9, 0.9], "points": DIAMOND}])

    def test_rectangular_nest_carries_no_points(self):
        self._post({"roi": None, "nests": [{"id": 1, "box": [0.1, 0.1, 0.2, 0.2]}]})
        self.device.refresh_from_db()
        self.assertEqual(self.device.nest_layout,
                         [{"id": 1, "box": [0.1, 0.1, 0.2, 0.2]}])

    def test_out_of_range_points_are_clamped(self):
        self._post({"roi": {"points": [[-1, 0.2], [0.5, 2], [0.9, 0.5]]}, "nests": []})
        self.device.refresh_from_db()
        self.assertEqual(self.device.roi_polygon, [[0.0, 0.2], [0.5, 1.0], [0.9, 0.5]])

    def test_too_few_points_is_not_a_polygon(self):
        # Two points enclose nothing: fall back to the box, if there is one.
        self._post({"roi": {"box": [0.1, 0.1, 0.6, 0.6], "points": [[0.1, 0.1], [0.6, 0.6]]},
                    "nests": []})
        self.device.refresh_from_db()
        self.assertIsNone(self.device.roi_polygon)
        self.assertEqual(self.device.roi_override, [0.1, 0.1, 0.6, 0.6])

    def test_degenerate_polygon_is_dropped(self):
        # A sliver thinner than the minimum box: not a usable region.
        self._post({"roi": {"points": [[0.5, 0.5], [0.501, 0.5], [0.5, 0.501]]},
                    "nests": []})
        self.device.refresh_from_db()
        self.assertIsNone(self.device.roi_override)
        self.assertIsNone(self.device.roi_polygon)

    def test_response_echoes_what_was_stored(self):
        r = self._post({"roi": {"points": DIAMOND}, "nests": []})
        body = r.json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["roi"], [0.1, 0.1, 0.9, 0.9])
        self.assertEqual(body["roi_polygon"], DIAMOND)


class RoiOverlayShapeTests(TestCase):
    """What the editor and the device page load the saved ROI as."""

    def test_saved_polygon_loads_as_box_plus_points(self):
        from apps.devices.views import _roi_shape

        user = User.objects.create_user("bob", password="x")
        device = Device.objects.create(owner=user, name="Trap 2", key_hash="h2",
                                       prefix="bmk_2", roi_override=[0.1, 0.1, 0.9, 0.9],
                                       roi_polygon=DIAMOND)
        self.assertEqual(_roi_shape(device), {"box": [0.1, 0.1, 0.9, 0.9], "points": DIAMOND})
