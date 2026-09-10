"""References the detector FOUND must reach the analyzer.

The Biodiversity Count batch showed the disconnect exactly: the job page
reported "Nests 4" while the batch page reported "0 REFERENCES — no references
were defined". The worker had found the four flowers and written them to
summary_stats["nest_bboxes"]; nothing local ever read that key, so every
analyzer ran against an empty reference list.

These boxes are PIXELS — the annotator draws them straight onto the frame —
while references must be normalised 0..1 to match the tracks. Getting that
wrong would not error, it would silently place every reference in the top-left
few percent of the frame and report zero visits, which looks identical to the
bug being fixed.
"""

from django.contrib.auth import get_user_model
from django.test import TestCase

from apps.pipelines import ops
from apps.videos.models import Video

User = get_user_model()


def result(nest_bboxes=None, hotel=None):
    stats = {}
    if nest_bboxes is not None:
        stats["nest_bboxes"] = nest_bboxes
    if hotel is not None:
        stats["hotel_bbox"] = hotel
    return {"summary_stats": stats}


class DetectedReferenceTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("dr", password="x")
        self.video = Video.objects.create(
            user=self.user, title="c", storage_key="dr/c.mp4", file_size_bytes=1,
            status=Video.Status.READY, width=1920, height=1080)

    def test_detected_boxes_become_references(self):
        refs = ops.detected_references(
            result({"1": [0, 0, 192, 108], "2": [960, 540, 1152, 648]}), self.video)

        self.assertEqual(len(refs), 2)
        self.assertEqual({r["id"] for r in refs}, {"nest_1", "nest_2"})

    def test_pixel_boxes_are_normalised_by_the_measured_frame_size(self):
        refs = ops.detected_references(result({"1": [960, 540, 1920, 1080]}), self.video)

        self.assertEqual(refs[0]["box"], (0.5, 0.5, 1.0, 1.0))

    def test_boxes_already_fractional_are_left_alone(self):
        refs = ops.detected_references(result({"1": [0.1, 0.1, 0.3, 0.3]}), self.video)

        self.assertEqual(refs[0]["box"], (0.1, 0.1, 0.3, 0.3))

    def test_without_a_frame_size_we_refuse_rather_than_guess(self):
        """Emitting pixel boxes as if fractional would put every reference in
        the top-left 0.1% of the frame and report zero visits — indistinguishable
        from the bug this fixes."""
        bare = Video.objects.create(user=self.user, title="n", storage_key="dr/n.mp4",
                                    file_size_bytes=1, status=Video.Status.READY)

        self.assertEqual(ops.detected_references(result({"1": [960, 540, 1920, 1080]}),
                                                 bare), [])

    def test_frame_size_can_come_from_the_summary_when_the_row_lacks_it(self):
        bare = Video.objects.create(user=self.user, title="n2", storage_key="dr/n2.mp4",
                                    file_size_bytes=1, status=Video.Status.READY)
        res = result({"1": [960, 540, 1920, 1080]})
        res["summary_stats"].update({"frame_width": 1920, "frame_height": 1080})

        refs = ops.detected_references(res, bare)

        self.assertEqual(refs[0]["box"], (0.5, 0.5, 1.0, 1.0))

    def test_the_hotel_is_used_only_when_nothing_finer_was_found(self):
        # Both present: the tubes win, or every episode would be counted twice.
        both = result({"1": [0.1, 0.1, 0.3, 0.3]}, hotel=[0.0, 0.0, 1.0, 1.0])
        self.assertEqual([r["id"] for r in ops.detected_references(both, self.video)],
                         ["nest_1"])

        only_hotel = result(hotel=[0.0, 0.0, 1.0, 1.0])
        self.assertEqual([r["id"] for r in ops.detected_references(only_hotel, self.video)],
                         ["hotel"])

    def test_a_job_that_detected_nothing_yields_nothing(self):
        self.assertEqual(ops.detected_references(result({}), self.video), [])
        self.assertEqual(ops.detected_references({}, self.video), [])
        self.assertEqual(ops.detected_references(None, self.video), [])

    def test_malformed_boxes_are_skipped_not_crashed(self):
        refs = ops.detected_references(
            result({"1": [0.1, 0.1, 0.3, 0.3], "2": "nonsense", "3": [1, 2]}),
            self.video)

        self.assertEqual([r["id"] for r in refs], ["nest_1"])

    def test_the_box_is_ordered_regardless_of_corner_order(self):
        refs = ops.detected_references(result({"1": [0.3, 0.3, 0.1, 0.1]}), self.video)

        x1, y1, x2, y2 = refs[0]["box"]
        self.assertLessEqual(x1, x2)
        self.assertLessEqual(y1, y2)
