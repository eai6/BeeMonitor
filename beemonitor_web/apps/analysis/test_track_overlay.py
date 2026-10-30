"""The live tracking overlay: its payload and the endpoint (memory/46, 60).

The stored payload is boxes only; events and references are served live
(``?part=events``) so a recount or a rule change shows at once."""
import gzip
import io
import json
from unittest import mock

from django.contrib.auth import get_user_model
from django.test import SimpleTestCase, TestCase
from django.urls import reverse
from django.utils import timezone

from apps.analysis import overlay
from apps.analysis.models import Job, JobResult
from apps.videos.models import Video

CSV = (
    "frame,track_id,x1,y1,x2,y2,cx,cy,detected,confidence,source,taxon,mode\n"
    "0,1,10.4,20,30,40,20,30,1,0.9,yolo,bee,lookback\n"
    "1,1,11,21,31,41,21,31,1,0.8,yolo,bee,lookback\n"
    "1,1,11,21,31,41,21,31,1,0.8,yolo,bee,tracking\n"   # same frame twice
    "2,1,11,21,31,41,25,35,0,,yolo,bee,tracking\n"      # predicted: tracker lost it
    "2,2,100,100,120,130,110,115,1,0.7,yolo,bee,tracking\n"
)

# Written before the tracker recorded `detected`: the box repeats on row 3,
# which used to be read as "lost". It is no longer inferred (memory/60).
OLD_CSV = (
    "frame,track_id,x1,y1,x2,y2,cx,cy,confidence\n"
    "1,1,11,21,31,41,21,31,0.8\n"
    "2,1,11,21,31,41,25,35,0.8\n"
)


class PayloadTests(SimpleTestCase):
    def test_rows_are_whole_pixels_one_per_track_per_frame(self):
        rows = overlay.frame_rows(CSV)
        self.assertEqual(rows[0], [0, 1, 10, 20, 30, 40, 0])
        self.assertEqual([r[:2] for r in rows], [[0, 1], [1, 1], [2, 1], [2, 2]])

    def test_a_predicted_row_is_lost_and_drawn_where_it_was_predicted(self):
        rows = overlay.frame_rows(CSV)
        predicted = rows[2]
        self.assertEqual(predicted[6], 1)
        self.assertEqual(predicted[2:6], [15, 25, 35, 45])  # the 20x20 box on (25, 35)
        self.assertEqual(rows[3][6], 0)

    def test_lost_is_never_inferred_from_a_repeated_box(self):
        self.assertEqual([r[6] for r in overlay.frame_rows(OLD_CSV)], [0, 0])

    def test_payload_holds_boxes_and_identity_but_no_events_or_regions(self):
        p = overlay.build(
            CSV, [{"track_id": "1", "species": "Osmia", "species_confidence": "0.8"}], 25)
        self.assertEqual(p["tracks"]["1"]["species"], "Osmia")
        # The detected span: frames 0-1; frame 2 was predicted.
        self.assertEqual((p["tracks"]["1"]["first"], p["tracks"]["1"]["last"],
                          p["tracks"]["1"]["frames_seen"]), (0, 1, 2))
        self.assertNotIn("events", p)
        self.assertNotIn("regions", p)
        self.assertEqual(len(p["rows"]), 4 * 7)

    def test_stored_beside_the_tracking_csv(self):
        self.assertEqual(overlay.stored_path("1/pl_x/tracking_results.csv"),
                         "1/pl_x/overlay_v3.json.gz")


class EndpointTests(TestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user("ov", password="x")
        video = Video.objects.create(
            user=self.user, title="clip", storage_key="ov/c.mp4", file_size_bytes=1,
            status=Video.Status.READY, recorded_at=timezone.now())
        self.job = Job.objects.create(user=self.user, video=video, status="completed")
        JobResult.objects.create(job=self.job, tracking_csv_path="1/pl_x/tracking_results.csv")
        self.url = reverse("analysis:track_overlay", kwargs={"pk": self.job.pk})

    def _s3(self):
        s3 = mock.Mock()

        def download(container, key, buf):
            if key.endswith(".gz"):
                raise FileNotFoundError(key)
            buf.write(CSV.encode())
        s3.download_to_stream.side_effect = download
        return s3

    def test_builds_stores_and_serves_gzip(self):
        s3 = self._s3()
        self.client.force_login(self.user)
        with mock.patch("apps.analysis.views.get_s3_client", return_value=s3):
            r = self.client.get(self.url, HTTP_ACCEPT_ENCODING="gzip")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r["Content-Encoding"], "gzip")
        payload = json.loads(gzip.decompress(r.content))
        self.assertEqual(len(payload["rows"]), 4 * 7)
        stored = s3.upload_stream.call_args
        self.assertEqual(stored.args[1], "1/pl_x/overlay_v3.json.gz")

    def test_events_are_served_live_and_never_stored(self):
        s3 = self._s3()
        saved = [{"frame": 4, "track_id": 1, "action": "stay", "reference": "reference_3",
                  "end_frame": 9, "at_edge": False}]
        self.client.force_login(self.user)
        with mock.patch("apps.analysis.views.get_s3_client", return_value=s3), \
             mock.patch("apps.pipelines.executors.primitives_for_job", return_value=saved), \
             mock.patch("apps.pipelines.executors.references_for_job", return_value=[]):
            r = self.client.get(self.url, {"part": "events"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(json.loads(r.content)["events"],
                         # + the event's key, your vote, and the accepted /
                         # rejected counts (memory/62).
                         [[4, 1, "stay", "reference_3", 9, False,
                           "stay|reference_3|1|4", None, 0, 0]])
        self.assertEqual(r["Cache-Control"], "private, no-store")
        s3.download_to_stream.assert_not_called()
        s3.upload_stream.assert_not_called()

    def test_a_stored_payload_is_served_without_rebuilding(self):
        body = overlay.encode({"v": 1, "rows": [], "tracks": {}})
        s3 = mock.Mock()
        s3.download_to_stream.side_effect = lambda c, k, buf: buf.write(body)
        self.client.force_login(self.user)
        with mock.patch("apps.analysis.views.get_s3_client", return_value=s3):
            r = self.client.get(self.url)  # no gzip accepted: plain JSON
        self.assertEqual(json.loads(r.content)["v"], 1)
        s3.upload_stream.assert_not_called()

    def test_a_stranger_gets_404(self):
        other = get_user_model().objects.create_user("x", password="x")
        self.client.force_login(other)
        self.assertEqual(self.client.get(self.url).status_code, 404)
