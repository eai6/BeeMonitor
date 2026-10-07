"""The live tracking overlay: its payload and the endpoint (memory/46)."""
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
    "frame,track_id,x1,y1,x2,y2,cx,cy,confidence,source,taxon,mode\n"
    "0,1,10.4,20,30,40,20,30,0.9,yolo,bee,lookback\n"
    "1,1,11,21,31,41,21,31,0.8,yolo,bee,lookback\n"
    "1,1,11,21,31,41,21,31,0.8,yolo,bee,tracking\n"   # same frame twice
    "2,1,11,21,31,41,25,35,0.8,yolo,bee,tracking\n"   # coasting: box repeats
    "2,2,100,100,120,130,110,115,0.7,yolo,bee,tracking\n"
)


class PayloadTests(SimpleTestCase):
    def test_rows_are_whole_pixels_one_per_track_per_frame(self):
        rows = overlay.frame_rows(CSV)
        self.assertEqual(rows[0], [0, 1, 10, 20, 30, 40, 0])
        self.assertEqual([r[:2] for r in rows], [[0, 1], [1, 1], [2, 1], [2, 2]])

    def test_a_coasting_track_is_lost_and_drawn_where_it_was_predicted(self):
        rows = overlay.frame_rows(CSV)
        coasting = rows[2]
        self.assertEqual(coasting[6], 1)
        self.assertEqual(coasting[2:6], [15, 25, 35, 45])  # the 20x20 box on (25, 35)
        self.assertEqual(rows[3][6], 0)

    def test_payload_carries_identity_events_and_regions(self):
        p = overlay.build(
            CSV, [{"track_id": "1", "species": "Osmia", "species_confidence": "0.8",
                   "first_frame": "0", "last_frame": "2", "frames_seen": "3"}],
            [{"frame": 1, "subject": 1, "action": "enter", "target": "nest 3"},
             {"frame": 2, "subject": "", "action": "x"}],
            25, {"hotel_roi": [0.1, 0.1, 0.9, 0.9],
                 "nest_layout": [{"id": 3, "box": [0.2, 0.2, 0.3, 0.3]}]})
        self.assertEqual(p["tracks"]["1"]["species"], "Osmia")
        self.assertEqual(p["events"], [[1, 1, "enter", "nest 3"]])
        self.assertEqual([r["label"] for r in p["regions"]], ["hotel", "reference 3"])
        self.assertEqual(len(p["rows"]), 4 * 7)

    def test_stored_beside_the_tracking_csv(self):
        self.assertEqual(overlay.stored_path("1/pl_x/tracking_results.csv"),
                         "1/pl_x/overlay_v2.json.gz")


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
        with mock.patch("apps.analysis.views.get_s3_client", return_value=s3), \
             mock.patch("apps.pipelines.executors.primitives_for_job", return_value=[]):
            r = self.client.get(self.url, HTTP_ACCEPT_ENCODING="gzip")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r["Content-Encoding"], "gzip")
        payload = json.loads(gzip.decompress(r.content))
        self.assertEqual(len(payload["rows"]), 4 * 7)
        stored = s3.upload_stream.call_args
        self.assertEqual(stored.args[1], "1/pl_x/overlay_v2.json.gz")

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
