"""An old batch exports the right answer without being re-run.

A batch analysed before the primitives existed stores only its retired
analyzer's output — `table_kind: "interaction"`, not `"interactions"` — so the
export had nothing to read and fell back to the worker's own CSV. That is the
file that matches an insect to a reference by centroid distance under a flat
50 px, so a bee inside a large flower never appears in it.

Nothing about the answer needs the GPU again: the tracking table and the
reference geometry are both saved, and the primitives are a pure function of
them. This pins that they are recomputed at export time.
"""

import csv
import tempfile
from pathlib import Path

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from apps.devices.models import Device
from apps.pipelines.models import Pipeline, PipelineRun
from apps.videos.models import Video

User = get_user_model()

# One insect parked at (0.29, 0.29) — inside the nest tube 0.1-0.3, but well
# away from its centre. This is the bee the worker's 50 px rule never saw.
TRACKING_ROWS = [{"frame": f, "track_id": 3, "cx": 0.29, "cy": 0.29}
                 for f in range(20)]


def _write_csv(rows, name):
    path = Path(tempfile.mkdtemp()) / name
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return str(path)


class ExportRecomputeTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("er", password="x")
        self.client.force_login(self.user)
        self.device = Device.objects.create(
            owner=self.user, name="beemonitor3", key_hash="her", prefix="bmk_er",
            roi_override=[0.0, 0.0, 0.6, 0.6],
            nest_layout=[{"id": 1, "box": [0.1, 0.1, 0.3, 0.3]}])
        self.pipeline = Pipeline.objects.create(user=self.user, title="Old")
        self.batch_id = "8f9a0b11-0000-4000-8000-0000000033cd"
        self.video = Video.objects.create(
            user=self.user, device=self.device, title="clip",
            storage_key="er/c.mp4", file_size_bytes=1, status=Video.Status.READY,
            recorded_at=timezone.now(), site_name="Meadow A", fps=25.0)
        self.tracking = _write_csv(TRACKING_ROWS, "tracking.csv")

        result = {"tracking_csv_path": self.tracking,
                  "events_csv_path": "e.csv",
                  "summary_stats": {"video_fps": 25.0}}
        PipelineRun.objects.create(
            pipeline=self.pipeline, user=self.user, batch_id=self.batch_id,
            status="completed",
            steps=[
                {"id": "v", "block_type": "input.video",
                 "config": {"video_id": str(self.video.pk)}},
                {"id": "d", "block_type": "detect.objects",
                 "config": {"label": "bee"}, "inputs": {"video": "v"}},
                {"id": "m", "block_type": "track.mot", "config": {},
                 "inputs": {"detections": "d"}},
                {"id": "r", "block_type": "reference.layout",
                 "config": {"source": "device_layout"}, "inputs": {"video": "v"}},
                # The RETIRED analyzer: table_kind "interaction", singular.
                {"id": "a", "block_type": "analyze.interaction", "config": {},
                 "inputs": {"tracks": "m", "rois": "r"}},
            ],
            context={"v": {"artifact": "video", "video_id": self.video.pk},
                     "m": {"artifact": "tracks", "result": result},
                     "a": {"artifact": "table", "table_kind": "interaction",
                           "interaction_count": 0, "rows": []}})

    def _download(self, kind):
        return self.client.get(reverse(
            "pipelines:batch_combined_csv",
            kwargs={"batch_id": self.batch_id, "kind": kind})).content.decode()

    def test_the_bee_inside_the_tube_reaches_the_export(self):
        body = self._download("interactions")

        self.assertIn("nest_1", body)
        self.assertIn("reference", body)
        self.assertIn("inside", body)

    def test_the_export_is_not_the_workers_centroid_distance_file(self):
        """Its columns are the giveaway: min_distance_px / avg_distance_px."""
        body = self._download("interactions")

        self.assertNotIn("avg_distance_px", body)
        self.assertIn("duration_sec", body)

    def test_events_are_recomputed_the_same_way(self):
        body = self._download("events")

        self.assertIn("enter", body)
        self.assertIn("exit", body)
        self.assertIn("nest_1", body)

    def test_recomputed_rows_still_carry_provenance(self):
        header, first = self._download("interactions").splitlines()[:2]

        self.assertTrue(header.startswith("video_title,video_recorded_at,absolute_time"))
        self.assertIn("beemonitor3", first)
        self.assertIn("Meadow A", first)

    def test_the_batch_offers_both_primitives(self):
        resp = self.client.get(reverse("pipelines:batch_detail",
                                       kwargs={"batch_id": self.batch_id}))

        kinds = {d["kind"] for d in resp.context["downloads"]}
        self.assertIn("interactions", kinds)
        self.assertIn("events", kinds)

    def test_a_run_with_no_tracking_table_is_skipped_not_crashed(self):
        from apps.pipelines import executors

        run = PipelineRun.objects.get(batch_id=self.batch_id)
        run.context["m"]["result"]["tracking_csv_path"] = ""

        self.assertEqual(executors.recompute_primitive(run, "interactions"), [])
