"""One clip must not have two different interaction tables.

The per-job results page rendered the worker's interactions.csv while the batch
export computed the primitives, so the same clip said different things
depending on which page you opened — and the worker's version matches an insect
to a reference by centroid distance under a flat 50 px, so a bee sitting inside
a large flower appears in neither.
"""

import csv
import tempfile
from pathlib import Path

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from apps.analysis.models import Job, JobResult
from apps.devices.models import Device
from apps.pipelines import executors
from apps.videos.models import Video

User = get_user_model()

# One insect inside the nest tube (0.1-0.3) but far from its centre — the case
# the worker's 50 px rule never recorded.
TRACKING = [{"frame": f, "track_id": 3, "cx": 0.29, "cy": 0.29} for f in range(20)]


def _csv(rows, name):
    path = Path(tempfile.mkdtemp()) / name
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return str(path)


class JobPrimitiveTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("jp", password="x")
        self.client.force_login(self.user)
        self.device = Device.objects.create(
            owner=self.user, name="Hotel", key_hash="hjp", prefix="bmk_jp",
            roi_override=[0.0, 0.0, 0.6, 0.6],
            nest_layout=[{"id": 1, "box": [0.1, 0.1, 0.3, 0.3]}])
        self.video = Video.objects.create(
            user=self.user, device=self.device, title="clip",
            storage_key="jp/c.mp4", file_size_bytes=1, status=Video.Status.READY,
            recorded_at=timezone.now(), fps=25.0, width=1920, height=1080)
        self.job = Job.objects.create(user=self.user, video=self.video,
                                      status="completed", modal_job_id="jp-1")
        JobResult.objects.create(
            job=self.job, tracking_csv_path=_csv(TRACKING, "t.csv"),
            interactions_csv_path="worker.csv",
            summary_stats={"video_fps": 25.0})

    def test_the_job_gets_the_computed_interactions(self):
        rows = executors.primitives_for_job(self.job, "interactions")

        self.assertTrue(rows)
        self.assertEqual(rows[0]["b_kind"], "reference")
        self.assertEqual(rows[0]["relation"], "inside")

    def test_it_works_without_a_pipeline_run_behind_the_job(self):
        """Jobs launched outside a pipeline still get an answer."""
        from apps.pipelines.models import PipelineRun

        self.assertFalse(PipelineRun.objects.exists())
        self.assertTrue(executors.primitives_for_job(self.job, "interactions"))

    def test_a_job_with_no_result_yields_nothing_rather_than_raising(self):
        bare = Job.objects.create(user=self.user, video=self.video,
                                  status="queued", modal_job_id="jp-2")

        self.assertEqual(executors.primitives_for_job(bare, "interactions"), [])

    def test_the_page_shows_the_computed_table_not_the_workers(self):
        html = self.client.get(reverse("analysis:results",
                                       kwargs={"pk": self.job.pk})).content.decode()

        self.assertIn("relation", html)
        self.assertNotIn("avg_distance_px", html)

    def test_the_pages_download_matches_its_table(self):
        resp = self.client.get(reverse("analysis:results_csv",
                                       kwargs={"pk": self.job.pk,
                                               "kind": "interactions"}))

        self.assertEqual(resp.status_code, 200)
        body = resp.content.decode()
        self.assertIn("relation", body.splitlines()[0])
        self.assertIn("inside", body)
        self.assertNotIn("avg_distance_px", body)

    def test_the_job_page_and_the_batch_export_agree(self):
        """The whole point: one clip, one answer."""
        from apps.pipelines import aggregate
        from apps.pipelines.models import Pipeline, PipelineRun

        pipeline = Pipeline.objects.create(user=self.user, title="P")
        run = PipelineRun.objects.create(
            pipeline=pipeline, user=self.user, batch_id="aaaa1111-0000-4000-8000-00000000beef",
            status="completed",
            steps=[{"id": "v", "block_type": "input.video",
                    "config": {"video_id": str(self.video.pk)}},
                   {"id": "r", "block_type": "reference.layout",
                    "config": {"source": "device_layout"}, "inputs": {"video": "v"}},
                   {"id": "a", "block_type": "analyze.interactions", "config": {},
                    "inputs": {"tracks": "m", "rois": "r"}}],
            context={"v": {"artifact": "video", "video_id": self.video.pk},
                     "m": {"artifact": "tracks",
                           "result": {"tracking_csv_path": self.job.result.tracking_csv_path,
                                      "summary_stats": {"video_fps": 25.0}}}})

        _fields, batch_rows = aggregate.primitive_csv([run], "interactions")
        job_rows = executors.primitives_for_job(self.job, "interactions")

        self.assertEqual(len(batch_rows), len(job_rows))
        self.assertEqual(batch_rows[0]["b"], job_rows[0]["b"])
        self.assertEqual(batch_rows[0]["duration_sec"], job_rows[0]["duration_sec"])

    def test_a_stranger_cannot_download_someone_elses_clip(self):
        self.client.force_login(User.objects.create_user("nosy3", password="x"))

        resp = self.client.get(reverse("analysis:results_csv",
                                       kwargs={"pk": self.job.pk,
                                               "kind": "interactions"}))

        self.assertEqual(resp.status_code, 404)

    def test_an_unknown_table_is_refused(self):
        resp = self.client.get(reverse("analysis:results_csv",
                                       kwargs={"pk": self.job.pk, "kind": "trips"}))

        self.assertEqual(resp.status_code, 404)
