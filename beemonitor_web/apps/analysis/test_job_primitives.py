"""One clip must not have two different interaction tables.

The per-job results page rendered the worker's interactions.csv while the batch
export computed the primitives, so the same clip said different things
depending on which page you opened — and the worker's version matches an insect
to a reference by centroid distance under a flat 50 px, so a bee sitting inside
a large flower appears in neither.
"""

import csv
import re
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
        self.assertEqual(rows[0]["with"], "reference_1")
        self.assertEqual(rows[0]["relation"], "in_reference")

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
        self.assertIn("in_reference", body)
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
                     "m": {"artifact": "tracks", "job_id": self.job.pk,
                           "result": {"tracking_csv_path": self.job.result.tracking_csv_path,
                                      "summary_stats": {"video_fps": 25.0}}}})
        # The run's own Interactions step counts its rows; both pages read them.
        from apps.pipelines.recount import recount_run
        recount_run(run)
        run.refresh_from_db()

        _fields, batch_rows = aggregate.primitive_csv([run], "interactions")
        job_rows = executors.primitives_for_job(self.job, "interactions")

        self.assertEqual(len(batch_rows), len(job_rows))
        self.assertEqual(batch_rows[0]["with"], job_rows[0]["with"])
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


class ResultsTableConsistencyTests(JobPrimitiveTests):
    """Every download on the page must be the table printed above it."""

    def _html(self):
        return self.client.get(reverse("analysis:results",
                                       kwargs={"pk": self.job.pk})).content.decode()

    def test_both_computed_tables_can_be_collapsed(self):
        html = self._html()

        self.assertEqual(html.count("Show Table"), html.count("Hide Table"))
        self.assertGreaterEqual(html.count("Show Table"), 2)

    def test_no_download_on_the_page_serves_the_workers_own_file(self):
        """Events and interactions must point at the computed endpoints.

        This used to assert no "X-Amz-Signature" anywhere in the page, which
        only held while presigning was failing: with working credentials the
        page legitimately signs the original video, the annotated video and the
        Tracking CSV — and for tracking the worker's file IS the answer. So the
        check is now about the two tables that are computed, not the whole page.
        """
        html = self._html()

        for kind in ("events", "interactions"):
            self.assertIn(
                reverse("analysis:results_csv",
                        kwargs={"pk": self.job.pk, "kind": kind}), html)
        # No signed link may point at a stored events/interactions CSV.
        for href in re.findall(r'href="([^"]+)"', html):
            if "X-Amz-Signature" in href:
                self.assertNotIn("events.csv", href)
                self.assertNotIn("interactions.csv", href)

    def test_the_events_download_matches_its_table(self):
        page = self._html()
        body = self.client.get(reverse("analysis:results_csv",
                                       kwargs={"pk": self.job.pk,
                                               "kind": "events"})).content.decode()

        self.assertEqual(body.splitlines()[0].split(",")[:8],
                         ["timestamp", "clip_sec", "track_id", "action", "reference",
                          "duration_sec", "at_edge", "ml_confidence"])
        self.assertIn("at_edge", page)
        self.assertNotIn("target_kind", page)


# Track 3 comes out of the tube and flies off; track 4 flies in and goes into
# it. Both are detected throughout, so this is a clip the current rules count.
AGREE_TRACKING = (
    [{"frame": f, "track_id": 3, "cx": 0.2 if f < 10 else 0.8, "cy": 0.2 if f < 10 else 0.8,
      "detected": 1} for f in range(20)]
    + [{"frame": f, "track_id": 4, "cx": 0.8 if f < 30 else 0.2, "cy": 0.8 if f < 30 else 0.2,
        "detected": 1} for f in range(20, 40)]
)


class OneClipAgreesTests(TestCase):
    """memory/60: the results table, the Events CSV, the overlay's events and
    the job's counts are one answer for one clip."""

    def setUp(self):
        from apps.pipelines.models import Pipeline, PipelineRun
        from apps.pipelines.recount import recount_run

        self.user = User.objects.create_user("agree", password="x")
        self.client.force_login(self.user)
        device = Device.objects.create(
            owner=self.user, name="Hotel", key_hash="hag", prefix="bmk_ag",
            nest_layout=[{"id": 1, "box": [0.1, 0.1, 0.3, 0.3]}])
        self.video = Video.objects.create(
            user=self.user, device=device, title="clip", storage_key="ag/c.mp4",
            file_size_bytes=1, status=Video.Status.READY, recorded_at=timezone.now(),
            fps=25.0, width=1920, height=1080, duration_seconds=4.0)
        self.job = Job.objects.create(user=self.user, video=self.video,
                                      status="completed", modal_job_id="ag-1")
        path = _csv(AGREE_TRACKING, "agree.csv")
        JobResult.objects.create(job=self.job, tracking_csv_path=path,
                                 summary_stats={"video_fps": 25.0},
                                 total_events=99, entry_count=50, exit_count=49)
        pipeline = Pipeline.objects.create(user=self.user, title="P")
        self.run = PipelineRun.objects.create(
            pipeline=pipeline, user=self.user, status="completed",
            started_at=self.job.created_at,
            steps=[{"id": "v", "block_type": "input.video",
                    "config": {"video_id": str(self.video.pk)}},
                   {"id": "r", "block_type": "reference.layout",
                    "config": {"source": "device_layout"}, "inputs": {"video": "v"}},
                   {"id": "e", "block_type": "analyze.events", "config": {},
                    "inputs": {"tracks": "m", "rois": "r"}}],
            context={"v": {"artifact": "video", "video_id": self.video.pk},
                     "m": {"artifact": "tracks", "job_id": self.job.pk,
                           "result": {"tracking_csv_path": path,
                                      "summary_stats": {"video_fps": 25.0}}}})
        self.counts = recount_run(self.run)

    def test_the_rule_counted_one_exit_and_one_entry(self):
        self.assertEqual(self.counts, {"events": 2})

    def test_page_csv_overlay_and_job_counts_agree(self):
        import csv as _csv_mod
        import io
        import json

        page = self.client.get(reverse("analysis:results", kwargs={"pk": self.job.pk}))
        page_rows = page.context["events_data"]["rows"]
        body = self.client.get(reverse("analysis:results_csv",
                                       kwargs={"pk": self.job.pk, "kind": "events"})).content
        csv_rows = list(_csv_mod.reader(io.StringIO(body.decode())))[1:]
        overlay = json.loads(self.client.get(
            reverse("analysis:track_overlay", args=[self.job.pk]), {"part": "events"}).content)
        result = JobResult.objects.get(job=self.job)

        actions = sorted(r[3] for r in csv_rows)
        self.assertEqual(actions, ["enter", "exit"])
        self.assertEqual(len(page_rows), len(csv_rows))
        self.assertEqual(sorted(e[2] for e in overlay["events"]), actions)
        self.assertEqual((result.total_events, result.entry_count, result.exit_count), (2, 1, 1))

    def test_the_overlay_draws_the_references_counted_against(self):
        import json

        overlay = json.loads(self.client.get(
            reverse("analysis:track_overlay", args=[self.job.pk]), {"part": "events"}).content)

        self.assertEqual([r["label"] for r in overlay["regions"]], ["reference_1"])


class OnlyWhatThePipelineHasTests(OneClipAgreesTests):
    """The batch said "—" for interactions while the clip page computed 103:
    a pipeline with an Events step and no Interactions step has events and no
    interactions everywhere (memory/60)."""

    def test_the_clip_page_shows_no_interactions_for_a_pipeline_without_the_step(self):
        from apps.pipelines import aggregate

        page = self.client.get(reverse("analysis:results", kwargs={"pk": self.job.pk}))
        tiles = {t["label"]: t for t in page.context["stat_tiles"]}
        self.assertEqual(tiles["Interactions"]["value"], "—")
        self.assertIn("No Interactions step", tiles["Interactions"]["hint"])
        self.assertEqual(tiles["Events"]["value"], 2)
        self.assertIsNone(aggregate.primitive_counts(self.run)["interactions"])

    def test_no_interactions_csv_or_overlay_spells_either(self):
        import json

        csv_resp = self.client.get(reverse("analysis:results_csv",
                                           kwargs={"pk": self.job.pk, "kind": "interactions"}))
        self.assertEqual(csv_resp.status_code, 404)
        live = json.loads(self.client.get(
            reverse("analysis:track_overlay", args=[self.job.pk]), {"part": "events"}).content)
        self.assertEqual(live["interactions"], [])
        self.assertEqual(len(live["events"]), 2)
