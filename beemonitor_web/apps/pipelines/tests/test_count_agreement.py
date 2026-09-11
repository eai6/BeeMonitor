"""One clip, one number — on the batch row, the clip's tiles, and the CSVs.

Every surface used to count from a different place. The batch table and the
per-clip stat tiles read JobResult.total_events / interaction_count — the
WORKER's counters — while the table under the tiles and the CSV beside it
carried the analyzers' computed rows. The worker matches an insect to a
reference by centroid distance under a flat 50 px, so the two legitimately
disagree, and a clip could head a 38-row events table with "Events 0".
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
from apps.pipelines import aggregate
from apps.pipelines.models import Pipeline, PipelineRun
from apps.videos.models import Video

User = get_user_model()

BATCH = "bbbb2222-0000-4000-8000-00000000feed"
# One insect parked inside the nest tube (0.1–0.3) but far from its centre: the
# case the worker's flat 50 px rule never records.
TRACKING = [{"frame": f, "track_id": 3, "cx": 0.29, "cy": 0.29} for f in range(20)]


def _csv_file(rows, name):
    path = Path(tempfile.mkdtemp()) / name
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return str(path)


class CountAgreementTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("ca", password="x")
        self.client.force_login(self.user)
        self.device = Device.objects.create(
            owner=self.user, name="Hotel", key_hash="hca", prefix="bmk_ca",
            nest_layout=[{"id": 1, "box": [0.1, 0.1, 0.3, 0.3]}])
        self.video = Video.objects.create(
            user=self.user, device=self.device, title="clip",
            storage_key="ca/c.mp4", file_size_bytes=1, status=Video.Status.READY,
            recorded_at=timezone.now(), fps=25.0, width=1920, height=1080,
            duration_seconds=33)
        self.job = Job.objects.create(user=self.user, video=self.video,
                                      status="completed", modal_job_id="ca-1")
        self.tracking = _csv_file(TRACKING, "t.csv")
        # The worker's own counters, deliberately wrong — they are what every
        # surface used to print.
        JobResult.objects.create(
            job=self.job, tracking_csv_path=self.tracking,
            events_csv_path="", interactions_csv_path="worker.csv",
            total_events=0, interaction_count=0, unique_tracks=6,
            summary_stats={"video_fps": 25.0})
        self.pipeline = Pipeline.objects.create(user=self.user, title="P")

    def _run(self, with_analyzer=True):
        steps = [{"id": "v", "block_type": "input.video",
                  "config": {"video_id": str(self.video.pk)}},
                 {"id": "r", "block_type": "reference.layout",
                  "config": {"source": "device_layout"}, "inputs": {"video": "v"}}]
        context = {"v": {"artifact": "video", "video_id": self.video.pk},
                   "m": {"artifact": "tracks", "job_id": self.job.pk,
                         "result": {"tracking_csv_path": self.tracking,
                                    "summary_stats": {"video_fps": 25.0}}}}
        if with_analyzer:
            steps.append({"id": "a", "block_type": "analyze.interactions",
                          "config": {}, "inputs": {"tracks": "m", "rois": "r"}})
        run = PipelineRun.objects.create(
            pipeline=self.pipeline, user=self.user, batch_id=BATCH,
            status="completed", steps=steps, context=context)
        if with_analyzer:
            from apps.pipelines import executors
            rows = executors.recompute_primitive(run, "interactions")
            run.context["a"] = {"artifact": "table", "table_kind": "interactions",
                                "interaction_count": len(rows), "rows": rows}
            run.save(update_fields=["context"])
        return run

    # ── the analyzers' count is what every surface reports ──────────────

    def test_the_batch_row_counts_the_analyzer_not_the_worker(self):
        run = self._run()
        expected = len(run.context["a"]["rows"])

        self.assertGreater(expected, 0)  # the worker recorded 0 for this clip
        row = aggregate.batch_rows([run])[0]

        self.assertEqual(row["primitives"]["interactions"], expected)
        self.assertEqual(row["result"].interaction_count, 0)  # still wrong, unused

    def test_a_pipeline_with_no_analyzer_reports_nothing_not_zero(self):
        """Zero is a measurement; this pipeline never asked the question."""
        row = aggregate.batch_rows([self._run(with_analyzer=False)])[0]

        self.assertIsNone(row["primitives"]["interactions"])
        self.assertIsNone(row["primitives"]["events"])

    def test_the_batch_column_and_the_batch_csv_agree(self):
        run = self._run()
        row = aggregate.batch_rows([run])[0]

        _fields, csv_rows = aggregate.primitive_csv([run], "interactions")

        self.assertEqual(row["primitives"]["interactions"], len(csv_rows))

    def test_the_clips_tiles_count_the_tables_on_the_same_page(self):
        self._run()
        resp = self.client.get(reverse("analysis:results", kwargs={"pk": self.job.pk}))
        ctx = resp.context

        tiles = {t["label"]: t["value"] for t in ctx["stat_tiles"]}
        self.assertEqual(tiles["Interactions"], ctx["interactions_data"]["total"])
        self.assertEqual(tiles["Events"], ctx["events_data"]["total"])
        self.assertGreater(tiles["Interactions"], 0)

    def test_the_clips_tile_and_its_own_csv_download_agree(self):
        self._run()
        page = self.client.get(reverse("analysis:results", kwargs={"pk": self.job.pk}))
        dl = self.client.get(reverse("analysis:results_csv",
                                     kwargs={"pk": self.job.pk, "kind": "interactions"}))

        tiles = {t["label"]: t["value"] for t in page.context["stat_tiles"]}
        body = [line for line in dl.content.decode().splitlines() if line.strip()]

        self.assertEqual(tiles["Interactions"], len(body) - 1)  # minus the header

    def test_every_surface_in_the_batch_lands_on_the_same_number(self):
        """The whole point: the row, the clip page and both CSVs agree."""
        run = self._run()
        row = aggregate.batch_rows([run])[0]
        _f, batch_csv = aggregate.primitive_csv([run], "interactions")
        page = self.client.get(reverse("analysis:results", kwargs={"pk": self.job.pk}))
        clip_csv = self.client.get(reverse(
            "analysis:results_csv",
            kwargs={"pk": self.job.pk, "kind": "interactions"})).content.decode()
        tiles = {t["label"]: t["value"] for t in page.context["stat_tiles"]}

        counts = {
            row["primitives"]["interactions"],
            len(batch_csv),
            tiles["Interactions"],
            len([ln for ln in clip_csv.splitlines() if ln.strip()]) - 1,
            page.context["interactions_data"]["total"],
        }

        self.assertEqual(len(counts), 1, f"surfaces disagree: {counts}")


class DownloadCompletenessTests(TestCase):
    """A download that quietly omits clips is not an accurate download."""

    def setUp(self):
        self.user = User.objects.create_user("dc", password="x")
        self.pipeline = Pipeline.objects.create(user=self.user, title="P")

    def _clip(self, recorded=True, **paths):
        n = Video.objects.count() + 1
        video = Video.objects.create(
            user=self.user, title=f"c{n}", storage_key=f"dc/{n}.mp4",
            file_size_bytes=1, status=Video.Status.READY,
            recorded_at=timezone.now() if recorded else None, fps=25.0)
        return PipelineRun.objects.create(
            pipeline=self.pipeline, user=self.user, batch_id=BATCH,
            status="completed",
            steps=[{"id": "v", "block_type": "input.video",
                    "config": {"video_id": str(video.pk)}}],
            context={"v": {"artifact": "video", "video_id": video.pk},
                     "m": {"artifact": "tracks", "result": paths}})

    def test_a_detection_only_run_joins_the_export(self):
        """The old gate demanded an events CSV — written when events was the
        only download, so the Detections button exported nothing."""
        run = self._clip(detections_csv_path="d.csv")

        sources, skipped = aggregate.collect_sources([run])

        self.assertEqual(len(sources), 1)
        self.assertEqual(skipped, [])
        self.assertEqual([d["kind"] for d in aggregate.available_downloads(sources)],
                         ["detections"])

    def test_a_clip_with_no_timestamp_still_joins(self):
        run = self._clip(recorded=False, tracking_csv_path="t.csv")

        sources, skipped = aggregate.collect_sources([run])

        self.assertEqual(len(sources), 1)
        self.assertEqual(skipped, [])

    def test_an_undated_clip_exports_with_a_blank_timestamp_not_a_crash(self):
        run = self._clip(recorded=False, tracking_csv_path="t.csv")
        sources, _ = aggregate.collect_sources([run])

        prov = aggregate._provenance(sources[0])

        self.assertEqual(prov["video_recorded_at"], "")

    def test_undated_clips_sort_last_without_comparing_to_none(self):
        dated = self._clip(tracking_csv_path="t.csv")
        undated = self._clip(recorded=False, tracking_csv_path="t.csv")

        sources, _ = aggregate.collect_sources([undated, dated])

        self.assertIsNotNone(sources[0]["recorded_at"])
        self.assertIsNone(sources[-1]["recorded_at"])

    def test_a_run_that_produced_no_csv_at_all_is_reported_not_hidden(self):
        run = self._clip()

        sources, skipped = aggregate.collect_sources([run])

        self.assertEqual(sources, [])
        self.assertEqual(len(skipped), 1)


class EventsTableMatchesItsDownloadTests(TestCase):
    """The events table used to fall back to the worker's CSV; its button did not.

    In exactly that case the page showed rows the download did not contain. The
    analyzer already folds the worker's own nest events into its output, so the
    fallback was substituting a different answer rather than adding one.
    """

    def setUp(self):
        self.user = User.objects.create_user("em", password="x")
        self.client.force_login(self.user)
        self.video = Video.objects.create(
            user=self.user, title="clip", storage_key="em/c.mp4",
            file_size_bytes=1, status=Video.Status.READY,
            recorded_at=timezone.now(), fps=25.0, width=1920, height=1080)
        self.job = Job.objects.create(user=self.user, video=self.video,
                                      status="completed", modal_job_id="em-1")
        # A worker events CSV with rows, and no tracking the analyzers can read.
        JobResult.objects.create(
            job=self.job,
            events_csv_path=_csv_file([{"frame": 1, "action": "Entry", "nest": "4"}],
                                      "e.csv"),
            total_events=1, summary_stats={"video_fps": 25.0})

    def test_the_table_the_tile_and_the_download_agree(self):
        page = self.client.get(reverse("analysis:results", kwargs={"pk": self.job.pk}))
        dl = self.client.get(reverse("analysis:results_csv",
                                     kwargs={"pk": self.job.pk, "kind": "events"}))

        tiles = {t["label"]: t["value"] for t in page.context["stat_tiles"]}
        body = [ln for ln in dl.content.decode().splitlines() if ln.strip()]
        download_rows = max(len(body) - 1, 0)

        self.assertEqual(page.context["events_data"]["total"], download_rows)
        self.assertEqual(tiles["Events"], download_rows)
