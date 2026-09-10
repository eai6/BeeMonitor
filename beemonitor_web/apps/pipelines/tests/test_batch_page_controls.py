"""What a finished batch page offers: the base tables, and another pipeline.

The clip table used to carry a Trips column whatever the pipeline measured, so
a Detection Count batch showed a column of zeros for a question it never asked.
The columns are the base measurements now — trips, visits and dwell are reads
over them, and the reader can do their own.
"""

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from apps.pipelines.models import Pipeline, PipelineRun
from apps.videos.models import Video

User = get_user_model()


class BatchPageControlTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("bp", password="x")
        self.client.force_login(self.user)
        self.pipeline = Pipeline.objects.create(user=self.user, title="Ran This")
        self.other = Pipeline.objects.create(user=self.user, title="Something Else")
        self.template = Pipeline.objects.create(user=self.user, title="A Template",
                                                is_template=True)
        self.batch_id = "7c1e9a44-0000-4000-8000-0000000012ab"
        self.video = Video.objects.create(
            user=self.user, title="clip", storage_key="bp/c.mp4", file_size_bytes=1,
            status=Video.Status.READY, recorded_at=timezone.now(),
            duration_seconds=37.0)
        PipelineRun.objects.create(
            pipeline=self.pipeline, user=self.user, batch_id=self.batch_id,
            status="completed",
            steps=[{"id": "v", "block_type": "input.video",
                    "config": {"video_id": str(self.video.pk)}}],
            context={"v": {"artifact": "video", "video_id": self.video.pk}})

    def _page(self):
        return self.client.get(
            reverse("pipelines:batch_detail", kwargs={"batch_id": self.batch_id}))

    def test_the_clip_table_reports_base_measurements(self):
        html = self._page().content.decode()

        for column in ("Tracks", "Events", "Interactions", "Length", "GPU time"):
            self.assertIn(f">{column}</div>", html, column)

    def test_the_trips_column_is_gone(self):
        """It showed zeros on every pipeline that never measured trips."""
        html = self._page().content.decode()

        self.assertNotIn('class="text-right">Trips</div>', html)

    def test_another_pipeline_can_be_run_over_the_same_clips(self):
        html = self._page().content.decode()

        self.assertIn(reverse("pipelines:run_on_videos"), html)
        self.assertIn(f'value="{self.video.pk}"', html)

    def test_the_pipeline_that_already_ran_is_not_offered(self):
        resp = self._page()

        titles = [p.title for p in resp.context["rerun_pipelines"]]
        self.assertNotIn("Ran This", titles)
        self.assertIn("Something Else", titles)

    def test_templates_are_offered_too(self):
        titles = [p.title for p in self._page().context["rerun_pipelines"]]

        self.assertIn("A Template", titles)

    def test_someone_elses_pipelines_are_not_offered(self):
        Pipeline.objects.create(
            user=User.objects.create_user("nosy", password="x"), title="Theirs")

        titles = [p.title for p in self._page().context["rerun_pipelines"]]

        self.assertNotIn("Theirs", titles)

    def test_the_visitation_panel_is_gone(self):
        """It answered a question Interactions answers, and with no references
        it rendered four zeros and an apology."""
        html = self._page().content.decode()

        self.assertNotIn("No references were defined", html)


class InlineClipReviewTests(TestCase):
    """Watching a clip without leaving the batch page.

    Scanning a day's footage is watching one clip after another; a round trip
    to a detail page between each turns a minute of review into ten.
    """

    def setUp(self):
        from apps.analysis.models import Job, JobResult

        self.user = User.objects.create_user("cv", password="x")
        self.client.force_login(self.user)
        self.pipeline = Pipeline.objects.create(user=self.user, title="P")
        self.batch_id = "9a2b3c44-0000-4000-8000-0000000099cd"
        self.video = Video.objects.create(
            user=self.user, title="clip", storage_key="cv/c.mp4", file_size_bytes=1,
            status=Video.Status.READY, recorded_at=timezone.now())
        self.job = Job.objects.create(user=self.user, video=self.video,
                                      status="completed")
        self.result = JobResult.objects.create(
            job=self.job, annotated_video_path="cv/annotated.mp4")
        PipelineRun.objects.create(
            pipeline=self.pipeline, user=self.user, batch_id=self.batch_id,
            status="completed",
            steps=[{"id": "v", "block_type": "input.video",
                    "config": {"video_id": str(self.video.pk)}}],
            context={"v": {"artifact": "video", "video_id": self.video.pk},
                     "t": {"job_id": self.job.pk,
                           "result": {"events_csv_path": "e.csv"}}})

    def _html(self):
        return self.client.get(
            reverse("pipelines:batch_detail", kwargs={"batch_id": self.batch_id})
        ).content.decode()

    def test_the_row_carries_both_sources(self):
        html = self._html()

        self.assertIn(
            reverse("videos:stream", kwargs={"pk": self.video.pk}), html)
        self.assertIn(
            reverse("analysis:video_proxy", kwargs={"pk": self.job.pk}), html)

    def test_a_run_without_an_annotated_video_offers_only_the_original(self):
        self.result.annotated_video_path = ""
        self.result.save(update_fields=["annotated_video_path"])

        html = self._html()

        self.assertIn(reverse("videos:stream", kwargs={"pk": self.video.pk}), html)
        self.assertNotIn("data-annotated=", html)

    def test_the_viewer_is_present_and_starts_hidden(self):
        html = self._html()

        self.assertIn('id="clip-viewer"', html)
        self.assertIn("hidden", html.split('id="clip-viewer"')[1][:40])

    def test_the_sources_are_lazy_endpoints_not_presigned_urls(self):
        """Presigning every clip up front would cost a signature per row on a
        3,000-clip batch, for videos nobody may open."""
        html = self._html()

        self.assertNotIn("X-Amz-Signature", html)


class LengthSelfHealTests(TestCase):
    """A clip's length is in its own file — "unknown" is a gap, not a fact."""

    def setUp(self):
        self.user = User.objects.create_user("lh", password="x")
        self.client.force_login(self.user)
        self.pipeline = Pipeline.objects.create(user=self.user, title="P")
        self.batch_id = "1b2c3d44-0000-4000-8000-0000000077ef"

    def _batch_of(self, n, duration=None):
        for i in range(n):
            v = Video.objects.create(
                user=self.user, title=f"c{i}", storage_key=f"lh/c{i}.mp4",
                file_size_bytes=1, status=Video.Status.READY,
                recorded_at=timezone.now(), duration_seconds=duration)
            PipelineRun.objects.create(
                pipeline=self.pipeline, user=self.user, batch_id=self.batch_id,
                status="completed",
                steps=[{"id": "v", "block_type": "input.video",
                        "config": {"video_id": str(v.pk)}}],
                context={"v": {"artifact": "video", "video_id": v.pk}})

    def _load(self):
        from unittest.mock import patch
        with patch("apps.videos.thumbnails.probe_on_demand") as probe:
            self.client.get(reverse("pipelines:batch_detail",
                                    kwargs={"batch_id": self.batch_id}))
        return probe

    def test_clips_of_unknown_length_are_probed(self):
        self._batch_of(3)

        self.assertEqual(self._load().call_count, 3)

    def test_clips_we_already_measured_are_left_alone(self):
        self._batch_of(3, duration=37.0)

        self._load().assert_not_called()

    def test_a_huge_batch_does_not_spawn_a_probe_per_clip(self):
        """Uncapped, a 3,000-clip batch spawns 3,000 threads per page load that
        almost all immediately give up on the semaphore."""
        from apps.pipelines.views import PROBES_PER_RENDER

        self._batch_of(PROBES_PER_RENDER + 5)

        self.assertEqual(self._load().call_count, PROBES_PER_RENDER)


class CombinedCsvProvenanceTests(TestCase):
    """A concatenated row must say where it came from.

    A batch can span devices and sites. Once every clip's rows are stacked into
    one file, a row that cannot name its device or site is not analysable — you
    cannot compare a treatment against a control if the two are
    indistinguishable in the export.
    """

    def setUp(self):
        from apps.devices.models import Device

        self.user = User.objects.create_user("prov", password="x")
        self.device = Device.objects.create(
            owner=self.user, name="BeeMonitor4", key_hash="hp2", prefix="bmk_p2",
            location="north hedgerow")
        self.video = Video.objects.create(
            user=self.user, device=self.device, title="clip",
            storage_key="prov/c.mp4", file_size_bytes=1,
            status=Video.Status.READY, recorded_at=timezone.now(),
            site_name="Meadow A")

    def _source(self):
        from apps.pipelines import aggregate
        return {"video": self.video, "title": "clip",
                "recorded_at": self.video.recorded_at, "fps": 25.0,
                "result": {}}

    def test_every_row_carries_device_site_and_location(self):
        from apps.pipelines import aggregate

        row = aggregate._provenance(self._source())

        self.assertEqual(row["device_name"], "BeeMonitor4")
        self.assertEqual(row["device_id"], self.device.id)
        self.assertEqual(row["site_name"], "Meadow A")
        self.assertEqual(row["location"], "north hedgerow")

    def test_provenance_leads_the_column_order(self):
        from apps.pipelines import aggregate

        self.assertEqual(aggregate.PROVENANCE_FIELDS[:3],
                         ["video_title", "video_recorded_at", "absolute_time"])
        for field in ("device_id", "device_name", "site_name", "location"):
            self.assertIn(field, aggregate.PROVENANCE_FIELDS)

    def test_a_clip_with_no_device_still_exports_cleanly(self):
        from apps.pipelines import aggregate

        self.video.device = None
        src = self._source()

        row = aggregate._provenance(src)

        self.assertEqual(row["device_name"], "")
        self.assertEqual(row["location"], "")
        self.assertEqual(row["site_name"], "Meadow A")

    def test_the_clips_own_site_is_used_not_the_devices_location(self):
        """A device can be moved between sites; the clip records where it was."""
        from apps.pipelines import aggregate

        row = aggregate._provenance(self._source())

        self.assertNotEqual(row["site_name"], row["location"])


class PrimitiveExportTests(TestCase):
    """The export must agree with the annotated video the user is watching.

    The Interactions download shipped the WORKER's interactions.csv, which
    matches an insect to a reference by centroid-to-centroid distance under a
    flat 50 px. A bee resting inside a 400 px flower sits ~200 px from its
    centre, so it appeared in the video with a box around it and not at all in
    the file. The analyzers ask containment; their rows are what gets exported.
    """

    def setUp(self):
        from apps.devices.models import Device

        self.user = User.objects.create_user("pe", password="x")
        self.client.force_login(self.user)
        self.device = Device.objects.create(owner=self.user, name="beemonitor3",
                                            key_hash="hpe", prefix="bmk_pe")
        self.pipeline = Pipeline.objects.create(user=self.user, title="P")
        self.batch_id = "5e6f7a88-0000-4000-8000-0000000055ab"
        self.video = Video.objects.create(
            user=self.user, device=self.device, title="clip",
            storage_key="pe/c.mp4", file_size_bytes=1, status=Video.Status.READY,
            recorded_at=timezone.now(), site_name="Meadow A", fps=25.0)

    def _run_with(self, output):
        return PipelineRun.objects.create(
            pipeline=self.pipeline, user=self.user, batch_id=self.batch_id,
            status="completed",
            steps=[{"id": "v", "block_type": "input.video",
                    "config": {"video_id": str(self.video.pk)}}],
            context={"v": {"artifact": "video", "video_id": self.video.pk},
                     "a": output})

    INTERACTION = {
        "artifact": "table", "table_kind": "interactions",
        "rows": [{"start_frame": 100, "end_frame": 200, "duration_sec": 4.0,
                  "a": 4, "a_kind": "organism", "b": "nest_1",
                  "b_kind": "reference", "relation": "inside",
                  "source": "derived"}],
    }

    def _download(self, kind):
        return self.client.get(reverse(
            "pipelines:batch_combined_csv",
            kwargs={"batch_id": self.batch_id, "kind": kind})).content.decode()

    def test_the_analyzers_reference_interactions_reach_the_export(self):
        self._run_with(self.INTERACTION)

        body = self._download("interactions")

        self.assertIn("organism", body)
        self.assertIn("nest_1", body)
        self.assertIn("inside", body)

    def test_exported_rows_carry_provenance_and_absolute_time(self):
        self._run_with(self.INTERACTION)

        header, first = self._download("interactions").splitlines()[:2]

        self.assertTrue(header.startswith("video_title,video_recorded_at,absolute_time"))
        self.assertIn("beemonitor3", first)
        self.assertIn("Meadow A", first)

    def test_absolute_time_uses_the_clips_real_frame_rate(self):
        self._run_with(self.INTERACTION)

        row = self._download("interactions").splitlines()[1]

        # frame 100 at 25 fps = 4s after the recording started.
        expected = (self.video.recorded_at
                    + __import__("datetime").timedelta(seconds=4.0)).isoformat()
        self.assertIn(expected, row)

    def test_columns_the_schema_does_not_fix_are_kept_not_dropped(self):
        out = dict(self.INTERACTION)
        out["rows"] = [{**self.INTERACTION["rows"][0], "b_label": "Tube 1",
                        "min_distance": 0.02}]
        self._run_with(out)

        header = self._download("interactions").splitlines()[0]

        self.assertIn("b_label", header)
        self.assertIn("min_distance", header)

    def test_events_export_the_same_way(self):
        self._run_with({"artifact": "events", "table_kind": "events",
                        "rows": [{"frame": 50, "subject": 4, "action": "enter",
                                  "target": "nest_1", "target_kind": "reference",
                                  "source": "derived"}]})

        body = self._download("events")

        self.assertIn("enter", body)
        self.assertIn("nest_1", body)

    def test_the_analyzer_table_is_flagged_on_the_page(self):
        self._run_with(self.INTERACTION)

        resp = self.client.get(reverse("pipelines:batch_detail",
                                       kwargs={"batch_id": self.batch_id}))

        by_kind = {d["kind"]: d for d in resp.context["downloads"]}
        self.assertTrue(by_kind["interactions"]["analyzed"])


class OutcomeRowLayoutTests(TestCase):
    """Outcome shares its row instead of leaving it two-thirds empty.

    With the failure card gone on a clean batch, the outcome panel sat alone in
    a 300px column with the rest of the row blank.
    """

    def setUp(self):
        self.user = User.objects.create_user("ol", password="x")
        self.client.force_login(self.user)
        self.pipeline = Pipeline.objects.create(user=self.user, title="P")
        Pipeline.objects.create(user=self.user, title="Another")
        self.batch_id = "4d5e6f88-0000-4000-8000-00000000cd12"

    def _run(self, status="completed", error=""):
        video = Video.objects.create(
            user=self.user, title="c", storage_key=f"ol/{status}.mp4",
            file_size_bytes=1, status=Video.Status.READY,
            recorded_at=timezone.now())
        return PipelineRun.objects.create(
            pipeline=self.pipeline, user=self.user, batch_id=self.batch_id,
            status=status, error_message=error,
            steps=[{"id": "v", "block_type": "input.video",
                    "config": {"video_id": str(video.pk)}}],
            context={"v": {"artifact": "video", "video_id": video.pk}})

    def _html(self):
        return self.client.get(reverse(
            "pipelines:batch_detail",
            kwargs={"batch_id": self.batch_id})).content.decode()

    def test_the_rerun_control_shares_the_outcome_row_on_a_clean_batch(self):
        self._run()
        html = self._html()

        grid = html.index("lg:grid-cols-[300px")
        table = html.index('id="clip-viewer"')
        self.assertLess(grid, html.index('id="rerun-pipeline"'))
        self.assertLess(html.index('id="rerun-pipeline"'), table)

    def test_it_is_rendered_exactly_once(self):
        """Both placements guarded, or the page shows two Run buttons."""
        self._run()

        self.assertEqual(self._html().count('id="rerun-pipeline"'), 1)

    def test_when_something_failed_the_breakdown_takes_that_space(self):
        self._run()
        self._run(status="failed", error="boom")

        html = self._html()

        self.assertIn("Why they failed", html)
        self.assertEqual(html.count('id="rerun-pipeline"'), 1)
        # Failure panel is in the row; the re-run control moved below it.
        self.assertLess(html.index("Why they failed"), html.index('id="rerun-pipeline"'))
