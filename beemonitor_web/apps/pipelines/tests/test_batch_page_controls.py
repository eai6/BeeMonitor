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
