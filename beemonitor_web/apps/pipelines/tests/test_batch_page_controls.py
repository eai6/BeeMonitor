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
