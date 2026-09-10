"""The frame rate must never be silently invented.

`Video.fps` was declared from the first migration and written by nothing, and
the batch-aggregation path looked for a `summary_stats["fps"]` key the GPU
backend never emits (it writes `video_fps`). Both terms were always empty, so
DEFAULT_FPS won every time — while the per-video run page read the real rate
through `ops.fps_of`. The same clip reported two different dwell times
depending on which page you opened.

These tests pin 25 fps specifically: that is what the devices record at, so a
regression to the 30 fps assumption shows up as the 20% error it really is.
"""

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.utils import timezone

from apps.analysis.models import Job, JobResult
from apps.pipelines import aggregate, ops
from apps.videos.models import Video


class FpsResolutionTests(TestCase):
    def test_the_measured_rate_on_the_video_wins(self):
        video = Video(fps=25.0)

        self.assertEqual(ops.fps_with_source({"video_fps": 30.0}, video), (25.0, "video"))

    def test_the_backends_video_fps_key_is_read(self):
        # The regression: this key is the one cloud/wrapper/pipeline.py writes.
        self.assertEqual(ops.fps_with_source({"video_fps": 25.0}, None), (25.0, "analysis"))

    def test_older_runs_that_wrote_fps_or_frame_rate_still_resolve(self):
        self.assertEqual(ops.fps_of({"fps": 25.0}), 25.0)
        self.assertEqual(ops.fps_of({"frame_rate": 25.0}), 25.0)

    def test_a_missing_rate_is_reported_as_assumed_not_as_fact(self):
        fps, source = ops.fps_with_source({}, None)

        self.assertEqual(fps, ops.DEFAULT_FPS)
        self.assertEqual(source, "assumed")

    def test_zero_and_junk_are_not_mistaken_for_a_rate(self):
        for summary in ({"video_fps": 0}, {"video_fps": None}, {"video_fps": "n/a"}):
            self.assertEqual(ops.fps_with_source(summary, None)[1], "assumed", summary)

    def test_a_video_row_with_no_rate_falls_through_to_the_analysis(self):
        self.assertEqual(
            ops.fps_with_source({"video_fps": 25.0}, Video(fps=None)), (25.0, "analysis"))

    def test_aggregate_reexports_the_shared_resolver(self):
        # foraging.py and the batch page must not drift apart again.
        self.assertIs(aggregate.fps_with_source, ops.fps_with_source)
        self.assertEqual(aggregate.DEFAULT_FPS, ops.DEFAULT_FPS)


class CollectSourcesFpsTests(TestCase):
    """The batch path end-to-end: a 25 fps run must not aggregate at 30."""

    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username="fps-tester", password="x")

    def _run_with(self, summary_stats, video_fps=None):
        from apps.pipelines.models import Pipeline, PipelineRun

        video = Video.objects.create(
            user=self.user, title="clip", storage_key="k/clip.mp4",
            status=Video.Status.READY, recorded_at=timezone.now(), fps=video_fps,
            file_size_bytes=1024)
        job = Job.objects.create(user=self.user, video=video, status="completed")
        JobResult.objects.create(job=job, events_csv_path="events.csv",
                                 summary_stats=summary_stats)
        pipeline = Pipeline.objects.create(user=self.user, title="p")
        run = PipelineRun.objects.create(
            user=self.user, pipeline=pipeline, status=PipelineRun.Status.COMPLETED,
            steps=[{"id": "v", "block_type": "input.video",
                    "config": {"video_id": video.pk}}],
            context={"track": {"result": {"events_csv_path": "events.csv",
                                          "summary_stats": summary_stats}}})
        return run, video

    def test_a_25fps_run_aggregates_at_25(self):
        run, _video = self._run_with({"video_fps": 25.0})

        sources, _skipped = aggregate.collect_sources([run])

        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["fps"], 25.0)
        self.assertEqual(sources[0]["fps_source"], "analysis")

    def test_a_run_with_no_recorded_rate_is_flagged_as_assumed(self):
        run, _video = self._run_with({})

        sources, _skipped = aggregate.collect_sources([run])

        self.assertEqual(sources[0]["fps_source"], "assumed")
