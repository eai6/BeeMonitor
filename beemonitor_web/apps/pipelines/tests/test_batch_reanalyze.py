"""Re-asking a finished batch a different question costs no GPU time.

Detection and tracking are the expensive part and they do not depend on which
analyzer reads them. `engine._gpu_cache_key` hashes the clip and the GPU step's
own config, so swapping a LOCAL analyzer leaves that key untouched and every
clip is served from cache. Before this, moving a twelve-clip batch onto a new
analyzer meant opening twelve runs and swapping each by hand.
"""

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from apps.pipelines import engine
from apps.pipelines.models import Pipeline, PipelineRun
from apps.videos.models import Video

User = get_user_model()


class BatchReanalyzeTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("ra", password="x")
        self.client.force_login(self.user)
        self.pipeline = Pipeline.objects.create(user=self.user, title="P", steps=[])
        self.batch_id = "3f9d1c22-0000-4000-8000-00000000abcd"
        self.videos = [
            Video.objects.create(user=self.user, title=f"clip{i}",
                                 storage_key=f"ra/c{i}.mp4", file_size_bytes=1,
                                 status=Video.Status.READY,
                                 recorded_at=timezone.now())
            for i in range(2)
        ]
        self.runs = [self._run(v, "completed") for v in self.videos]

    def _steps(self, video, analyzer="analyze.visitation"):
        return [
            {"id": "v", "block_type": "input.video",
             "config": {"video_id": str(video.pk)}},
            {"id": "d", "block_type": "detect.objects", "config": {"label": "bee"},
             "inputs": {"video": "v"}},
            {"id": "m", "block_type": "track.mot", "config": {"tracker": "beetrack"},
             "inputs": {"detections": "d"}},
            {"id": "a", "block_type": analyzer, "config": {}, "inputs": {"tracks": "m"}},
        ]

    def _run(self, video, status, analyzer="analyze.visitation"):
        return PipelineRun.objects.create(
            pipeline=self.pipeline, user=self.user, batch_id=self.batch_id,
            status=status, steps=self._steps(video, analyzer),
            context={"v": {"artifact": "video", "video_id": video.pk},
                     "m": {"artifact": "tracks",
                           "result": {"tracking_csv_path": "t.csv"}}},
        )

    def _post(self, block_type):
        return self.client.post(
            reverse("pipelines:batch_reanalyze", kwargs={"batch_id": self.batch_id}),
            {"block_type": block_type})

    def test_it_launches_a_new_batch_with_the_swapped_analyzer(self):
        before = PipelineRun.objects.count()

        resp = self._post("analyze.interactions")

        self.assertEqual(resp.status_code, 302)
        new = PipelineRun.objects.exclude(batch_id=self.batch_id)
        self.assertEqual(new.count(), PipelineRun.objects.count() - before)
        for run in new:
            kinds = [s["block_type"] for s in run.steps
                     if s["block_type"].startswith("analyze.")]
            self.assertEqual(kinds, ["analyze.interactions"])

    def test_the_original_batch_keeps_its_own_results(self):
        self._post("analyze.interactions")

        for run in self.runs:
            run.refresh_from_db()
            kinds = [s["block_type"] for s in run.steps
                     if s["block_type"].startswith("analyze.")]
            self.assertEqual(kinds, ["analyze.visitation"])

    def test_each_new_run_is_bound_to_its_own_clip(self):
        self._post("analyze.interactions")

        bound = set()
        for run in PipelineRun.objects.exclude(batch_id=self.batch_id):
            for step in run.steps:
                if step["block_type"] == "input.video":
                    bound.add(step["config"]["video_id"])
        self.assertEqual(bound, {str(v.pk) for v in self.videos})

    def test_the_new_runs_are_not_marked_fresh_so_the_cache_is_used(self):
        """Reuse is the whole point — `fresh` would re-pay for the GPU."""
        self._post("analyze.interactions")

        for run in PipelineRun.objects.exclude(batch_id=self.batch_id):
            self.assertFalse(run.fresh)

    def test_the_new_block_gets_its_own_default_config(self):
        self._post("analyze.interactions")

        run = PipelineRun.objects.exclude(batch_id=self.batch_id).first()
        cfg = next(s["config"] for s in run.steps
                   if s["block_type"] == "analyze.interactions")
        self.assertEqual(cfg["interaction_type"], "all")
        self.assertEqual(cfg["gap_frames"], 15)

    def test_failed_clips_are_left_out_rather_than_failing_again(self):
        video = Video.objects.create(user=self.user, title="bad",
                                     storage_key="ra/bad.mp4", file_size_bytes=1,
                                     status=Video.Status.READY,
                                     recorded_at=timezone.now())
        self._run(video, "failed")

        self._post("analyze.interactions")

        launched = {s["config"]["video_id"]
                    for run in PipelineRun.objects.exclude(batch_id=self.batch_id)
                    for s in run.steps if s["block_type"] == "input.video"}
        self.assertNotIn(str(video.pk), launched)

    def test_a_batch_with_nothing_completed_is_refused_with_a_reason(self):
        PipelineRun.objects.filter(batch_id=self.batch_id).update(status="failed")

        resp = self._post("analyze.interactions")

        self.assertEqual(resp.status_code, 302)
        self.assertEqual(PipelineRun.objects.exclude(batch_id=self.batch_id).count(), 0)

    def test_a_non_analyzer_block_is_refused(self):
        resp = self._post("detect.objects")

        self.assertEqual(resp.status_code, 302)
        self.assertEqual(PipelineRun.objects.exclude(batch_id=self.batch_id).count(), 0)

    def test_someone_elses_batch_is_forbidden(self):
        other = User.objects.create_user("nosy", password="x")
        self.client.force_login(other)

        resp = self._post("analyze.interactions")

        self.assertIn(resp.status_code, (403, 404))


class LaunchBatchStepsOverrideTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("lb", password="x")
        self.pipeline = Pipeline.objects.create(user=self.user, title="P", steps=[
            {"id": "v", "block_type": "input.video", "config": {}},
        ])

    def test_the_override_is_used_instead_of_the_saved_graph(self):
        video = Video.objects.create(user=self.user, title="c", storage_key="lb/c.mp4",
                                     file_size_bytes=1, status=Video.Status.READY)
        override = [{"id": "v", "block_type": "input.video", "config": {}}]

        _batch, launched, _invalid = engine.launch_batch(
            self.pipeline, [video], self.user, steps=override)

        self.assertEqual(launched, [video.pk])

    def test_the_override_is_not_mutated_across_clips(self):
        """One override serves the whole batch — a shared dict bound to the
        first clip would send every run at the same video."""
        videos = [Video.objects.create(user=self.user, title=f"c{i}",
                                       storage_key=f"lb/c{i}.mp4", file_size_bytes=1,
                                       status=Video.Status.READY) for i in range(3)]
        override = [{"id": "v", "block_type": "input.video", "config": {}}]

        engine.launch_batch(self.pipeline, videos, self.user, steps=override)

        self.assertEqual(override[0]["config"], {})   # untouched
        bound = {s["config"]["video_id"]
                 for run in PipelineRun.objects.all()
                 for s in run.steps if s["block_type"] == "input.video"}
        self.assertEqual(bound, {str(v.pk) for v in videos})
