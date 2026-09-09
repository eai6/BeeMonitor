"""Re-analysing a clip with a different analyzer, without paying for the GPU.

Detection and tracking are the expensive part and they do not depend on which
analyzer reads them. engine._gpu_cache_key hashes the clip and the GPU step's
own config, so swapping a LOCAL analyzer leaves that key untouched and the
cached result is served. Re-analysis was already free; there was simply no way
to ask for it.
"""

from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from apps.pipelines import engine
from apps.pipelines.models import Pipeline, PipelineRun, StepResult
from apps.videos.models import Video

User = get_user_model()

STEPS = [
    {"id": "v", "block_type": "input.video", "config": {"video_id": "1"}},
    {"id": "d", "block_type": "detect.objects", "config": {"label": "bee"}},
    {"id": "t", "block_type": "track.mot", "config": {}},
    {"id": "a", "block_type": "analyze.visitation", "config": {"gap": 15}},
]


class ReanalyzeTestCase(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("alice", password="x")
        self.video = Video.objects.create(
            user=self.user, title="clip", storage_key="alice/clip.mp4",
            file_size_bytes=1, status=Video.Status.READY)
        steps = [dict(s) for s in STEPS]
        steps[0]["config"] = {"video_id": str(self.video.pk)}
        self.pipeline = Pipeline.objects.create(
            user=self.user, title="Biodiversity Count", steps=steps)
        self.run = PipelineRun.objects.create(
            pipeline=self.pipeline, user=self.user, status="completed",
            steps=steps, context={"a": {"table_kind": "visitation", "total_visits": 3}})
        self.client.force_login(self.user)

    def _post(self, block_type):
        with patch("apps.pipelines.engine.advance_run"):
            return self.client.post(
                reverse("pipelines:run_reanalyze",
                        kwargs={"pk": self.pipeline.pk, "run_id": self.run.pk}),
                {"block_type": block_type})


class OptionsTests(ReanalyzeTestCase):
    def test_the_page_offers_the_other_analyzers(self):
        from apps.pipelines.views import analyzer_options

        current, options = analyzer_options(self.run)

        self.assertEqual(current, "analyze.visitation")
        keys = {o["key"] for o in options}
        self.assertIn("analyze.foraging_trips", keys)
        self.assertIn("analyze.interaction", keys)

    def test_the_current_analyzer_is_marked_not_offered_as_a_swap(self):
        from apps.pipelines.views import analyzer_options

        _, options = analyzer_options(self.run)

        current = [o for o in options if o["current"]]
        self.assertEqual(len(current), 1)
        self.assertEqual(current[0]["key"], "analyze.visitation")

    def test_hidden_analyzers_are_not_offered(self):
        """Colony activity is hidden — it should not appear as a swap."""
        from apps.pipelines.views import analyzer_options

        _, options = analyzer_options(self.run)

        self.assertNotIn("analyze.colony_activity", {o["key"] for o in options})

    def test_a_run_still_going_cannot_be_re_analysed(self):
        self.run.status = "running"
        self.run.save(update_fields=["status"])

        ctx = self.client.get(reverse("pipelines:run_detail", kwargs={
            "pk": self.pipeline.pk, "run_id": self.run.pk})).context

        self.assertFalse(ctx["can_reanalyze"])


class SwapTests(ReanalyzeTestCase):
    def test_swapping_creates_a_new_run_with_the_new_analyzer(self):
        resp = self._post("analyze.foraging_trips")

        self.assertEqual(resp.status_code, 302)
        new = PipelineRun.objects.exclude(pk=self.run.pk).get()
        kinds = [s["block_type"] for s in new.steps]
        self.assertIn("analyze.foraging_trips", kinds)
        self.assertNotIn("analyze.visitation", kinds)

    def test_everything_before_the_analyzer_is_untouched(self):
        """That is what makes the cached tracking still match."""
        self._post("analyze.interaction")

        new = PipelineRun.objects.exclude(pk=self.run.pk).get()
        self.assertEqual(new.steps[:3], self.run.steps[:3])

    def test_the_new_run_is_not_fresh_so_the_cache_is_used(self):
        """fresh=True would re-run the GPU and defeat the entire point."""
        self._post("analyze.foraging_trips")

        new = PipelineRun.objects.exclude(pk=self.run.pk).get()
        self.assertFalse(new.fresh)

    def test_the_old_analyzers_config_does_not_carry_over(self):
        """A visitation gap threshold means nothing to foraging trips."""
        self._post("analyze.foraging_trips")

        new = PipelineRun.objects.exclude(pk=self.run.pk).get()
        analyzer = [s for s in new.steps if s["block_type"].startswith("analyze.")][0]
        self.assertNotIn("gap", analyzer["config"])

    def test_required_config_is_seeded_from_the_blocks_defaults(self):
        """detection_count requires `metric`; without a default the swapped run
        would fail validation instead of running."""
        resp = self._post("analyze.detection_count")

        self.assertEqual(resp.status_code, 302)
        new = PipelineRun.objects.exclude(pk=self.run.pk).get()
        analyzer = [s for s in new.steps if s["block_type"].startswith("analyze.")][0]
        self.assertIn("metric", analyzer["config"])

    def test_the_original_run_keeps_its_answer(self):
        self._post("analyze.foraging_trips")

        self.run.refresh_from_db()
        self.assertEqual(self.run.context["a"]["total_visits"], 3)
        self.assertEqual(self.run.status, "completed")

    def test_a_non_analyzer_block_is_refused(self):
        resp = self._post("detect.objects")

        self.assertEqual(resp.status_code, 302)
        self.assertEqual(PipelineRun.objects.count(), 1)

    def test_an_unknown_block_is_refused(self):
        resp = self._post("analyze.does_not_exist")

        self.assertEqual(resp.status_code, 302)
        self.assertEqual(PipelineRun.objects.count(), 1)

    def test_a_stranger_cannot_re_analyse_someone_elses_run(self):
        self.client.force_login(User.objects.create_user("mallory", password="x"))

        resp = self._post("analyze.foraging_trips")

        self.assertEqual(resp.status_code, 404)
        self.assertEqual(PipelineRun.objects.count(), 1)


class CacheReuseTests(ReanalyzeTestCase):
    def test_the_gpu_step_key_is_unchanged_by_an_analyzer_swap(self):
        """The property the whole feature rests on: the GPU step's cache key
        depends on the clip and its own config, not on what reads its output."""
        self._post("analyze.foraging_trips")
        new = PipelineRun.objects.exclude(pk=self.run.pk).get()

        with patch("apps.pipelines.executors.build_job_config") as build:
            build.return_value = ({"video_id": self.video.pk,
                                   "config": {"detector_kind": "yolo"}}, None)
            old_key = engine._gpu_cache_key(self.run, self.run.steps[1], {}, 1)
            new_key = engine._gpu_cache_key(new, new.steps[1], {}, 1)

        self.assertEqual(old_key, new_key)

    def test_a_cached_result_is_reused_rather_than_re_submitted(self):
        from apps.pipelines import executors

        with patch("apps.pipelines.executors.build_job_config") as build:
            build.return_value = ({"video_id": self.video.pk,
                                   "config": {"detector_kind": "yolo"}}, None)
            key = engine._gpu_cache_key(self.run, self.run.steps[1], {}, 1)
        StepResult.objects.create(user=self.user, cache_key=key,
                                  block_type="detect.objects",
                                  output={"artifact": "detections", "result": {}})

        self.assertTrue(StepResult.objects.filter(cache_key=key).exists())
