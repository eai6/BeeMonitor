"""Fixing a clip's tubes doesn't make a re-run track it again (memory/65).

Tracking uses only the ROI; the tube positions (``nest_layout``) went to the
worker's own events, which nothing reads any more (memory/60). They were in the
GPU cache key, so a clip whose tubes were fixed after its batch ran tracked
again on a re-run. They're out of the key now, and a result cached under the
old key is still found through its job's config.
"""

import uuid

from apps.analysis.models import Job
from apps.pipelines import engine, executors
from apps.pipelines.models import PipelineRun, StepResult

from .test_primitive_blocks import PrimitiveBlockTestCase


class TubesNotInCacheKeyTests(PrimitiveBlockTestCase):
    def _run(self):
        steps = self._steps("analyze.events")
        run = PipelineRun.objects.create(pipeline=self.pipeline, user=self.user, steps=steps)
        run.context = {"v": {"artifact": "video", "video_id": self.video.pk}}
        run.save()
        return run, steps

    def _gpu(self, steps):
        return next(i for i, s in enumerate(steps)
                    if (executors.get_block(s["block_type"]) or {}).get("backend") == "gpu")

    def _key(self):
        run, steps = self._run()
        i = self._gpu(steps)
        return engine._gpu_cache_key(run, steps[i], run.context, i)

    def test_moving_the_tubes_keeps_the_key(self):
        run, steps = self._run()
        i = self._gpu(steps)
        built, _ = executors.build_job_config(steps[i], run, run.context, i)
        self.assertIn("nest_layout", built["config"])    # still sent to the worker
        before = self._key()
        self.device.nest_layout = [{"id": 1, "box": [0.15, 0.12, 0.33, 0.31]},
                                   {"id": 2, "box": [0.4, 0.1, 0.5, 0.2]}]
        self.device.save()
        self.assertEqual(self._key(), before)

    def test_moving_the_roi_changes_the_key(self):
        before = self._key()
        self.device.roi_override = [0.0, 0.0, 0.7, 0.7]
        self.device.save()
        self.assertNotEqual(self._key(), before)

    def test_a_result_cached_under_the_old_key_is_still_found(self):
        run, steps = self._run()
        i = self._gpu(steps)
        built, _ = executors.build_job_config(steps[i], run, run.context, i)
        old_config = {**built["config"], "nest_layout": [{"id": 9, "box": [0, 0, 0.1, 0.1]}],
                      "pipeline_run_id": str(uuid.uuid4()), "pipeline_step_id": "d"}
        job = Job.objects.create(user=self.user, video=self.video, status="completed",
                                 modal_job_id="old", config=old_config)
        StepResult.objects.create(user=self.user, cache_key="o" * 64,
                                  block_type=steps[i]["block_type"],
                                  output={"artifact": "tracks", "job_id": job.pk})
        key = engine._gpu_cache_key(run, steps[i], run.context, i)

        hit = engine._cached_step(run, steps[i], run.context, i, key)

        self.assertIsNotNone(hit)
        self.assertEqual(hit.output["job_id"], job.pk)
        # Saved under the new key, so the next run finds it directly.
        self.assertTrue(StepResult.objects.filter(user=self.user, cache_key=key).exists())

    def test_a_job_with_a_different_roi_is_not_reused(self):
        run, steps = self._run()
        i = self._gpu(steps)
        built, _ = executors.build_job_config(steps[i], run, run.context, i)
        job = Job.objects.create(user=self.user, video=self.video, status="completed",
                                 modal_job_id="old",
                                 config={**built["config"], "hotel_roi": [0, 0, 1, 1]})
        StepResult.objects.create(user=self.user, cache_key="o" * 64,
                                  block_type=steps[i]["block_type"],
                                  output={"artifact": "tracks", "job_id": job.pk})
        key = engine._gpu_cache_key(run, steps[i], run.context, i)
        self.assertIsNone(engine._cached_step(run, steps[i], run.context, i, key))
