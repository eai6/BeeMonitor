"""End-to-end run advancement, plus the GPU step cache.

No SageMaker is involved: a GPU step only creates an ``analysis.Job`` row in
QUEUED, and ``_drain_queue`` (which would spawn it) is never called here. The run
is then driven forward by handing ``on_job_finished`` a completed Job, exactly as
the poller and the reconciler do.
"""

from django.contrib.auth import get_user_model
from django.test import TestCase

from apps.analysis.models import Job, JobResult
from apps.devices.models import Device
from apps.pipelines import engine
from apps.pipelines.models import Pipeline, PipelineRun, StepResult
from apps.videos.models import Video

User = get_user_model()


class EngineTestCase(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("alice", password="x")
        self.device = Device.objects.create(
            owner=self.user, name="Danniella", key_hash="h1", prefix="bmk_1",
            roi_override=[0.0, 0.0, 0.6, 0.6],
        )
        self.video = Video.objects.create(
            user=self.user, device=self.device, title="clip",
            storage_key="alice/clip.mp4", file_size_bytes=1,
            status=Video.Status.READY,
        )

    def _steps(self, extra=None, detector_config=None):
        steps = [
            {"id": "v", "block_type": "input.video",
             "config": {"video_id": str(self.video.pk)}},
            {"id": "d", "block_type": "detect.objects",
             "config": detector_config or {"reference_source": "device_layout"},
             "inputs": {"video": "v"}},
            {"id": "m", "block_type": "track.mot", "config": {"tracker": "beetrack"},
             "inputs": {"detections": "d"}},
            {"id": "f", "block_type": "analyze.foraging_trips",
             "config": {"event_confidence": 0.6}, "inputs": {"tracks": "m"}},
        ]
        return steps + (extra or [])

    def _start(self, steps):
        pipeline = Pipeline.objects.create(user=self.user, title="P", steps=steps)
        run = PipelineRun.objects.create(pipeline=pipeline, user=self.user)
        engine.start_run(run, steps=steps)
        run.refresh_from_db()
        return run

    def _finish_job(self, job, **result_fields):
        JobResult.objects.create(job=job, **({
            # Local (nonexistent) path: analyzers fall back to the job summary
            # without any network round-trip.
            "tracking_csv_path": "/tmp/bm-test-tracking.csv",
            "foraging_trip_count": 3,
            "avg_trip_duration_sec": 42.0,
            "unique_tracks": 5,
        } | result_fields))
        job.status = Job.Status.COMPLETED
        job.save(update_fields=["status"])
        engine.on_job_finished(job)


class RunAdvancementTests(EngineTestCase):
    def test_gpu_step_queues_a_tagged_job_and_parks_the_run(self):
        run = self._start(self._steps())

        job = Job.objects.get()
        self.assertEqual(job.status, Job.Status.QUEUED)
        self.assertEqual(job.config["pipeline_run_id"], str(run.pk))
        self.assertEqual(job.config["pipeline_step_id"], "d")
        self.assertEqual(run.status, PipelineRun.Status.RUNNING)
        self.assertEqual(run.step_state("d"), PipelineRun.STEP_RUNNING)
        self.assertEqual(run.step_state("m"), PipelineRun.STEP_PENDING)

    def test_completion_advances_through_mot_to_the_analyzer(self):
        run = self._start(self._steps())
        self._finish_job(Job.objects.get())

        run.refresh_from_db()
        self.assertEqual(run.status, PipelineRun.Status.COMPLETED)
        self.assertEqual(run.step_state("m"), PipelineRun.STEP_DONE)
        # MOT relabelled the fused result; the analyzer read the trips off it.
        self.assertEqual(run.context["m"]["artifact"], "tracks")
        self.assertEqual(run.context["m"]["unique_tracks"], 5)
        self.assertEqual(run.context["f"]["foraging_trip_count"], 3)

    def test_failed_job_cascades_to_downstream_steps(self):
        run = self._start(self._steps())
        job = Job.objects.get()
        job.status = Job.Status.FAILED
        job.error_message = "endpoint exploded"
        job.save(update_fields=["status", "error_message"])
        engine.on_job_finished(job)

        run.refresh_from_db()
        self.assertEqual(run.status, PipelineRun.Status.FAILED)
        for step_id in ("d", "m", "f"):
            self.assertEqual(run.step_state(step_id), PipelineRun.STEP_FAILED, step_id)

    def test_reference_only_detector_fails_the_mot_step_with_a_reason(self):
        steps = self._steps(detector_config={
            "reference_source": "device_layout", "run_scope": "reference_only"})
        run = self._start(steps)
        self._finish_job(Job.objects.get(), tracking_csv_path="")

        run.refresh_from_db()
        self.assertEqual(run.step_state("d"), PipelineRun.STEP_DONE)
        self.assertEqual(run.step_state("m"), PipelineRun.STEP_FAILED)
        self.assertIn("Reference only", run.context["m"]["error"])

    def test_steps_with_video_binds_every_input_step(self):
        pipeline = Pipeline.objects.create(
            user=self.user, title="P",
            steps=[{"id": "v", "block_type": "input.video", "config": {}}],
        )
        bound = engine.steps_with_video(pipeline, self.video.pk)
        self.assertEqual(bound[0]["config"]["video_id"], str(self.video.pk))

    def test_launch_batch_shares_one_batch_id(self):
        second = Video.objects.create(
            user=self.user, device=self.device, title="clip2",
            storage_key="alice/clip2.mp4", file_size_bytes=1,
            status=Video.Status.READY,
        )
        pipeline = Pipeline.objects.create(user=self.user, title="P", steps=self._steps())

        batch_id, launched, invalid = engine.launch_batch(
            pipeline, [self.video, second], self.user,
        )

        self.assertEqual(len(launched), 2)
        self.assertEqual(invalid, 0)
        self.assertEqual(
            set(PipelineRun.objects.values_list("batch_id", flat=True)), {batch_id},
        )


class StepCacheTests(EngineTestCase):
    def test_identical_rerun_reuses_the_gpu_result(self):
        run = self._start(self._steps())
        self._finish_job(Job.objects.get())
        self.assertEqual(StepResult.objects.count(), 1)

        second = self._start(self._steps())

        self.assertEqual(Job.objects.count(), 1)  # no new SageMaker work
        self.assertTrue(second.context["d"]["_cached"])
        self.assertEqual(second.status, PipelineRun.Status.COMPLETED)

    def test_changing_confidence_forces_a_new_job(self):
        self._start(self._steps())
        self._finish_job(Job.objects.get())

        self._start(self._steps(detector_config={
            "reference_source": "device_layout", "confidence": 0.9}))

        self.assertEqual(Job.objects.count(), 2)

    def test_adding_a_marker_node_does_not_bust_the_cache(self):
        """Regression: identify.marker used to force a full GPU re-run.

        It set identify_bees/marker_method on the job config — keys nothing
        downstream consumed — and the cache key hashes that config, so authoring
        a marker node silently re-billed an already-computed video.
        """
        self._start(self._steps())
        self._finish_job(Job.objects.get())

        with_marker = self._steps(extra=[
            {"id": "i", "block_type": "identify.marker",
             "config": {"marker_type": "qr"}, "inputs": {"tracks": "m"}},
        ])
        second = self._start(with_marker)

        self.assertEqual(Job.objects.count(), 1)
        self.assertTrue(second.context["d"]["_cached"])


class ReconcileTests(EngineTestCase):
    def test_replays_a_missed_completion(self):
        run = self._start(self._steps())
        job = Job.objects.get()
        JobResult.objects.create(job=job, tracking_csv_path="/tmp/bm-test-tracking.csv",
                                 foraging_trip_count=1)
        # Job finished but the hook never fired (deploy restart mid-poll).
        Job.objects.filter(pk=job.pk).update(status=Job.Status.COMPLETED)

        engine.reconcile_user_runs(self.user)

        run.refresh_from_db()
        self.assertEqual(run.status, PipelineRun.Status.COMPLETED)

    def test_vanished_job_fails_the_step(self):
        run = self._start(self._steps())
        Job.objects.all().delete()

        engine.reconcile_user_runs(self.user)

        run.refresh_from_db()
        self.assertEqual(run.status, PipelineRun.Status.FAILED)
        self.assertIn("no longer exists", run.context["d"]["error"])
