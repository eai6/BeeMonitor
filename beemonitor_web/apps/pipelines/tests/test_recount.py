"""Old runs are recounted from their saved tracks, with no GPU (memory/55).

The batch table, its CSVs and the clip's results page all read the rows the
run saved, so recounting a run moves them all together.
"""

from django.urls import reverse
from django.utils import timezone

from apps.analysis.models import Job, JobResult
from apps.pipelines import aggregate, executors, recount
from apps.pipelines.models import PipelineRun

from .test_primitive_blocks import PrimitiveBlockTestCase

OLD_ROWS = [{"frame": f, "action": a, "subject": 1, "target": "reference_1"}
            for f in range(0, 40, 2) for a in ("enter", "exit")]


class RecountTests(PrimitiveBlockTestCase):
    def _old_run(self, status=PipelineRun.Status.COMPLETED):
        steps = self._steps("analyze.events")
        run = PipelineRun.objects.create(pipeline=self.pipeline, user=self.user, status=status,
                                         steps=steps, counting_rules=1,
                                         started_at=timezone.now(), completed_at=timezone.now())
        run.context = {
            "v": {"artifact": "video", "video_id": self.video.pk},
            "m": {"artifact": "tracks", "result": {"tracking_csv_path": self.tracking_csv,
                                                   "summary_stats": {"video_fps": 25.0}}},
            "a": {"artifact": "events", "table_kind": "events", "rows": OLD_ROWS,
                  "event_count": len(OLD_ROWS)},
        }
        run.save()
        return run

    def test_a_recount_replaces_the_saved_rows(self):
        run = self._old_run()
        self.assertEqual(aggregate.primitive_counts(run)["events"], 40)

        counts = recount.recount_run(run)

        run.refresh_from_db()
        self.assertEqual(counts, {"events": 2})
        self.assertEqual(aggregate.primitive_counts(run)["events"], 2)
        self.assertIn("counted_at", run.context["a"])

    def test_the_sweep_recounts_each_old_finished_run_once(self):
        old = self._old_run()
        running = self._old_run(status=PipelineRun.Status.RUNNING)

        self.assertEqual(recount.sweep(), 1)
        self.assertEqual(recount.sweep(), 0)

        old.refresh_from_db()
        running.refresh_from_db()
        self.assertEqual(old.counting_rules, recount.CURRENT_RULES)
        self.assertEqual(len(old.context["a"]["rows"]), 2)
        self.assertEqual(running.counting_rules, 1)

    def test_new_runs_are_already_on_the_current_rules(self):
        run = PipelineRun.objects.create(pipeline=self.pipeline, user=self.user)
        self.assertEqual(run.counting_rules, recount.CURRENT_RULES)


class ResultsPageAgreesWithBatchTests(RecountTests):
    def _job_for(self, run):
        job = Job.objects.create(user=self.user, video=self.video, status=Job.Status.COMPLETED)
        JobResult.objects.create(job=job, tracking_csv_path=self.tracking_csv,
                                 summary_stats={"video_fps": 25.0})
        run.context["m"]["job_id"] = job.pk
        PipelineRun.objects.filter(pk=run.pk).update(context=run.context)
        return job

    def test_the_clip_page_reads_the_runs_own_rows(self):
        run = self._old_run()
        job = self._job_for(run)

        self.assertEqual(len(executors.primitives_for_job(job, "events")), 40)

    def test_the_recount_button_updates_both(self):
        run = self._old_run()
        job = self._job_for(run)
        self.client.force_login(self.user)

        self.client.post(reverse("analysis:recount", args=[job.pk]))

        run.refresh_from_db()
        self.assertEqual(aggregate.primitive_counts(run)["events"], 2)
        self.assertEqual(len(executors.primitives_for_job(job, "events")), 2)

    def test_someone_elses_clip_cant_be_recounted(self):
        from django.contrib.auth import get_user_model

        run = self._old_run()
        job = self._job_for(run)
        other = get_user_model().objects.create_user("other", password="x")
        self.client.force_login(other)

        self.assertEqual(self.client.post(reverse("analysis:recount", args=[job.pk])).status_code, 404)
