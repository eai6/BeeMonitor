"""The batch page as a review surface: causes, counts, money, and re-running.

Batch 5af72b17 was nine failures of two kinds. The page showed a status pill per
run and nothing else, so triaging it meant opening nine pages — and there was no
way to re-run a batch at all.
"""

from datetime import datetime, timezone as dt_tz
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from apps.analysis.models import Job, JobResult
from apps.devices.models import Device
from apps.pipelines import failures
from apps.pipelines.models import Pipeline, PipelineRun
from apps.videos.models import Video

User = get_user_model()

CPU_ERROR = ("SageMaker inference failed: Amazon SageMaker could not get a "
             "response from the beemonitor-sm-dev-sam3 endpoint.")
SAM3_ERROR = ("ImportError: cannot import name 'Sam3Model' from 'transformers' "
              "(/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)")


class ClassifyTests(TestCase):
    def test_the_two_causes_of_the_real_batch(self):
        self.assertEqual(failures.classify(CPU_ERROR)["key"], "endpoint_unresponsive")
        self.assertEqual(failures.classify(SAM3_ERROR)["key"], "sam3_import_race")

    def test_an_unmatched_message_still_gets_a_cause(self):
        cause = failures.classify("something nobody has seen before")

        self.assertEqual(cause["key"], "unknown")
        self.assertTrue(cause["title"])

    def test_empty_and_none_are_safe(self):
        for message in ("", None):
            self.assertEqual(failures.classify(message)["key"], "unknown")

    def test_grouping_orders_by_count(self):
        old = datetime(2026, 9, 9, 1, 22, tzinfo=dt_tz.utc)
        rows = [(1, SAM3_ERROR, old)] * 3 + [(2, CPU_ERROR, old)] * 6

        groups = failures.group(rows)

        self.assertEqual([g["count"] for g in groups], [6, 3])
        self.assertEqual(groups[0]["cause"]["key"], "endpoint_unresponsive")

    def test_a_cause_is_only_fixed_when_every_run_predates_the_fix(self):
        before = datetime(2026, 9, 9, 1, 0, tzinfo=dt_tz.utc)
        after = datetime(2026, 9, 10, 1, 0, tzinfo=dt_tz.utc)

        self.assertTrue(failures.group([(1, CPU_ERROR, before)])[0]["fixed"])
        self.assertFalse(failures.group([(1, CPU_ERROR, after)])[0]["fixed"])
        # One recent failure is enough to withdraw the claim.
        mixed = failures.group([(1, CPU_ERROR, before), (2, CPU_ERROR, after)])
        self.assertFalse(mixed[0]["fixed"])

    def test_upstream_is_never_mistaken_for_a_cause(self):
        """It is the consequence of another step dying, not a reason itself."""
        self.assertEqual(failures.classify("Upstream step failed.")["key"], "upstream")
        self.assertFalse(failures.classify("Upstream step failed.").get("retryable", True))


class BatchPageTestCase(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("alice", password="x")
        self.device = Device.objects.create(owner=self.user, name="BeeMonitor4",
                                            key_hash="h1", prefix="bmk_1")
        self.pipeline = Pipeline.objects.create(
            user=self.user, title="Biodiversity Count",
            steps=[{"id": "v", "block_type": "input.video", "config": {}},
                   {"id": "d", "block_type": "detect.objects",
                    "config": {"label": "bee"}}])
        self.batch = "5af72b17-0a72-4000-aa0e-cfd1d11edc30"
        self.client.force_login(self.user)

    def _run(self, hour, status, error="", tracks=None):
        video = Video.objects.create(
            user=self.user, device=self.device,
            title=f"clip{hour}-{Video.objects.count()}",
            storage_key=f"alice/{hour}-{Video.objects.count()}.mp4", file_size_bytes=1,
            status=Video.Status.READY,
            recorded_at=datetime(2026, 8, 5, hour, 0, tzinfo=dt_tz.utc))
        job = Job.objects.create(user=self.user, video=video, status=status,
                                 modal_job_id=f"j{Job.objects.count()}",
                                 execution_seconds=120, compute_cost_usd="0.0246")
        if tracks is not None:
            JobResult.objects.create(job=job, unique_tracks=tracks,
                                     total_events=tracks // 2,
                                     foraging_trip_count=tracks // 4)
        run = PipelineRun.objects.create(
            pipeline=self.pipeline, user=self.user, batch_id=self.batch,
            status=("completed" if status == "completed" else "failed"),
            started_at=datetime(2026, 9, 9, 1, 22, tzinfo=dt_tz.utc),
            steps=[{"id": "v", "block_type": "input.video",
                    "config": {"video_id": str(video.pk)}},
                   {"id": "d", "block_type": "detect.objects",
                    "config": {"label": "bee"}}],
            context={"d": {"job_id": job.pk, **({"error": error} if error else {})}},
        )
        return run, video

    def _seed_real_batch(self):
        for h in range(6):
            self._run(10 + h, "failed", CPU_ERROR)
        for h in range(3):
            self._run(16 + h, "failed", SAM3_ERROR)
        for h in range(3):
            self._run(19 + h, "completed", tracks=24)


class BatchSummaryTests(BatchPageTestCase):
    def test_the_page_reports_nine_failed_three_completed(self):
        self._seed_real_batch()

        ctx = self.client.get(reverse("pipelines:batch_detail",
                                      kwargs={"batch_id": self.batch})).context

        self.assertEqual(ctx["outcome"]["failed"], 9)
        self.assertEqual(ctx["outcome"]["completed"], 3)
        self.assertEqual(ctx["outcome"]["total"], 12)

    def test_failures_collapse_to_two_causes(self):
        self._seed_real_batch()

        groups = self.client.get(reverse("pipelines:batch_detail",
                                         kwargs={"batch_id": self.batch})).context["failure_groups"]

        self.assertEqual(len(groups), 2)
        self.assertEqual([g["count"] for g in groups], [6, 3])

    def test_time_spent_on_failures_is_reported_separately(self):
        """Work that produced nothing still consumed the GPU, and that was
        invisible. Reported as time, not money — seconds are a fact about the
        work; a price is a claim about a rate card that drifts."""
        self._seed_real_batch()

        outcome = self.client.get(reverse("pipelines:batch_detail",
                                          kwargs={"batch_id": self.batch})).context["outcome"]

        self.assertEqual(outcome["gpu_seconds_failed"], 120 * 9)
        self.assertEqual(outcome["gpu_seconds"], 120 * 12)

    def test_no_money_appears_on_the_batch_page(self):
        self._seed_real_batch()

        html = self.client.get(reverse("pipelines:batch_detail",
                                       kwargs={"batch_id": self.batch})).content.decode()

        self.assertNotIn("Cost</", html)
        self.assertNotIn("$0.", html)

    def test_a_completed_row_carries_its_numbers_and_a_failed_row_its_reason(self):
        self._seed_real_batch()

        rows = self.client.get(reverse("pipelines:batch_detail",
                                       kwargs={"batch_id": self.batch})).context["rows"]

        # Rows are newest-first, so which failure comes first is a property of
        # the clip times, not of the cause — assert on the set.
        done = [r for r in rows if r["status"] == "completed"]
        bad = [r for r in rows if r["status"] == "failed"]
        self.assertTrue(all(r["result"].unique_tracks == 24 for r in done))
        self.assertTrue(all(r["result"] is None for r in bad),
                        "a failed run has no result to show")
        errors = " ".join(r["error"] for r in bad)
        self.assertIn("could not get a response", errors)
        self.assertIn("Sam3Model", errors)

    def test_upstream_errors_do_not_mask_the_real_one(self):
        run, _ = self._run(9, "failed", CPU_ERROR)
        run.context["z"] = {"error": "Upstream step failed."}
        run.save(update_fields=["context"])

        groups = self.client.get(reverse("pipelines:batch_detail",
                                         kwargs={"batch_id": self.batch})).context["failure_groups"]

        self.assertEqual(groups[0]["cause"]["key"], "endpoint_unresponsive")


class BatchRerunTests(BatchPageTestCase):
    def _post(self, **data):
        with patch("apps.pipelines.engine.start_run") as started, \
             patch("apps.analysis.views._drain_queue"):
            resp = self.client.post(
                reverse("pipelines:batch_rerun", kwargs={"batch_id": self.batch}), data)
        return resp, started

    def test_re_running_failed_launches_only_the_failures(self):
        self._seed_real_batch()

        resp, started = self._post(scope="failed")

        self.assertEqual(resp.status_code, 302)
        self.assertEqual(started.call_count, 9)

    def test_re_running_all_launches_every_clip(self):
        self._seed_real_batch()

        _, started = self._post(scope="all", fresh="1")

        self.assertEqual(started.call_count, 12)

    def test_one_cause_can_be_re_run_on_its_own(self):
        self._seed_real_batch()

        _, started = self._post(scope="cause", cause="sam3_import_race")

        self.assertEqual(started.call_count, 3)

    def test_re_running_all_asks_for_a_fresh_run(self):
        """The cache is keyed on clip + config, not on the build — without this
        a re-run after a GPU fix returns the old result and reports success."""
        self._seed_real_batch()

        _, started = self._post(scope="all", fresh="1")

        self.assertTrue(all(c.kwargs.get("fresh") for c in started.call_args_list))

    def test_re_running_failures_reuses_the_cache(self):
        """Failures were never cached, so there is nothing stale to return."""
        self._seed_real_batch()

        _, started = self._post(scope="failed")

        self.assertFalse(any(c.kwargs.get("fresh") for c in started.call_args_list))

    def test_the_new_batch_is_separate_from_the_old_one(self):
        self._seed_real_batch()

        resp, _ = self._post(scope="failed")

        self.assertNotIn(self.batch, resp.url)

    def test_a_stranger_cannot_re_run_someone_elses_batch(self):
        self._seed_real_batch()
        self.client.force_login(User.objects.create_user("mallory", password="x"))

        resp = self.client.post(
            reverse("pipelines:batch_rerun", kwargs={"batch_id": self.batch}),
            {"scope": "all"})

        self.assertIn(resp.status_code, (403, 404))

    def test_nothing_to_re_run_is_a_message_not_a_crash(self):
        self._run(9, "completed", tracks=4)

        resp, started = self._post(scope="failed")

        self.assertEqual(resp.status_code, 302)
        self.assertEqual(started.call_count, 0)


class FreshRunTests(BatchPageTestCase):
    def test_a_fresh_run_records_the_flag(self):
        from apps.pipelines import engine

        run = PipelineRun.objects.create(pipeline=self.pipeline, user=self.user)
        with patch("apps.pipelines.engine.advance_run"):
            engine.start_run(run, steps=self.pipeline.steps, fresh=True)

        run.refresh_from_db()
        self.assertTrue(run.fresh)

    def test_an_ordinary_run_does_not(self):
        from apps.pipelines import engine

        run = PipelineRun.objects.create(pipeline=self.pipeline, user=self.user)
        with patch("apps.pipelines.engine.advance_run"):
            engine.start_run(run, steps=self.pipeline.steps)

        run.refresh_from_db()
        self.assertFalse(run.fresh)



class RunHistoryPagingTests(BatchPageTestCase):
    """Run history is paginated and filterable.

    It used to render 100 runs beneath every batch ever launched — on an account
    with a 3,250-video batch that is one scroll with no end, and nothing older
    than the last hundred is reachable at all.
    """

    def test_a_page_holds_25_runs_not_everything(self):
        for h in range(30):
            self._run(h % 24, "completed", tracks=1)

        page = self.client.get(reverse("pipelines:run_list")).context["page"]

        self.assertEqual(len(page.object_list), 25)
        self.assertTrue(page.has_next())
        self.assertEqual(page.paginator.count, 30)

    def test_older_runs_are_reachable(self):
        for h in range(30):
            self._run(h % 24, "completed", tracks=1)

        page = self.client.get(reverse("pipelines:run_list"), {"page": 2}).context["page"]

        self.assertEqual(len(page.object_list), 5)
        self.assertFalse(page.has_next())

    def test_failures_can_be_isolated(self):
        self._seed_real_batch()

        ctx = self.client.get(reverse("pipelines:run_list"), {"status": "failed"}).context

        self.assertEqual(ctx["page"].paginator.count, 9)
        self.assertTrue(all(r["run"].status == "failed" for r in ctx["rows"]))

    def test_the_counts_describe_the_whole_history_not_the_page(self):
        self._seed_real_batch()

        counts = self.client.get(reverse("pipelines:run_list")).context["run_counts"]

        self.assertEqual(counts["all"], 12)
        self.assertEqual(counts["failed"], 9)
        self.assertEqual(counts["completed"], 3)

    def test_a_filter_that_matches_nothing_is_not_an_error(self):
        self._seed_real_batch()

        resp = self.client.get(reverse("pipelines:run_list"), {"status": "running"})

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.context["page"].paginator.count, 0)

    def test_only_recent_batches_are_listed(self):
        """Batches are the index into history; the list stays short enough to read."""
        ctx = self.client.get(reverse("pipelines:run_list")).context

        self.assertLessEqual(len(ctx["batches"]), 12)
