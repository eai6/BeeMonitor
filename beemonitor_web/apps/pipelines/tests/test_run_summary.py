"""Pages that list many clips read each run's slim copy (memory/76): they must
show exactly what the whole results show, and never write the copy back."""

import re
import uuid
from unittest import mock

from django.urls import reverse

from apps.pipelines import runsummary
from apps.pipelines.models import AddedEvent, EventReview, PipelineRun
from apps.pipelines.tests.test_batch_select import BatchSelectTests
from apps.pipelines.tests.test_event_review import EventReviewTests


def _whole(context):
    """A 'summary' that is the whole context: the page as it was before."""
    return {"v": runsummary.VERSION, "context": context or {}}


def _page(test, url, method="get", data=None):
    res = getattr(test.client, method)(url, data or {})
    test.assertEqual(res.status_code, 200)
    html = res.content.decode()
    return re.sub(r'name="csrfmiddlewaretoken" value="[^"]+"', "", html)


class SameAsWhole:
    def _resummarise(self, build):
        for pk, ctx in PipelineRun.objects.values_list("pk", "context"):
            PipelineRun.objects.filter(pk=pk).update(summary=build(ctx))

    def assert_same(self, url, method="get", data=None):
        self._resummarise(runsummary.build)
        slim = _page(self, url, method, data)
        self._resummarise(_whole)
        whole = _page(self, url, method, data)
        self.assertEqual(slim, whole)


class EventsBatchSummaryTests(SameAsWhole, EventReviewTests):
    def test_the_batch_page_and_metrics_read_the_same_from_the_slim_copy(self):
        self.vote(self.owner, "rejected", "wrong tube")
        AddedEvent.objects.create(run=self.run, user=self.owner, frame=3, action="enter",
                                  reference="reference_1")
        self.client.force_login(self.owner)
        self.assert_same(reverse("pipelines:batch_detail", args=[self.batch]))
        self.assert_same(reverse("pipelines:batch_review_metrics", args=[self.batch]),
                         "post", {"run": [self.run.pk]})

    def test_the_slim_copy_is_small_and_keeps_the_event_keys(self):
        ctx = PipelineRun.objects.get(pk=self.run.pk).context
        ctx["i"] = {"table_kind": "interactions", "rows": [{"start_frame": i} for i in range(5000)]}
        slim = runsummary.build(ctx)["context"]
        self.assertEqual(slim["i"]["rows"], [])
        self.assertEqual(slim["i"]["interaction_count"], 5000)
        from apps.pipelines.review import event_key
        whole = [event_key(r) for o in ctx.values() if isinstance(o, dict)
                 and o.get("table_kind") == "events" for r in o["rows"]]
        kept = [event_key(r) for o in slim.values() if isinstance(o, dict)
                and o.get("table_kind") == "events" for r in o["rows"]]
        self.assertEqual(kept, whole)

    def test_a_run_read_through_its_slim_copy_cannot_be_saved(self):
        run = runsummary.slim_runs(PipelineRun.objects.filter(pk=self.run.pk))[0]
        with self.assertRaises(ValueError):
            run.save()
        run.batch_id = uuid.uuid4()
        run.save(update_fields=["batch_id"])          # not its context: fine
        self.assertTrue(PipelineRun.objects.get(pk=self.run.pk).context)

    def test_saving_the_context_keeps_the_summary_in_step(self):
        run = PipelineRun.objects.get(pk=self.run.pk)
        run.context = {**run.context, "x": {"table_kind": "interactions", "rows": [{}] * 3}}
        run.save(update_fields=["context"])
        self.assertEqual(PipelineRun.objects.get(pk=run.pk).summary["context"]["x"]["interaction_count"], 3)

    def test_an_old_summary_is_rebuilt_when_read(self):
        PipelineRun.objects.filter(pk=self.run.pk).update(summary={})
        run = runsummary.slim_runs(PipelineRun.objects.filter(pk=self.run.pk))[0]
        self.assertTrue(runsummary.is_current(PipelineRun.objects.get(pk=run.pk).summary))

    def test_the_batch_page_never_loads_whole_results(self):
        from django.db import connection
        from django.test.utils import CaptureQueriesContext

        self.client.force_login(self.owner)
        with CaptureQueriesContext(connection) as q:
            self.client.get(reverse("pipelines:batch_detail", args=[self.batch]))
        whole = [s["sql"] for s in q.captured_queries
                 if '"pipelines_pipelinerun"."context"' in s["sql"]]
        self.assertEqual(whole, [])


class GroupsBatchSummaryTests(SameAsWhole, BatchSelectTests):
    def test_a_batch_with_insect_groups_reads_the_same(self):
        self.assert_same(reverse("pipelines:batch_detail", args=[self.batch]))


class FillCommandTests(EventReviewTests):
    def test_the_command_fills_missing_summaries(self):
        from django.core.management import call_command
        from io import StringIO

        PipelineRun.objects.filter(pk=self.run.pk).update(summary={})
        out = StringIO()
        call_command("fill_run_summaries", stdout=out)
        self.assertTrue(runsummary.is_current(PipelineRun.objects.get(pk=self.run.pk).summary))
        self.assertIn("1 filled", out.getvalue())
        out = StringIO()
        call_command("fill_run_summaries", stdout=out)
        self.assertIn("all current", out.getvalue())


class OnlyTheAwaitedJobTests(EventReviewTests):
    """memory/76: a finished job lands only on the step waiting for it."""

    def _finish(self, job_pk, step_status, waiting):
        from types import SimpleNamespace

        from apps.pipelines import engine

        PipelineRun.objects.filter(pk=self.run.pk).update(
            step_status={**self.run.step_status, "m": step_status})
        ctx = PipelineRun.objects.get(pk=self.run.pk).context
        ctx["m"] = {**ctx["m"], "job_id": waiting}
        PipelineRun.write_context(self.run.pk, ctx)
        job = SimpleNamespace(pk=job_pk, status="completed", error_message="",
                              config={"pipeline_run_id": self.run.pk, "pipeline_step_id": "m"})
        with mock.patch.object(engine, "_job_result_summary",
                               return_value={"summary_stats": {"from_job": job_pk}}), \
                mock.patch.object(engine, "advance_run"):
            engine.on_job_finished(job)
        return PipelineRun.objects.get(pk=self.run.pk).context["m"]["result"]

    def test_an_old_job_of_a_clip_run_again_is_ignored(self):
        result = self._finish(job_pk=111, step_status=PipelineRun.STEP_RUNNING, waiting=222)
        self.assertNotIn("from_job", result.get("summary_stats") or {})

    def test_the_awaited_job_lands(self):
        result = self._finish(job_pk=222, step_status=PipelineRun.STEP_RUNNING, waiting=222)
        self.assertEqual(result["summary_stats"]["from_job"], 222)

    def test_a_finished_step_is_not_rewritten_again(self):
        result = self._finish(job_pk=222, step_status=PipelineRun.STEP_DONE, waiting=222)
        self.assertNotIn("from_job", result.get("summary_stats") or {})


class ScaleTests(EventReviewTests):
    """memory/76: what a page that lists a batch costs must not grow with the
    size of each clip's results. 150 clips, each ~150 KB of interaction rows:
    whole, that is ~22 MB of JSON and several times that in Python."""

    N, ROWS = 150, 1400

    def _big_batch(self):
        from apps.analysis.models import Job

        blob = [{"start_frame": i, "end_frame": i + 5, "track_id": i % 40,
                 "with": f"reference_{i % 60}", "relation": "in_reference",
                 "duration_sec": 0.2} for i in range(self.ROWS)]
        jobs = Job.objects.bulk_create([Job(user=self.owner, video=self.video, status="completed",
                                            modal_job_id=f"s{i}") for i in range(self.N)])
        for j in jobs:
            ctx = {**self.run.context, "m": {**self.run.context["m"], "job_id": j.pk},
                   "i": {"artifact": "table", "table_kind": "interactions", "rows": blob}}
            PipelineRun.objects.create(pipeline=self.run.pipeline, user=self.owner,
                                       status="completed", batch_id=self.batch,
                                       started_at=self.job.created_at, steps=self.run.steps,
                                       context=ctx)

    def _peak_mb(self, fn):
        import tracemalloc

        tracemalloc.start()
        try:
            res = fn()
            if getattr(res, "streaming", False):
                for _chunk in res.streaming_content:
                    pass
            self.assertEqual(res.status_code, 200)
            return tracemalloc.get_traced_memory()[1] / 2**20
        finally:
            tracemalloc.stop()

    def test_batch_pages_stay_within_a_memory_budget(self):
        from apps.pipelines import sharing

        self._big_batch()
        self.client.force_login(self.owner)
        for pk in PipelineRun.objects.values_list("pk", flat=True):     # as after deploy
            PipelineRun.objects.filter(pk=pk).update(
                summary=runsummary.build(PipelineRun.objects.get(pk=pk).context))
        share = sharing.turn_on(self.owner, self.batch)
        pages = {
            "batch page": lambda: self.client.get(reverse("pipelines:batch_detail", args=[self.batch])),
            "review metrics": lambda: self.client.post(
                reverse("pipelines:batch_review_metrics", args=[self.batch]),
                {"run": list(PipelineRun.objects.values_list("pk", flat=True))}),
            "share page": lambda: self.client.get(reverse("public_batch", args=[share.token])),
            "interactions CSV": lambda: self.client.get(reverse(
                "pipelines:batch_combined_csv", kwargs={"batch_id": self.batch, "kind": "interactions"})),
        }
        for name, fn in pages.items():
            with self.subTest(name):
                self.assertLess(self._peak_mb(fn), 40, name)
