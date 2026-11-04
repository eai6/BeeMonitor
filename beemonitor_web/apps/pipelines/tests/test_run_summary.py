"""Pages that list many clips read each run's slim copy (memory/76): they must
show exactly what the whole results show, and never write the copy back."""

import re
import uuid
from unittest import mock

from django.urls import reverse
from django.utils import timezone

from apps.pipelines import aggregate, engine, runsummary
from apps.pipelines.models import AddedEvent, EventReview, PipelineRun
from apps.pipelines.tests.test_batch_select import BatchSelectTests
from apps.pipelines.tests.test_event_review import EventReviewTests
from apps.videos.models import Video


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


class BigBatchTests(EventReviewTests):
    """memory/77: a batch of thousands of clips, and launching one."""

    def _clips(self, n):
        return [Video.objects.create(
            user=self.owner, device=self.video.device, title=f"c{i}", storage_key=f"big/{i}.mp4",
            file_size_bytes=2_000_000, status=Video.Status.READY, recorded_at=timezone.now(),
            fps=25.0, width=1920, height=1080, duration_seconds=4.0) for i in range(n)]

    def _launch(self, clips, **extra):
        self.client.force_login(self.owner)
        return self.client.post(
            reverse("pipelines:run_on_videos"), {"video_ids": [c.pk for c in clips], **extra},
            headers={"X-Requested-With": "XMLHttpRequest"}).json()

    def test_a_big_launch_goes_in_turns_and_resending_skips_what_started(self):
        pipeline = self.run.pipeline
        pipeline.steps = [{"id": "v", "block_type": "input.video", "config": {}}]
        pipeline.save()
        clips = self._clips(5)
        with mock.patch("apps.pipelines.views.LAUNCH_CHUNK", 2):
            first = self._launch(clips, pipeline=pipeline.pk)
            self.assertEqual((first["submitted"], first["more"]), (2, 3))
            second = self._launch(clips, add_to=first["batch_id"])
            self.assertEqual((second["submitted"], second["more"]), (2, 1))
            last = self._launch(clips, add_to=first["batch_id"])
        self.assertEqual((last["submitted"], last["more"]), (1, 0))
        self.assertEqual(aggregate.batch_video_ids(first["batch_id"]), {c.pk for c in clips})
        self.assertEqual(PipelineRun.objects.filter(batch_id=first["batch_id"]).count(), 5)

    def test_runs_of_a_batch_share_their_steps_and_pipeline(self):
        other = PipelineRun.objects.create(
            pipeline=self.run.pipeline, user=self.owner, status="completed", batch_id=self.batch,
            started_at=self.job.created_at, context=self.run.context,
            steps=[{**self.run.steps[0], "config": {"video_id": "999"}}, *self.run.steps[1:]])
        one, two = runsummary.share_steps(
            PipelineRun.objects.filter(pk__in=[self.run.pk, other.pk]).order_by("id"))
        self.assertIs(one.steps[1], two.steps[1])
        self.assertIs(one.pipeline, two.pipeline)
        self.assertEqual({aggregate.run_video_id(one), aggregate.run_video_id(two)},
                         {self.video.pk, 999})

    def test_a_big_running_batch_stops_reloading_itself(self):
        PipelineRun.objects.filter(pk=self.run.pk).update(status="running")
        self.client.force_login(self.owner)
        url = reverse("pipelines:batch_detail", args=[self.batch])
        self.assertContains(self.client.get(url), "pollThenReload")
        with mock.patch("apps.pipelines.views.BIG_BATCH", 0):
            res = self.client.get(url)
        self.assertNotContains(res, "pollThenReload")
        self.assertContains(res, "Reload the page")

    def test_the_batch_page_reads_no_job_config_or_result_summary(self):
        from django.db import connection
        from django.test.utils import CaptureQueriesContext

        self.client.force_login(self.owner)
        with CaptureQueriesContext(connection) as seen:
            self.client.get(reverse("pipelines:batch_detail", args=[self.batch]))
        for q in seen.captured_queries:
            self.assertNotIn('"analysis_jobresult"."summary_stats"', q["sql"])
            self.assertNotIn('"analysis_job"."config"', q["sql"])

    def test_the_loop_checks_each_user_with_running_runs_once(self):
        from apps.analysis import reconcile

        PipelineRun.objects.filter(pk=self.run.pk).update(status="running")
        for _ in range(3):
            PipelineRun.objects.create(pipeline=self.run.pipeline, user=self.owner,
                                       status="running", started_at=timezone.now(), steps=[])
        with mock.patch("apps.pipelines.engine.reconcile_user_runs") as check, \
                mock.patch("apps.analysis.views._poll_sagemaker_results", return_value=0), \
                mock.patch("apps.analysis.views._drain_queue", return_value=0):
            reconcile.reconcile_all()
        self.assertEqual(check.call_count, 1)

    def test_runs_that_started_first_are_rechecked_first(self):
        from datetime import timedelta

        PipelineRun.objects.filter(pk=self.run.pk).update(
            status="running", started_at=timezone.now() - timedelta(days=1),
            step_status={"m": PipelineRun.STEP_RUNNING})
        for _ in range(2):
            PipelineRun.objects.create(pipeline=self.run.pipeline, user=self.owner,
                                       status="running", started_at=timezone.now(), steps=[])
        with mock.patch("apps.pipelines.engine.on_job_finished") as finished:
            PipelineRun.objects.filter(pk=self.run.pk).update(
                steps=[{"id": "m", "block_type": "track.mot"}])
            engine.reconcile_user_runs(self.owner, limit=1)
        finished.assert_called_once()

    def _batch_of(self, n, status="completed"):
        for video in self._clips(n):
            PipelineRun.objects.create(
                pipeline=self.run.pipeline, user=self.owner, status=status, batch_id=self.batch,
                started_at=timezone.now(), context={"v": {"artifact": "video", "video_id": video.pk}},
                steps=[{"id": "v", "block_type": "input.video",
                        "config": {"video_id": str(video.pk)}}])

    def test_a_big_batch_sends_a_page_of_rows_and_everything_the_filter_needs(self):
        self._batch_of(4)
        self.client.force_login(self.owner)
        url = reverse("pipelines:batch_detail", args=[self.batch])
        with mock.patch("apps.pipelines.views.BIG_BATCH", 2), \
                mock.patch("apps.pipelines.views.CLIPS_PER_PAGE", 2):
            res = self.client.get(url)
        html = res.content.decode()
        self.assertEqual(html.count('class="clip-row'), 2)
        self.assertEqual(len(res.context["clip_index"]), 5)
        self.assertIn('id="clip-index"', html)
        self.assertIn('id="clip-pager"', html)
        self.assertIn("csrfmiddlewaretoken", html[html.index('id="clip-pager"'):])
        first = res.context["clip_index"][0]
        self.assertIn(f'data-f-run="{first["run"]}"', html)
        self.assertEqual(first["status"], "completed")

    def test_a_small_batch_keeps_every_row_and_no_pager(self):
        self._batch_of(2)
        self.client.force_login(self.owner)
        html = self.client.get(reverse("pipelines:batch_detail", args=[self.batch])).content.decode()
        self.assertEqual(html.count('class="clip-row'), 3)
        self.assertNotIn('id="clip-index"', html)
        self.assertNotIn('id="clip-pager"', html)

    def test_rows_of_a_page_come_in_the_order_asked_for(self):
        self._batch_of(3)
        self.client.force_login(self.owner)
        pks = [str(pk) for pk in PipelineRun.objects.filter(batch_id=self.batch)
               .order_by("started_at").values_list("pk", flat=True)][::-1][:3]
        url = reverse("pipelines:batch_rows", args=[self.batch])
        html = self.client.post(url, {"run": pks}).content.decode()
        self.assertEqual(re.findall(r'data-f-run="([^"]+)"', html), pks)
        self.assertNotIn("rows", "".join(re.findall(r'back=([^"]*)"', html)))

    def test_a_stranger_gets_no_rows(self):
        self.client.force_login(self.stranger)
        res = self.client.post(reverse("pipelines:batch_rows", args=[self.batch]),
                               {"run": [str(self.run.pk)]})
        self.assertIn(res.status_code, (403, 404))

    def _queued(self, n):
        from apps.analysis.models import Job

        jobs = []
        for i, video in enumerate(self._clips(n)):
            job = Job.objects.create(user=self.owner, video=video, status="queued",
                                     modal_job_id=f"q{i}")
            jobs.append(job)
            PipelineRun.objects.create(
                pipeline=self.run.pipeline, user=self.owner, status="running", batch_id=self.batch,
                started_at=timezone.now(), step_status={"m": PipelineRun.STEP_RUNNING},
                context={"m": {"job_id": job.pk}},
                steps=[{"id": "v", "block_type": "input.video", "config": {"video_id": str(video.pk)}},
                       {"id": "m", "block_type": "track.mot"}])
        return jobs

    def _turn(self, name, **data):
        self.client.force_login(self.owner)
        return self.client.post(reverse(f"pipelines:{name}", args=[self.batch]), data,
                                headers={"X-Requested-With": "XMLHttpRequest"}).json()

    def test_cancel_stops_every_waiting_job_at_once_and_closes_runs_in_turns(self):
        jobs = self._queued(5)
        with mock.patch("apps.pipelines.views.TURN_CLIPS", 2):
            first = self._turn("batch_cancel")
            self.assertEqual(first, {"done": 2, "left": 3})
            for job in jobs:
                job.refresh_from_db()
                self.assertEqual(job.status, "cancelled")
            second = self._turn("batch_cancel", done=first["done"])
            self.assertEqual(second, {"done": 4, "left": 1})
            self.assertEqual(self._turn("batch_cancel", done=second["done"]), {"done": 5, "left": 0})
        self.assertFalse(PipelineRun.objects.filter(batch_id=self.batch, status="running").exists())

    def test_rerun_goes_in_turns_and_each_turn_moves_on(self):
        self._queued(5)
        with mock.patch("apps.pipelines.views.TURN_CLIPS", 10):
            self._turn("batch_cancel")
        with mock.patch("apps.pipelines.views.TURN_CLIPS", 2):
            first = self._turn("batch_retry_failed")
            self.assertEqual((first["done"], first["left"]), (2, 3))
            second = self._turn("batch_retry_failed", done=first["done"], after=first["after"])
            self.assertEqual((second["done"], second["left"]), (4, 1))
            # These clips fail again at once: the turns still come to an end.
            last = self._turn("batch_retry_failed", done=second["done"], after=second["after"])
        self.assertEqual(last, {"done": 5, "left": 0})

    def test_a_clip_row_opens_its_results_with_no_inline_viewer(self):
        self.client.force_login(self.owner)
        html = self.client.get(reverse("pipelines:batch_detail", args=[self.batch])).content.decode()
        url = reverse("analysis:results", args=[self.job.pk]) + f"?run={self.run.pk}"
        self.assertIn(f'data-href="{url}"', html)
        self.assertIn(f'<a href="{url}"', html)
        self.assertNotIn("crops &amp; CSVs", html)
        self.assertNotIn('id="clip-viewer"', html)
        self.assertNotIn("js/clip_viewer.js", html)


class TripsNodeTests(EventReviewTests):
    def test_the_trips_node_needs_no_update_and_is_no_swap_for_an_analyzer(self):
        from apps.pipelines import regroup
        from apps.pipelines.registry import get_block
        from apps.pipelines.views import analyzer_options

        pipeline = self.run.pipeline
        pipeline.steps = [*self.run.steps, {"id": "t", "block_type": "analyze.trips",
                                            "config": {"min_trip_seconds": 5},
                                            "inputs": {"events": "e"}}]
        changes = regroup.pipeline_changes(self.run, pipeline)
        self.assertEqual((changes["web"], changes["gpu"]), ([], []))
        self.assertFalse(get_block("analyze.trips").get("hidden"))
        self.assertNotIn("analyze.trips", [o["key"] for o in analyzer_options(self.run)[1]])

    def test_a_pipeline_with_the_trips_node_runs(self):
        from apps.pipelines.engine import validate_steps

        steps = [{"id": "v", "block_type": "input.video", "config": {"video_id": str(self.video.pk)}},
                 {"id": "r", "block_type": "reference.layout", "config": {"source": "device_layout"},
                  "inputs": {"video": "v"}},
                 {"id": "e", "block_type": "analyze.events", "config": {},
                  "inputs": {"tracks": "m", "rois": "r"}},
                 {"id": "t", "block_type": "analyze.trips", "config": {}, "inputs": {"events": "e"}}]
        self.assertEqual([p for p in validate_steps(steps) if "Trips" in str(p)], [])
