"""A big batch's page from its clips' listings and stored totals, and the
layout editor's day saves (memory/81).

A 6,306-clip batch page took 24 s on average: it read every clip's results
and counted every event's review on every load.
"""

import json
from datetime import datetime, timedelta, timezone as dt_timezone
from unittest import mock

from django.db import connection
from django.test import TestCase
from django.test.utils import CaptureQueriesContext
from django.urls import reverse

from apps.analysis.models import Job, JobResult
from apps.pipelines import batchstats, listing, recount
from apps.pipelines.models import BatchStats, EventReview, PipelineRun
from apps.pipelines.tests import test_event_review
from apps.videos.models import Layout, Site, Video

BOX = [0.1, 0.1, 0.3, 0.3]
MOVED = [0.5, 0.5, 0.7, 0.7]


class Base(TestCase):
    # The event review tests' batch of one analysed clip, without their tests.
    setUp = test_event_review.EventReviewTests.setUp
    vote = test_event_review.EventReviewTests.vote


class ListingTests(Base):
    def test_a_runs_listing_follows_its_results(self):
        lst = self.run.listing
        self.assertEqual((lst["v"], lst["video"], lst["events"]),
                         (listing.VERSION, self.video.pk, len(self.events)))
        self.assertIsNotNone(self.run.results_at)
        ctx = self.run.context
        ctx["e"] = {**ctx["e"], "error": "Boom"}
        PipelineRun.write_context(self.run.pk, ctx)
        self.run.refresh_from_db()
        self.assertEqual(self.run.listing["errors"], ["Boom"])

    def test_a_run_from_before_listings_gets_one_when_listed(self):
        PipelineRun.objects.filter(pk=self.run.pk).update(listing={})
        rows = listing.rows(PipelineRun.objects.filter(batch_id=self.batch))
        self.assertEqual(rows[0]["primitives"]["events"], len(self.events))
        self.run.refresh_from_db()
        self.assertTrue(listing.is_current(self.run.listing))

    def test_rows_say_who_reviewed_a_clip(self):
        self.vote(self.owner, "accepted")
        row = listing.rows(PipelineRun.objects.filter(batch_id=self.batch))[0]
        self.assertEqual((row["reviewed"], row["reviewers"]), (True, ["owner"]))


class BigBatchPageTests(Base):
    def setUp(self):
        Base.setUp(self)
        self.other = self._clip_run("other")
        self.client.force_login(self.owner)
        self.page_url = reverse("pipelines:batch_detail", args=[self.batch])

    def _clip_run(self, title):
        video = Video.objects.create(
            user=self.owner, device=self.video.device, title=title, storage_key=f"bl/{title}.mp4",
            file_size_bytes=1, status=Video.Status.READY,
            recorded_at=self.video.recorded_at + timedelta(minutes=10),
            fps=25.0, width=1920, height=1080, duration_seconds=4.0)
        steps = json.loads(json.dumps(self.run.steps))
        steps[0]["config"]["video_id"] = str(video.pk)
        ctx = json.loads(json.dumps(self.run.context))
        ctx["v"]["video_id"] = video.pk
        job = Job.objects.create(user=self.owner, video=video, status="completed",
                                 modal_job_id=f"bl-{title}")
        JobResult.objects.create(job=job, tracking_csv_path=self.job.result.tracking_csv_path)
        ctx["m"]["job_id"] = job.pk
        return PipelineRun.objects.create(pipeline=self.run.pipeline, user=self.owner,
                                          status="completed", batch_id=self.batch,
                                          started_at=self.run.started_at, steps=steps, context=ctx)

    def _get(self):
        with mock.patch("apps.pipelines.views.BIG_BATCH", 1), \
                mock.patch("apps.pipelines.views.CLIPS_PER_PAGE", 1):
            return self.client.get(self.page_url)

    def test_the_page_reads_only_the_shown_clips_results(self):
        self._get()                                   # works the totals out
        self.assertTrue(BatchStats.objects.filter(batch_id=self.batch).exists())
        with CaptureQueriesContext(connection) as q:
            res = self._get()
        self.assertEqual(res.status_code, 200)
        self.assertEqual(len(res.context["clip_index"]), 2)
        self.assertFalse(res.context["stats_stale"])
        slim = [s["sql"] for s in q.captured_queries
                if '"pipelines_pipelinerun"."summary"' in s["sql"]]
        self.assertTrue(slim)
        self.assertFalse([s for s in slim if self.other.pk.hex in s])
        self.assertEqual(res.context["downloads"][0]["kind"], "tracking")

    def test_the_totals_are_kept_until_the_batch_changes(self):
        self._get()
        with mock.patch.object(batchstats, "compute") as compute:
            self._get()
        compute.assert_not_called()
        self.vote(self.owner, "accepted")
        self.client.force_login(self.owner)
        res = self._get()
        self.assertEqual(res.context["review_summary"]["events"]["accepted"], 1)

    def test_a_pipeline_just_saved_can_be_applied_before_the_totals_are_redone(self):
        """memory/82: the card read "Up to date" with Update greyed out while a
        6,305-clip batch's totals were worked out again."""
        self._get()                                   # totals stored: up to date
        pipeline = self.run.pipeline
        steps = json.loads(json.dumps(self.run.steps))
        ev = next(s for s in steps if s["block_type"] == "analyze.events")
        ev["config"] = {**(ev.get("config") or {}), "exit_settle_seconds": 0.9}
        pipeline.steps = steps
        pipeline.save()
        with mock.patch.object(batchstats, "start", return_value=None):   # still working
            res = self._get()
        self.assertTrue(res.context["stats_stale"])
        self.assertTrue(res.context["changes"]["web"])
        self.assertIsNone(res.context["changes"]["behind"])
        self.assertContains(res, "Changed in ")

    def test_one_web_process_at_a_time_works_the_totals_out(self):
        """memory/82: both web workers held the whole batch at once."""
        self.assertTrue(batchstats._claim(self.batch))
        self.assertFalse(batchstats._claim(self.batch))
        BatchStats.objects.filter(batch_id=self.batch).update(working_since=None)
        self.assertTrue(batchstats._claim(self.batch))

    def test_fresh_totals_rest_before_being_redone(self):
        self._get()
        BatchStats.objects.filter(batch_id=self.batch).update(fingerprint="old")
        with mock.patch.object(batchstats, "_testing", return_value=False), \
                mock.patch.object(batchstats, "start") as start:
            data, current = batchstats.stored(self.batch, self.run.pipeline, None)
        start.assert_not_called()
        self.assertFalse(current)
        self.assertTrue(data)

    def test_a_big_batch_page_is_compressed(self):
        res = self.client.get(self.page_url, headers={"Accept-Encoding": "gzip"})
        self.assertEqual(res.headers.get("Content-Encoding"), "gzip")


class RefBase(Base):
    """A clip counted with its site's layout, and its site's next day."""

    def setUp(self):
        Base.setUp(self)
        self.site = Site.objects.create(user=self.owner, name="mendels", tz_name="UTC")
        self.day1 = datetime(2024, 4, 17, 9, tzinfo=dt_timezone.utc)
        Video.objects.filter(pk=self.video.pk).update(site=self.site, site_name="mendels",
                                                      device=None, recorded_at=self.day1)
        self.video.refresh_from_db()
        Layout.save_new(site=self.site, roi=None, polygon=None,
                        nests=[{"id": 1, "box": BOX}], user=self.owner)
        self.next_clip = Video.objects.create(
            user=self.owner, site=self.site, site_name="mendels", title="next",
            storage_key="ref/next.mp4", file_size_bytes=1, status=Video.Status.READY,
            recorded_at=self.day1 + timedelta(days=1))
        recount.recount_run(PipelineRun.objects.get(pk=self.run.pk))
        self.client.force_login(self.owner)


class EditorTests(RefBase):
    def _save(self, **body):
        return self.client.post(reverse("videos:layout", args=[self.video.pk]), json.dumps({
            "roi": None, "nests": [{"id": 1, "box": MOVED}], **body}),
            content_type="application/json").json()

    def test_saving_a_day_recounts_it(self):
        out = self._save(scope="day", recount=True)
        self.assertEqual(out["queued"], 1)
        self.run.refresh_from_db()
        self.assertLess(self.run.counting_rules, recount.CURRENT_RULES)

    def test_saving_without_the_tick_leaves_the_counts(self):
        out = self._save(scope="day")
        self.assertEqual(out["queued"], 0)

    def test_a_site_save_can_take_over_the_clips_own_layouts(self):
        Layout.save_new(video=self.video, roi=None, polygon=None,
                        nests=[{"id": 1, "box": BOX}], user=self.owner)
        page = self.client.get(reverse("videos:layout", args=[self.video.pk]))
        self.assertEqual(page.context["own_at_site"]["clips"], 1)
        self._save(scope="site", override_own=True)
        self.assertFalse(Layout.objects.filter(video=self.video).exists())

    def test_the_next_day_opens_with_this_days_drawing(self):
        page = self.client.get(reverse("videos:layout", args=[self.video.pk]) + "?scope=day")
        nxt = page.context["nav"]["next_day"]
        self.assertEqual(nxt["date"].isoformat(), "2024-04-18")
        carried = self.client.get(nxt["carry_url"])
        self.assertEqual(carried.context["carried"], self.video)
        self.assertIn(str(BOX[2]), carried.context["nests_json"])
