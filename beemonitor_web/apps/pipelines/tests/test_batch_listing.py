"""A big batch's page from its clips' listings and stored totals, and its
References page (memory/81).

A 6,306-clip batch page took 24 s on average: it read every clip's results
and counted every event's review on every load. Fixing the batch's tube
references went a clip at a time, with nothing to say which days were done.
"""

import json
from datetime import datetime, timedelta, timezone as dt_timezone
from unittest import mock

from django.db import connection
from django.test import TestCase
from django.test.utils import CaptureQueriesContext
from django.urls import reverse

from apps.analysis.models import Job, JobResult
from apps.pipelines import batchstats, listing, recount, references
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
        self.page_url = reverse("pipelines:batch_references", args=[self.batch])

    def _days(self):
        return references.days(listing.rows(PipelineRun.objects.filter(batch_id=self.batch)))


class ReferencesTests(RefBase):
    """The References page."""

    def test_a_run_counted_with_the_layout_its_clip_has_is_up_to_date(self):
        day = self._days()[0]
        self.assertEqual((day["clips"], day["analysed"], day["up_to_date"], day["stale"]),
                         (1, 1, 1, 0))
        self.assertEqual(day["layout"]["source"], "site")

    def test_a_layout_fixed_but_not_recounted_shows_as_out_of_date(self):
        """memory/81: days saved but never recounted, with nothing to say so."""
        Layout.save_new(video=self.video, roi=None, polygon=None,
                        nests=[{"id": 1, "box": MOVED}], user=self.owner)
        day = self._days()[0]
        self.assertEqual((day["stale"], day["own"]), (1, 1))
        page = self.client.get(self.page_url)
        self.assertContains(page, "Recount everything out of date (1)")

        self.client.post(reverse("pipelines:batch_references_action", args=[self.batch]),
                         {"action": "recount"})
        self.run.refresh_from_db()
        self.assertLess(self.run.counting_rules, recount.CURRENT_RULES)
        recount.sweep()
        self.assertEqual(self._days()[0]["stale"], 0)

    def test_a_day_goes_back_on_its_sites_layout(self):
        Layout.save_new(video=self.video, roi=None, polygon=None,
                        nests=[{"id": 1, "box": MOVED}], user=self.owner)
        self.client.post(reverse("pipelines:batch_references_action", args=[self.batch]),
                         {"action": "site", "day": self.video.pk})
        self.assertFalse(Layout.objects.filter(video=self.video).exists())
        self.assertEqual(self._days()[0]["layout"]["source"], "site")

    def test_a_day_takes_the_day_befores_layout(self):
        Layout.save_new(video=self.video, roi=None, polygon=None,
                        nests=[{"id": 1, "box": MOVED}], user=self.owner)
        # The next day's clip, in the batch.
        steps = json.loads(json.dumps(self.run.steps))
        steps[0]["config"]["video_id"] = str(self.next_clip.pk)
        PipelineRun.objects.create(pipeline=self.run.pipeline, user=self.owner, status="failed",
                                   batch_id=self.batch, steps=steps,
                                   context={"v": {"artifact": "video", "video_id": self.next_clip.pk}})
        page = self.client.get(self.page_url)
        self.assertContains(page, "Copy 17 Apr's")
        self.client.post(reverse("pipelines:batch_references_action", args=[self.batch]),
                         {"action": "copy", "day": self.next_clip.pk, "from": self.video.pk})
        self.assertEqual(Layout.current(video=self.next_clip).nest_layout, [{"id": 1, "box": MOVED}])

    def test_only_the_launcher_sees_it(self):
        self.client.force_login(self.stranger)
        self.assertEqual(self.client.get(self.page_url).status_code, 404)


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
