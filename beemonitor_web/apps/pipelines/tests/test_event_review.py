"""Reviewers' yes / no on detected events (memory/62).

A vote sits beside the event and never changes it; the Events CSV gains
review, reviewed_by and review_notes; only the launcher and invited reviewers
may vote; a recount never loses a vote.
"""

import csv
import io
import json
import uuid

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from apps.analysis.models import Job, JobResult
from apps.analysis.test_job_primitives import AGREE_TRACKING, _csv
from apps.devices.models import Device
from apps.pipelines import aggregate, review
from apps.pipelines.models import BatchMember, EventReview, Pipeline, PipelineRun
from apps.pipelines.recount import recount_run
from apps.videos.models import Video

User = get_user_model()


class EventReviewTests(TestCase):
    def setUp(self):
        self.owner = User.objects.create_user("owner", password="x")
        self.maria = User.objects.create_user("maria", password="x")
        self.stranger = User.objects.create_user("stranger", password="x")
        device = Device.objects.create(owner=self.owner, name="Hotel", key_hash="hrv2",
                                       prefix="bmk_rv2",
                                       nest_layout=[{"id": 1, "box": [0.1, 0.1, 0.3, 0.3]}])
        self.video = Video.objects.create(
            user=self.owner, device=device, title="clip", storage_key="rv2/c.mp4",
            file_size_bytes=1, status=Video.Status.READY, recorded_at=timezone.now(),
            fps=25.0, width=1920, height=1080, duration_seconds=4.0)
        self.job = Job.objects.create(user=self.owner, video=self.video,
                                      status="completed", modal_job_id="rv2-1")
        path = _csv(AGREE_TRACKING, "review.csv")
        JobResult.objects.create(job=self.job, tracking_csv_path=path,
                                 summary_stats={"video_fps": 25.0})
        self.batch = uuid.uuid4()
        self.run = PipelineRun.objects.create(
            pipeline=Pipeline.objects.create(user=self.owner, title="P"), user=self.owner,
            status="completed", batch_id=self.batch, started_at=self.job.created_at,
            steps=[{"id": "v", "block_type": "input.video",
                    "config": {"video_id": str(self.video.pk)}},
                   {"id": "r", "block_type": "reference.layout",
                    "config": {"source": "device_layout"}, "inputs": {"video": "v"}},
                   {"id": "e", "block_type": "analyze.events", "config": {},
                    "inputs": {"tracks": "m", "rois": "r"}}],
            context={"v": {"artifact": "video", "video_id": self.video.pk},
                     "m": {"artifact": "tracks", "job_id": self.job.pk,
                           "result": {"tracking_csv_path": path,
                                      "summary_stats": {"video_fps": 25.0}}}})
        recount_run(self.run)
        self.run.refresh_from_db()
        self.events = [r for k, o in aggregate.analyzer_outputs(self.run) if k == "events"
                       for r in o["rows"]]
        self.key = review.event_key(self.events[0])
        self.url = reverse("analysis:event_review", args=[self.job.pk])

    def vote(self, user, verdict, reason="", key=None):
        self.client.force_login(user)
        return self.client.post(self.url, json.dumps(
            {"event_key": key or self.key, "verdict": verdict, "reason": reason}),
            content_type="application/json")

    def csv_rows(self):
        self.client.force_login(self.owner)
        body = self.client.get(reverse("analysis:results_csv",
                                       kwargs={"pk": self.job.pk, "kind": "events"})).content
        return list(csv.DictReader(io.StringIO(body.decode())))

    def test_the_launcher_votes_changes_and_takes_back(self):
        self.assertEqual(self.vote(self.owner, "accepted").json()["verdict"], "accepted")
        self.assertEqual(EventReview.objects.get().verdict, "accepted")
        r = self.vote(self.owner, "rejected", "wrong tube").json()
        self.assertEqual((r["verdict"], r["counts"]["rejected"]), ("rejected", 1))
        self.assertEqual(EventReview.objects.get().reason, "wrong tube")
        self.assertIsNone(self.vote(self.owner, None).json()["verdict"])
        self.assertFalse(EventReview.objects.exists())

    def test_a_vote_never_changes_the_detected_events(self):
        self.vote(self.owner, "rejected", "not an event")
        rows = self.csv_rows()
        self.assertEqual(len(rows), len(self.events))
        rejected = [r for r in rows if r["review"] == "rejected"]
        self.assertEqual(len(rejected), 1)
        self.assertEqual((rejected[0]["reviewed_by"], rejected[0]["review_notes"]),
                         ("owner", "not an event"))

    def test_two_reviewers_who_disagree_make_it_disputed(self):
        BatchMember.objects.create(batch_id=self.batch, user=self.maria,
                                   role=BatchMember.Role.REVIEWER)
        self.vote(self.owner, "accepted")
        self.assertEqual(self.vote(self.maria, "rejected", "can't tell").status_code, 200)
        row = [r for r in self.csv_rows() if r["review"]][0]
        self.assertEqual((row["review"], row["reviewed_by"]), ("disputed", "maria;owner"))

    def test_only_the_launcher_and_invited_reviewers_may_vote(self):
        self.assertEqual(self.vote(self.stranger, "accepted").status_code, 404)
        BatchMember.objects.create(batch_id=self.batch, user=self.stranger,
                                   role=BatchMember.Role.VIEWER)
        self.assertEqual(self.vote(self.stranger, "accepted").status_code, 404)
        self.assertFalse(EventReview.objects.exists())

    def test_a_bad_verdict_or_reason_is_refused(self):
        self.assertEqual(self.vote(self.owner, "maybe").status_code, 400)
        self.assertEqual(self.vote(self.owner, "rejected", "too blurry").status_code, 400)

    def test_a_vote_outlives_a_recount_that_drops_its_event(self):
        self.vote(self.owner, "accepted", key="exit|reference_9|77|5")
        recount_run(self.run)
        self.assertEqual(review.event_review_counts(self.run)["gone"], 1)
        self.assertTrue(EventReview.objects.filter(event_key="exit|reference_9|77|5").exists())

    def test_the_player_gets_each_events_key_and_votes(self):
        self.vote(self.owner, "accepted")
        live = json.loads(self.client.get(reverse("analysis:track_overlay", args=[self.job.pk]),
                                          {"part": "events"}).content)
        voted = [e for e in live["events"] if e[6] == self.key][0]
        self.assertEqual((voted[7], voted[8], voted[9]), ("accepted", 1, 0))

    def test_the_page_offers_review_to_the_launcher_and_counts_it(self):
        from unittest import mock

        self.vote(self.owner, "rejected")
        with mock.patch("apps.analysis.views._generate_presigned_url", return_value="https://x"):
            page = self.client.get(reverse("analysis:results", args=[self.job.pk]))
        self.assertTrue(page.context["can_review"])
        self.assertEqual(page.context["review_counts"]["rejected"], 1)
        self.assertContains(page, "data-rv-start")

    def test_the_batch_table_reports_review_progress(self):
        self.vote(self.owner, "rejected")
        counts = review.batch_review_counts([self.run])[self.run.pk]
        self.assertEqual((counts["reviewed"], counts["events"], counts["rejected"]),
                         (1, len(self.events), 1))
