"""Period review and trip review (memory/62 §4, §4b).

Two clips an hour apart on one unit: a bee leaves tube 1 in the first and
comes back in the second. The period puts both on one axis, suggests the trip
across clips (the same marker first), and a trip is data only once a reviewer
calls it.
"""

import csv
import io
import json
import uuid
from datetime import datetime, timedelta, timezone as dt_timezone
from unittest import mock

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from apps.analysis.models import Job, JobResult
from apps.analysis.test_job_primitives import _csv
from apps.devices.models import Device
from apps.pipelines import period
from apps.pipelines.models import (BatchMember, EventReview, Pipeline, PipelineRun,
                                   TripReview)
from apps.pipelines.recount import recount_run
from apps.videos.models import Video

User = get_user_model()

# Clip A: track 3 starts in tube 1 (0.1–0.3) and flies off — an exit at frame 0.
EXIT_ROWS = [{"frame": f, "track_id": 3, "cx": 0.2 if f < 10 else 0.8,
              "cy": 0.2 if f < 10 else 0.8, "detected": 1} for f in range(20)]
# Clip B: track 4 comes in and ends in tube 1 (frame 39); track 6 later (frame 79).
ENTER_ROWS = (
    [{"frame": f, "track_id": 4, "cx": 0.8 if f < 30 else 0.2, "cy": 0.8 if f < 30 else 0.2,
      "detected": 1} for f in range(20, 40)]
    + [{"frame": f, "track_id": 6, "cx": 0.8 if f < 70 else 0.2, "cy": 0.8 if f < 70 else 0.2,
        "detected": 1} for f in range(60, 80)])


class PeriodBase(TestCase):
    def setUp(self):
        self.owner = User.objects.create_user("owner", password="x")
        self.maria = User.objects.create_user("maria", password="x")
        self.stranger = User.objects.create_user("stranger", password="x")
        self.device = Device.objects.create(owner=self.owner, name="Hotel", key_hash="hpr",
                                            prefix="bmk_pr",
                                            nest_layout=[{"id": 1, "box": [0.1, 0.1, 0.3, 0.3]}])
        self.pipeline = Pipeline.objects.create(user=self.owner, title="P")
        self.batch = uuid.uuid4()
        t0 = datetime(2024, 5, 8, 10, 0, tzinfo=dt_timezone.utc)
        self.a = self.clip("a", t0, EXIT_ROWS, {"3": {"marker": "red"}})
        self.b = self.clip("b", t0 + timedelta(hours=1), ENTER_ROWS,
                           {"4": {"marker": "blue"}, "6": {"marker": "red"}})

    def clip(self, name, when, rows, by_track):
        video = Video.objects.create(
            user=self.owner, device=self.device, title=name, storage_key=f"pr/{name}.mp4",
            file_size_bytes=1, status=Video.Status.READY, recorded_at=when,
            fps=25.0, width=1920, height=1080, duration_seconds=4.0)
        job = Job.objects.create(user=self.owner, video=video, status="completed",
                                 modal_job_id=f"pr-{name}")
        path = _csv(rows, f"{name}.csv")
        JobResult.objects.create(job=job, tracking_csv_path=path, summary_stats={
            "video_fps": 25.0, "identification": {"by_track": by_track}})
        run = PipelineRun.objects.create(
            pipeline=self.pipeline, user=self.owner, status="completed", batch_id=self.batch,
            started_at=job.created_at,
            steps=[{"id": "v", "block_type": "input.video", "config": {"video_id": str(video.pk)}},
                   {"id": "r", "block_type": "reference.layout",
                    "config": {"source": "device_layout"}, "inputs": {"video": "v"}},
                   {"id": "e", "block_type": "analyze.events", "config": {},
                    "inputs": {"tracks": "m", "rois": "r"}}],
            context={"v": {"artifact": "video", "video_id": video.pk},
                     "m": {"artifact": "tracks", "job_id": job.pk,
                           "result": {"tracking_csv_path": path,
                                      "summary_stats": {"video_fps": 25.0}}}})
        recount_run(run)
        run.refresh_from_db()
        return {"video": video, "job": job, "run": run}

    def page(self, user=None, **params):
        self.client.force_login(user or self.owner)
        q = {"device": self.device.pk, "from": "2024-05-08T09:00", "to": "2024-05-08T13:00"}
        q.update(params)
        return self.client.get(reverse("analysis:period_review"), q)

    def payload(self, resp):
        return resp.context["payload"]


class GatherTests(PeriodBase):
    def test_the_period_holds_both_clips_in_order_on_one_axis(self):
        p = self.payload(self.page())
        self.assertEqual([c["title"] for c in p["clips"]], ["a", "b"])
        self.assertEqual([c["offset"] for c in p["clips"]], [3600.0, 7200.0])
        enter = [e for e in p["clips"][1]["events"] if e["action"] == "enter"][0]
        self.assertAlmostEqual(enter["t"], 7200 + 39 / 25, places=2)
        self.assertTrue(enter["clock"].startswith("2024-05-08T11:00:01"))

    def test_clips_it_cannot_place_are_listed_apart(self):
        Video.objects.create(user=self.owner, device=self.device, title="no time",
                             storage_key="pr/n.mp4", file_size_bytes=1,
                             recorded_at=datetime(2024, 5, 8, 10, 30, tzinfo=dt_timezone.utc),
                             metadata={"recorded_at_source": "upload_time"})
        Video.objects.create(user=self.owner, device=self.device, title="not analysed",
                             storage_key="pr/u.mp4", file_size_bytes=1,
                             recorded_at=datetime(2024, 5, 8, 12, 0, tzinfo=dt_timezone.utc))
        skipped = {s["title"]: s["reason"] for s in self.page().context["data"]["skipped"]}
        self.assertEqual(skipped["no time"], "recording time unknown")
        self.assertEqual(skipped["not analysed"], "no finished run with an Events step")

    def test_a_period_is_capped_at_seven_days_and_says_so(self):
        data = self.page(to="2024-05-20T00:00").context["data"]
        self.assertEqual(data["end"] - data["start"], timedelta(days=7))
        self.assertTrue(any("7 days" in n for n in data["notes"]))

    def test_and_at_a_number_of_clips(self):
        with mock.patch.object(period, "MAX_CLIPS", 1):
            data = self.page().context["data"]
        self.assertEqual(len(data["clips"]), 1)
        self.assertTrue(any("at most 1 clips" in n for n in data["notes"]))

    def test_a_batch_fills_in_its_device_and_range(self):
        self.client.force_login(self.owner)
        r = self.client.get(reverse("analysis:period_review"), {"batch": str(self.batch)})
        self.assertEqual([c["title"] for c in r.context["payload"]["clips"]], ["a", "b"])

    def test_the_batch_and_device_pages_link_to_it(self):
        self.client.force_login(self.owner)
        batch = self.client.get(reverse("pipelines:batch_detail", args=[self.batch]))
        self.assertIn("?batch=", batch.context["period_review_url"])


class AccessTests(PeriodBase):
    def test_a_stranger_gets_nothing(self):
        self.assertEqual(self.page(self.stranger).status_code, 404)

    def test_an_invited_member_sees_the_batchs_clips(self):
        BatchMember.objects.create(batch_id=self.batch, user=self.maria,
                                   role=BatchMember.Role.VIEWER)
        p = self.payload(self.page(self.maria))
        self.assertEqual([c["title"] for c in p["clips"]], ["a", "b"])
        self.assertFalse(any(c["can_review"] for c in p["clips"]))

    def test_the_stream_admits_members_too(self):
        BatchMember.objects.create(batch_id=self.batch, user=self.maria)
        self.client.force_login(self.maria)
        with mock.patch("apps.videos.views.stream_redirect",
                        return_value=__import__("django.http").http.HttpResponse("ok")):
            r = self.client.get(reverse("analysis:period_stream", args=[self.a["job"].pk]))
        self.assertEqual(r.status_code, 200)
        self.client.force_login(self.stranger)
        self.assertEqual(self.client.get(reverse("analysis:period_stream",
                                                 args=[self.a["job"].pk])).status_code, 404)


class SuggestionTests(PeriodBase):
    def test_the_trip_spans_clips_and_prefers_the_same_marker(self):
        trips = self.payload(self.page())["trips"]
        self.assertEqual(len(trips), 1)
        t = trips[0]
        self.assertEqual((t["exit"]["clip"], t["entry"]["clip"]), (0, 1))
        self.assertTrue(t["entry"]["key"].startswith("enter|reference_1|6|"))
        self.assertTrue(t["same_marker"])

    def test_without_markers_the_next_entry_wins_and_never_an_earlier_one(self):
        ev = lambda a, t, track, marker="": {"action": a, "t": t, "reference": "reference_1",
                                             "key": f"{a}|{track}", "marker": marker}
        trips = period.suggest_trips([(0, ev("enter", 5, 1)), (0, ev("exit", 10, 2)),
                                      (1, ev("enter", 50, 3)), (1, ev("enter", 90, 4))])
        self.assertEqual(len(trips), 1)
        self.assertEqual(trips[0]["entry"][1]["key"], "enter|3")

    def test_events_from_period_review_land_on_their_own_clip(self):
        p = self.payload(self.page())
        self.assertEqual(p["clips"][1]["review_url"],
                         reverse("analysis:event_review", args=[self.b["job"].pk]))
        key = p["clips"][1]["events"][0]["key"]
        self.client.post(p["clips"][1]["review_url"], json.dumps(
            {"event_key": key, "verdict": "accepted"}), content_type="application/json")
        self.assertEqual(EventReview.objects.get().run, self.b["run"])


class TripVoteTests(PeriodBase):
    def setUp(self):
        super().setUp()
        self.trip = self.payload(self.page())["trips"][0]
        self.clips = self.payload(self.page())["clips"]

    def vote(self, verdict, entry=True, user=None, **extra):
        self.client.force_login(user or self.owner)
        body = {"exit_job": self.clips[0]["job"], "exit_key": self.trip["exit"]["key"],
                "entry_job": self.clips[1]["job"] if entry else None,
                "entry_key": self.trip["entry"]["key"] if entry else "", "verdict": verdict}
        body.update(extra)
        return self.client.post(reverse("analysis:trip_review"), json.dumps(body),
                                content_type="application/json")

    def csv_rows(self):
        self.client.force_login(self.owner)
        body = self.client.get(reverse("analysis:period_trips_csv"), {
            "device": self.device.pk, "from": "2024-05-08T09:00",
            "to": "2024-05-08T13:00"}).content.decode()
        return list(csv.DictReader(io.StringIO(body)))

    def test_no_trip_is_exported_until_someone_reviews_it(self):
        self.assertEqual(self.csv_rows(), [])

    def test_an_accepted_trip_is_exported_with_its_times(self):
        self.assertEqual(self.vote("accepted").json()["verdict"], "accepted")
        row = self.csv_rows()[0]
        self.assertEqual((row["reference"], row["exit_track"], row["entry_track"], row["review"]),
                         ("reference_1", "3", "6", "accepted"))
        self.assertTrue(row["exit_time"].startswith("2024-05-08T10:00:00"))
        self.assertAlmostEqual(float(row["duration_sec"]), 3600 + 79 / 25, places=1)
        self.assertEqual((row["exit_clip"], row["entry_clip"]), ("a", "b"))

    def test_relink_and_didnt_come_back_and_taking_it_back(self):
        other = [e for e in self.clips[1]["events"] if "|4|" in e["key"]][0]
        self.vote("accepted", entry_key=other["key"], relinked=True)
        t = TripReview.objects.get()
        self.assertTrue(t.relinked)
        self.assertEqual((t.entry_track, t.reason), ("4", "relinked"))
        self.vote("rejected", entry=False, reason="didn't come back")
        t.refresh_from_db()
        self.assertEqual((t.verdict, t.entry_key, t.reason), ("rejected", "", "didn't come back"))
        self.assertEqual(self.csv_rows()[0]["review_notes"], "didn't come back")
        self.vote(None)
        self.assertFalse(TripReview.objects.exists())

    def test_two_reviewers_who_disagree_make_it_disputed(self):
        BatchMember.objects.create(batch_id=self.batch, user=self.maria)
        self.vote("accepted")
        self.assertEqual(self.vote("rejected", user=self.maria, reason="can't tell").status_code, 200)
        row = self.csv_rows()[0]
        self.assertEqual((row["review"], row["reviewed_by"]), ("disputed", "maria;owner"))

    def test_viewers_and_strangers_cannot_vote(self):
        BatchMember.objects.create(batch_id=self.batch, user=self.maria,
                                   role=BatchMember.Role.VIEWER)
        self.assertEqual(self.vote("accepted", user=self.maria).status_code, 404)
        self.assertEqual(self.vote("accepted", user=self.stranger).status_code, 404)
        self.assertEqual(self.vote("maybe").status_code, 400)
        self.assertFalse(TripReview.objects.exists())

    def test_the_batch_offers_reviewed_trips_only_after_a_review(self):
        self.client.force_login(self.owner)
        page = self.client.get(reverse("pipelines:batch_detail", args=[self.batch]))
        self.assertNotIn("trips", [d["kind"] for d in page.context["downloads"]])
        self.vote("accepted")
        page = self.client.get(reverse("pipelines:batch_detail", args=[self.batch]))
        self.assertIn("trips", [d["kind"] for d in page.context["downloads"]])
        body = self.client.get(reverse("pipelines:batch_combined_csv",
                                       kwargs={"batch_id": self.batch, "kind": "trips"})).content.decode()
        self.assertIn("reference_1", body)
