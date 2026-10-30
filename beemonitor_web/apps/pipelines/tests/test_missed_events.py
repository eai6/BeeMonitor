"""Events the system missed, added by reviewers, and the review metrics
(memory/62 §3b)."""

import json

from django.urls import reverse

from apps.pipelines import review
from apps.pipelines.models import AddedEvent, BatchMember, TripReview
from apps.pipelines.tests.test_event_review import EventReviewTests


class MissedEventTests(EventReviewTests):
    def add(self, user=None, **body):
        self.client.force_login(user or self.owner)
        payload = {"frame": 400, "action": "enter", "reference": "1", "track_id": ""}
        payload.update(body)
        return self.client.post(reverse("analysis:add_missed_event", args=[self.job.pk]),
                                json.dumps(payload), content_type="application/json")

    def test_a_reviewer_adds_a_missed_event(self):
        r = self.add(track_id="9", note="bee went in behind another")
        self.assertEqual(r.status_code, 200, r.content)
        a = AddedEvent.objects.get()
        self.assertEqual((a.action, a.reference, a.track_id, a.frame), ("enter", "reference_1", "9", 400))
        self.assertEqual(r.json()["counts"]["added"], 1)

    def test_one_next_to_a_detected_event_asks_first(self):
        detected = self.events[0]
        frame = int(detected["frame"]) + 10
        r = self.add(frame=frame, action=detected["action"], reference=detected["reference"])
        self.assertEqual(r.status_code, 409)
        self.assertTrue(r.json()["duplicate"])
        self.assertFalse(AddedEvent.objects.exists())
        r = self.add(frame=frame, action=detected["action"], reference=detected["reference"], force=True)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(AddedEvent.objects.count(), 1)

    def test_a_stay_needs_its_end(self):
        self.assertEqual(self.add(action="stay").status_code, 400)
        self.assertEqual(self.add(action="stay", end_frame=450).status_code, 200)

    def test_bad_input_is_refused(self):
        self.assertEqual(self.add(action="hover").status_code, 400)
        self.assertEqual(self.add(reference="").status_code, 400)

    def test_only_reviewers_add_and_only_the_adder_or_owner_removes(self):
        self.assertEqual(self.add(user=self.stranger).status_code, 404)
        BatchMember.objects.create(batch_id=self.batch, user=self.maria, role="reviewer")
        BatchMember.objects.create(batch_id=self.batch, user=self.stranger, role="viewer")
        self.assertEqual(self.add(user=self.stranger).status_code, 404)
        pk = self.add(user=self.maria).json()["id"]
        url = reverse("analysis:add_missed_event", args=[self.job.pk])
        third = self.owner.__class__.objects.create_user("third", password="x")
        BatchMember.objects.create(batch_id=self.batch, user=third, role="reviewer")
        self.client.force_login(third)
        self.assertEqual(self.client.post(url, json.dumps({"remove": pk}),
                                          content_type="application/json").status_code, 404)
        self.client.force_login(self.owner)
        self.assertEqual(self.client.post(url, json.dumps({"remove": pk}),
                                          content_type="application/json").status_code, 200)
        self.assertFalse(AddedEvent.objects.exists())

    def test_the_csv_carries_it_beside_the_detected_rows(self):
        self.add(track_id="9", note="behind another bee")
        rows = self.csv_rows()
        added = [r for r in rows if r["review"] == "added"]
        self.assertEqual(len(rows), len(self.events) + 1)
        self.assertEqual((added[0]["action"], added[0]["reference"], added[0]["track_id"],
                          added[0]["reviewed_by"], added[0]["review_notes"]),
                         ("enter", "reference_1", "9", "owner", "behind another bee"))

    def test_the_player_gets_it(self):
        self.add(track_id="9")
        live = json.loads(self.client.get(reverse("analysis:track_overlay", args=[self.job.pk]),
                                          {"part": "events"}).content)
        a = live["added"][0]
        self.assertEqual((a[0], a[1], a[2], a[3], a[6], a[7]), (400, 9, "enter", "reference_1", "owner", True))

    def test_precision_and_recall(self):
        self.assertEqual(review.quality(9, 1, 1), (90, 90))
        self.assertEqual(review.quality(0, 0, 0), (None, None))
        self.vote(self.owner, "accepted")
        self.add()
        counts = review.event_review_counts(self.run)
        self.assertEqual((counts["precision"], counts["recall"]), (100, 50))


class BatchReviewSummaryTests(EventReviewTests):
    def test_nothing_reviewed_shows_nothing(self):
        self.assertIsNone(review.batch_review_summary([self.run]))
        self.client.force_login(self.owner)
        page = self.client.get(reverse("pipelines:batch_detail", args=[self.batch]))
        self.assertIsNone(page.context["review_summary"])

    def test_events_and_trips_metrics(self):
        self.vote(self.owner, "accepted")
        self.client.post(reverse("analysis:add_missed_event", args=[self.job.pk]),
                         json.dumps({"frame": 400, "action": "enter", "reference": "1"}),
                         content_type="application/json")
        TripReview.objects.create(exit_run=self.run, exit_key="k1", user=self.owner,
                                  verdict="accepted", relinked=True, reason="relinked")
        TripReview.objects.create(exit_run=self.run, exit_key="k2", user=self.owner,
                                  verdict="rejected", reason="didn't come back")
        TripReview.objects.create(exit_run=self.run, exit_key="k3", user=self.owner, verdict="accepted")
        s = review.batch_review_summary([self.run])
        self.assertEqual((s["events"]["accepted"], s["events"]["added"], s["events"]["precision"],
                          s["events"]["recall"]), (1, 1, 100, 50))
        self.assertEqual((s["trips"]["reviewed"], s["trips"]["accepted"], s["trips"]["relinked"],
                          s["trips"]["didnt_come_back"], s["trips"]["suggestion_right"]), (3, 2, 1, 1, 33))
        self.client.force_login(self.owner)
        page = self.client.get(reverse("pipelines:batch_detail", args=[self.batch]))
        self.assertContains(page, "Events review")
        self.assertContains(page, "Trips review")
