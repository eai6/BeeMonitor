"""Review metrics count the actions the batch chooses (memory/70).

Foraging needs exits and entries: stays are still reviewed, but the batch's
precision and recall can leave them out. A classifier's cut-off can be tried
on the reviewed events it scored.
"""

import json

from django.urls import reverse

from apps.pipelines import batches, review
from apps.pipelines.models import AddedEvent, EventReview

from .test_period_review import PeriodBase


class ReviewActionsTests(PeriodBase):
    def _vote(self, run, row, verdict):
        key = review.event_key(row)
        a, ref, t, f = key.split("|")
        EventReview.objects.create(run=run, event_key=key, action=a, reference=ref, track_id=t,
                                   frame=int(f), user=self.owner, verdict=verdict)

    def setUp(self):
        super().setUp()
        self.runs = [self.a["run"], self.b["run"]]
        rows = [(run, r) for run in self.runs for r in run.context["e"]["rows"]]
        self.actions = sorted({r["action"] for _run, r in rows})
        for run, r in rows:
            self._vote(run, r, "accepted")
        # A made-up stay, reviewed as wrong, and a missed stay.
        run = self.a["run"]
        run.context["e"]["rows"].append({"action": "stay", "reference": "reference_1",
                                         "track_id": 77, "frame": 5, "end_frame": 9})
        run.save()
        self._vote(run, run.context["e"]["rows"][-1], "rejected")
        AddedEvent.objects.create(run=run, frame=20, action="stay", reference="reference_1",
                                  user=self.owner)

    def test_all_actions_count_by_default(self):
        self.assertEqual(batches.review_actions(self.batch), ["exit", "enter", "stay"])
        ev = review.batch_review_summary(self.runs)["events"]
        self.assertEqual(ev["rejected"], 1)
        self.assertEqual(ev["added"], 1)

    def test_exits_and_entries_only_leave_stays_out(self):
        batches.set_review_actions(self.batch, self.owner, ["exit", "enter"])
        chosen = batches.review_actions(self.batch)
        ev = review.batch_review_summary(self.runs, actions=chosen)["events"]
        self.assertEqual((ev["rejected"], ev["added"]), (0, 0))
        self.assertEqual(ev["precision"], 100)
        # The stay vote is still there to see in review.
        self.assertTrue(EventReview.objects.filter(action="stay").exists())

    def test_the_batch_page_saves_the_choice(self):
        self.client.force_login(self.owner)
        url = reverse("pipelines:batch_review_actions", args=[self.batch])
        self.client.post(url, {"actions": ["exit", "enter"]})
        page = self.client.get(reverse("pipelines:batch_detail", args=[self.batch]))
        self.assertEqual(page.context["review_actions"], ["exit", "enter"])
        self.assertContains(page, "left out of these numbers")
        self.assertEqual(page.context["review_summary"]["events"]["rejected"], 0)

    def test_others_cannot_change_it(self):
        self.client.force_login(self.stranger)
        res = self.client.post(reverse("pipelines:batch_review_actions", args=[self.batch]),
                               {"actions": ["stay"]})
        self.assertEqual(res.status_code, 404)

    def test_scored_reviewed_events_feed_the_cut_off_panel(self):
        run = self.a["run"]
        for r in run.context["e"]["rows"]:
            r["ml_confidence"] = 0.9 if r["action"] != "stay" else 0.2
        run.save()
        self.client.force_login(self.owner)
        page = self.client.get(reverse("pipelines:batch_detail", args=[self.batch]))
        data = json.loads(page.context["cutoff_json"])
        self.assertIn(["stay", 0.2, "rejected"], data["scored"])
        self.assertEqual(data["added"], 1)
        self.assertContains(page, "Classifier cut-off")
