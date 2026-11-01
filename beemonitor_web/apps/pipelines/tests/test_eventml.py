"""Event classifiers trained and evaluated on review data (memory/70)."""

import io
import random
from unittest import mock

from django.test import SimpleTestCase
from django.urls import reverse

from apps.pipelines import eventml, ml_scoring, review
from apps.pipelines.models import AddedEvent, EventReview
from apps.training.models import CustomModel

from .test_period_review import PeriodBase

STORE = {}


def fake_store(payload, user_id, name):
    key = f"k{len(STORE)}"
    STORE[key] = payload
    return key


def synthetic(n, kind="exit_enter", left_out_every=4, seed=0):
    """Rows whose label follows one feature, spread over days."""
    rnd = random.Random(seed)
    names = eventml._features(kind)
    rows = []
    for i in range(n):
        label = i % 2
        feats = {k: rnd.random() for k in names}
        feats[names[1]] = label * 5 + rnd.random()          # the informative one
        day = (i // 2) % 8
        rows.append({"kind": kind, "label": label, "action": "exit" if kind != "stay" else "stay",
                     "features": feats, "site": "mendels", "day": day,
                     "left_out": day % left_out_every == 0, "clip": i, "batch": "b",
                     "run": "r", "key": str(i)})
    return rows


class TrainingTests(SimpleTestCase):
    def test_trains_on_the_other_days_and_tests_on_the_left_out_ones(self):
        rows = synthetic(80)
        model, report = eventml.train(rows, {"missed_rows": []}, "exit_enter")
        self.assertIsNotNone(model)
        self.assertEqual(report["test_events"], sum(r["left_out"] for r in rows))
        self.assertEqual(report["train_events"] + report["test_events"], 80)
        self.assertGreater(report["test"]["auc"], 0.9)
        self.assertIn("auc", report["validation"])          # grouped by day

    def test_missed_events_count_against_pipeline_recall(self):
        probs, labels = [0.9, 0.9, 0.1, 0.8], [1, 1, 0, 0]
        plain = eventml.evaluate(probs, labels, missed=0)
        with_missed = eventml.evaluate(probs, labels, missed=2)
        self.assertEqual(plain["pipeline_recall"], 100)
        self.assertEqual(with_missed["pipeline_recall"], 50)
        self.assertEqual(plain["pipeline_precision"], 67)

    def test_too_little_data_says_so(self):
        model, report = eventml.train(synthetic(6), {"missed_rows": []}, "stay")
        self.assertIsNone(model)
        self.assertIn("Not enough", report["note"])

    def test_stay_features(self):
        f = ml_scoring.stay_features([(10, 10), (11, 10), (12, 11)], [0, 1, 2], (0, 0, 20, 20), [8, 8])
        self.assertEqual(set(f), set(ml_scoring.STAY_FEATURES))
        self.assertEqual(f["duration_frames"], 2)
        self.assertEqual(f["body_length"], 8.0)


class DatasetTests(PeriodBase):
    def setUp(self):
        super().setUp()
        for clip in (self.a, self.b):
            run = clip["run"]
            for r in run.context["e"]["rows"]:
                key = review.event_key(r)
                a, ref, t, f = key.split("|")
                EventReview.objects.create(run=run, event_key=key, action=a, reference=ref,
                                           track_id=t, frame=int(f), user=self.owner,
                                           verdict="accepted")
        AddedEvent.objects.create(run=self.a["run"], frame=3, action="enter",
                                  reference="reference_1", user=self.owner)

    def test_each_reviewed_event_becomes_a_labelled_row(self):
        rows, summary = eventml.dataset(self.owner, [str(self.batch)], "2024-05-08")
        self.assertEqual(len(rows), summary["events"])
        self.assertTrue(rows)
        self.assertTrue(all(r["label"] == 1 and r["left_out"] for r in rows))
        self.assertEqual(set(rows[0]["features"]), set(eventml._features("exit_enter")))
        self.assertEqual(summary["missed"], 1)
        csv = eventml.csv_text(rows)
        self.assertTrue(csv.startswith("kind,action,label,left_out,site,day"))

    def test_the_page_shows_the_data_and_trains(self):
        self.client.force_login(self.owner)
        url = reverse("training:event_classifiers")
        page = self.client.get(url, {"batch": str(self.batch), "show": "1", "left_out": ""})
        self.assertContains(page, "reviewed events")
        with mock.patch("threading.Thread") as thread:
            self.client.post(url, {"batch": str(self.batch), "kind": ["exit_enter", "stay"],
                                   "left_out": ""})
        self.assertEqual(thread.call_count, 2)
        self.assertEqual(CustomModel.objects.filter(model_type="event_classifier",
                                                    status="training").count(), 2)
        csv = self.client.get(reverse("training:event_dataset_csv"), {"batch": str(self.batch)})
        self.assertEqual(csv["Content-Type"], "text/csv")


class ScoringTests(PeriodBase):
    def test_a_trained_model_scores_the_events_node(self):
        import joblib

        rows = synthetic(60)
        model, _report = eventml.train(rows, {"missed_rows": []}, "exit_enter")
        buf = io.BytesIO()
        joblib.dump({"model": model, "feature_names": eventml._features("exit_enter"),
                     "kind": "exit_enter", "actions": ["exit", "enter"]}, buf)
        STORE["m1"] = buf.getvalue()
        m = CustomModel.objects.create(user=self.owner, name="Exit/enter model v1",
                                       model_type="event_classifier", base_model="random_forest",
                                       classes=["exit", "enter"], storage_key="m1")
        ml_scoring._trained.cache_clear()
        run = self.a["run"]
        run.steps[2]["config"] = {"ml_scoring": f"model:{m.pk}", "ml_min_confidence": "1.1"}
        run.save()
        from apps.pipelines.recount import recount_run
        with mock.patch.object(eventml, "read_model", side_effect=lambda k: STORE[k]):
            recount_run(run)
        run.refresh_from_db()
        out = run.context["e"]
        self.assertEqual(out["rows"], [])                      # all below the 1.1 cut
        self.assertIn("left out", out["note"])
