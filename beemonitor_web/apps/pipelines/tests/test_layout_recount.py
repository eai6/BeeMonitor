"""Moving reference objects, then recounting clips already analysed (memory/63).

References are only used after tracking, so a clip's saved tracks are counted
again against the layout it uses now — no GPU.
"""

import json

from django.urls import reverse

from apps.pipelines import recount
from apps.pipelines.models import PipelineRun

from . import test_recount as base
from .test_primitive_blocks import PrimitiveBlockTestCase

LAYOUT = {"roi": {"box": [0, 0, 1, 1], "points": None},
          "nests": [{"id": 1, "box": [0.1, 0.1, 0.2, 0.2], "points": None}], "scope": "clip"}


class LayoutRecountTests(PrimitiveBlockTestCase):
    # The recount tests' old run and its job, without re-running their tests.
    def setUp(self):
        super().setUp()
        self.tracking_csv = base._write_csv([{**r, "detected": 1} for r in base.TRACKING_ROWS],
                                            "tracking_detected.csv")

    _old_run = base.RecountTests._old_run
    _job_for = base.ResultsPageAgreesWithBatchTests._job_for

    def _analysed(self):
        run = self._old_run()
        self._job_for(run)
        return run

    def test_the_clips_runs_are_found_through_their_jobs(self):
        run = self._analysed()
        self._old_run()                       # no job: not this clip's
        self.assertEqual([r.pk for r in recount.runs_for_clips([self.video])], [run.pk])

    def test_saving_says_how_many_runs_were_counted_with_the_old_layout(self):
        self._analysed()
        self.client.force_login(self.user)
        d = self.client.post(reverse("videos:layout", args=[self.video.pk]), json.dumps(LAYOUT),
                             content_type="application/json").json()
        self.assertTrue(d["ok"])
        self.assertEqual((d["analysed"], d["scope"]), (1, "clip"))
        self.assertIn("roi_changed", d)

    def test_recount_counts_the_saved_tracks_again(self):
        run = self._analysed()
        self.client.force_login(self.user)
        d = self.client.post(reverse("videos:layout_recount", args=[self.video.pk]),
                             json.dumps({"scope": "clip"}), content_type="application/json").json()
        self.assertEqual((d["recounted"], d["queued"]), (1, 0))
        run.refresh_from_db()
        self.assertIn("counted_at", run.context["a"])
        self.assertEqual(run.counting_rules, recount.CURRENT_RULES)

    def test_many_runs_go_to_the_sweep(self):
        run = self._analysed()
        out = recount.recount_clips([self.video], now=0)
        self.assertEqual(out["queued"], 1)
        self.assertEqual(PipelineRun.objects.get(pk=run.pk).counting_rules, 0)
        self.assertEqual(recount.sweep(), 1)

    def test_only_the_clips_owner_may_recount(self):
        from django.contrib.auth import get_user_model
        other = get_user_model().objects.create_user("other", password="x")
        self.client.force_login(other)
        r = self.client.post(reverse("videos:layout_recount", args=[self.video.pk]),
                             json.dumps({"scope": "clip"}), content_type="application/json")
        self.assertEqual(r.status_code, 404)
