"""A total without a denominator is not a usable number.

Every aggregate reported `clips = len(outputs)` — the clips that *succeeded*.
The batch where 9 of 12 runs failed rendered as a clean 3-clip day with nothing
saying the other 75% of the footage is missing. Coverage carries the
denominator through so a partial result reads as partial.
"""

from django.test import SimpleTestCase

from apps.pipelines import aggregate


class CoverageTests(SimpleTestCase):
    def test_a_complete_batch_says_so(self):
        cov = aggregate.coverage_of([{}, {}, {}], attempted=3)

        self.assertTrue(cov["complete"])
        self.assertEqual((cov["analysed"], cov["attempted"], cov["missing"]), (3, 3, 0))
        self.assertEqual(cov["pct"], 100)

    def test_the_batch_that_lost_nine_of_twelve_reports_the_gap(self):
        cov = aggregate.coverage_of([{}, {}, {}], attempted=12)

        self.assertFalse(cov["complete"])
        self.assertEqual(cov["missing"], 9)
        self.assertEqual(cov["pct"], 25)

    def test_clips_resting_on_a_guessed_frame_rate_are_counted(self):
        outputs = [{"fps_source": "analysis"}, {"fps_source": "assumed"},
                   {"fps_source": "assumed"}]

        self.assertEqual(aggregate.coverage_of(outputs, 3)["assumed_fps"], 2)

    def test_an_empty_batch_does_not_divide_by_zero(self):
        cov = aggregate.coverage_of([], attempted=0)

        self.assertEqual(cov["pct"], 0)
        self.assertEqual(cov["missing"], 0)

    def test_more_outputs_than_runs_never_reports_negative_missing(self):
        # A branched graph can emit two outputs of one kind from one run.
        cov = aggregate.coverage_of([{}, {}, {}], attempted=2)

        self.assertEqual(cov["missing"], 0)
        self.assertTrue(cov["complete"])


class TripPanelVisibilityTests(SimpleTestCase):
    """Trips are a read over events, so an Events pipeline still gets them.

    The panel used to key off the Foraging Trips *block*. With that block
    retired, a pipeline built on the Events primitive would have silently lost
    the trip section — the one thing it exists to produce.
    """

    @staticmethod
    def _show(kinds):
        results = [{"kind": k} for k in kinds]
        return any(a["kind"] in ("events", "foraging_trips") for a in results) or not results

    def test_an_events_pipeline_shows_trips(self):
        self.assertTrue(self._show(["events"]))

    def test_a_legacy_foraging_pipeline_still_shows_trips(self):
        self.assertTrue(self._show(["foraging_trips"]))

    def test_a_detection_count_pipeline_does_not(self):
        self.assertFalse(self._show(["detection_count"]))

    def test_an_interactions_only_pipeline_does_not(self):
        self.assertFalse(self._show(["interactions"]))

    def test_a_batch_with_no_analyzer_output_falls_back_to_showing_them(self):
        self.assertTrue(self._show([]))
