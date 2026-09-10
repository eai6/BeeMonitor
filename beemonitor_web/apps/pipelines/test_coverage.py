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
