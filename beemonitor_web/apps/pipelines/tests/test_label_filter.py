"""A class filter must not empty a table it cannot speak the language of.

Analyzers take their own rows out of a shared GPU table by taxon, because one
GPU pass serves several Detect nodes. The Detect node carries the class the
USER configured ("bee"); the tracking CSV carries the class the DETECTOR
emitted ("insect"). When those disagree the filter matched nothing, the tracks
vanished, normalized_tracks saw an empty frame, and every analyzer reported
"tracking CSV not available" — a clip full of insects read as a quiet one.
"""

import pandas as pd
from django.test import SimpleTestCase

from apps.pipelines import ops


def rows(*taxa):
    return pd.DataFrame([
        {"frame": i, "track_id": i, "cx": 0.2, "cy": 0.7, "taxon": t}
        for i, t in enumerate(taxa)])


class LabelFilterTests(SimpleTestCase):
    def test_a_matching_class_is_selected(self):
        df = ops.filter_by_label(rows("bee", "nest", "bee"), "bee")

        self.assertEqual(len(df), 2)

    def test_a_single_class_table_survives_a_vocabulary_mismatch(self):
        """The whole table is 'insect' and the node asks for 'bee'. The filter
        cannot tell those apart, and emptying the table hides the clip."""
        df = ops.filter_by_label(rows("insect", "insect", "insect"), "bee")

        self.assertEqual(len(df), 3)

    def test_a_mixed_table_still_reports_a_genuinely_absent_class(self):
        """Here the filter CAN tell: the table distinguishes classes and none
        of them is a wasp. Passing bees through as wasps would be worse than
        returning nothing."""
        df = ops.filter_by_label(rows("bee", "nest"), "wasp")

        self.assertEqual(len(df), 0)

    def test_several_wanted_classes_are_honoured(self):
        df = ops.filter_by_label(rows("bee", "nest", "wasp"), "bee, wasp")

        self.assertEqual(len(df), 2)

    def test_matching_ignores_case_and_padding(self):
        df = ops.filter_by_label(rows(" Bee ", "nest"), "bee")

        self.assertEqual(len(df), 1)

    def test_no_label_means_no_filter(self):
        self.assertEqual(len(ops.filter_by_label(rows("bee", "nest"), "")), 2)

    def test_a_table_without_a_taxon_column_passes_through(self):
        df = pd.DataFrame([{"frame": 0, "track_id": 1, "cx": 0.2, "cy": 0.7}])

        self.assertEqual(len(ops.filter_by_label(df, "bee")), 1)

    def test_the_mismatch_does_not_silently_empty_the_analysis(self):
        """End to end: the tracks must still normalise, which is what every
        analyzer needs before it can compute anything at all."""
        df = ops.filter_by_label(rows(*(["insect"] * 5)), "bee")

        self.assertIsNotNone(ops.normalized_tracks(df, {}))
