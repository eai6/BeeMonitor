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


class TaxonAgnosticTests(SimpleTestCase):
    """Renaming a class must move rows between branches, not empty one.

    The Detect node carries the class the user configured; the tracking table
    carries the class the detector emitted. Those drift — a detector retrained
    to say "insect" instead of "bee", a node renamed to a species. The pipeline
    has to keep working across that.
    """

    def test_an_exact_match_is_reported_as_exact(self):
        taxa, how = ops.resolve_taxa(rows("bee", "flower"), "bee")

        self.assertEqual((taxa, how), ({"bee"}, "exact"))

    def test_the_subject_branch_is_whatever_the_reference_branch_is_not(self):
        """The detector says 'insect'; the node says 'bee'. A sibling Detect
        node claims 'flower', so the rest of the table is this branch."""
        taxa, how = ops.resolve_taxa(rows("insect", "flower"), "bee",
                                     exclude=["flower"])

        self.assertEqual((taxa, how), ({"insect"}, "role"))

    def test_role_resolution_survives_renaming_the_subject_class(self):
        for detector_says in ("insect", "apis", "osmia bicornis", "arthropod"):
            taxa, how = ops.resolve_taxa(rows(detector_says, "flower"), "bee",
                                         exclude=["flower"])
            self.assertEqual(taxa, {detector_says}, detector_says)
            self.assertEqual(how, "role")

    def test_role_resolution_survives_renaming_the_reference_class(self):
        taxa, how = ops.resolve_taxa(rows("insect", "petal"), "bee",
                                     exclude=["petal"])

        self.assertEqual((taxa, how), ({"insect"}, "role"))

    def test_a_single_class_table_needs_no_sibling_to_resolve(self):
        taxa, how = ops.resolve_taxa(rows("insect", "insect"), "bee")

        self.assertEqual((taxa, how), ({"insect"}, "only"))

    def test_a_genuinely_absent_class_is_still_reported_absent(self):
        """Two classes, neither is a wasp, and no sibling claim disambiguates.
        Handing back bees here would be worse than an empty answer."""
        taxa, how = ops.resolve_taxa(rows("bee", "flower"), "wasp")

        self.assertEqual((taxa, how), (set(), "absent"))

    def test_a_sibling_claim_that_matches_nothing_does_not_resolve_by_role(self):
        """If the excluded class is not in the table either, the exclusion says
        nothing about which rows are ours."""
        taxa, how = ops.resolve_taxa(rows("bee", "flower"), "wasp",
                                     exclude=["beetle"])

        self.assertEqual(how, "absent")

    def test_an_unconfigured_branch_takes_everything(self):
        taxa, how = ops.resolve_taxa(rows("bee", "flower"), "")

        self.assertIsNone(taxa)

    def test_the_filter_and_the_resolver_agree(self):
        df = rows("insect", "flower", "insect")

        kept = ops.filter_by_label(df, "bee", exclude=["flower"])

        self.assertEqual(len(kept), 2)
        self.assertEqual(set(kept["taxon"]), {"insect"})
