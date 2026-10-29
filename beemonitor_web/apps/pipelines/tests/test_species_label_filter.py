"""The species vote must not empty the analyzers (memory/60).

Identify species writes each track's voted species into the tracking table's
``taxon`` column. The analyzers used to take their rows by matching that column
against the Detect class ("bee"), so a pipeline with species ID on found no
tracks, and no events. With one Detect class every track is the subject; with
several, rows are taken by the detector's own ``class`` column.
"""

import pandas as pd
from django.test import SimpleTestCase

from apps.pipelines import executors, ops

SPECIES_TABLE = pd.DataFrame({
    "frame": [0, 1, 0, 1],
    "track_id": [1, 1, 2, 2],
    "cx": [10, 11, 50, 51], "cy": [10, 11, 50, 51],
    "taxon": ["Osmia cornifrons", "Osmia cornifrons", "Xylocopa virginica", "Xylocopa virginica"],
})


class _Run:
    def __init__(self, labels):
        self.steps = [{"id": f"d{i}", "block_type": "detect.objects", "config": {"label": label}}
                      for i, label in enumerate(labels)]


class SpeciesVoteTests(SimpleTestCase):
    def test_one_detect_class_takes_every_track_whatever_the_species(self):
        label = executors._tracks_label(_Run(["bee"]), {"tracks": {"label": "bee"}})
        self.assertEqual(label, "")
        self.assertEqual(len(ops.filter_by_label(SPECIES_TABLE, label)), 4)

    def test_matching_the_class_against_species_names_finds_nothing(self):
        # What happened before: "bee" against species names.
        self.assertEqual(len(ops.filter_by_label(SPECIES_TABLE, "bee")), 0)

    def test_several_classes_are_split_by_the_detectors_own_class_column(self):
        table = SPECIES_TABLE.assign(**{"class": ["bee", "bee", "flower", "flower"]})
        label = executors._tracks_label(_Run(["bee", "flower"]), {"tracks": {"label": "bee"}})
        self.assertEqual(label, "bee")
        kept = ops.filter_by_label(table, label)
        self.assertEqual(sorted(kept["track_id"].unique()), [1])
