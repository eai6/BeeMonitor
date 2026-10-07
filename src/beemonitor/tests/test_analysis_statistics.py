"""get_statistics returns its stats when there are events.

It fell off the end and returned None whenever a clip had any event, so the
cloud worker crashed ('NoneType' object does not support item assignment)
after tracking, rendering and uploading — losing the whole run.
"""

import unittest

import pandas as pd

from beemonitor.core.analysis_results import AnalysisResults


class StatisticsTests(unittest.TestCase):
    def _results(self, events):
        r = object.__new__(AnalysisResults)
        r.events = events
        r.tracks = pd.DataFrame({"track_id": [1, 1, 2]})
        r.nests = {"nests": {"1": (0, 0, 1, 1), "2": (2, 2, 3, 3)}}
        return r

    def test_with_events(self):
        events = pd.DataFrame({"nest": ["1", "1"], "action": ["Entry", "Exit"]})
        stats = self._results(events).get_statistics()
        self.assertEqual((stats["total_events"], stats["total_entries"], stats["total_exits"],
                          stats["total_tracks"]), (2, 1, 1, 2))

    def test_without_events(self):
        self.assertEqual(self._results(pd.DataFrame()).get_statistics()["total_events"], 0)


if __name__ == "__main__":
    unittest.main()
