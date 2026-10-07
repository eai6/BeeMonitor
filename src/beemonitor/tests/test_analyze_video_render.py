"""The annotated video is rendered only when asked for.

analyze_video rendered it on every run. On the cloud worker (visualize=False)
that re-read and re-encoded a 14.5k-frame 1080p clip on the CPU after tracking
— 13+ minutes nobody used — and helped push long clips past SageMaker's hour.
"""

import tempfile
import unittest
from unittest import mock

import pandas as pd

from beemonitor.core.config import Config
from beemonitor.core.video_analyzer import BeeMonitor


class RenderOnlyWhenAskedTests(unittest.TestCase):
    def _run(self, visualize):
        tracks = pd.DataFrame({"frame": [0], "track_id": [1], "cx": [1.0], "cy": [1.0]})
        monitor = object.__new__(BeeMonitor)   # __init__ loads models
        monitor.config = Config.default()
        with tempfile.TemporaryDirectory() as tmp, \
                tempfile.NamedTemporaryFile(suffix=".mp4") as clip, \
                mock.patch.object(BeeMonitor, "get_motion_tracking", return_value=(tracks, tracks)), \
                mock.patch.object(BeeMonitor, "process_motion_tracking", return_value=pd.DataFrame()), \
                mock.patch.object(BeeMonitor, "synthesize_csv", return_value=pd.DataFrame()), \
                mock.patch("beemonitor.core.video_analyzer.AnalysisResults") as results:
            monitor.analyze_video(clip.name, output_folder=tmp, visualize=visualize,
                                  manual_nests={"hotel": (0, 0, 10, 10), "nests": {}})
        return results.return_value

    def test_no_video_unless_asked(self):
        self.assertFalse(self._run(visualize=False).save_video.called)

    def test_video_when_asked(self):
        self.assertTrue(self._run(visualize=True).save_video.called)


if __name__ == "__main__":
    unittest.main()
