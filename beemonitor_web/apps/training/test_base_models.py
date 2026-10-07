"""Every "train from scratch" choice names weights Ultralytics can fetch.

YOLOv11n/s were sent as yolov11n.pt; Ultralytics calls them yolo11n.pt, so
training from them failed before it started.
"""

import importlib.util
import unittest

from django.test import SimpleTestCase

from apps.training.models import TrainingJob
from apps.training.views import ultralytics_weights

# Ultralytics' release asset names (ultralytics/utils/downloads.py).
EXPECTED = {
    "yolov8n": "yolov8n", "yolov8s": "yolov8s", "yolov8m": "yolov8m",
    "yolov11n": "yolo11n", "yolov11s": "yolo11s",
    "yolo26n": "yolo26n", "yolo26s": "yolo26s", "yolo26m": "yolo26m",
}


class BaseModelWeightsTests(SimpleTestCase):
    def test_every_choice_maps_to_its_ultralytics_weights(self):
        self.assertEqual({v: ultralytics_weights(v) for v in TrainingJob.BaseModel.values},
                         EXPECTED)

    # The Django CI job does not install Ultralytics; checked where it is.
    @unittest.skipUnless(importlib.util.find_spec("ultralytics"), "ultralytics not installed")
    def test_ultralytics_knows_every_name(self):
        from ultralytics.utils.downloads import GITHUB_ASSETS_NAMES

        for name in EXPECTED.values():
            self.assertIn(f"{name}.pt", GITHUB_ASSETS_NAMES)
