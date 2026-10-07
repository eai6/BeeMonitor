"""Every task reports how long the handler ran.

detect_photo returned no execution_seconds, so photo runs showed "GPU time 0s"
and were priced at nothing.
"""

import importlib.util
from pathlib import Path
from unittest import mock

_spec = importlib.util.spec_from_file_location(
    "sm_inference", Path(__file__).resolve().parents[2] / "sagemaker_backend" / "inference.py")
inference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(inference)


def test_a_photo_run_reports_its_time():
    with mock.patch.object(inference, "_profiler", return_value=None), \
            mock.patch.object(inference, "_detect_device", return_value="cuda:T4"), \
            mock.patch.object(inference, "_detect_photo", return_value={"status": "completed"}):
        out = inference.predict_fn({"task": "detect_photo"}, pipeline=None)
    assert out["status"] == "completed"
    assert "execution_seconds" in out and out["execution_seconds"] >= 0


def test_a_task_that_times_itself_keeps_its_own_figure():
    with mock.patch.object(inference, "_profiler", return_value=None), \
            mock.patch.object(inference, "_detect_device", return_value="cuda:T4"), \
            mock.patch.object(inference, "_transcode",
                              return_value={"status": "completed", "execution_seconds": 42.0}):
        out = inference.predict_fn({"task": "transcode"}, pipeline=None)
    assert out["execution_seconds"] == 42.0
