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


# ── Cancelled clips (memory/49) ──────────────────────────────────────────────

class _Storage:
    def __init__(self, cancelled):
        self.cancelled = cancelled
        self.asked = []

    def blob_exists(self, container, key):
        self.asked.append((container, key))
        return self.cancelled


class _Pipeline:
    def __init__(self, cancelled, process=None):
        self._storage = _Storage(cancelled)
        self.processed = False
        self._process = process

    def process(self, **kw):
        self.processed = True
        return self._process(**kw)


_PAYLOAD = {"job_id": "pl_abc-c1", "user_id": 1, "video_blob_path": "v.mp4"}


def test_a_clip_cancelled_while_queued_is_answered_without_processing():
    pipeline = _Pipeline(cancelled=True)
    with mock.patch.object(inference, "_profiler", return_value=None), \
            mock.patch.object(inference, "_detect_device", return_value="cuda:T4"):
        out = inference.predict_fn(dict(_PAYLOAD), pipeline)
    assert out["status"] == "cancelled"
    assert not pipeline.processed
    assert pipeline._storage.asked == [("processed", "cancel/pl_abc")]


def test_a_clip_cancelled_mid_run_returns_cancelled_not_failed():
    from beemonitor.cancellation import JobCancelled

    def stops(**kw):
        raise JobCancelled("Cancelled by user.")
    pipeline = _Pipeline(cancelled=False, process=stops)
    with mock.patch.object(inference, "_profiler", return_value=None), \
            mock.patch.object(inference, "_detect_device", return_value="cuda:T4"):
        out = inference.predict_fn(dict(_PAYLOAD), pipeline)
    assert out["status"] == "cancelled"
    assert pipeline.processed
