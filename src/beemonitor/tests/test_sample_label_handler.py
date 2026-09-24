"""The GPU endpoint's ``sample_label`` task, end to end with fakes for S3 and
SAM 3: moving detections only, one bad clip never fails the batch, the result
lands at the fixed key, and batched SAM 3 falls back to single frames."""

import json
import os
import shutil
import sys
import tempfile
from unittest import mock

import pytest

from beemonitor.tests.test_sample_label import _write, detect_dark

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "sagemaker_backend"))
import inference  # noqa: E402  (sagemaker_backend/inference.py, as the image imports it)


class FakeStorage:
    def __init__(self, clips):
        self.clips, self.uploaded = clips, {}

    def download_file(self, container, blob, dest):
        assert container == "raw-videos"
        if blob not in self.clips:
            raise FileNotFoundError(blob)
        shutil.copy(self.clips[blob], dest)
        return dest

    def upload_stream(self, container, key, stream, overwrite=True, content_type=None):
        assert container == "processed" and content_type == "image/jpeg"
        self.uploaded[key] = len(stream.read())


class FakeDet:
    def __init__(self, label, conf, bbox):
        self.label, self.confidence, self.bbox = label, conf, bbox


class FakeSam3:
    """Boxes every dark object (bees and nest plugs), labelled with the prompt."""
    def __init__(self, prompt="bee", **kw):
        self.prompt = prompt.split(",")[0]

    def detect_many(self, frames, batch_size=8):
        return [[FakeDet(self.prompt, b["confidence"], (b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]))
                 for b in boxes] for boxes in detect_dark(frames)]


@pytest.fixture
def clips():
    d = tempfile.mkdtemp()
    bee = os.path.join(d, "bee.avi")
    _write(bee)
    yield {"users/1/devices/2/2026/07/a.mp4": bee}
    shutil.rmtree(d)


def test_batch_writes_moving_frames_and_isolates_a_bad_clip(clips):
    storage = FakeStorage(clips)
    pipeline = mock.Mock(_storage=storage)
    put = mock.Mock()
    payload = {"task": "sample_label", "batch_id": "b1", "result_bucket": "out", "result_key": "sampling-results/b1.json",
               "classes": ["bee"], "candidates": 10, "min_gap_s": 0.2,
               "clips": [{"task_id": 7, "video_blob_path": "users/1/devices/2/2026/07/a.mp4", "max_frames": 3},
                         {"task_id": 8, "video_blob_path": "users/1/devices/2/2026/07/missing.mp4"}]}
    with mock.patch("beemonitor.detection.sam3_detector.Sam3Detector", FakeSam3), \
         mock.patch("boto3.client", return_value=mock.Mock(put_object=put)):
        assert inference.input_fn(json.dumps(payload), "application/json")["batch_id"] == "b1"
        out = inference._sample_label_batch(payload, pipeline)

    good, bad = out["clips"]
    assert good["task_id"] == 7 and 1 <= len(good["frames"]) <= 3
    for f in good["frames"]:
        assert f["key"] == f"frames/users_1_devices_2_2026_07_a.mp4/f{f['n']:06d}.jpg"
        assert f["key"] in storage.uploaded
        assert f["boxes"] and all(b["class"] == "bee" and b["y"] < 200 for b in f["boxes"])
    assert good["motion"]["picked"] and "seconds" in good
    assert bad["task_id"] == 8 and "error" in bad and bad["frames"] == []
    put.assert_called_once()
    assert put.call_args.kwargs["Key"] == "sampling-results/b1.json"


def test_a_sample_label_payload_needs_no_single_video():
    body = json.dumps({"task": "sample_label", "batch_id": "b", "clips": [{}]})
    assert inference.input_fn(body, "application/json")["task"] == "sample_label"
    with pytest.raises(ValueError):
        inference.input_fn(json.dumps({"task": "sample_label", "clips": []}), "application/json")


def test_detect_many_falls_back_to_single_frames():
    import numpy as np
    from beemonitor.detection.sam3_detector import Sam3Detector

    det = Sam3Detector(prompt="bee", conf_threshold=0.3)
    det._ensure_model = lambda: None
    det._segment_batch = mock.Mock(side_effect=RuntimeError("CUDA out of memory"))
    det._segment = mock.Mock(return_value=[(1.0, 2.0, 11.0, 12.0, 0.9)])
    frames = [np.zeros((20, 20, 3), np.uint8) for _ in range(3)]
    out = det.detect_many(frames, batch_size=8)
    assert [len(d) for d in out] == [1, 1, 1]
    assert det._segment.call_count == 3 and det._segment_batch.call_count == 1
