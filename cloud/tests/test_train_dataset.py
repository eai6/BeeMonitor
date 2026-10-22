"""The trainer builds its dataset from saved frames, not whole videos.

A 683-clip project once made it download ~300 GB of video to cut out 5,296
frames, and the instance ran out of space. Saved frame JPEGs are downloaded as
they are; only frames with none are cut from their video, one video at a time,
deleted afterwards.
"""

import importlib
import sys
from pathlib import Path

import boto3
import cv2
import numpy as np
import pytest
from moto import mock_aws

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sagemaker_backend"))


@pytest.fixture
def train(tmp_path, monkeypatch):
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        for b in ("raw", "processed"):
            s3.create_bucket(Bucket=b)
        mod = importlib.import_module("train")
        monkeypatch.setattr(mod, "_s3", s3)
        monkeypatch.setattr(mod, "RAW_VIDEOS_BUCKET", "raw")
        monkeypatch.setattr(mod, "PROCESSED_BUCKET", "processed")
        monkeypatch.setattr(mod, "WORK", tmp_path / "work")
        (tmp_path / "work").mkdir()
        yield mod, s3, tmp_path


def _video(path, frames=10):
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for i in range(frames):
        out.write(np.full((48, 64, 3), i * 20, np.uint8))
    out.release()


def test_saved_frames_are_used_and_only_the_rest_are_cut_from_video(train):
    mod, s3, tmp = train
    ok, jpg = cv2.imencode(".jpg", np.zeros((48, 64, 3), np.uint8))
    s3.put_object(Bucket="processed", Key="frames/a_0001.jpg", Body=jpg.tobytes())
    _video(tmp / "clip.mp4")
    s3.upload_file(str(tmp / "clip.mp4"), "raw", "1/clip.mp4")
    downloads = []
    real = s3.download_file
    s3.download_file = lambda b, k, p: (downloads.append((b, k)), real(b, k, p))[1]

    manifest = {"classes": ["insect"], "val_percent": 20, "video_annotations": [
        {"video_blob_path": "1/saved_only.mp4",
         "frames": [{"frame_number": 1, "filename": "a_f000001.jpg", "label": "0 0.5 0.5 0.1 0.1",
                     "image_key": "frames/a_0001.jpg"}]},
        {"video_blob_path": "1/clip.mp4",
         "frames": [{"frame_number": 3, "filename": "b_f000003.jpg", "label": "0 0.4 0.4 0.1 0.1"},
                    {"frame_number": 7, "filename": "b_f000007.jpg", "label": ""}]},
    ]}
    data_yaml, n_train, n_val = mod._build_dataset(manifest, tmp / "ds")

    assert ("raw", "1/saved_only.mp4") not in downloads        # its frame was saved
    assert ("processed", "frames/a_0001.jpg") in downloads
    assert ("raw", "1/clip.mp4") in downloads
    assert (n_train, n_val) == (3, 3)                            # tiny: val reuses train
    images = sorted({p.name for p in (tmp / "ds" / "images" / "train").glob("*.jpg")})
    assert images == ["a_f000001.jpg", "b_f000003.jpg", "b_f000007.jpg"]
    assert not list((tmp / "work" / "videos").iterdir())        # the video was deleted
    assert data_yaml.exists()
