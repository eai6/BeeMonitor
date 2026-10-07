"""Tests for cloud.wrapper.pipeline."""

from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
import pandas as pd

import pytest

from cloud.wrapper.pipeline import CloudPipeline, PipelineResult
from cloud.wrapper.model_manager import ModelPaths


class TestPipelineResult:
    def test_to_dict(self):
        result = PipelineResult(
            job_id="job1",
            user_id="user1",
            total_events=10,
            entry_count=5,
            exit_count=5,
            unique_tracks=8,
            nest_count=48,
            events_csv_path="processed/user1/job1/events.csv",
            tracking_csv_path="processed/user1/job1/tracking_results.csv",
            foraging_trips_csv_path="processed/user1/job1/foraging_trips.csv",
            interactions_csv_path="processed/user1/job1/interactions.csv",
            crops_csv_path="processed/user1/job1/track_crops.csv",
            annotated_video_path="processed/user1/job1/annotated_video.mp4",
            foraging_trip_count=3,
            avg_trip_duration_sec=120.5,
            interaction_count=12,
            summary_stats={"total_events": 10},
        )
        d = result.to_dict()
        assert d["job_id"] == "job1"
        assert d["total_events"] == 10
        assert d["entry_count"] == 5


class TestCloudPipeline:
    @pytest.fixture
    def mock_storage(self):
        return MagicMock()

    @pytest.fixture
    def mock_models(self):
        mm = MagicMock()
        mm.ensure_models.return_value = ModelPaths(
            nest_detection="/tmp/models/nest_detection.pt",
            bee_tracking="/tmp/models/bee_tracking.pt",
            event_classifier="/tmp/models/event_classifier_model.pkl",
        )
        return mm

    @pytest.fixture
    def pipeline(self, mock_storage, mock_models, tmp_path):
        config = MagicMock()
        config.raw_videos_container = "raw-videos"
        config.processed_container = "processed"
        return CloudPipeline(
            storage_client=mock_storage,
            storage_config=config,
            model_manager=mock_models,
            local_work_dir=str(tmp_path / "work"),
        )

    def test_cleanup(self, pipeline, tmp_path):
        job_dir = Path(pipeline.work_dir) / "test_job"
        job_dir.mkdir(parents=True)
        (job_dir / "test.txt").write_text("hello")

        pipeline.cleanup("test_job")
        assert not job_dir.exists()

    def test_cleanup_nonexistent(self, pipeline):
        # Should not raise
        pipeline.cleanup("nonexistent_job")

    @patch("cloud.wrapper.pipeline.CloudPipeline._run_analysis")
    def test_process_calls_steps_in_order(self, mock_run, pipeline, mock_storage, mock_models):
        # Setup mock analysis result
        mock_result = MagicMock()
        mock_result.events = pd.DataFrame({
            "action": ["Entry", "Exit", "Entry"],
            "nest": [1, 2, 3],
        })
        mock_result.tracks = pd.DataFrame({
            "track_id": [1, 1, 2],
            "frame": [10, 20, 30],
        })
        mock_result.get_statistics.return_value = {
            "total_events": 3,
            "total_entries": 2,
            "total_exits": 1,
            "total_nests": 48,
            "total_tracks": 2,
        }
        mock_run.return_value = mock_result

        # Create a fake video to "download"
        def fake_download(container, blob_path, local_path):
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            Path(local_path).write_text("fake video")
            return local_path

        mock_storage.download_file.side_effect = fake_download
        mock_storage.upload_file.return_value = "processed/user1/job1/events.csv"

        result = pipeline.process(
            job_id="job1",
            user_id="user1",
            video_blob_path="user1/upload1/video.mp4",
        )

        # Verify download was called
        mock_storage.download_file.assert_called_once()

        # Verify models were ensured
        mock_models.ensure_models.assert_called_once()

        # Verify analysis ran
        mock_run.assert_called_once()

        # Verify uploads happened (events, tracking, video, metadata = 4 calls min)
        assert mock_storage.upload_file.call_count >= 1

        # Verify result structure
        assert isinstance(result, PipelineResult)
        assert result.job_id == "job1"
        assert result.total_events == 3
        assert result.entry_count == 2
        assert result.exit_count == 1


class TestLayout:
    """No ROI drawn → the whole frame; no tubes drawn → none (the nest model
    may fill them). Tracking must run either way: a new hotel the model has
    never seen used to skip the whole analysis and report an empty run."""

    @pytest.fixture
    def clip(self, tmp_path):
        cv2 = pytest.importorskip("cv2")
        import numpy as np
        path = str(tmp_path / "c.avi")
        out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 240))
        for _ in range(3):
            out.write(np.zeros((240, 320, 3), np.uint8))
        out.release()
        return path

    def test_no_roi_is_the_whole_frame(self, clip):
        layout = CloudPipeline._build_manual_nests(clip, None, None)
        assert layout == {"hotel": (0, 0, 320, 240), "nests": {}}

    def test_a_drawn_roi_is_used_without_tubes(self, clip):
        layout = CloudPipeline._build_manual_nests(clip, [0.5, 0.5, 1.0, 1.0], [])
        assert layout["hotel"] == (160, 120, 320, 240)
        assert layout["nests"] == {}

    def test_drawn_tubes_are_kept(self, clip):
        layout = CloudPipeline._build_manual_nests(
            clip, None, [{"id": 1, "box": [0, 0, 0.5, 0.5]}])
        assert layout["hotel"] == (0, 0, 320, 240)
        assert layout["nests"] == {1: (0, 0, 160, 120)}

    @staticmethod
    def _fill(clip, found):
        """_fill_nests with the nest model stubbed. The stubs stand in for the
        whole import chain (ultralytics, beemonitor.*) so this runs in CI's
        cloud job, which installs neither."""
        import sys
        import types

        det = MagicMock()
        det.return_value.get_nests_and_hotel_detections.return_value = found
        cfg = MagicMock()
        mods = {name: types.ModuleType(name) for name in (
            "ultralytics", "beemonitor", "beemonitor.core", "beemonitor.core.config",
            "beemonitor.detection", "beemonitor.detection.nest_detector")}
        mods["ultralytics"].YOLO = MagicMock()
        mods["beemonitor.core.config"].Config = cfg
        mods["beemonitor.detection.nest_detector"].NestDetector = det
        with patch.dict(sys.modules, mods):
            return CloudPipeline._fill_nests(
                CloudPipeline._build_manual_nests(clip, None, None), clip, "m.pt")

    def test_no_nests_found_still_returns_a_layout(self, clip):
        assert self._fill(clip, None) == {"hotel": (0, 0, 320, 240), "nests": {}}

    def test_model_nests_fill_in_but_the_roi_stays(self, clip):
        layout = self._fill(clip, {"hotel": (10, 10, 20, 20), "nests": {"a": (1, 2, 3, 4)}})
        assert layout == {"hotel": (0, 0, 320, 240), "nests": {"a": (1, 2, 3, 4)}}


class TestBestCrops:
    """The job page shows each track's sharpest crops first, in frame order."""

    def test_sharpest_kept_in_frame_order(self):
        from cloud.wrapper.pipeline import _best_keys
        rows = [(0, "a", 1.0), (1, "b", 9.0), (2, "c", 5.0), (3, "d", 7.0)]
        assert _best_keys(rows, 2) == ["b", "d"]

    def test_blur_scores_lower(self, tmp_path):
        import cv2
        import numpy as np
        from cloud.wrapper.pipeline import _sharpness
        sharp = (np.indices((64, 64)).sum(axis=0) % 2 * 255).astype(np.uint8)
        cv2.imwrite(str(tmp_path / "s.png"), sharp)
        cv2.imwrite(str(tmp_path / "b.png"), cv2.GaussianBlur(sharp, (9, 9), 3))
        assert _sharpness(tmp_path / "s.png") > _sharpness(tmp_path / "b.png")


class TestCropUploads:
    """Only each track's sharpest crops leave the worker."""

    def test_at_most_the_cap_per_track_ranked_by_the_trackers_scores(self, tmp_path, monkeypatch):
        import cloud.wrapper.pipeline as pl
        from cloud.wrapper.pipeline import CloudPipeline

        monkeypatch.setattr(pl, "UPLOAD_CROPS_PER_TRACK", 2)
        clip = tmp_path / "crops" / "clip"
        d = clip / "track_0001"
        d.mkdir(parents=True)
        for f in range(4):
            (d / f"frame_{f:06d}.jpg").write_bytes(b"x")
        (clip / "sharpness.csv").write_text(
            "crop,sharpness\n" + "".join(
                f"track_0001/frame_{f:06d}.jpg,{s}\n" for f, s in [(0, 1), (1, 9), (2, 5), (3, 7)]))
        storage = MagicMock()
        cfg = MagicMock()
        pipe = CloudPipeline(storage_client=storage, storage_config=cfg,
                             model_manager=MagicMock(), local_work_dir=str(tmp_path / "w"))

        out = pipe._upload_results("j", "u", tmp_path, str(tmp_path / "clip.mp4"))

        crop_puts = sorted(c.args[1] for c in storage.upload_file.call_args_list
                           if c.args[1].endswith(".jpg"))
        assert crop_puts == ["u/j/crops/track_0001/frame_000001.jpg",
                             "u/j/crops/track_0001/frame_000003.jpg"]
        assert out["crops_total"] == 2


class TestIdentifyTracks:
    """The worker writes each track's vote into the tracking CSV the platform reads."""

    def _setup(self, tmp_path):
        import cv2
        import numpy as np
        for tid in (1, 2):
            d = tmp_path / "crops" / "clip" / f"track_{tid:04d}"
            d.mkdir(parents=True)
            for f in range(3):
                cv2.imwrite(str(d / f"frame_{f:06d}.jpg"), np.zeros((40, 40, 3), np.uint8))
        pd.DataFrame({
            "frame": [0, 1, 0], "track_id": [1, 1, 2],
            "taxon": ["bee", "bee", "bee"], "taxon_confidence": [None] * 3,
            "taxon_votes": [0, 0, 0], "bee_id": [None] * 3,
            "bee_id_method": [None] * 3, "bee_id_confidence": [0.0] * 3,
        }).to_csv(tmp_path / "clip_tracking_results.csv", index=False)

    def test_species_written_back_and_status_reported(self, tmp_path):
        from cloud.wrapper.pipeline import identify_tracks
        self._setup(tmp_path)

        class Species:
            def classify_images(self, images):
                return [("Osmia lignaria", 0.4)] * len(images)

        status = identify_tracks(tmp_path, species=Species(),
                                 species_status={"model": "beemachine", "loaded": True})
        df = pd.read_csv(tmp_path / "clip_tracking_results.csv")
        assert set(df["taxon"]) == {"Osmia lignaria"}
        assert set(df["taxon_votes"]) == {3}
        assert status["species"]["identified"] == 2
        assert status["species"]["crops"] == 6
        assert (tmp_path / "track_votes.csv").exists()

    def test_a_track_with_no_reading_keeps_the_detector_label(self, tmp_path):
        from cloud.wrapper.pipeline import identify_tracks
        self._setup(tmp_path)

        class Species:
            def classify_images(self, images):
                return [None] * len(images)

        identify_tracks(tmp_path, species=Species(),
                        species_status={"model": "beemachine", "loaded": True})
        df = pd.read_csv(tmp_path / "clip_tracking_results.csv")
        assert set(df["taxon"]) == {"bee"}
        assert set(df["taxon_votes"]) == {0}
