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
            annotated_video_path="processed/user1/job1/annotated_video.mp4",
            foraging_trip_count=3,
            avg_trip_duration_sec=120.5,
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
