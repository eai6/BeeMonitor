"""Tests for cloud.wrapper.model_manager."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cloud.wrapper.model_manager import ModelManager, MODEL_FILES, ModelPaths


class TestModelManager:
    @pytest.fixture
    def mock_client(self):
        return MagicMock()

    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.models_container = "models"
        return config

    @pytest.fixture
    def manager(self, mock_client, mock_config, tmp_path):
        return ModelManager(
            storage_client=mock_client,
            storage_config=mock_config,
            local_cache_dir=str(tmp_path / "cache"),
            model_version="v1",
        )

    def test_ensure_models_downloads_when_not_cached(self, manager, mock_client):
        paths = manager.ensure_models()

        assert mock_client.download_file.call_count == 3
        assert isinstance(paths, ModelPaths)
        assert "nest_detection.pt" in paths.nest_detection
        assert "bee_tracking.pt" in paths.bee_tracking
        assert "event_classifier_model.pkl" in paths.event_classifier

    def test_ensure_models_skips_cached(self, manager, mock_client):
        # Pre-create cached files
        for filename in MODEL_FILES.values():
            cached = Path(manager.cache_dir) / "v1" / filename
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_text("fake model")

        paths = manager.ensure_models()

        # Should NOT download since files exist
        mock_client.download_file.assert_not_called()
        assert isinstance(paths, ModelPaths)

    def test_upload_models(self, manager, mock_client, tmp_path):
        # Create fake model files
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        for filename in MODEL_FILES.values():
            (models_dir / filename).write_text("fake")

        uploaded = manager.upload_models(str(models_dir))

        assert len(uploaded) == 3
        assert mock_client.upload_file.call_count == 3

    def test_upload_models_skips_missing(self, manager, mock_client, tmp_path):
        # Empty dir — no model files
        models_dir = tmp_path / "empty_models"
        models_dir.mkdir()

        uploaded = manager.upload_models(str(models_dir))
        assert len(uploaded) == 0
        mock_client.upload_file.assert_not_called()

    def test_verify_models(self, manager, mock_client, tmp_path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        test_file = models_dir / "nest_detection.pt"
        test_file.write_bytes(b"x" * 100)

        mock_client.get_blob_properties.return_value = {"size": 100}

        results = manager.verify_models(str(models_dir))
        assert results["nest_detection"] is True
        # Others don't exist locally
        assert results["bee_tracking"] is False
        assert results["event_classifier"] is False

    def test_clear_cache(self, manager):
        # Create cache dir with a file
        cache_path = Path(manager.cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        (cache_path / "test.txt").write_text("hello")

        manager.clear_cache()
        assert not cache_path.exists()
