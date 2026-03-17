"""Tests for Modal function logic (without actually running on Modal).

These test the internal helper functions by patching Modal imports.
"""

from unittest.mock import MagicMock, patch
import sys

import pytest


# We need to mock Modal's module-level decorators before importing
# the function modules. Use a dedicated helper to do this cleanly.

@pytest.fixture(autouse=True)
def _mock_modal_app(monkeypatch):
    """Prevent Modal from initializing during import."""
    mock_modal = MagicMock()
    mock_modal.Image.debian_slim.return_value = MagicMock()
    mock_modal.Volume.from_name.return_value = MagicMock()
    mock_modal.Secret.from_name.return_value = MagicMock()
    mock_modal.App.return_value = MagicMock()

    # Patch modal at sys.modules level so imports resolve
    monkeypatch.setitem(sys.modules, "modal", mock_modal)

    # Clear cached modules that may have imported the real modal
    for key in list(sys.modules.keys()):
        if key.startswith("cloud.modal_app"):
            monkeypatch.delitem(sys.modules, key, raising=False)

    yield mock_modal


class TestIngestVideoHelpers:
    def test_download_s3_dispatch(self):
        from cloud.modal_app.functions.ingest_video import _download_from_source

        with patch("cloud.modal_app.functions.ingest_video._download_s3") as mock:
            mock.return_value = "/tmp/video.mp4"
            result = _download_from_source("aws_s3", {"bucket": "b", "key": "k"}, "/tmp")
            mock.assert_called_once()
            assert result == "/tmp/video.mp4"

    def test_download_azure_dispatch(self):
        from cloud.modal_app.functions.ingest_video import _download_from_source

        with patch("cloud.modal_app.functions.ingest_video._download_azure") as mock:
            mock.return_value = "/tmp/video.mp4"
            _download_from_source("azure_blob", {}, "/tmp")
            mock.assert_called_once()

    def test_download_gcs_dispatch(self):
        from cloud.modal_app.functions.ingest_video import _download_from_source

        with patch("cloud.modal_app.functions.ingest_video._download_gcs") as mock:
            mock.return_value = "/tmp/video.mp4"
            _download_from_source("gcs", {}, "/tmp")
            mock.assert_called_once()

    def test_download_gdrive_dispatch(self):
        from cloud.modal_app.functions.ingest_video import _download_from_source

        with patch("cloud.modal_app.functions.ingest_video._download_gdrive") as mock:
            mock.return_value = "/tmp/video.mp4"
            _download_from_source("google_drive", {}, "/tmp")
            mock.assert_called_once()

    def test_download_direct(self):
        from cloud.modal_app.functions.ingest_video import _download_from_source

        result = _download_from_source("direct", {"local_path": "/data/vid.mp4"}, "/tmp")
        assert result == "/data/vid.mp4"

    def test_unsupported_source_raises(self):
        from cloud.modal_app.functions.ingest_video import _download_from_source

        with pytest.raises(ValueError, match="Unsupported"):
            _download_from_source("ftp", {}, "/tmp")


class TestProcessVideoImport:
    def test_module_imports(self):
        """Verify process_video module loads without errors."""
        from cloud.modal_app.functions import process_video

        assert hasattr(process_video, "process_video")
