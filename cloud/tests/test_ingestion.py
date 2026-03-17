"""Tests for multi-cloud ingestion connectors."""

from unittest.mock import MagicMock

import pytest

from cloud.ingestion.base_connector import RemoteFile
from cloud.ingestion.credential_manager import CredentialManager
from cloud.ingestion.direct_upload import DirectUploadHandler


class TestRemoteFile:
    def test_creation(self):
        f = RemoteFile(name="video.mp4", path="folder/video.mp4", size=1000)
        assert f.name == "video.mp4"
        assert f.size == 1000


class TestCredentialManager:
    def test_round_trip(self):
        cm = CredentialManager()
        creds = {"access_key": "AKIA...", "secret": "s3cr3t"}
        encrypted = cm.encrypt(creds)
        decrypted = cm.decrypt(encrypted)
        assert decrypted == creds

    def test_different_keys_fail(self):
        cm1 = CredentialManager()
        cm2 = CredentialManager()
        encrypted = cm1.encrypt({"key": "value"})
        with pytest.raises(Exception):
            cm2.decrypt(encrypted)


class TestDirectUploadHandler:
    @pytest.fixture
    def handler(self, tmp_path):
        mock_storage = MagicMock()
        mock_config = MagicMock()
        mock_config.raw_videos_container = "raw-videos"
        return DirectUploadHandler(
            storage_client=mock_storage,
            storage_config=mock_config,
            upload_dir=str(tmp_path / "uploads"),
        )

    def test_init_upload(self, handler):
        result = handler.init_upload("user1", "video.mp4", 1000, 2)
        assert "session_id" in result
        assert result["total_chunks"] == 2

    def test_upload_chunks_and_finalize(self, handler):
        session = handler.init_upload("user1", "video.mp4", 10, 2)
        sid = session["session_id"]
        handler.upload_chunk(sid, 0, b"hello")
        r2 = handler.upload_chunk(sid, 1, b"world")
        assert r2["complete"] is True
        result = handler.finalize_upload(sid)
        assert result["file_size"] == 10
        assert "user1" in result["azure_blob_path"]

    def test_finalize_with_missing_chunks(self, handler):
        session = handler.init_upload("user1", "video.mp4", 10, 3)
        sid = session["session_id"]
        handler.upload_chunk(sid, 0, b"part1")
        with pytest.raises(ValueError, match="Missing chunks"):
            handler.finalize_upload(sid)

    def test_get_session_status(self, handler):
        session = handler.init_upload("user1", "video.mp4", 10, 2)
        sid = session["session_id"]
        handler.upload_chunk(sid, 0, b"data1")
        status = handler.get_session_status(sid)
        assert status["received"] == 1
        assert status["complete"] is False

    def test_unknown_session(self, handler):
        with pytest.raises(ValueError, match="Unknown session"):
            handler.upload_chunk("fake_id", 0, b"data")


class TestS3Connector:
    def _make_connector(self, mock_client=None):
        from cloud.ingestion.s3_connector import S3Connector

        client = mock_client or MagicMock()
        return S3Connector(
            bucket="my-bucket",
            access_key_id="key",
            secret_access_key="secret",
            _boto3_client=client,
        )

    def test_test_connection_success(self):
        conn = self._make_connector()
        ok, msg = conn.test_connection()
        assert ok is True
        assert "my-bucket" in msg

    def test_test_connection_failure(self):
        mock_client = MagicMock()
        mock_client.head_bucket.side_effect = Exception("AccessDenied")
        conn = self._make_connector(mock_client)
        ok, msg = conn.test_connection()
        assert ok is False

    def test_list_files(self):
        mock_client = MagicMock()
        mock_client.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "videos/test.mp4", "Size": 5000},
            ]
        }
        conn = self._make_connector(mock_client)
        files = conn.list_files(prefix="videos/")
        assert len(files) == 1
        assert files[0].name == "test.mp4"
        assert files[0].size == 5000

    def test_download_to_file(self, tmp_path):
        mock_client = MagicMock()
        conn = self._make_connector(mock_client)
        local = str(tmp_path / "out.mp4")
        conn.download_to_file("videos/test.mp4", local)
        mock_client.download_file.assert_called_once_with("my-bucket", "videos/test.mp4", local)
