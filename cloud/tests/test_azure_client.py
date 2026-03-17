"""Tests for AzureBlobClient using mocked Azure SDK.

These tests verify the client logic without needing a real Azure account.
For integration tests with Azurite, see test_azure_integration.py.
"""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

from cloud.storage.config import StorageConfig


class TestAzureBlobClientInit:
    """Test client initialization paths."""

    @patch("cloud.storage.azure_client.BlobServiceClient")
    def test_init_with_connection_string(self, mock_bsc):
        mock_bsc.from_connection_string.return_value = MagicMock()
        config = StorageConfig(connection_string="DefaultEndpointsProtocol=https;AccountName=test;AccountKey=dGVzdA==;EndpointSuffix=core.windows.net")

        from cloud.storage.azure_client import AzureBlobClient
        client = AzureBlobClient(config)

        mock_bsc.from_connection_string.assert_called_once_with(config.connection_string)

    @patch("cloud.storage.azure_client.BlobServiceClient")
    def test_init_with_account_name_key(self, mock_bsc):
        mock_bsc.from_connection_string.return_value = MagicMock()
        # No connection string, so it falls back to account_name + key
        config = StorageConfig(
            connection_string="",
            account_name="myaccount",
            account_key="mykey",
        )

        from cloud.storage.azure_client import AzureBlobClient
        client = AzureBlobClient(config)

    def test_init_fails_without_credentials(self):
        config = StorageConfig(connection_string="", account_name="", account_key="")

        from cloud.storage.azure_client import AzureBlobClient
        with pytest.raises(ValueError):
            AzureBlobClient(config)


class TestAzureBlobClientOperations:
    """Test upload/download/list operations with mocked service."""

    @pytest.fixture
    def client(self):
        with patch("cloud.storage.azure_client.BlobServiceClient") as mock_bsc:
            mock_service = MagicMock()
            mock_bsc.from_connection_string.return_value = mock_service

            config = StorageConfig(
                connection_string="DefaultEndpointsProtocol=https;AccountName=test;AccountKey=dGVzdA==;EndpointSuffix=core.windows.net"
            )

            from cloud.storage.azure_client import AzureBlobClient
            c = AzureBlobClient(config)
            c._mock_service = mock_service
            return c

    def test_create_container(self, client):
        client.create_container("test-container")
        client._mock_service.create_container.assert_called_once_with("test-container")

    def test_create_container_already_exists(self, client):
        client._mock_service.create_container.side_effect = Exception("ContainerAlreadyExists")
        # Should not raise
        client.create_container("test-container")

    def test_create_all_containers(self, client):
        client.create_all_containers()
        assert client._mock_service.create_container.call_count == 4

    def test_list_blobs(self, client):
        mock_container = MagicMock()
        mock_blob1 = MagicMock()
        mock_blob1.name = "file1.mp4"
        mock_blob2 = MagicMock()
        mock_blob2.name = "file2.mp4"
        mock_container.list_blobs.return_value = [mock_blob1, mock_blob2]
        client._mock_service.get_container_client.return_value = mock_container

        result = client.list_blobs("raw-videos", prefix="user1/")
        assert result == ["file1.mp4", "file2.mp4"]

    def test_blob_exists_true(self, client):
        mock_blob = MagicMock()
        client._mock_service.get_blob_client.return_value = mock_blob
        assert client.blob_exists("models", "v1/nest_detection.pt") is True

    def test_blob_exists_false(self, client):
        mock_blob = MagicMock()
        mock_blob.get_blob_properties.side_effect = Exception("BlobNotFound")
        client._mock_service.get_blob_client.return_value = mock_blob
        assert client.blob_exists("models", "v1/nonexistent.pt") is False

    def test_delete_blob(self, client):
        mock_blob = MagicMock()
        client._mock_service.get_blob_client.return_value = mock_blob
        client.delete_blob("processed", "user1/job1/events.csv")
        mock_blob.delete_blob.assert_called_once()

    def test_upload_file(self, client, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello")

        mock_blob = MagicMock()
        client._mock_service.get_blob_client.return_value = mock_blob

        result = client.upload_file("raw-videos", "user1/video.mp4", str(test_file))
        assert result == "raw-videos/user1/video.mp4"
        mock_blob.upload_blob.assert_called_once()

    def test_download_file(self, client, tmp_path):
        out_path = tmp_path / "downloaded.txt"

        mock_blob = MagicMock()
        mock_stream = MagicMock()
        mock_blob.download_blob.return_value = mock_stream
        client._mock_service.get_blob_client.return_value = mock_blob

        result = client.download_file("raw-videos", "user1/video.mp4", str(out_path))
        assert result == str(out_path)
        mock_stream.readinto.assert_called_once()
