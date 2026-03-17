"""Azure Blob connector — optimized same-cloud transfers."""

import logging
import os
from pathlib import Path
from typing import BinaryIO

from cloud.ingestion.base_connector import BaseConnector, RemoteFile

logger = logging.getLogger(__name__)


class AzureConnector(BaseConnector):
    """Connect to an external Azure Blob Storage container.

    Optimized: uses server-side copy when transferring between Azure accounts.
    """

    source_type = "azure_blob"

    def __init__(self, container: str, connection_string: str = ""):
        self.container = container
        self._connection_string = connection_string or os.environ.get(
            "AZURE_STORAGE_CONNECTION_STRING", ""
        )
        self._service = None

    def _get_service(self):
        if self._service is None:
            from azure.storage.blob import BlobServiceClient

            self._service = BlobServiceClient.from_connection_string(self._connection_string)
        return self._service

    def test_connection(self) -> tuple[bool, str]:
        try:
            svc = self._get_service()
            client = svc.get_container_client(self.container)
            client.get_container_properties()
            return True, f"Connected to Azure container: {self.container}"
        except Exception as e:
            return False, f"Azure connection failed: {e}"

    def list_files(self, prefix: str = "", limit: int = 100) -> list[RemoteFile]:
        svc = self._get_service()
        client = svc.get_container_client(self.container)
        files = []
        for blob in client.list_blobs(name_starts_with=prefix or None):
            files.append(
                RemoteFile(
                    name=Path(blob.name).name,
                    path=blob.name,
                    size=blob.size,
                    content_type=blob.content_settings.content_type if blob.content_settings else "",
                    last_modified=blob.last_modified.isoformat() if blob.last_modified else "",
                )
            )
            if len(files) >= limit:
                break
        return files

    def download_to_file(self, remote_path: str, local_path: str) -> str:
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        svc = self._get_service()
        blob = svc.get_blob_client(self.container, remote_path)
        with open(local_path, "wb") as fh:
            stream = blob.download_blob()
            stream.readinto(fh)
        logger.info("Downloaded azure://%s/%s -> %s", self.container, remote_path, local_path)
        return local_path

    def download_to_stream(self, remote_path: str, stream: BinaryIO) -> int:
        svc = self._get_service()
        blob = svc.get_blob_client(self.container, remote_path)
        downloader = blob.download_blob()
        data = downloader.readall()
        stream.write(data)
        return len(data)

    def download_to_azure(
        self,
        remote_path: str,
        azure_client,
        azure_container: str,
        azure_blob_path: str,
    ) -> str:
        """Server-side copy between Azure containers (fast, no egress)."""
        svc = self._get_service()
        source_blob = svc.get_blob_client(self.container, remote_path)
        source_url = source_blob.url

        dest_blob = azure_client._service.get_blob_client(azure_container, azure_blob_path)
        dest_blob.start_copy_from_url(source_url)
        logger.info(
            "Server-side copy azure://%s/%s -> %s/%s",
            self.container, remote_path, azure_container, azure_blob_path,
        )
        return azure_blob_path
