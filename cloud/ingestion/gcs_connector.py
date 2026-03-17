"""Google Cloud Storage connector."""

import logging
from pathlib import Path
from typing import BinaryIO

from cloud.ingestion.base_connector import BaseConnector, RemoteFile

logger = logging.getLogger(__name__)


class GCSConnector(BaseConnector):
    """Connect to a Google Cloud Storage bucket."""

    source_type = "gcs"

    def __init__(self, bucket: str, service_account_json: dict | None = None):
        self.bucket_name = bucket
        self._sa_json = service_account_json
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google.cloud import storage as gcs_storage

            if self._sa_json:
                self._client = gcs_storage.Client.from_service_account_info(self._sa_json)
            else:
                self._client = gcs_storage.Client()
        return self._client

    def test_connection(self) -> tuple[bool, str]:
        try:
            client = self._get_client()
            bucket = client.get_bucket(self.bucket_name)
            return True, f"Connected to gs://{self.bucket_name}"
        except Exception as e:
            return False, f"GCS connection failed: {e}"

    def list_files(self, prefix: str = "", limit: int = 100) -> list[RemoteFile]:
        client = self._get_client()
        bucket = client.bucket(self.bucket_name)
        blobs = bucket.list_blobs(prefix=prefix or None, max_results=limit)
        files = []
        for blob in blobs:
            files.append(
                RemoteFile(
                    name=Path(blob.name).name,
                    path=blob.name,
                    size=blob.size or 0,
                    content_type=blob.content_type or "",
                    last_modified=blob.updated.isoformat() if blob.updated else "",
                )
            )
        return files

    def download_to_file(self, remote_path: str, local_path: str) -> str:
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        client = self._get_client()
        bucket = client.bucket(self.bucket_name)
        blob = bucket.blob(remote_path)
        blob.download_to_filename(local_path)
        logger.info("Downloaded gs://%s/%s -> %s", self.bucket_name, remote_path, local_path)
        return local_path

    def download_to_stream(self, remote_path: str, stream: BinaryIO) -> int:
        client = self._get_client()
        bucket = client.bucket(self.bucket_name)
        blob = bucket.blob(remote_path)
        blob.download_to_file(stream)
        return blob.size or 0
