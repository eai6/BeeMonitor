"""
Thin wrapper that maps DataSource model instances to the appropriate
cloud/ingestion connector for downloading files.
"""

from __future__ import annotations

import json
import logging
import tempfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from apps.sources.models import DataSource

logger = logging.getLogger(__name__)


class BaseConnector:
    """Abstract base for all source connectors."""

    def __init__(self, config: dict):
        self.config = config

    def test_connection(self) -> tuple[bool, str]:
        raise NotImplementedError

    def download(self, remote_path: str) -> str:
        """Download a remote file and return the local temporary path."""
        raise NotImplementedError


class S3Connector(BaseConnector):
    """AWS S3 connector."""

    def test_connection(self) -> tuple[bool, str]:
        try:
            import boto3

            client = boto3.client(
                "s3",
                aws_access_key_id=self.config.get("access_key_id"),
                aws_secret_access_key=self.config.get("secret_access_key"),
                region_name=self.config.get("region", "us-east-1"),
            )
            client.list_buckets()
            return True, "Connected to AWS S3."
        except Exception as exc:
            return False, str(exc)

    def download(self, remote_path: str) -> str:
        import boto3

        client = boto3.client(
            "s3",
            aws_access_key_id=self.config.get("access_key_id"),
            aws_secret_access_key=self.config.get("secret_access_key"),
            region_name=self.config.get("region", "us-east-1"),
        )
        bucket = self.config["bucket"]
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        client.download_file(bucket, remote_path, tmp.name)
        return tmp.name


class GCSConnector(BaseConnector):
    """Google Cloud Storage connector."""

    def test_connection(self) -> tuple[bool, str]:
        try:
            from google.cloud import storage as gcs
            from google.oauth2 import service_account

            creds = service_account.Credentials.from_service_account_info(
                self.config
            )
            client = gcs.Client(credentials=creds)
            list(client.list_buckets(max_results=1))
            return True, "Connected to Google Cloud Storage."
        except Exception as exc:
            return False, str(exc)

    def download(self, remote_path: str) -> str:
        from google.cloud import storage as gcs
        from google.oauth2 import service_account

        creds = service_account.Credentials.from_service_account_info(
            self.config
        )
        client = gcs.Client(credentials=creds)
        bucket = client.bucket(self.config["bucket"])
        blob = bucket.blob(remote_path)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        blob.download_to_filename(tmp.name)
        return tmp.name


class GoogleDriveConnector(BaseConnector):
    """Google Drive connector."""

    def test_connection(self) -> tuple[bool, str]:
        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build

            creds = service_account.Credentials.from_service_account_info(
                self.config,
                scopes=["https://www.googleapis.com/auth/drive.readonly"],
            )
            service = build("drive", "v3", credentials=creds)
            service.files().list(pageSize=1).execute()
            return True, "Connected to Google Drive."
        except Exception as exc:
            return False, str(exc)

    def download(self, remote_path: str) -> str:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build

        creds = service_account.Credentials.from_service_account_info(
            self.config,
            scopes=["https://www.googleapis.com/auth/drive.readonly"],
        )
        service = build("drive", "v3", credentials=creds)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        request = service.files().get_media(fileId=remote_path)
        with open(tmp.name, "wb") as f:
            f.write(request.execute())
        return tmp.name


# ── Connector registry ────────────────────────────────────────────────────────

_CONNECTOR_MAP: dict[str, type[BaseConnector]] = {
    "aws_s3": S3Connector,
    "gcs": GCSConnector,
    "google_drive": GoogleDriveConnector,
}


def get_connector(source: DataSource) -> BaseConnector:
    """
    Instantiate the appropriate connector for a DataSource.

    Decodes the encrypted config (stored as bytes) back into a dict and
    passes it to the matching connector class.
    """
    connector_cls = _CONNECTOR_MAP.get(source.source_type)
    if connector_cls is None:
        raise ValueError(f"Unsupported source type: {source.source_type}")

    config = json.loads(bytes(source.config_encrypted).decode("utf-8"))
    return connector_cls(config)
