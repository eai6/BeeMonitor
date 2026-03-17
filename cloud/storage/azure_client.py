"""Azure Blob Storage client for BeeMonitor Cloud.

Handles upload, download, SAS URL generation, and container management.
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import BinaryIO, Optional

from azure.storage.blob import (
    BlobServiceClient,
    ContainerClient,
    BlobClient,
    BlobSasPermissions,
    generate_blob_sas,
)

from cloud.storage.config import StorageConfig

logger = logging.getLogger(__name__)


class AzureBlobClient:
    """Thin wrapper around Azure Blob SDK for BeeMonitor operations."""

    def __init__(self, config: Optional[StorageConfig] = None):
        self.config = config or StorageConfig()
        self.config.validate()

        if self.config.connection_string:
            self._service = BlobServiceClient.from_connection_string(
                self.config.connection_string
            )
        else:
            from azure.storage.blob import BlobServiceClient as _Bsc

            self._service = _Bsc(
                account_url=f"https://{self.config.account_name}.blob.core.windows.net",
                credential=self.config.account_key,
            )

    # ------------------------------------------------------------------
    # Container helpers
    # ------------------------------------------------------------------

    def create_container(self, name: str) -> None:
        """Create a container if it doesn't exist."""
        try:
            self._service.create_container(name)
            logger.info("Created container: %s", name)
        except Exception as e:
            if "ContainerAlreadyExists" in str(e):
                logger.debug("Container already exists: %s", name)
            else:
                raise

    def create_all_containers(self) -> None:
        """Create every container defined in config."""
        for name in self.config.all_containers:
            self.create_container(name)

    def list_containers(self) -> list[str]:
        return [c.name for c in self._service.list_containers()]

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------

    def upload_file(
        self,
        container: str,
        blob_path: str,
        local_path: str,
        overwrite: bool = True,
        content_type: Optional[str] = None,
    ) -> str:
        """Upload a local file to Azure Blob Storage.

        Returns the full blob path ``container/blob_path``.
        """
        blob = self._service.get_blob_client(container, blob_path)
        kwargs: dict = {"overwrite": overwrite}
        if content_type:
            from azure.storage.blob import ContentSettings

            kwargs["content_settings"] = ContentSettings(content_type=content_type)

        with open(local_path, "rb") as fh:
            blob.upload_blob(fh, **kwargs)

        logger.info("Uploaded %s -> %s/%s", local_path, container, blob_path)
        return f"{container}/{blob_path}"

    def upload_stream(
        self,
        container: str,
        blob_path: str,
        stream: BinaryIO,
        overwrite: bool = True,
        content_type: Optional[str] = None,
    ) -> str:
        """Upload a file-like stream to Azure Blob Storage."""
        blob = self._service.get_blob_client(container, blob_path)
        kwargs: dict = {"overwrite": overwrite}
        if content_type:
            from azure.storage.blob import ContentSettings

            kwargs["content_settings"] = ContentSettings(content_type=content_type)

        blob.upload_blob(stream, **kwargs)
        logger.info("Uploaded stream -> %s/%s", container, blob_path)
        return f"{container}/{blob_path}"

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download_file(self, container: str, blob_path: str, local_path: str) -> str:
        """Download a blob to a local file. Returns the local path."""
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        blob = self._service.get_blob_client(container, blob_path)
        with open(local_path, "wb") as fh:
            stream = blob.download_blob()
            stream.readinto(fh)

        logger.info("Downloaded %s/%s -> %s", container, blob_path, local_path)
        return local_path

    def download_to_stream(self, container: str, blob_path: str, stream: BinaryIO) -> None:
        """Download a blob into an open writable stream."""
        blob = self._service.get_blob_client(container, blob_path)
        downloader = blob.download_blob()
        downloader.readinto(stream)

    # ------------------------------------------------------------------
    # SAS URLs
    # ------------------------------------------------------------------

    def generate_sas_url(
        self,
        container: str,
        blob_path: str,
        expiry_hours: Optional[int] = None,
        permissions: str = "r",
    ) -> str:
        """Generate a time-limited SAS URL for a blob.

        Args:
            container: Container name.
            blob_path: Path within container.
            expiry_hours: Hours until expiry (default from config).
            permissions: 'r' for read, 'w' for write, 'rw' for both.
        """
        if expiry_hours is None:
            expiry_hours = self.config.sas_expiry_hours

        perm = BlobSasPermissions(read="r" in permissions, write="w" in permissions)
        expiry = datetime.now(timezone.utc) + timedelta(hours=expiry_hours)

        account_name = self._service.account_name
        # Extract account key — needed for SAS generation
        account_key = self.config.account_key
        if not account_key and self.config.connection_string:
            for part in self.config.connection_string.split(";"):
                if part.startswith("AccountKey="):
                    account_key = part.split("=", 1)[1]
                    break

        token = generate_blob_sas(
            account_name=account_name,
            container_name=container,
            blob_name=blob_path,
            account_key=account_key,
            permission=perm,
            expiry=expiry,
        )
        return f"https://{account_name}.blob.core.windows.net/{container}/{blob_path}?{token}"

    # ------------------------------------------------------------------
    # Metadata / list / delete
    # ------------------------------------------------------------------

    def list_blobs(self, container: str, prefix: str = "") -> list[str]:
        """List blob names in a container, optionally filtered by prefix."""
        client = self._service.get_container_client(container)
        return [b.name for b in client.list_blobs(name_starts_with=prefix or None)]

    def blob_exists(self, container: str, blob_path: str) -> bool:
        blob = self._service.get_blob_client(container, blob_path)
        try:
            blob.get_blob_properties()
            return True
        except Exception:
            return False

    def get_blob_properties(self, container: str, blob_path: str) -> dict:
        blob = self._service.get_blob_client(container, blob_path)
        props = blob.get_blob_properties()
        return {
            "name": props.name,
            "size": props.size,
            "content_type": props.content_settings.content_type,
            "last_modified": props.last_modified.isoformat() if props.last_modified else None,
            "etag": props.etag,
        }

    def delete_blob(self, container: str, blob_path: str) -> None:
        blob = self._service.get_blob_client(container, blob_path)
        blob.delete_blob()
        logger.info("Deleted %s/%s", container, blob_path)

    def delete_prefix(self, container: str, prefix: str) -> int:
        """Delete all blobs under a prefix. Returns count deleted."""
        blobs = self.list_blobs(container, prefix=prefix)
        for name in blobs:
            self.delete_blob(container, name)
        return len(blobs)
