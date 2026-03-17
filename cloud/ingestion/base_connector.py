"""Abstract base class for cloud storage connectors."""

import abc
import logging
from dataclasses import dataclass
from typing import BinaryIO, Optional

logger = logging.getLogger(__name__)


@dataclass
class RemoteFile:
    """Metadata about a file in a remote storage source."""

    name: str
    path: str
    size: int  # bytes
    content_type: str = ""
    last_modified: str = ""


class BaseConnector(abc.ABC):
    """Abstract connector for external cloud storage.

    Each implementation handles one source type (S3, GCS, Azure, GDrive).
    """

    source_type: str = "base"

    @abc.abstractmethod
    def test_connection(self) -> tuple[bool, str]:
        """Test that the configured credentials are valid.

        Returns:
            (success, message) tuple.
        """

    @abc.abstractmethod
    def list_files(self, prefix: str = "", limit: int = 100) -> list[RemoteFile]:
        """List files in the remote storage, optionally filtered by prefix."""

    @abc.abstractmethod
    def download_to_file(self, remote_path: str, local_path: str) -> str:
        """Download a remote file to a local path.

        Returns the local path.
        """

    @abc.abstractmethod
    def download_to_stream(self, remote_path: str, stream: BinaryIO) -> int:
        """Download a remote file into an open writable stream.

        Returns bytes written.
        """

    def download_to_azure(
        self,
        remote_path: str,
        azure_client,
        azure_container: str,
        azure_blob_path: str,
    ) -> str:
        """Transfer a file from remote source directly to Azure Blob.

        Default implementation streams through memory. Subclasses may
        override for optimized transfers (e.g., Azure-to-Azure copy).

        Returns the Azure blob path.
        """
        import io

        buf = io.BytesIO()
        self.download_to_stream(remote_path, buf)
        buf.seek(0)
        azure_client.upload_stream(azure_container, azure_blob_path, buf)
        logger.info("Transferred %s -> %s/%s", remote_path, azure_container, azure_blob_path)
        return azure_blob_path
