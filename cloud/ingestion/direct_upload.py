"""Direct chunked upload handler for videos sent via REST API.

Manages multi-part upload sessions: init, receive chunks, finalize.
"""

import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from cloud.storage.azure_client import AzureBlobClient
from cloud.storage.config import StorageConfig

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 5 * 1024 * 1024  # 5 MB


@dataclass
class UploadSession:
    """Tracks state for an in-progress chunked upload."""

    session_id: str
    user_id: str
    filename: str
    total_size: int
    total_chunks: int
    received_chunks: set = field(default_factory=set)
    local_path: str = ""


class DirectUploadHandler:
    """Handle chunked video uploads to Azure Blob Storage.

    Flow:
        1. Client calls init_upload() → gets session_id
        2. Client sends chunks via upload_chunk()
        3. Client calls finalize_upload() → file assembled and uploaded to Azure
    """

    def __init__(
        self,
        storage_client: Optional[AzureBlobClient] = None,
        storage_config: Optional[StorageConfig] = None,
        upload_dir: str = "/tmp/beemonitor_uploads",
    ):
        self._config = storage_config or StorageConfig()
        self._storage = storage_client or AzureBlobClient(self._config)
        self._upload_dir = Path(upload_dir)
        self._sessions: dict[str, UploadSession] = {}

    def init_upload(
        self,
        user_id: str,
        filename: str,
        total_size: int,
        total_chunks: int,
    ) -> dict:
        """Initialize an upload session.

        Returns:
            Dict with session_id and upload instructions.
        """
        session_id = str(uuid.uuid4())[:12]
        session_dir = self._upload_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        session = UploadSession(
            session_id=session_id,
            user_id=user_id,
            filename=filename,
            total_size=total_size,
            total_chunks=total_chunks,
            local_path=str(session_dir / filename),
        )
        self._sessions[session_id] = session

        logger.info(
            "Upload session %s: %s (%d bytes, %d chunks)",
            session_id, filename, total_size, total_chunks,
        )
        return {
            "session_id": session_id,
            "chunk_size": DEFAULT_CHUNK_SIZE,
            "total_chunks": total_chunks,
        }

    def upload_chunk(
        self, session_id: str, chunk_index: int, data: bytes
    ) -> dict:
        """Receive a single chunk for an upload session."""
        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"Unknown session: {session_id}")

        chunk_path = self._upload_dir / session_id / f"chunk_{chunk_index:06d}"
        chunk_path.write_bytes(data)
        session.received_chunks.add(chunk_index)

        return {
            "session_id": session_id,
            "chunk_index": chunk_index,
            "received": len(session.received_chunks),
            "total": session.total_chunks,
            "complete": len(session.received_chunks) == session.total_chunks,
        }

    def finalize_upload(self, session_id: str, upload_id: str | None = None) -> dict:
        """Assemble chunks and upload the complete file to Azure.

        Returns:
            Dict with azure_blob_path and file_size.
        """
        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"Unknown session: {session_id}")

        if len(session.received_chunks) != session.total_chunks:
            missing = set(range(session.total_chunks)) - session.received_chunks
            raise ValueError(f"Missing chunks: {sorted(missing)}")

        # Assemble chunks into final file
        final_path = Path(session.local_path)
        with open(final_path, "wb") as out:
            for i in range(session.total_chunks):
                chunk_path = self._upload_dir / session_id / f"chunk_{i:06d}"
                out.write(chunk_path.read_bytes())

        file_size = final_path.stat().st_size
        if upload_id is None:
            upload_id = session_id

        # Upload to Azure
        blob_path = f"{session.user_id}/{upload_id}/{session.filename}"
        self._storage.upload_file(
            self._config.raw_videos_container,
            blob_path,
            str(final_path),
            content_type="video/mp4",
        )

        # Cleanup
        self._cleanup_session(session_id)

        return {
            "upload_id": upload_id,
            "azure_blob_path": blob_path,
            "file_size": file_size,
        }

    def get_session_status(self, session_id: str) -> dict | None:
        session = self._sessions.get(session_id)
        if session is None:
            return None
        return {
            "session_id": session.session_id,
            "filename": session.filename,
            "received": len(session.received_chunks),
            "total": session.total_chunks,
            "complete": len(session.received_chunks) == session.total_chunks,
        }

    def _cleanup_session(self, session_id: str) -> None:
        import shutil

        session_dir = self._upload_dir / session_id
        if session_dir.exists():
            shutil.rmtree(session_dir)
        self._sessions.pop(session_id, None)
