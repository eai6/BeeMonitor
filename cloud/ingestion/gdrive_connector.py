"""Google Drive connector for video ingestion."""

import logging
import urllib.request
from pathlib import Path
from typing import BinaryIO

from cloud.ingestion.base_connector import BaseConnector, RemoteFile

logger = logging.getLogger(__name__)

GDRIVE_API = "https://www.googleapis.com/drive/v3"


class GoogleDriveConnector(BaseConnector):
    """Connect to Google Drive via OAuth2 token."""

    source_type = "google_drive"

    def __init__(self, oauth_token: str, folder_id: str = "root"):
        self._token = oauth_token
        self.folder_id = folder_id

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._token}"}

    def _api_get(self, url: str) -> dict:
        import json

        req = urllib.request.Request(url, headers=self._headers())
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())

    def test_connection(self) -> tuple[bool, str]:
        try:
            data = self._api_get(f"{GDRIVE_API}/about?fields=user")
            email = data.get("user", {}).get("emailAddress", "unknown")
            return True, f"Connected as {email}"
        except Exception as e:
            return False, f"Google Drive connection failed: {e}"

    def list_files(self, prefix: str = "", limit: int = 100) -> list[RemoteFile]:
        query = f"'{self.folder_id}' in parents and mimeType contains 'video'"
        if prefix:
            query += f" and name contains '{prefix}'"
        url = (
            f"{GDRIVE_API}/files?q={urllib.request.quote(query)}"
            f"&fields=files(id,name,size,mimeType,modifiedTime)"
            f"&pageSize={limit}"
        )
        data = self._api_get(url)
        files = []
        for f in data.get("files", []):
            files.append(
                RemoteFile(
                    name=f["name"],
                    path=f["id"],
                    size=int(f.get("size", 0)),
                    content_type=f.get("mimeType", ""),
                    last_modified=f.get("modifiedTime", ""),
                )
            )
        return files

    def download_to_file(self, remote_path: str, local_path: str) -> str:
        """remote_path is a Google Drive file ID."""
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        url = f"{GDRIVE_API}/files/{remote_path}?alt=media"
        req = urllib.request.Request(url, headers=self._headers())
        with urllib.request.urlopen(req) as resp, open(local_path, "wb") as fh:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                fh.write(chunk)
        logger.info("Downloaded gdrive:%s -> %s", remote_path, local_path)
        return local_path

    def download_to_stream(self, remote_path: str, stream: BinaryIO) -> int:
        url = f"{GDRIVE_API}/files/{remote_path}?alt=media"
        req = urllib.request.Request(url, headers=self._headers())
        total = 0
        with urllib.request.urlopen(req) as resp:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                stream.write(chunk)
                total += len(chunk)
        return total
