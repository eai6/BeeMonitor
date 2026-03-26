"""
Cloud Client for BeeMonitor Desktop
=====================================
Handles all communication with the BeeMonitor cloud REST API,
including authentication (login/register) and analysis operations.
"""

import json
from pathlib import Path
from typing import Callable, Optional

import requests

CLOUD_CONFIG_DIR = Path.home() / ".beemonitor"
CLOUD_CONFIG_FILE = CLOUD_CONFIG_DIR / "cloud_config.json"

DEFAULT_API_URL = "https://beemonitor-web-app.azurewebsites.net"


def load_cloud_config() -> dict:
    """Load cloud config from ~/.beemonitor/cloud_config.json."""
    if CLOUD_CONFIG_FILE.exists():
        try:
            return json.loads(CLOUD_CONFIG_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {"api_url": DEFAULT_API_URL, "api_key": "", "username": "", "tier": ""}


def save_cloud_config(
    api_url: str,
    api_key: str,
    username: str = "",
    email: str = "",
    tier: str = "",
) -> None:
    """Persist cloud config to ~/.beemonitor/cloud_config.json."""
    CLOUD_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CLOUD_CONFIG_FILE.write_text(
        json.dumps(
            {
                "api_url": api_url.rstrip("/"),
                "api_key": api_key,
                "username": username,
                "email": email,
                "tier": tier,
            },
            indent=2,
        )
    )


def clear_cloud_config() -> None:
    """Remove saved credentials (logout)."""
    if CLOUD_CONFIG_FILE.exists():
        CLOUD_CONFIG_FILE.unlink()


class BeeMonitorCloudClient:
    """REST API client for BeeMonitor cloud backend."""

    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {api_key}"

    # -- Auth ------------------------------------------------------------------

    @staticmethod
    def login(api_url: str, username: str, password: str) -> dict:
        """Authenticate with username/password and get an API key.

        Returns dict with: api_key, username, email, tier, monthly_credits, used_credits.
        Raises requests.HTTPError on failure.
        """
        url = f"{api_url.rstrip('/')}/api/v1/auth/login/"
        resp = requests.post(url, json={"username": username, "password": password}, timeout=15)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def register(api_url: str, username: str, email: str, password: str) -> dict:
        """Create a new account and get an API key.

        Returns same dict as login().
        Raises requests.HTTPError on failure.
        """
        url = f"{api_url.rstrip('/')}/api/v1/auth/register/"
        resp = requests.post(
            url,
            json={"username": username, "email": email, "password": password},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()

    # -- Profile ---------------------------------------------------------------

    def get_profile(self) -> dict:
        """Fetch the authenticated user's profile."""
        resp = self.session.get(f"{self.api_url}/api/v1/auth/profile/", timeout=10)
        resp.raise_for_status()
        return resp.json()

    # -- Connection test -------------------------------------------------------

    def test_connection(self) -> tuple[bool, str]:
        """Test API connectivity and authentication."""
        try:
            resp = self.session.get(f"{self.api_url}/api/v1/videos/", timeout=10)
            if resp.status_code == 200:
                return True, "Connected"
            if resp.status_code == 401:
                return False, "Authentication failed - check API key"
            return False, f"Server returned {resp.status_code}"
        except requests.ConnectionError:
            return False, "Cannot reach server"
        except requests.Timeout:
            return False, "Connection timed out"
        except Exception as e:
            return False, str(e)

    # -- Upload ----------------------------------------------------------------

    def upload_video(
        self,
        file_path: str,
        site_name: str = "",
        on_progress: Optional[Callable[[int], None]] = None,
    ) -> dict:
        """Upload a video file to the cloud.

        Returns the Video dict from the API (id, title, azure_blob_path, ...).
        """
        path = Path(file_path)
        file_size = path.stat().st_size

        class _ProgressFile:
            """Wrapper that reports read progress."""

            def __init__(self, fp, total, callback):
                self._fp = fp
                self._total = total
                self._cb = callback
                self._read = 0

            def read(self, size=-1):
                data = self._fp.read(size)
                self._read += len(data)
                if self._cb and self._total > 0:
                    self._cb(min(100, int(100 * self._read / self._total)))
                return data

            def __len__(self):
                return self._total

        with open(file_path, "rb") as f:
            wrapped = _ProgressFile(f, file_size, on_progress) if on_progress else f
            files = {"video_file": (path.name, wrapped, "video/mp4")}
            data = {}
            if site_name:
                data["site_name"] = site_name

            resp = self.session.post(
                f"{self.api_url}/api/v1/videos/upload_file/",
                files=files,
                data=data,
                timeout=1800,  # 30 min for very large files
            )

        resp.raise_for_status()
        return resp.json()

    # -- Job lifecycle ---------------------------------------------------------

    def create_and_submit_job(self, video_id: int, config: Optional[dict] = None) -> dict:
        """Create a job for the video and immediately submit it for processing."""
        payload = {"video_id": video_id, "config": config or {}}
        resp = self.session.post(f"{self.api_url}/api/v1/jobs/", json=payload, timeout=30)
        resp.raise_for_status()
        job = resp.json()

        job_id = job["id"]
        resp = self.session.post(
            f"{self.api_url}/api/v1/jobs/{job_id}/submit/",
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def batch_submit(self, video_ids: list[int], config: Optional[dict] = None) -> dict:
        """Submit multiple videos as a batch for chunked GPU processing.

        Returns dict with 'jobs' list, 'submitted' count, 'skipped' count.
        """
        payload = {"video_ids": video_ids, "config": config or {}}
        resp = self.session.post(
            f"{self.api_url}/api/v1/jobs/batch_submit/",
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()

    def poll_job(self, job_id: int) -> dict:
        """Get current job status and result."""
        resp = self.session.get(f"{self.api_url}/api/v1/jobs/{job_id}/", timeout=30)
        resp.raise_for_status()
        return resp.json()

    # -- Result download -------------------------------------------------------

    def get_download_url(self, job_id: int, file_field: str) -> Optional[str]:
        """Get a temporary SAS URL for a result file."""
        resp = self.session.get(
            f"{self.api_url}/api/v1/jobs/{job_id}/download/",
            params={"file": file_field},
            timeout=30,
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json().get("url")

    def download_file(
        self,
        url: str,
        local_path: str,
        on_progress: Optional[Callable[[int], None]] = None,
    ) -> None:
        """Download a file from a URL with progress reporting."""
        resp = requests.get(url, stream=True, timeout=1800)
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0))
        downloaded = 0

        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
                downloaded += len(chunk)
                if on_progress and total > 0:
                    on_progress(min(100, int(100 * downloaded / total)))
