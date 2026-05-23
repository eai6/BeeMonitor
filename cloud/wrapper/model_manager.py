"""Manage ML model files between S3 and local disk.

Downloads models on demand from the S3 ``models`` bucket and caches them.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from cloud.storage.config import S3Config
from cloud.storage.s3_client import S3StorageClient

logger = logging.getLogger(__name__)

MODEL_VERSION = "v1"

MODEL_FILES = {
    "nest_detection": "nest_detection.pt",
    "bee_tracking": "bee_tracking.pt",
    "event_classifier": "event_classifier_model.pkl",
}


@dataclass
class ModelPaths:
    """Resolved local paths for all required models."""

    nest_detection: str
    bee_tracking: str
    event_classifier: str


class ModelManager:
    """Download, cache, and resolve model paths from the S3 models bucket."""

    CONTAINER = "models"

    def __init__(
        self,
        storage_client: Optional[S3StorageClient] = None,
        storage_config: Optional[S3Config] = None,
        local_cache_dir: str = "/tmp/beemonitor_models",
        model_version: str = MODEL_VERSION,
    ):
        self._config = storage_config or S3Config()
        self._client = storage_client or S3StorageClient(self._config)
        self.cache_dir = Path(local_cache_dir)
        self.model_version = model_version

    def _remote_key(self, filename: str) -> str:
        return f"{self.model_version}/{filename}"

    def _local_path(self, filename: str) -> Path:
        return self.cache_dir / self.model_version / filename

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ensure_models(self) -> ModelPaths:
        """Download all models if not already cached. Returns resolved paths."""
        paths = {}
        for key, filename in MODEL_FILES.items():
            paths[key] = self._ensure_single(filename)
        return ModelPaths(**paths)

    def ensure_custom_model(self, remote_key: str) -> str:
        """Download a custom model from S3 if not cached. Returns local path."""
        local = self.cache_dir / "custom" / remote_key.replace("/", "_")
        if local.exists():
            logger.debug("Custom model cached: %s", local)
            return str(local)

        local.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading custom model %s/%s -> %s", self.CONTAINER, remote_key, local)
        self._client.download_file(self.CONTAINER, remote_key, str(local))
        return str(local)

    def _ensure_single(self, filename: str) -> str:
        """Download a single model file if not cached locally."""
        local = self._local_path(filename)
        if local.exists():
            logger.debug("Model cached locally: %s", local)
            return str(local)

        remote = self._remote_key(filename)
        logger.info("Downloading model %s/%s -> %s", self.CONTAINER, remote, local)
        self._client.download_file(self.CONTAINER, remote, str(local))
        return str(local)

    def upload_models(self, models_dir: str) -> dict[str, str]:
        """Upload local model files to the S3 models bucket.

        Returns a dict mapping model key to ``models/<key>`` storage path.
        """
        uploaded = {}
        src = Path(models_dir)
        for key, filename in MODEL_FILES.items():
            local_file = src / filename
            if not local_file.exists():
                logger.warning("Model file not found, skipping: %s", local_file)
                continue

            remote = self._remote_key(filename)
            self._client.upload_file(
                self.CONTAINER, remote, str(local_file), overwrite=True
            )
            uploaded[key] = f"{self.CONTAINER}/{remote}"
        return uploaded

    def verify_models(self, models_dir: str) -> dict[str, bool]:
        """Check that local models match what's in S3 (by size)."""
        results = {}
        src = Path(models_dir)
        for key, filename in MODEL_FILES.items():
            local_file = src / filename
            if not local_file.exists():
                results[key] = False
                continue

            remote = self._remote_key(filename)
            try:
                props = self._client.get_blob_properties(self.CONTAINER, remote)
                results[key] = local_file.stat().st_size == props["size"]
            except Exception:
                results[key] = False
        return results

    def clear_cache(self) -> None:
        """Remove all cached model files."""
        import shutil

        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            logger.info("Cleared model cache: %s", self.cache_dir)
