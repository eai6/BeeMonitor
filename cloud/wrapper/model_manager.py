"""Manage ML model files between Azure Blob Storage and local disk.

Downloads models on demand, caches them locally, and verifies checksums.
"""

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from cloud.storage.azure_client import AzureBlobClient
from cloud.storage.config import StorageConfig

logger = logging.getLogger(__name__)

# Default model version prefix in Azure
MODEL_VERSION = "v1"

# Expected model filenames
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
    """Download, cache, and resolve model paths from Azure Blob Storage."""

    def __init__(
        self,
        storage_client: Optional[AzureBlobClient] = None,
        storage_config: Optional[StorageConfig] = None,
        local_cache_dir: str = "/tmp/beemonitor_models",
        model_version: str = MODEL_VERSION,
    ):
        self._client = storage_client or AzureBlobClient(storage_config)
        self._config = storage_config or StorageConfig()
        self.cache_dir = Path(local_cache_dir)
        self.model_version = model_version

    @property
    def _container(self) -> str:
        return self._config.models_container

    def _azure_path(self, filename: str) -> str:
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

    def ensure_custom_model(self, azure_blob_path: str) -> str:
        """Download a custom model from Azure if not cached. Returns local path."""
        local = self.cache_dir / "custom" / azure_blob_path.replace("/", "_")
        if local.exists():
            logger.debug("Custom model cached: %s", local)
            return str(local)

        local.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading custom model %s/%s -> %s", self._container, azure_blob_path, local)
        self._client.download_file(self._container, azure_blob_path, str(local))
        return str(local)

    def _ensure_single(self, filename: str) -> str:
        """Download a single model file if not cached locally."""
        local = self._local_path(filename)
        if local.exists():
            logger.debug("Model cached locally: %s", local)
            return str(local)

        azure_path = self._azure_path(filename)
        logger.info("Downloading model %s/%s -> %s", self._container, azure_path, local)
        self._client.download_file(self._container, azure_path, str(local))
        return str(local)

    def upload_models(self, models_dir: str) -> dict[str, str]:
        """Upload local model files to Azure Blob Storage.

        Args:
            models_dir: Directory containing model files.

        Returns:
            Dict mapping model key to azure blob path.
        """
        uploaded = {}
        src = Path(models_dir)
        for key, filename in MODEL_FILES.items():
            local_file = src / filename
            if not local_file.exists():
                logger.warning("Model file not found, skipping: %s", local_file)
                continue

            azure_path = self._azure_path(filename)
            self._client.upload_file(
                self._container, azure_path, str(local_file), overwrite=True
            )
            uploaded[key] = f"{self._container}/{azure_path}"
        return uploaded

    def verify_models(self, models_dir: str) -> dict[str, bool]:
        """Check that local models match what's in Azure (by size)."""
        results = {}
        src = Path(models_dir)
        for key, filename in MODEL_FILES.items():
            local_file = src / filename
            if not local_file.exists():
                results[key] = False
                continue

            azure_path = self._azure_path(filename)
            try:
                props = self._client.get_blob_properties(self._container, azure_path)
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
