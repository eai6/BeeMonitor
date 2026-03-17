"""One-time Azure Blob container setup and lifecycle policies."""

import logging

from cloud.storage.azure_client import AzureBlobClient
from cloud.storage.config import StorageConfig

logger = logging.getLogger(__name__)


def setup_containers(config: StorageConfig | None = None) -> None:
    """Create all required containers in Azure Blob Storage."""
    client = AzureBlobClient(config)
    client.create_all_containers()

    existing = client.list_containers()
    for name in (config or StorageConfig()).all_containers:
        status = "OK" if name in existing else "MISSING"
        logger.info("  %s: %s", name, status)

    logger.info("Container setup complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    setup_containers()
