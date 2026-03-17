"""Storage configuration loaded from environment variables."""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StorageConfig:
    """Azure Blob Storage configuration.

    All values can be overridden via environment variables.
    """

    connection_string: str = field(
        default_factory=lambda: os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "")
    )
    account_name: str = field(
        default_factory=lambda: os.environ.get("AZURE_STORAGE_ACCOUNT_NAME", "")
    )
    account_key: str = field(
        default_factory=lambda: os.environ.get("AZURE_STORAGE_ACCOUNT_KEY", "")
    )

    # Container names
    raw_videos_container: str = "raw-videos"
    processed_container: str = "processed"
    models_container: str = "models"
    user_configs_container: str = "user-configs"

    # Lifecycle
    hot_tier_days: int = 30
    cool_tier_days: int = 90

    # SAS token defaults
    sas_expiry_hours: int = 24

    @property
    def all_containers(self) -> list[str]:
        return [
            self.raw_videos_container,
            self.processed_container,
            self.models_container,
            self.user_configs_container,
        ]

    def validate(self) -> None:
        """Raise if connection info is missing."""
        if not self.connection_string and not (self.account_name and self.account_key):
            raise ValueError(
                "Set AZURE_STORAGE_CONNECTION_STRING or both "
                "AZURE_STORAGE_ACCOUNT_NAME and AZURE_STORAGE_ACCOUNT_KEY"
            )
