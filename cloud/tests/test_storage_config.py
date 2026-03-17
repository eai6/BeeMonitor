"""Tests for cloud.storage.config."""

import os
import pytest

from cloud.storage.config import StorageConfig


class TestStorageConfig:
    def test_defaults(self):
        config = StorageConfig()
        assert config.raw_videos_container == "raw-videos"
        assert config.processed_container == "processed"
        assert config.models_container == "models"
        assert config.user_configs_container == "user-configs"
        assert config.sas_expiry_hours == 24

    def test_all_containers(self):
        config = StorageConfig()
        assert len(config.all_containers) == 4
        assert "raw-videos" in config.all_containers

    def test_validate_raises_without_credentials(self):
        config = StorageConfig(connection_string="", account_name="", account_key="")
        with pytest.raises(ValueError, match="AZURE_STORAGE_CONNECTION_STRING"):
            config.validate()

    def test_validate_passes_with_connection_string(self):
        config = StorageConfig(connection_string="DefaultEndpointsProtocol=https;AccountName=test;AccountKey=key;EndpointSuffix=core.windows.net")
        config.validate()  # Should not raise

    def test_validate_passes_with_account_name_and_key(self):
        config = StorageConfig(account_name="myaccount", account_key="mykey")
        config.validate()  # Should not raise

    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("AZURE_STORAGE_CONNECTION_STRING", "test_conn_string")
        config = StorageConfig()
        assert config.connection_string == "test_conn_string"
