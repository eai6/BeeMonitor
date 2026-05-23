"""Project-level storage helpers.

Single entry point for getting an ``S3StorageClient`` from inside Django code.
View / task / management-command code should call ``get_s3_client()`` rather
than importing boto3 or building ``S3Config`` themselves — so future changes
to credential / region handling live here only.
"""

from __future__ import annotations

from functools import lru_cache

from django.conf import settings

from cloud.storage.config import S3Config
from cloud.storage.s3_client import S3StorageClient


def _build_config() -> S3Config:
    """Construct S3Config from Django settings, not raw env."""
    return S3Config(
        region=settings.AWS_REGION,
        raw_videos_bucket=settings.AWS_S3_BUCKET_RAW_VIDEOS,
        processed_bucket=settings.AWS_S3_BUCKET_PROCESSED,
        models_bucket=settings.AWS_S3_BUCKET_MODELS,
        user_configs_bucket=settings.AWS_S3_BUCKET_USER_CONFIGS,
    )


@lru_cache(maxsize=1)
def get_s3_client() -> S3StorageClient:
    """Process-wide singleton S3 client.

    Cached: boto3 client construction does a slow chain-of-credentials walk
    on first call, and there's no per-request state to keep separate.
    """
    return S3StorageClient(_build_config())
