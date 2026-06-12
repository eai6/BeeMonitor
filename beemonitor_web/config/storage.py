"""Project-level storage helpers.

Single entry point for getting an ``S3StorageClient`` from inside Django code.
View / task / management-command code should call ``get_s3_client()`` rather
than importing boto3 or building ``S3Config`` themselves — so future changes
to credential / region handling live here only.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from django.conf import settings

from cloud.storage.config import S3Config
from cloud.storage.s3_client import S3StorageClient

logger = logging.getLogger(__name__)


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


def presigned_get(storage_key: str, container: str = "raw-videos",
                  expiry_hours: int = 1) -> "str | None":
    """Short-lived presigned GET URL for a stored blob, or None on failure.

    The single place the browser-facing presign contract lives — shared by the
    device pages and the activity pages so it can't drift between them.
    """
    if not storage_key:
        return None
    try:
        return get_s3_client().generate_presigned_url(
            container, storage_key, expiry_hours=expiry_hours)
    except Exception as e:  # pragma: no cover - an S3 hiccup must not 500 the page
        logger.warning("presign failed for %s: %s", storage_key, e)
        return None
