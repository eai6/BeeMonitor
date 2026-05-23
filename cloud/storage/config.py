"""Storage configuration loaded from environment variables."""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class S3Config:
    """AWS S3 configuration.

    Each logical area (raw videos, processed outputs, models, user configs)
    is its own bucket. Bucket names come exclusively from environment
    variables — never hardcoded — so the same code runs in dev, prod, and
    test without knowing the account it targets.

    Credentials follow the standard boto3 chain: on AWS, the ECS/EC2 task
    role; locally, ``AWS_PROFILE=ecomorph`` (SSO) or an env-var key pair.
    """

    region: str = field(
        default_factory=lambda: os.environ.get("AWS_REGION", "us-east-1")
    )
    endpoint_url: Optional[str] = field(
        default_factory=lambda: os.environ.get("AWS_S3_ENDPOINT_URL") or None
    )

    raw_videos_bucket: str = field(
        default_factory=lambda: os.environ.get("AWS_S3_BUCKET_RAW_VIDEOS", "")
    )
    processed_bucket: str = field(
        default_factory=lambda: os.environ.get("AWS_S3_BUCKET_PROCESSED", "")
    )
    models_bucket: str = field(
        default_factory=lambda: os.environ.get("AWS_S3_BUCKET_MODELS", "")
    )
    user_configs_bucket: str = field(
        default_factory=lambda: os.environ.get("AWS_S3_BUCKET_USER_CONFIGS", "")
    )

    presigned_expiry_hours: int = 24

    @property
    def bucket_for_container(self) -> dict[str, str]:
        """Map the legacy ``container`` name to its actual bucket."""
        return {
            "raw-videos": self.raw_videos_bucket,
            "processed": self.processed_bucket,
            "models": self.models_bucket,
            "user-configs": self.user_configs_bucket,
        }

    @property
    def all_buckets(self) -> list[str]:
        return [
            self.raw_videos_bucket,
            self.processed_bucket,
            self.models_bucket,
            self.user_configs_bucket,
        ]

    def validate(self) -> None:
        missing = [
            env
            for env, val in (
                ("AWS_S3_BUCKET_RAW_VIDEOS", self.raw_videos_bucket),
                ("AWS_S3_BUCKET_PROCESSED", self.processed_bucket),
                ("AWS_S3_BUCKET_MODELS", self.models_bucket),
                ("AWS_S3_BUCKET_USER_CONFIGS", self.user_configs_bucket),
            )
            if not val
        ]
        if missing:
            raise ValueError(
                f"S3Config is missing required env vars: {', '.join(missing)}"
            )
