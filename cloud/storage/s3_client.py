"""S3 storage client for BeeMonitor.

Each logical area (raw-videos, processed, models, user-configs) lives in its
own bucket. The argument pair ``(container, blob_path)`` resolves to
``s3://<bucket-for-container>/<blob_path>`` — the container name is the
routing key, the blob path is the S3 key verbatim.

Credentials follow the standard boto3 chain: on AWS, the ECS/EC2 task role;
locally, the ``ecomorph`` SSO profile (export ``AWS_PROFILE=ecomorph``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import BinaryIO, Optional

import boto3
from botocore.client import Config as BotoConfig
from botocore.exceptions import ClientError

from cloud.storage.config import S3Config

logger = logging.getLogger(__name__)


class S3StorageClient:
    """boto3-backed storage client (bucket per logical container)."""

    def __init__(self, config: Optional[S3Config] = None):
        self.config = config or S3Config()
        self.config.validate()

        self._client = boto3.client(
            "s3",
            region_name=self.config.region,
            endpoint_url=self.config.endpoint_url,
            # Required for presigned URLs against buckets in non-us-east-1.
            config=BotoConfig(signature_version="s3v4"),
        )

    # ------------------------------------------------------------------
    # Routing helpers
    # ------------------------------------------------------------------

    def _bucket(self, container: str) -> str:
        try:
            return self.config.bucket_for_container[container]
        except KeyError as exc:
            raise ValueError(
                f"Unknown container {container!r}; expected one of "
                f"{list(self.config.bucket_for_container)}"
            ) from exc

    @staticmethod
    def _key(blob_path: str) -> str:
        # Strip any leading slash so callers can pass "a/b" or "/a/b".
        return blob_path.lstrip("/")

    # ------------------------------------------------------------------
    # Container helpers — S3 buckets are pre-provisioned via Pulumi/CLI
    # ------------------------------------------------------------------

    def create_container(self, name: str) -> None:
        """No-op — buckets are provisioned out-of-band, not by app code."""
        logger.debug("S3: create_container(%s) is a no-op", name)

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------

    def upload_file(
        self,
        container: str,
        blob_path: str,
        local_path: str,
        overwrite: bool = True,
        content_type: Optional[str] = None,
    ) -> str:
        bucket = self._bucket(container)
        key = self._key(blob_path)
        if not overwrite and self.blob_exists(container, blob_path):
            raise FileExistsError(f"s3://{bucket}/{key} already exists")

        extra: dict = {}
        if content_type:
            extra["ContentType"] = content_type

        self._client.upload_file(
            Filename=local_path,
            Bucket=bucket,
            Key=key,
            ExtraArgs=extra or None,
        )
        logger.info("Uploaded %s -> s3://%s/%s", local_path, bucket, key)
        return f"{container}/{blob_path}"

    def upload_stream(
        self,
        container: str,
        blob_path: str,
        stream: BinaryIO,
        overwrite: bool = True,
        content_type: Optional[str] = None,
    ) -> str:
        bucket = self._bucket(container)
        key = self._key(blob_path)
        if not overwrite and self.blob_exists(container, blob_path):
            raise FileExistsError(f"s3://{bucket}/{key} already exists")

        extra: dict = {}
        if content_type:
            extra["ContentType"] = content_type

        self._client.upload_fileobj(
            Fileobj=stream,
            Bucket=bucket,
            Key=key,
            ExtraArgs=extra or None,
        )
        logger.info("Uploaded stream -> s3://%s/%s", bucket, key)
        return f"{container}/{blob_path}"

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download_file(self, container: str, blob_path: str, local_path: str) -> str:
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        bucket = self._bucket(container)
        key = self._key(blob_path)
        self._client.download_file(
            Bucket=bucket,
            Key=key,
            Filename=local_path,
        )
        logger.info("Downloaded s3://%s/%s -> %s", bucket, key, local_path)
        return local_path

    def download_to_stream(
        self, container: str, blob_path: str, stream: BinaryIO
    ) -> None:
        bucket = self._bucket(container)
        key = self._key(blob_path)
        self._client.download_fileobj(
            Bucket=bucket,
            Key=key,
            Fileobj=stream,
        )

    # ------------------------------------------------------------------
    # Presigned URLs
    # ------------------------------------------------------------------

    def generate_presigned_url(
        self,
        container: str,
        blob_path: str,
        expiry_hours: Optional[int] = None,
        permissions: str = "r",
    ) -> str:
        if expiry_hours is None:
            expiry_hours = self.config.presigned_expiry_hours
        expires_in = int(expiry_hours * 3600)

        bucket = self._bucket(container)
        key = self._key(blob_path)
        client_method = "put_object" if "w" in permissions.lower() else "get_object"

        return self._client.generate_presigned_url(
            ClientMethod=client_method,
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=expires_in,
        )

    # ------------------------------------------------------------------
    # Metadata / list / delete
    # ------------------------------------------------------------------

    def list_blobs(self, container: str, prefix: str = "") -> list[str]:
        bucket = self._bucket(container)
        paginator = self._client.get_paginator("list_objects_v2")
        out: list[str] = []
        for page in paginator.paginate(Bucket=bucket, Prefix=self._key(prefix)):
            for obj in page.get("Contents", []) or []:
                out.append(obj["Key"])
        return out

    def blob_exists(self, container: str, blob_path: str) -> bool:
        bucket = self._bucket(container)
        key = self._key(blob_path)
        try:
            self._client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code")
            if code in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise

    def get_blob_properties(self, container: str, blob_path: str) -> dict:
        bucket = self._bucket(container)
        key = self._key(blob_path)
        head = self._client.head_object(Bucket=bucket, Key=key)
        last_modified = head.get("LastModified")
        return {
            "name": key,
            "size": head.get("ContentLength"),
            "content_type": head.get("ContentType"),
            "last_modified": last_modified.isoformat() if last_modified else None,
            "etag": head.get("ETag", "").strip('"'),
        }

    def delete_blob(self, container: str, blob_path: str) -> None:
        bucket = self._bucket(container)
        key = self._key(blob_path)
        self._client.delete_object(Bucket=bucket, Key=key)
        logger.info("Deleted s3://%s/%s", bucket, key)

    def delete_prefix(self, container: str, prefix: str) -> int:
        keys = self.list_blobs(container, prefix=prefix)
        if not keys:
            return 0
        bucket = self._bucket(container)
        deleted = 0
        for i in range(0, len(keys), 1000):  # delete_objects max 1000/call
            batch = keys[i:i + 1000]
            self._client.delete_objects(
                Bucket=bucket,
                Delete={
                    "Objects": [{"Key": k} for k in batch],
                    "Quiet": True,
                },
            )
            deleted += len(batch)
        logger.info(
            "Deleted %d objects under s3://%s/%s", deleted, bucket, prefix,
        )
        return deleted
