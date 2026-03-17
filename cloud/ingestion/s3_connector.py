"""AWS S3 connector for ingesting videos into BeeMonitor Cloud."""

import logging
from pathlib import Path
from typing import BinaryIO, Optional

from cloud.ingestion.base_connector import BaseConnector, RemoteFile

logger = logging.getLogger(__name__)


class S3Connector(BaseConnector):
    """Connect to an AWS S3 bucket."""

    source_type = "aws_s3"

    def __init__(
        self,
        bucket: str,
        access_key_id: str = "",
        secret_access_key: str = "",
        region: str = "us-east-1",
        _boto3_client=None,
    ):
        self.bucket = bucket
        self._access_key = access_key_id
        self._secret_key = secret_access_key
        self._region = region
        self._client = _boto3_client

    def _get_client(self):
        if self._client is None:
            import boto3

            kwargs = {"region_name": self._region}
            if self._access_key:
                kwargs["aws_access_key_id"] = self._access_key
                kwargs["aws_secret_access_key"] = self._secret_key
            self._client = boto3.client("s3", **kwargs)
        return self._client

    def test_connection(self) -> tuple[bool, str]:
        try:
            client = self._get_client()
            client.head_bucket(Bucket=self.bucket)
            return True, f"Connected to s3://{self.bucket}"
        except Exception as e:
            return False, f"S3 connection failed: {e}"

    def list_files(self, prefix: str = "", limit: int = 100) -> list[RemoteFile]:
        client = self._get_client()
        kwargs = {"Bucket": self.bucket, "MaxKeys": limit}
        if prefix:
            kwargs["Prefix"] = prefix

        resp = client.list_objects_v2(**kwargs)
        files = []
        for obj in resp.get("Contents", []):
            files.append(
                RemoteFile(
                    name=Path(obj["Key"]).name,
                    path=obj["Key"],
                    size=obj["Size"],
                    last_modified=obj["LastModified"].isoformat() if obj.get("LastModified") else "",
                )
            )
        return files

    def download_to_file(self, remote_path: str, local_path: str) -> str:
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        client = self._get_client()
        client.download_file(self.bucket, remote_path, local_path)
        logger.info("Downloaded s3://%s/%s -> %s", self.bucket, remote_path, local_path)
        return local_path

    def download_to_stream(self, remote_path: str, stream: BinaryIO) -> int:
        client = self._get_client()
        resp = client.get_object(Bucket=self.bucket, Key=remote_path)
        body = resp["Body"]
        total = 0
        for chunk in body.iter_chunks(chunk_size=8192):
            stream.write(chunk)
            total += len(chunk)
        return total
