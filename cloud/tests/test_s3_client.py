"""Tests for S3StorageClient using moto's in-process S3 mock."""

import io

import pytest

boto3 = pytest.importorskip("boto3")
moto = pytest.importorskip("moto")
from moto import mock_aws

from cloud.storage.config import S3Config
from cloud.storage.s3_client import S3StorageClient


REGION = "us-east-1"
BUCKETS = {
    "raw-videos": "bm-test-raw-videos",
    "processed": "bm-test-processed",
    "models": "bm-test-models",
    "user-configs": "bm-test-user-configs",
}


@pytest.fixture
def client():
    with mock_aws():
        s3 = boto3.client("s3", region_name=REGION)
        for name in BUCKETS.values():
            s3.create_bucket(Bucket=name)
        yield S3StorageClient(
            S3Config(
                region=REGION,
                raw_videos_bucket=BUCKETS["raw-videos"],
                processed_bucket=BUCKETS["processed"],
                models_bucket=BUCKETS["models"],
                user_configs_bucket=BUCKETS["user-configs"],
            )
        )


class TestS3ConfigValidation:
    def test_init_fails_when_buckets_unset(self, monkeypatch):
        for var in (
            "AWS_S3_BUCKET_RAW_VIDEOS",
            "AWS_S3_BUCKET_PROCESSED",
            "AWS_S3_BUCKET_MODELS",
            "AWS_S3_BUCKET_USER_CONFIGS",
        ):
            monkeypatch.delenv(var, raising=False)
        with pytest.raises(ValueError, match="missing required env vars"):
            S3StorageClient(S3Config())

    def test_unknown_container_raises(self, client):
        with pytest.raises(ValueError, match="Unknown container"):
            client.blob_exists("not-a-container", "x")


class TestS3StorageClientOperations:
    def test_create_container_is_noop(self, client):
        client.create_container("raw-videos")  # should not raise

    def test_upload_file_and_exists(self, client, tmp_path):
        f = tmp_path / "v.mp4"
        f.write_bytes(b"hello world")

        result = client.upload_file("raw-videos", "user1/v.mp4", str(f))
        assert result == "raw-videos/user1/v.mp4"
        assert client.blob_exists("raw-videos", "user1/v.mp4") is True
        assert client.blob_exists("raw-videos", "missing.mp4") is False
        # And not in a different container's bucket.
        assert client.blob_exists("processed", "user1/v.mp4") is False

    def test_upload_stream_sets_content_type(self, client):
        client.upload_stream(
            "raw-videos", "user2/v.mp4", io.BytesIO(b"streamed"),
            content_type="video/mp4",
        )
        props = client.get_blob_properties("raw-videos", "user2/v.mp4")
        assert props["size"] == len(b"streamed")
        assert props["content_type"] == "video/mp4"

    def test_download_file(self, client, tmp_path):
        src = tmp_path / "src.mp4"
        src.write_bytes(b"some video bytes")
        client.upload_file("raw-videos", "u/src.mp4", str(src))

        dst = tmp_path / "out" / "downloaded.mp4"
        client.download_file("raw-videos", "u/src.mp4", str(dst))
        assert dst.read_bytes() == b"some video bytes"

    def test_download_to_stream(self, client):
        client.upload_stream("processed", "j1/events.csv", io.BytesIO(b"a,b,c\n1,2,3"))
        buf = io.BytesIO()
        client.download_to_stream("processed", "j1/events.csv", buf)
        assert buf.getvalue() == b"a,b,c\n1,2,3"

    def test_list_blobs_scopes_to_container(self, client):
        client.upload_stream("raw-videos", "u/a.mp4", io.BytesIO(b"a"))
        client.upload_stream("raw-videos", "u/b.mp4", io.BytesIO(b"b"))
        client.upload_stream("processed", "u/c.csv", io.BytesIO(b"c"))

        assert sorted(client.list_blobs("raw-videos")) == ["u/a.mp4", "u/b.mp4"]
        assert client.list_blobs("processed") == ["u/c.csv"]
        assert sorted(client.list_blobs("raw-videos", prefix="u/")) == [
            "u/a.mp4", "u/b.mp4",
        ]
        assert client.list_blobs("raw-videos", prefix="nothing/") == []

    def test_delete_blob(self, client):
        client.upload_stream("raw-videos", "to/delete.mp4", io.BytesIO(b"x"))
        assert client.blob_exists("raw-videos", "to/delete.mp4")
        client.delete_blob("raw-videos", "to/delete.mp4")
        assert not client.blob_exists("raw-videos", "to/delete.mp4")

    def test_delete_prefix(self, client):
        for i in range(5):
            client.upload_stream(
                "raw-videos", f"batch/v{i}.mp4", io.BytesIO(f"x{i}".encode())
            )
        client.upload_stream("raw-videos", "keep.mp4", io.BytesIO(b"keep"))

        deleted = client.delete_prefix("raw-videos", "batch/")
        assert deleted == 5
        assert client.list_blobs("raw-videos") == ["keep.mp4"]

    def test_presigned_url_for_read(self, client):
        client.upload_stream("raw-videos", "u/v.mp4", io.BytesIO(b"data"))
        url = client.generate_presigned_url("raw-videos", "u/v.mp4")
        assert BUCKETS["raw-videos"] in url
        assert "u/v.mp4" in url
        assert "X-Amz-Signature" in url
        assert "X-Amz-Expires" in url

    def test_presigned_url_for_write(self, client):
        url = client.generate_presigned_url(
            "raw-videos", "u/new.mp4", permissions="w"
        )
        assert BUCKETS["raw-videos"] in url
        assert "u/new.mp4" in url
        assert "X-Amz-Signature" in url
