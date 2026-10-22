"""A user's own S3 bucket, read with the user's own keys (memory/54).

Everything that touches a connected bucket goes through here, so two rules
hold everywhere:

* **Never our AWS role.** boto3 falls back to the instance role when keys are
  missing; a source with no keys would then list *our* buckets. ``client``
  refuses instead.
* **Plain errors.** What a user sees is "Access denied to that bucket", never
  a boto traceback (which can echo keys and ARNs). Details go to the log.
"""

from __future__ import annotations

import logging
import mimetypes

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".h264")
PAGE_SIZE = 200


class SourceError(Exception):
    """Something a user can fix (wrong key, no access); ``str()`` is safe to show."""


def remote_uri(bucket: str, key: str) -> str:
    """How a clip that lives in the user's bucket is stored: ``s3://bucket/key``."""
    return f"s3://{bucket}/{key}"


def parse_remote(video) -> tuple[str, str]:
    """(bucket, key) of a clip in a user's bucket. Older imports stored
    ``s3://key`` with the bucket only in metadata."""
    meta = video.metadata or {}
    if meta.get("remote_bucket") and meta.get("remote_key"):
        return meta["remote_bucket"], meta["remote_key"]
    rest = (video.storage_key or "").removeprefix("s3://")
    bucket, _, key = rest.partition("/")
    return bucket, key


def client(creds: dict):
    """An S3 client on the user's keys. Raises SourceError when they're missing."""
    import boto3
    from botocore.config import Config

    key_id, secret = creds.get("access_key_id"), creds.get("secret_access_key")
    if not (key_id and secret and creds.get("bucket")):
        raise SourceError("This source is missing its bucket, access key or secret. Connect it again.")
    return boto3.client("s3", aws_access_key_id=key_id, aws_secret_access_key=secret,
                        aws_session_token=None, region_name=creds.get("region") or "us-east-1",
                        config=Config(retries={"max_attempts": 3}, connect_timeout=10, read_timeout=60))


def friendly(exc: Exception) -> str:
    """A message for the user; the exception itself only goes to the log."""
    if isinstance(exc, SourceError):
        return str(exc)
    code = ""
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        code = str((response.get("Error") or {}).get("Code") or "")
    logger.info("S3 source error (%s): %s", code or type(exc).__name__, exc)
    if code in ("InvalidAccessKeyId", "SignatureDoesNotMatch", "InvalidToken"):
        return "The access key or secret isn't right."
    if code in ("AccessDenied", "403", "AllAccessDisabled"):
        return "Access denied to that bucket or folder. The key needs s3:ListBucket and s3:GetObject on it."
    if code in ("NoSuchBucket", "404", "NotFound"):
        return "There's no bucket by that name (or it's in another region)."
    if code in ("PermanentRedirect", "AuthorizationHeaderMalformed", "IllegalLocationConstraintException"):
        return "That bucket is in a different region. Check the region."
    return "Couldn't reach the bucket. Check the name, region and key."


def test(creds: dict) -> str:
    """Prove the key can read this bucket (and folder). Returns a success line."""
    try:
        c = client(creds)
        c.head_bucket(Bucket=creds["bucket"])
        c.list_objects_v2(Bucket=creds["bucket"], Prefix=creds.get("prefix") or "", MaxKeys=1)
    except Exception as exc:
        raise SourceError(friendly(exc)) from None
    folder = f"/{creds['prefix']}" if creds.get("prefix") else ""
    return f"Connected to s3://{creds['bucket']}{folder}."


def browse(creds: dict, path: str = "", token: str = "") -> dict:
    """One page of a folder: ``{folders, files, next_token, path}``. ``path``
    is relative to the source's prefix; only video files are listed."""
    base = creds.get("prefix") or ""
    prefix = base + path
    try:
        page = client(creds).list_objects_v2(
            Bucket=creds["bucket"], Prefix=prefix, Delimiter="/", MaxKeys=PAGE_SIZE,
            **({"ContinuationToken": token} if token else {}))
    except Exception as exc:
        raise SourceError(friendly(exc)) from None
    folders = [p["Prefix"][len(base):] for p in page.get("CommonPrefixes") or []]
    files = [{"key": o["Key"], "name": o["Key"].rsplit("/", 1)[-1], "size": o.get("Size", 0),
              "modified": o.get("LastModified")}
             for o in page.get("Contents") or []
             if o["Key"].lower().endswith(VIDEO_EXTENSIONS)]
    return {"folders": folders, "files": files, "path": path,
            "next_token": page.get("NextContinuationToken") or ""}


def list_videos(creds: dict, path: str = "", limit: int = 2000) -> list[dict]:
    """Every video under a folder (recursively), up to ``limit``."""
    base = creds.get("prefix") or ""
    out = []
    try:
        pages = client(creds).get_paginator("list_objects_v2").paginate(
            Bucket=creds["bucket"], Prefix=base + path)
        for page in pages:
            for o in page.get("Contents") or []:
                if o["Key"].lower().endswith(VIDEO_EXTENSIONS):
                    out.append({"key": o["Key"], "name": o["Key"].rsplit("/", 1)[-1], "size": o.get("Size", 0)})
                    if len(out) >= limit:
                        return out
    except Exception as exc:
        raise SourceError(friendly(exc)) from None
    return out


def head(creds: dict, key: str) -> int:
    """Size in bytes of one object (and proof it's readable)."""
    try:
        return int(client(creds).head_object(Bucket=creds["bucket"], Key=key)["ContentLength"])
    except Exception as exc:
        raise SourceError(friendly(exc)) from None


def presign(creds: dict, bucket: str, key: str, expires: int = 3600) -> str:
    """A short-lived GET URL on the user's keys (frames, probing, playback)."""
    return client({**creds, "bucket": bucket}).generate_presigned_url(
        "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=expires)


def content_type(name: str) -> str:
    return mimetypes.guess_type(name)[0] or "video/mp4"


def stream_copy(creds: dict, bucket: str, key: str, dest_key: str, progress=None) -> int:
    """Copy one object from the user's bucket into our raw-videos bucket,
    streamed (multipart, no local disk, any size). Returns the bytes copied."""
    from boto3.s3.transfer import TransferConfig

    from config.storage import get_s3_client

    ours = get_s3_client()
    try:
        body = client({**creds, "bucket": bucket}).get_object(Bucket=bucket, Key=key)["Body"]
    except Exception as exc:
        raise SourceError(friendly(exc)) from None
    done = [0]

    def _cb(n):
        done[0] += n
        if progress:
            progress(done[0])

    ours._client.upload_fileobj(
        Fileobj=body, Bucket=ours._bucket("raw-videos"), Key=ours._key(dest_key),
        ExtraArgs={"ContentType": content_type(key)},
        # Parts upload in this thread: the progress callback writes to the
        # database, and boto's worker threads would each open a connection.
        Config=TransferConfig(multipart_chunksize=64 * 1024 * 1024, use_threads=False),
        Callback=_cb)
    return done[0]
