"""CPU function for ingesting videos from external cloud sources into Azure.

Handles streaming downloads from AWS S3, Azure Blob, GCS, Google Drive,
and direct uploads — all stored into the raw-videos container.
"""

import logging
import os
import tempfile
import uuid
from pathlib import Path

import modal

from cloud.modal_app.app import app
from cloud.modal_app.secrets import azure_secret

logger = logging.getLogger(__name__)


@app.function(
    timeout=3600,  # 1 hour max
    secrets=[azure_secret],
    memory=4096,  # 4 GiB
)
def ingest_video(
    user_id: str,
    source_type: str,
    source_config: dict,
    upload_id: str | None = None,
) -> dict:
    """Download a video from an external source and store in Azure Blob.

    Args:
        user_id: Owner user ID.
        source_type: One of 'aws_s3', 'azure_blob', 'gcs', 'google_drive', 'direct'.
        source_config: Source-specific configuration dict.
            aws_s3: {bucket, key, access_key_id?, secret_access_key?}
            azure_blob: {container, blob_path, connection_string?}
            gcs: {bucket, key, service_account_json?}
            google_drive: {file_id, oauth_token}
            direct: {local_path} (for testing)
        upload_id: Optional upload ID (generated if not provided).

    Returns:
        Dict with upload_id, azure_blob_path, file_size.
    """
    from cloud.storage.azure_client import AzureBlobClient
    from cloud.storage.config import StorageConfig

    if upload_id is None:
        upload_id = str(uuid.uuid4())[:12]

    config = StorageConfig()
    storage = AzureBlobClient(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Download from source
        local_path = _download_from_source(source_type, source_config, tmpdir)
        filename = Path(local_path).name
        file_size = Path(local_path).stat().st_size

        # Upload to Azure raw-videos container
        blob_path = f"{user_id}/{upload_id}/{filename}"
        storage.upload_file(
            config.raw_videos_container,
            blob_path,
            local_path,
            content_type="video/mp4",
        )

    logger.info(
        "Ingested %s -> %s/%s (%d bytes)",
        source_type, config.raw_videos_container, blob_path, file_size,
    )

    return {
        "upload_id": upload_id,
        "azure_blob_path": blob_path,
        "file_size": file_size,
        "container": config.raw_videos_container,
    }


def _download_from_source(source_type: str, source_config: dict, tmpdir: str) -> str:
    """Download a video file from the specified source to a temp directory."""
    if source_type == "aws_s3":
        return _download_s3(source_config, tmpdir)
    elif source_type == "azure_blob":
        return _download_azure(source_config, tmpdir)
    elif source_type == "gcs":
        return _download_gcs(source_config, tmpdir)
    elif source_type == "google_drive":
        return _download_gdrive(source_config, tmpdir)
    elif source_type == "direct":
        return source_config["local_path"]
    else:
        raise ValueError(f"Unsupported source type: {source_type}")


def _download_s3(config: dict, tmpdir: str) -> str:
    import boto3

    session_kwargs = {}
    if config.get("access_key_id"):
        session_kwargs["aws_access_key_id"] = config["access_key_id"]
        session_kwargs["aws_secret_access_key"] = config["secret_access_key"]

    s3 = boto3.client("s3", **session_kwargs)
    key = config["key"]
    filename = Path(key).name
    local_path = str(Path(tmpdir) / filename)
    s3.download_file(config["bucket"], key, local_path)
    return local_path


def _download_azure(config: dict, tmpdir: str) -> str:
    from azure.storage.blob import BlobServiceClient

    conn = config.get("connection_string", os.environ.get("AZURE_STORAGE_CONNECTION_STRING", ""))
    service = BlobServiceClient.from_connection_string(conn)
    blob = service.get_blob_client(config["container"], config["blob_path"])
    filename = Path(config["blob_path"]).name
    local_path = str(Path(tmpdir) / filename)
    with open(local_path, "wb") as fh:
        stream = blob.download_blob()
        stream.readinto(fh)
    return local_path


def _download_gcs(config: dict, tmpdir: str) -> str:
    from google.cloud import storage as gcs_storage

    if config.get("service_account_json"):
        client = gcs_storage.Client.from_service_account_info(
            config["service_account_json"]
        )
    else:
        client = gcs_storage.Client()

    bucket = client.bucket(config["bucket"])
    blob = bucket.blob(config["key"])
    filename = Path(config["key"]).name
    local_path = str(Path(tmpdir) / filename)
    blob.download_to_filename(local_path)
    return local_path


def _download_gdrive(config: dict, tmpdir: str) -> str:
    import urllib.request

    file_id = config["file_id"]
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
    headers = {"Authorization": f"Bearer {config['oauth_token']}"}

    req = urllib.request.Request(url, headers=headers)
    local_path = str(Path(tmpdir) / f"{file_id}.mp4")
    with urllib.request.urlopen(req) as resp, open(local_path, "wb") as fh:
        while True:
            chunk = resp.read(8192)
            if not chunk:
                break
            fh.write(chunk)
    return local_path
