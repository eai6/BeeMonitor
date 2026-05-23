import logging

from celery import shared_task

from apps.sources.models import DataSource
from apps.videos.models import Video

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=3, default_retry_delay=60)
def ingest_from_source(
    self,
    video_id: int,
    source_id: int,
    remote_path: str,
) -> None:
    """
    Download a video file from an external data source and upload it to the
    raw-videos S3 bucket.

    Steps:
        1. Resolve the DataSource connector.
        2. Stream the file from the remote path.
        3. Upload to S3 (raw-videos bucket).
        4. Update the Video record with the storage key and metadata.
    """
    try:
        video = Video.objects.get(pk=video_id)
        source = DataSource.objects.get(pk=source_id)
    except (Video.DoesNotExist, DataSource.DoesNotExist) as exc:
        logger.error(
            "Video %s or DataSource %s not found; aborting.",
            video_id,
            source_id,
        )
        return

    from apps.sources.connectors import get_connector

    connector = get_connector(source)

    try:
        video.status = Video.Status.UPLOADING
        video.save(update_fields=["status"])

        # Download from remote source to a temp file
        local_tmp_path = connector.download(remote_path)

        import os
        from config.storage import get_s3_client

        blob_path = (
            f"videos/{video.user_id}/{video.id}/{remote_path.split('/')[-1]}"
        )
        get_s3_client().upload_file(
            "raw-videos", blob_path, local_tmp_path, content_type="video/mp4",
        )

        file_size = (
            os.path.getsize(local_tmp_path)
            if os.path.exists(local_tmp_path)
            else 0
        )

        video.storage_key = blob_path
        video.file_size_bytes = file_size
        video.status = Video.Status.READY
        video.save(
            update_fields=["storage_key", "file_size_bytes", "status"]
        )

        logger.info(
            "Video %s ingested from source %s -> %s",
            video_id,
            source_id,
            blob_path,
        )

    except Exception as exc:
        video.status = Video.Status.ARCHIVED
        video.save(update_fields=["status"])
        logger.exception(
            "Failed to ingest video %s from source %s.", video_id, source_id
        )
        raise self.retry(exc=exc)
