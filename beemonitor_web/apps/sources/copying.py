"""Copying clips from users' buckets into ours, in the background (memory/54).

No Celery in production, so the reconcile loop (apps/analysis/reconcile.py)
calls ``tick`` every couple of minutes. Each tick:

* puts back copies a restart interrupted (running for over an hour),
* claims up to ``PARALLEL`` waiting copies, one conditional UPDATE each, so a
  clip is copied once even with several workers ticking,
* runs each in its own thread: streamed bucket to bucket, then the clip is
  measured (duration, fps, frame size) and gets a thumbnail,
* removes copies made for a single analysis a day after they were made.

``ensure_readable`` is the synchronous path the GPU spawn takes for a clip kept
in the user's bucket: a temporary copy, recorded so the tick removes it.
"""

from __future__ import annotations

import logging
import threading
from datetime import timedelta

from django.db import close_old_connections
from django.utils import timezone

from .models import SourceCopy

logger = logging.getLogger(__name__)

PARALLEL = 2
STALE_AFTER = timedelta(hours=1)
TEMP_KEEP = timedelta(days=1)


def queue(video, purpose) -> SourceCopy:
    """Queue a copy (or a measure) of a clip, once: an unfinished one of the
    same purpose is returned instead of a second."""
    existing = SourceCopy.objects.filter(
        video=video, purpose=purpose,
        status__in=[SourceCopy.Status.QUEUED, SourceCopy.Status.RUNNING]).first()
    if existing:
        return existing
    return SourceCopy.objects.create(video=video, source=video.source, purpose=purpose,
                                     bytes_total=video.file_size_bytes or 0)


def _dest_key(video, tag=""):
    from . import s3
    _bucket, key = s3.parse_remote(video)
    name = key.rsplit("/", 1)[-1] or f"clip-{video.pk}.mp4"
    return f"{video.user_id}/{video.pk}/{tag + '/' if tag else ''}{name}"


def tick() -> dict:
    now = timezone.now()
    SourceCopy.objects.filter(status=SourceCopy.Status.RUNNING,
                              started_at__lt=now - STALE_AFTER).update(status=SourceCopy.Status.QUEUED)
    removed = _remove_temp_copies(now)
    running = SourceCopy.objects.filter(status=SourceCopy.Status.RUNNING).count()
    started = 0
    for pk in SourceCopy.objects.filter(status=SourceCopy.Status.QUEUED).values_list("pk", flat=True)[
            :max(PARALLEL - running, 0)]:
        claimed = SourceCopy.objects.filter(pk=pk, status=SourceCopy.Status.QUEUED).update(
            status=SourceCopy.Status.RUNNING, started_at=now, bytes_done=0, error="")
        if claimed:
            threading.Thread(target=_run_in_thread, args=(pk,), daemon=True).start()
            started += 1
    return {"copies_started": started, "temp_copies_removed": removed}


def _run_in_thread(pk):
    try:
        run(pk)
    finally:
        close_old_connections()


def run(pk) -> None:
    """Do one claimed copy or measure. Never raises: failures are recorded."""
    from apps.videos.models import Video

    from . import crypto, s3

    job = SourceCopy.objects.select_related("video", "source").get(pk=pk)
    video = job.video
    try:
        if job.purpose == SourceCopy.Purpose.KEEP and video.storage_key.startswith("s3://"):
            if job.source is None:
                raise s3.SourceError("The storage this clip came from was removed.")
            creds = crypto.decrypt(job.source)
            bucket, key = s3.parse_remote(video)
            dest = _dest_key(video)
            last = [0]

            def progress(done):
                # A write every ~2%, not every 8 MB part.
                if job.bytes_total and done - last[0] < job.bytes_total / 50:
                    return
                last[0] = done
                SourceCopy.objects.filter(pk=pk).update(bytes_done=done)

            size = s3.stream_copy(creds, bucket, key, dest, progress)
            meta = {**(video.metadata or {}), "remote_bucket": bucket, "remote_key": key,
                    "copied_at": timezone.now().isoformat()}
            Video.everything.filter(pk=video.pk).update(storage_key=dest, file_size_bytes=size, metadata=meta)
            SourceCopy.objects.filter(pk=pk).update(dest_key=dest, bytes_done=size, bytes_total=size)
            video.refresh_from_db()
        measure(video)
        SourceCopy.objects.filter(pk=pk).update(status=SourceCopy.Status.DONE, finished_at=timezone.now())
    except Exception as exc:
        message = s3.friendly(exc)
        logger.warning("source copy %s (video %s) failed: %s", pk, video.pk, exc)
        SourceCopy.objects.filter(pk=pk).update(status=SourceCopy.Status.FAILED, error=message[:300],
                                                finished_at=timezone.now())


def measure(video) -> None:
    """Fill the clip's duration, fps and frame size, and give it a thumbnail."""
    import io

    from apps.videos import frames
    from apps.videos.thumbnails import THUMBNAIL_WIDTH, SAMPLE_AT_SECONDS, _store_props, thumbnail_key
    from config.storage import get_s3_client

    props = frames.probe(video)
    if props:
        _store_props(video, props)
    if video.thumbnail_key:
        return
    still = frames.frame_jpeg(video, SAMPLE_AT_SECONDS, width=THUMBNAIL_WIDTH)
    if still:
        key = thumbnail_key(video.storage_key.removeprefix("s3://"))
        get_s3_client().upload_stream("processed", key, io.BytesIO(still), content_type="image/jpeg")
        type(video).everything.filter(pk=video.pk).update(thumbnail_key=key)


def ensure_readable(video, tag: str) -> str:
    """A key in our raw-videos bucket the GPU can read for this clip.

    A clip in our storage: its key. A clip kept in the user's bucket: a
    temporary copy for this run (``tag`` keeps runs apart), removed by the
    tick a day later; the clip itself stays in the user's bucket."""
    from . import crypto, s3

    if not (video.storage_key or "").startswith("s3://"):
        return video.storage_key
    if video.source is None:
        raise s3.SourceError("The storage this clip came from was removed, so it can't be analysed.")
    bucket, key = s3.parse_remote(video)
    dest = _dest_key(video, f"run-{tag}")
    row = SourceCopy.objects.create(video=video, source=video.source, purpose=SourceCopy.Purpose.TEMP,
                                    status=SourceCopy.Status.RUNNING, started_at=timezone.now(),
                                    bytes_total=video.file_size_bytes or 0, dest_key=dest)
    try:
        size = s3.stream_copy(crypto.decrypt(video.source), bucket, key, dest)
    except Exception as exc:
        SourceCopy.objects.filter(pk=row.pk).update(status=SourceCopy.Status.FAILED,
                                                    error=s3.friendly(exc)[:300], finished_at=timezone.now())
        raise
    SourceCopy.objects.filter(pk=row.pk).update(status=SourceCopy.Status.DONE, bytes_done=size,
                                                finished_at=timezone.now())
    return dest


def _remove_temp_copies(now) -> int:
    from config.storage import get_s3_client

    done = SourceCopy.objects.filter(purpose=SourceCopy.Purpose.TEMP, status=SourceCopy.Status.DONE,
                                     finished_at__lt=now - TEMP_KEEP).exclude(dest_key="")[:200]
    removed = 0
    for row in done:
        try:
            get_s3_client().delete_blob("raw-videos", row.dest_key)
        except Exception:
            logger.warning("could not remove temporary copy %s", row.dest_key)
            continue
        SourceCopy.objects.filter(pk=row.pk).update(status=SourceCopy.Status.REMOVED)
        removed += 1
    return removed
