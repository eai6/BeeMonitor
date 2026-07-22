"""Frame sampling — decoupled from SAM 3 auto-labelling.

Getting frames into the annotation editor used to require a SAM 3 pre-annotation
run: sampling was a payload key (``sample_interval``/``max_frames``) consumed
*inside* the GPU handler, so every re-sample paid for a g5 pass over the whole
video whether or not the user wanted machine labels. That is the waste this
module removes.

Decoding is plain OpenCV on the web container's CPU (already installed there for
the editor's frame fallback), so sampling costs nothing but a little wall-clock.
Frames are written to the **same** processed-bucket key convention the GPU worker
uses — ``frames/{video_blob_path with / → _}/f{frame:06d}.jpg`` — so
``FrameImageView`` serves them with no change, and a later SAM 3 run over the
same frames overwrites rather than duplicates.

Each sampled frame also gets an empty ``Annotation`` row, which is what makes it
navigable: the editor builds its prev/next list from ``Annotation``. Those rows
carry ``sampled_only=True`` so training can tell "nobody has looked at this yet"
apart from "a human marked this frame empty" — the latter is a real negative
example and must keep counting.
"""

import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

from django.db import connection
from django.utils import timezone

logger = logging.getLogger(__name__)

# Decode is CPU-bound and shares the web container with request handling, so keep
# this narrow. Mirrors the bounded pool used for pre-annotation spawns.
_SAMPLE_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="sample-frames")

# Clamps — a runaway interval/count would pin a worker for minutes.
MIN_INTERVAL, MAX_INTERVAL = 1, 600
MIN_FRAMES, MAX_FRAMES = 1, 2000


def clamp_params(params):
    """Normalise user-supplied sampling knobs into safe bounds."""
    def _clamp(name, default, lo, hi):
        try:
            return max(lo, min(hi, int(params.get(name, default))))
        except (TypeError, ValueError):
            return default

    return {
        "sample_interval": _clamp("sample_interval", 30, MIN_INTERVAL, MAX_INTERVAL),
        "max_frames": _clamp("max_frames", 100, MIN_FRAMES, MAX_FRAMES),
    }


def frame_key(video_blob_path, frame_number):
    """The processed-bucket key for one extracted frame.

    Must stay identical to the GPU worker's convention (see
    ``sagemaker_backend/inference.py``) or the editor's image view won't find it.
    """
    return f"frames/{video_blob_path.replace('/', '_')}/f{frame_number:06d}.jpg"


def sample_frames_for_task(task):
    """Decode + upload this task's frames, writing an Annotation row per frame.

    Returns the number of frames written. Raises on unrecoverable errors; the
    caller records them on the task.
    """
    import io

    import cv2

    from config.storage import get_s3_client

    from .models import Annotation

    video = task.video
    blob_path = video.storage_key or ""
    if not blob_path:
        raise ValueError("This video has no stored file to sample.")
    if blob_path.startswith("s3://"):
        raise ValueError(
            "This video still lives in an external bucket. Open it in the editor "
            "once to ingest it, then sample."
        )

    params = clamp_params(task.params or {})
    interval = params["sample_interval"]
    max_frames = params["max_frames"]

    s3 = get_s3_client()
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    written = 0
    try:
        s3.download_file("raw-videos", blob_path, tmp.name)
        cap = cv2.VideoCapture(tmp.name)
        if not cap.isOpened():
            raise ValueError("Could not open the video for decoding.")
        try:
            # Sequential decode, keeping every Nth frame. Much faster than
            # seeking with CAP_PROP_POS_FRAMES per target, which re-seeks to the
            # preceding keyframe every time.
            frame_index = 0
            while written < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_index % interval == 0:
                    height, width = frame.shape[:2]
                    ok_enc, buf = cv2.imencode(
                        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ok_enc:
                        key = frame_key(blob_path, frame_index)
                        s3.upload_stream("processed", key, io.BytesIO(buf.tobytes()),
                                         content_type="image/jpeg")
                        _record_frame(Annotation, task, frame_index, key, width, height)
                        written += 1
                frame_index += 1
        finally:
            cap.release()
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
    return written


def _record_frame(Annotation, task, frame_number, key, width, height):
    """Upsert the navigable placeholder row for one sampled frame.

    Never clobbers existing work: a frame that already carries boxes (from a SAM 3
    pass or a human) keeps them and only gains the image path.
    """
    existing = Annotation.objects.filter(
        project=task.project, video=task.video, frame_number=frame_number,
    ).first()
    if existing:
        existing.frame_image_path = key
        existing.image_width = width
        existing.image_height = height
        existing.save(update_fields=["frame_image_path", "image_width", "image_height"])
        return
    Annotation.objects.create(
        project=task.project, video=task.video, frame_number=frame_number,
        boxes=[], frame_image_path=key,
        image_width=width, image_height=height,
        sampled_only=True,
    )


def run_sampling_task(task_pk):
    """Run one task to completion, recording status. Never raises."""
    from .models import FrameSamplingTask

    try:
        # Claim: only move QUEUED → PROCESSING, so a cancel mid-flight wins.
        claimed = FrameSamplingTask.objects.filter(
            pk=task_pk, status=FrameSamplingTask.Status.QUEUED,
        ).update(status=FrameSamplingTask.Status.PROCESSING,
                 started_at=timezone.now())
        if not claimed:
            return 0

        task = FrameSamplingTask.objects.select_related("video", "project").get(pk=task_pk)
        written = sample_frames_for_task(task)

        # Re-check: the user may have cancelled while we decoded.
        FrameSamplingTask.objects.filter(
            pk=task_pk, status=FrameSamplingTask.Status.PROCESSING,
        ).update(status=FrameSamplingTask.Status.COMPLETED,
                 frames_written=written, completed_at=timezone.now())
        logger.info("sampled %d frame(s) for task %s", written, task_pk)
        return written
    except Exception as exc:
        logger.exception("frame sampling task %s failed", task_pk)
        try:
            FrameSamplingTask.objects.filter(pk=task_pk).update(
                status=FrameSamplingTask.Status.FAILED,
                error_message=str(exc)[:500], completed_at=timezone.now(),
            )
        except Exception:
            logger.exception("could not record sampling failure for %s", task_pk)
        return 0
    finally:
        connection.close()


def spawn_sampling_async(task_pk):
    """Hand a queued task to the bounded decode pool."""
    _SAMPLE_POOL.submit(run_sampling_task, task_pk)


def poll_frame_sampling_tasks(limit=10):
    """Reconciler hook: pick up tasks that were never spawned.

    The pool lives in the web process, so a deploy mid-decode leaves rows stuck.
    Re-queueing PROCESSING rows is safe because the work is idempotent (same S3
    keys, upserted Annotation rows).
    """
    from .models import FrameSamplingTask

    started = 0
    try:
        stale = list(
            FrameSamplingTask.objects
            .filter(status=FrameSamplingTask.Status.QUEUED)
            .order_by("created_at")[:limit]
        )
        for task in stale:
            spawn_sampling_async(task.pk)
            started += 1
    except Exception:
        logger.exception("poll_frame_sampling_tasks failed")
    return started
