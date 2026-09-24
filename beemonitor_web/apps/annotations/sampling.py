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


# Motion sampling ("Most activity"): the defaults a newly added clip gets.
MOTION_DEFAULT_FRAMES = 20
MIN_GAP_CHOICES = (0.5, 1.0, 2.0, 5.0)
# A frame counts as active when at least this share of the ROI is moving
# foreground after noise cleanup — about one bee at the scoring resolution.
MOTION_MIN_FRACTION = float(os.environ.get("BEEMONITOR_SAMPLE_MOTION_MIN", "0.0015"))
MOTION_SCORE_WIDTH = 320          # frames are scored at this width (greyscale)
MOTION_WARMUP_SECONDS = 1.0       # background model settles before scoring
PROFILE_BUCKETS = 60              # the clip row's motion strip


def clamp_params(params):
    """Normalise user-supplied sampling knobs into safe bounds.

    ``method`` is "motion" (the most active frames, spaced apart) or "interval"
    (every Nth frame). A params dict without ``method`` is from before motion
    sampling existed and means "interval".
    """
    def _clamp(name, default, lo, hi):
        try:
            return max(lo, min(hi, int(params.get(name, default))))
        except (TypeError, ValueError):
            return default

    method = "motion" if params.get("method") == "motion" else "interval"
    out = {
        "method": method,
        "sample_interval": _clamp("sample_interval", 30, MIN_INTERVAL, MAX_INTERVAL),
        "max_frames": _clamp("max_frames",
                             MOTION_DEFAULT_FRAMES if method == "motion" else 100,
                             MIN_FRAMES, MAX_FRAMES),
    }
    if method == "motion":
        try:
            gap = float(params.get("min_gap_s", 1.0))
        except (TypeError, ValueError):
            gap = 1.0
        out["min_gap_s"] = min(MIN_GAP_CHOICES, key=lambda c: abs(c - gap))
        out["roi"] = "frame" if params.get("roi") == "frame" else "device"
        out["replace"] = str(params.get("replace", "1")).lower() not in ("0", "false", "off", "")
    return out


def motion_params(**overrides):
    """The "Most activity" defaults (what a newly added clip is sampled with)."""
    return clamp_params({"method": "motion", **overrides})


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
        if params["method"] == "motion":
            return _sample_by_motion(task, params, tmp.name, blob_path, s3)
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


def _roi_for(task, params):
    """(box, polygon) in normalized coords to score motion inside, or (None, None)
    for the whole frame. The layout in use when the clip was recorded."""
    if params.get("roi") != "device" or task.video.device_id is None:
        return None, None
    from apps.devices.layouts import layout_for_video
    layout = layout_for_video(task.video)
    return layout["roi_override"], layout["roi_polygon"]


def score_motion(path, roi=None, polygon=None):
    """Per-frame motion for a clip: the share of the ROI that is moving.

    Greyscale at MOTION_SCORE_WIDTH, cropped to ``roi`` (normalized
    [x1, y1, x2, y2]) and masked to ``polygon``; MOG2 background subtraction
    (as the device's recorder) plus a morphological open to drop sensor noise
    and leaf flicker. Frames inside the warm-up read 0. Returns
    ``(scores, fps)``.
    """
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Could not open the video for decoding.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    warmup = int(round(fps * MOTION_WARMUP_SECONDS))
    subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=16,
                                                    detectShadows=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    scores, crop, mask, area = [], None, None, 1
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            small_h = max(1, int(h * MOTION_SCORE_WIDTH / w))
            gray = cv2.cvtColor(cv2.resize(frame, (MOTION_SCORE_WIDTH, small_h),
                                           interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            if crop is None:
                x1, y1, x2, y2 = (roi or [0, 0, 1, 1])
                cx1, cy1 = int(x1 * MOTION_SCORE_WIDTH), int(y1 * small_h)
                cx2 = max(cx1 + 1, int(x2 * MOTION_SCORE_WIDTH))
                cy2 = max(cy1 + 1, int(y2 * small_h))
                crop = (cx1, cy1, cx2, cy2)
                if polygon:
                    mask = np.zeros((cy2 - cy1, cx2 - cx1), np.uint8)
                    pts = np.array([[int(px * MOTION_SCORE_WIDTH) - cx1, int(py * small_h) - cy1]
                                    for px, py in polygon], np.int32)
                    cv2.fillPoly(mask, [pts], 255)
                    area = max(1, int(np.count_nonzero(mask)))
                else:
                    area = (cy2 - cy1) * (cx2 - cx1)
            region = gray[crop[1]:crop[3], crop[0]:crop[2]]
            fg = subtractor.apply(region)
            if len(scores) < warmup:
                scores.append(0.0)
                continue
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
            if mask is not None:
                fg = cv2.bitwise_and(fg, mask)
            scores.append(float(np.count_nonzero(fg)) / area)
    finally:
        cap.release()
    return scores, fps


def pick_active_frames(scores, count, min_gap_frames, min_fraction=None):
    """The ``count`` busiest frames, each at least ``min_gap_frames`` from any
    other pick, ignoring frames below ``min_fraction``. Sorted by frame."""
    floor = MOTION_MIN_FRACTION if min_fraction is None else min_fraction
    picks = []
    for i in sorted(range(len(scores)), key=lambda k: scores[k], reverse=True):
        if len(picks) >= count or scores[i] < floor:
            break
        if all(abs(i - p) >= min_gap_frames for p in picks):
            picks.append(i)
    return sorted(picks)


def motion_profile(scores, picks, buckets=PROFILE_BUCKETS):
    """A compact strip for the clip row: peak motion per bucket (0-100) and
    which buckets hold a picked frame."""
    n = len(scores)
    if not n:
        return {"profile": [], "picked": [], "frames": 0}
    size = max(1, -(-n // buckets))
    peaks = [max(scores[i:i + size]) for i in range(0, n, size)]
    top = max(peaks) or 1.0
    return {"profile": [round(v / top * 100) for v in peaks],
            "picked": sorted({p // size for p in picks}),
            "frames": n}


def _clear_unlabelled(task):
    """Drop this clip's earlier sampled frames that nobody has touched: no
    boxes, never reviewed. The frame images stay in storage (the key is per
    video, shared across projects)."""
    from .models import Annotation
    return Annotation.objects.filter(
        project=task.project, video=task.video, sampled_only=True,
        reviewed=False, boxes=[]).delete()[0]


def _sample_by_motion(task, params, path, blob_path, s3):
    """Score every frame, keep the most active, write only those."""
    import io

    import cv2

    from .models import Annotation, FrameSamplingTask

    roi, polygon = _roi_for(task, params)
    scores, fps = score_motion(path, roi, polygon)
    picks = pick_active_frames(scores, params["max_frames"],
                               max(1, int(round(params["min_gap_s"] * fps))))
    FrameSamplingTask.objects.filter(pk=task.pk).update(
        motion=motion_profile(scores, picks))

    if params.get("replace"):
        _clear_unlabelled(task)
    if not picks:
        return 0

    wanted = set(picks)
    written = 0
    cap = cv2.VideoCapture(path)
    try:
        index, last = 0, picks[-1]
        while index <= last:
            ok, frame = cap.read()
            if not ok:
                break
            if index in wanted:
                height, width = frame.shape[:2]
                ok_enc, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok_enc:
                    key = frame_key(blob_path, index)
                    s3.upload_stream("processed", key, io.BytesIO(buf.tobytes()),
                                     content_type="image/jpeg")
                    _record_frame(Annotation, task, index, key, width, height)
                    written += 1
            index += 1
    finally:
        cap.release()
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


def _not_gpu_task():
    """Rows the web process may sample: anything but a GPU ``sample_label`` task.

    Spelled out because ``exclude(params__method=...)`` also drops rows with no
    ``method`` key at all (NULL comparisons), which is every older task.
    """
    from django.db.models import Q
    return Q(params__method__isnull=True) | ~Q(params__method="sample_label")


def run_sampling_task(task_pk):
    """Run one task to completion, recording status. Never raises."""
    from .models import FrameSamplingTask

    try:
        # Claim: only move QUEUED → PROCESSING, so a cancel mid-flight wins.
        # GPU sample_label tasks are never run in the web process.
        claimed = FrameSamplingTask.objects.filter(
            pk=task_pk, status=FrameSamplingTask.Status.QUEUED,
        ).filter(_not_gpu_task()).update(status=FrameSamplingTask.Status.PROCESSING,
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
            # GPU tasks are sent by sampling_remote.dispatch, never run here.
            .filter(_not_gpu_task())
            .order_by("created_at")[:limit]
        )
        for task in stale:
            spawn_sampling_async(task.pk)
            started += 1
    except Exception:
        logger.exception("poll_frame_sampling_tasks failed")
    return started
