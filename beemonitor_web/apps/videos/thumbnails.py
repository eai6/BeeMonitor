"""One sampled frame per clip, so the review grid is scannable at rest.

A video file carries no image, so a grid of clips has nothing to show until
something opens each one. Playing 200 clips to find the one with a bee in it is
the problem the grid exists to solve, so we extract a still once, at upload.

**Not frame 0.** The recorder keeps a rolling pre-roll buffer and writes it in
front of every motion clip (``PRE_ROLL = 3.0`` in ``hardware/motion/config.py``,
applied via ``CircularOutput`` in ``recorder.py``), so a clip always begins
three seconds BEFORE anything moved. Frame 0 is reliably an empty hotel — 331
thumbnails of nothing, a grid exactly as uniform as no thumbnails at all. We
sample at the pre-roll mark instead: the moment motion was detected, and the
frame most likely to contain the bee. Continuous-mode clips have no pre-roll,
so there it is simply an early frame — no special case needed.
"""

import io
import logging
import os
import tempfile
import threading

logger = logging.getLogger(__name__)

# Seconds into the clip to sample. Mirrors the device's BEEMONITOR_PRE_ROLL.
SAMPLE_AT_SECONDS = float(os.environ.get("BEEMONITOR_THUMBNAIL_AT", "3.0"))

# Long edge in px. Cards render ~270px wide; 2x covers retina without paying
# for full frames 200 at a time.
THUMBNAIL_WIDTH = 560
JPEG_QUALITY = 78

# Clips uploaded before stills existed have none, and a backfill needs someone
# to run it — so the grid extracts what it is asked for, on demand, and keeps
# the result. A page fills in as you scroll and is instant forever after.
#
# Bounded, because a screen of 20 lazy <img>s would otherwise start 20
# simultaneous downloads and decodes on a web dyno. Requests that cannot get a
# slot promptly give up rather than queue: the card stays dark and the next
# scroll past it tries again.
ON_DEMAND = os.environ.get("BEEMONITOR_THUMBNAIL_ON_DEMAND", "1") == "1"
ON_DEMAND_SLOTS = int(os.environ.get("BEEMONITOR_THUMBNAIL_SLOTS", "3"))
ON_DEMAND_WAIT_SECONDS = float(os.environ.get("BEEMONITOR_THUMBNAIL_WAIT", "2.0"))

_slots = threading.BoundedSemaphore(ON_DEMAND_SLOTS)


def _needs_probe(video) -> bool:
    """True when we still don't know this clip's own basic properties.

    Frame rate and duration are read from the container, so "we don't know how
    long the video is" is a gap we can always close — not a limitation.
    """
    return not (getattr(video, "fps", None) and getattr(video, "duration_seconds", None))


def probe_on_demand(video) -> None:
    """Fill in a clip's measured properties in the background, if a slot is free.

    Pages that want a clip's length call this when it is missing; the value
    appears on a later load rather than blocking this one. Best-effort by
    design — one S3 read, bounded by the same semaphore as thumbnails, and a
    clip whose file cannot be read simply stays unknown.
    """
    import threading

    if not _needs_probe(video) or not (video.storage_key or ""):
        return

    def _run():
        if not _slots.acquire(timeout=0):
            return                       # busy: a later page load will retry
        try:
            extract_thumbnail(video, force=False)
        except Exception:
            logger.exception("probe: failed for video %s", video.pk)
        finally:
            _slots.release()

    threading.Thread(target=_run, daemon=True).start()


def extract_on_demand(video) -> str:
    """Extract now if a slot is free, else return "" and let the card retry."""
    if not ON_DEMAND or (video.thumbnail_key and not _needs_probe(video)):
        return video.thumbnail_key
    if not _slots.acquire(timeout=ON_DEMAND_WAIT_SECONDS):
        logger.info("thumbnail: busy, deferring video %s", video.pk)
        return ""
    try:
        return extract_thumbnail(video)
    finally:
        _slots.release()


def thumbnail_key(blob_path: str) -> str:
    """Where a clip's still lives in the processed bucket."""
    return f"thumbs/{blob_path.replace('/', '_')}.jpg"


def extract_thumbnail(video, *, force: bool = False) -> str:
    """Sample one frame from ``video`` and store it. Returns the key, or "".

    Never raises: a clip without a usable still is a cosmetic loss, not a
    failed upload. Callers may fire this inline or from a worker.
    """
    # A clip may already have its still but predate the property probe, so the
    # early return is conditional on BOTH being done — otherwise "we have a
    # thumbnail" silently means "we will never learn this clip's length".
    if video.thumbnail_key and not force and not _needs_probe(video):
        return video.thumbnail_key

    blob_path = video.storage_key or ""
    if not blob_path or blob_path.startswith("s3://"):
        # Not in our bucket yet (external-source ingest happens later).
        return ""

    import cv2

    from config.storage import get_s3_client

    s3 = get_s3_client()
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    try:
        s3.download_file("raw-videos", blob_path, tmp.name)
        frame, props = _grab_frame(cv2, tmp.name)
        # Persist first: a clip whose frame rate we now know is worth recording
        # even when no frame renders, since durations depend on it and the
        # still does not.
        _store_props(video, props)
        if frame is None:
            logger.info("thumbnail: no decodable frame in video %s", video.pk)
            return ""

        if video.thumbnail_key and not force:
            return video.thumbnail_key   # probed above; the still is already stored

        frame = _downscale(cv2, frame)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            return ""

        key = thumbnail_key(blob_path)
        s3.upload_stream("processed", key, io.BytesIO(buf.tobytes()),
                         content_type="image/jpeg")
        video.thumbnail_key = key
        video.save(update_fields=["thumbnail_key"])
        return key
    except Exception:
        logger.exception("thumbnail: extraction failed for video %s", video.pk)
        return ""
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def _grab_frame(cv2, path: str):
    """``(frame, props)`` — the still, plus what the container says about itself.

    The frame is the one at SAMPLE_AT_SECONDS, falling back toward the start: a
    clip shorter than the pre-roll (or one whose seek fails) still gets a still
    via the midpoint, then the very first frame, rather than nothing.

    ``props`` carries ``fps``, ``duration_seconds``, ``width`` and ``height``,
    omitting any the container did not report. These come free — the capture is
    already open and the frame rate is already read to place the sample — and
    they are the only measurement of a clip's real frame rate the system takes.
    Without them every frame→seconds conversion falls back to an assumption.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, {}
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        props = {}
        if fps > 0:
            props["fps"] = round(float(fps), 3)
            if total > 0:
                props["duration_seconds"] = round(float(total) / float(fps), 2)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width > 0 and height > 0:
            props["width"], props["height"] = width, height

        targets = []
        if fps > 0:
            targets.append(int(SAMPLE_AT_SECONDS * fps))
        if total > 0:
            targets.append(int(total // 2))
        targets.append(0)

        for target in targets:
            if total and target >= total:
                continue
            if target > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ok, frame = cap.read()
            if ok and frame is not None:
                return frame, props
        return None, props
    finally:
        cap.release()


def _store_props(video, props: dict) -> None:
    """Write measured container properties onto the row, filling blanks only.

    An existing value is left alone: the field may have been corrected by hand,
    and a probe is not authoritative enough to overwrite that.
    """
    fields = [f for f, value in props.items()
              if value and not getattr(video, f, None) and hasattr(video, f)]
    if not fields:
        return
    for f in fields:
        setattr(video, f, props[f])
    video.save(update_fields=fields)
    logger.info("probe: video %s recorded %s", video.pk,
                ", ".join(f"{f}={props[f]}" for f in fields))


def _downscale(cv2, frame):
    height, width = frame.shape[:2]
    if width <= THUMBNAIL_WIDTH:
        return frame
    scale = THUMBNAIL_WIDTH / float(width)
    return cv2.resize(frame, (THUMBNAIL_WIDTH, max(1, int(height * scale))),
                      interpolation=cv2.INTER_AREA)


def queue_thumbnail(video) -> None:
    """Extract in the background, so an upload never waits on a decode.

    Best-effort by design: the clip is already stored and listed by the time
    this runs, and a missing still degrades the grid rather than the data.
    """
    import threading

    def _run():
        try:
            extract_thumbnail(video)
        except Exception:  # pragma: no cover - extract_thumbnail already guards
            logger.exception("thumbnail: background extraction failed for %s", video.pk)

    threading.Thread(target=_run, name=f"thumb-{video.pk}", daemon=True).start()
