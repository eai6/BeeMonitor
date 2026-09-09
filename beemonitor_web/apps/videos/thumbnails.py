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


def extract_on_demand(video) -> str:
    """Extract now if a slot is free, else return "" and let the card retry."""
    if not ON_DEMAND or video.thumbnail_key:
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
    if video.thumbnail_key and not force:
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
        frame = _grab_frame(cv2, tmp.name)
        if frame is None:
            logger.info("thumbnail: no decodable frame in video %s", video.pk)
            return ""

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
    """The frame at SAMPLE_AT_SECONDS, falling back toward the start.

    A clip shorter than the pre-roll (or one whose seek fails) still gets a
    still: we fall back to the midpoint, then to the very first frame, rather
    than returning nothing.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
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
                return frame
        return None
    finally:
        cap.release()


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
