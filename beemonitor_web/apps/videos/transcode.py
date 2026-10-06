"""Uploaded AVIs become MP4s (memory/44).

The browser can't play AVI, so an uploaded one is converted on the GPU
endpoint (``task="transcode"``, sagemaker_backend/inference.py), which writes
the MP4 beside the original in raw-videos. State lives on the clip:

    metadata.needs_transcode = True                      set at upload
    metadata.transcode = {state, output_key, failure_uri, at, attempts}

The reconciler tick drives it: ``dispatch`` sends clips that need it,
``collect`` switches a clip to its MP4 once that exists (the original key is
kept in ``metadata.original_key``), and gives up after a failure record or a
timeout, retrying once. The clip stays usable throughout — the analysis
reads AVI directly; only browser playback waits for the MP4.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import PurePosixPath

from django.conf import settings
from django.utils import timezone

logger = logging.getLogger(__name__)

TASK = "transcode"
GIVE_UP_AFTER = timedelta(hours=2)
MAX_ATTEMPTS = 2
PER_TICK = 5


def output_key_for(storage_key: str) -> str:
    return str(PurePosixPath(storage_key).with_suffix(".mp4"))


def _enabled() -> bool:
    return bool(getattr(settings, "SAGEMAKER_ENDPOINT_NAME", ""))


def dispatch() -> int:
    """Send clips waiting for conversion. Returns how many were sent."""
    from apps.analysis import views as analysis_views
    from .models import Video

    if not _enabled():
        return 0
    sent = 0
    waiting = (Video.objects.filter(metadata__needs_transcode=True)
               .filter(metadata__transcode__isnull=True)[:PER_TICK])
    for video in waiting:
        out_key = output_key_for(video.storage_key)
        inference_id = f"tc_{video.pk}_{int(timezone.now().timestamp())}"
        payload = {"task": TASK, "job_id": inference_id, "user_id": str(video.user_id),
                   "video_blob_path": video.storage_key, "output_key": out_key}
        attempts = int((video.metadata or {}).get("transcode_attempts", 0)) + 1
        try:
            input_uri = analysis_views._put_inference_payload(inference_id, payload)
            _out, failure_uri = analysis_views._invoke_endpoint_async(inference_id, input_uri)
        except Exception:
            logger.exception("transcode dispatch failed for video %s", video.pk)
            continue
        video.metadata = {**(video.metadata or {}), "transcode_attempts": attempts,
                          "transcode": {"state": "invoked", "output_key": out_key,
                                        "failure_uri": failure_uri,
                                        "at": timezone.now().isoformat()}}
        video.save(update_fields=["metadata"])
        sent += 1
    return sent


def collect() -> int:
    """Switch finished clips to their MP4. Returns how many finished."""
    from django.utils.dateparse import parse_datetime

    from config.storage import get_s3_client
    from .models import Video

    s3 = get_s3_client()
    finished = 0
    for video in Video.objects.filter(metadata__transcode__state="invoked")[:50]:
        meta = dict(video.metadata or {})
        state = dict(meta.get("transcode") or {})
        out_key = state.get("output_key") or ""
        try:
            ready = bool(out_key) and s3.blob_exists("raw-videos", out_key)
        except Exception:
            logger.exception("transcode check failed for video %s", video.pk)
            continue
        if ready:
            meta.update({"original_key": video.storage_key, "needs_transcode": False,
                         "transcode": {**state, "state": "done", "done_at": timezone.now().isoformat()}})
            video.storage_key = out_key
            video.metadata = meta
            video.save(update_fields=["storage_key", "metadata"])
            try:
                from .thumbnails import queue_thumbnail
                queue_thumbnail(video)
            except Exception:
                logger.exception("thumbnail after transcode failed for video %s", video.pk)
            finished += 1
            continue
        failed = _failure_recorded(s3, state.get("failure_uri"))
        sent_at = parse_datetime(state.get("at") or "")
        timed_out = sent_at is not None and timezone.now() - sent_at > GIVE_UP_AFTER
        if failed or timed_out:
            if int(meta.get("transcode_attempts", 1)) < MAX_ATTEMPTS:
                meta.pop("transcode", None)          # dispatch sends it again
            else:
                meta["transcode"] = {**state, "state": "failed",
                                     "reason": "conversion failed" if failed else "timed out"}
            video.metadata = meta
            video.save(update_fields=["metadata"])
    return finished


def _failure_recorded(s3, failure_uri) -> bool:
    """SageMaker wrote a failure record for the invocation."""
    if not failure_uri or not failure_uri.startswith("s3://"):
        return False
    bucket, _, key = failure_uri[5:].partition("/")
    try:
        s3._client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def tick() -> dict:
    return {"transcoded": collect(), "transcode_sent": dispatch()}
