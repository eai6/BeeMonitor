"""The batch's Files list (memory/80): one row per clip, whatever became of it.

The other downloads only carry clips that produced rows, so a clip that failed,
was cancelled or found nothing left no trace in them. This one lists every
clip in the batch with when it was recorded, how that time was worked out, and
what happened to its run, so the dates and times of everything processed can
be checked in one place.

Read from the slim runs and a few hundred clips at a time (memory/76, 77).
"""

from __future__ import annotations

import re
from datetime import timedelta

from apps.videos import timing

from . import aggregate, failures

FIELDS = [
    "video_title", "site_name", "device_name",
    "date", "start_time", "end_time", "time_zone",
    "clip_start", "clip_end", "time_source", "filename_time",
    "duration_min", "fps", "size_mb",
    "run_status", "failure_cause", "failure_detail", "next_step", "error", "gpu_hours",
    "run_id", "video_id",
]

CHUNK = 500

# site_YYYY-MM-DD_HH_MM_SS, the recorder's file names.
_NAME_TIME = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{2})_(\d{2})_(\d{2})")


def filename_time(title: str) -> str:
    """The wall-clock written in a clip's file name, or ""."""
    m = _NAME_TIME.search(title or "")
    return f"{m.group(1)} {m.group(2)}:{m.group(3)}:{m.group(4)}" if m else ""


def zone_name(start) -> str:
    """The zone a clip's times are given in: an IANA name, "UTC", or "" when
    the start is a wall-clock with no zone known."""
    if start is None or start.tzinfo is None:
        return ""
    return getattr(start.tzinfo, "key", None) or start.tzname() or ""


def run_status(run, error: str) -> str:
    if run.status == run.Status.FAILED and "Cancelled by user" in error:
        return "cancelled"
    return run.status


def _row(run, video, job):
    error = aggregate.run_error(run)
    status = run_status(run, error)
    # Why it didn't finish, in the words of the batch page's failures panel.
    cause = failures.classify(error) if status in ("failed", "cancelled") else {}
    start = timing.clip_start(video) if video else None
    seconds = float(video.duration_seconds) if video and video.duration_seconds else None
    end = start + timedelta(seconds=seconds) if start is not None and seconds else None
    title = getattr(video, "title", "") if video else ""
    return {
        "video_title": title,
        "site_name": getattr(video, "site_name", "") or "" if video else "",
        "device_name": getattr(getattr(video, "device", None), "name", "") or "",
        "date": start.date().isoformat() if start else "",
        "start_time": start.strftime("%H:%M:%S") if start else "",
        "end_time": end.strftime("%H:%M:%S") if end else "",
        "time_zone": zone_name(start),
        "clip_start": timing.iso(start),
        "clip_end": timing.iso(end),
        # Where the start came from: device, filename, file, user, upload_time.
        "time_source": ((video.metadata or {}).get("recorded_at_source", "") if video else ""),
        "filename_time": filename_time(title),
        "duration_min": round(seconds / 60, 2) if seconds else "",
        "fps": float(video.fps) if video and video.fps else "",
        "size_mb": round(video.file_size_bytes / 1e6, 1) if video and video.file_size_bytes else "",
        "run_status": status,
        "failure_cause": cause.get("title", ""),
        "failure_detail": cause.get("detail", ""),
        "next_step": cause.get("action", ""),
        "error": error,
        "gpu_hours": (round(float(job.execution_seconds) / 3600, 3)
                      if job and job.execution_seconds else 0),
        "run_id": str(run.pk),
        "video_id": video.pk if video else "",
    }


def rows(runs):
    """Every run's row, in recording order (clips with no known start last)."""
    from apps.analysis.models import Job
    from apps.videos.models import Video

    runs = list(runs)
    vid_of = {run.pk: aggregate.run_video_id(run) for run in runs}
    when = dict(Video.everything.filter(pk__in=[v for v in vid_of.values() if v])
                .values_list("pk", "recorded_at"))
    runs.sort(key=lambda r: (when.get(vid_of[r.pk]) is None, when.get(vid_of[r.pk]) or 0,
                             str(r.pk)))
    for i in range(0, len(runs), CHUNK):
        chunk = runs[i:i + CHUNK]
        videos = {v.pk: v for v in Video.everything.filter(
            pk__in=[vid_of[r.pk] for r in chunk if vid_of[r.pk]]).select_related("device", "site")}
        job_ids = [j for r in chunk if (j := aggregate.run_job_id(r))]
        jobs = {j.pk: j for j in Job.objects.filter(pk__in=job_ids).only("pk", "execution_seconds")}
        for run in chunk:
            yield _row(run, videos.get(vid_of[run.pk]), jobs.get(aggregate.run_job_id(run)))
