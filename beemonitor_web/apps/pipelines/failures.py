"""Why a run failed, grouped so nine failures read as two problems.

The batch page showed a status per run and nothing else, so triaging a failed
batch meant opening every run to find that most of them died of the same thing.
Batch 5af72b17 was nine failures and two causes.

A cause is matched on the text the job recorded. That text comes from six
different writers (``analysis/views.py`` for platform-level deaths, the
container's own traceback for anything raised inside the analyzer), so matching
on substrings is deliberate: the taxonomy has to describe what actually gets
written, not what we wish were written.

``fixed_at`` marks a cause we have since shipped a fix for. A run that predates
it gets a "fixed" badge — which is a claim about chronology, not a promise, and
is why it compares against the run's own timestamp rather than asserting the
cause can no longer happen.
"""

from datetime import datetime, timezone

# Ordered: the first match wins, so put specific causes before general ones.
CAUSES = [
    {
        "key": "endpoint_unresponsive",
        "match": ("could not get a response",),
        "title": "The endpoint ran out of CPU",
        "detail": (
            "SageMaker gave up waiting for the container. Several jobs were "
            "sharing one instance and its CPUs were fully committed, so nothing "
            "could answer in time. Nothing was wrong with the clip."
        ),
        "action": "Re-run — one job per instance now, across four instances.",
        "fixed_at": datetime(2026, 9, 9, 21, 0, tzinfo=timezone.utc),
    },
    {
        "key": "sam3_import_race",
        "match": ("cannot import name 'Sam3Model'", 'cannot import name "Sam3Model"'),
        "title": "SAM 3 failed to load under concurrency",
        "detail": (
            "Two jobs raced the same lazy import of transformers, and one saw a "
            "half-registered module. Intermittent, and only when jobs overlap."
        ),
        "action": "Re-run — the model now loads once per process behind a lock.",
        "fixed_at": datetime(2026, 9, 9, 21, 0, tzinfo=timezone.utc),
    },
    {
        "key": "no_detections",
        "match": ("No detections to track",),
        "title": "The detector was set to reference only",
        "detail": (
            "The Detect node was asked for references and nothing else, so there "
            "was no tracking data for the MOT step to name. A clip the detector "
            "simply found nothing in no longer fails — that completes with zero "
            "rows."
        ),
        "action": "Set the Detect node's Run scope to 'Objects + reference', then re-run.",
        "fixed_at": datetime(2026, 9, 10, 23, 0, tzinfo=timezone.utc),
    },
    {
        "key": "container_failure",
        "match": ("SageMaker inference failed",),
        "title": "The container failed mid-clip",
        "detail": (
            "The analyzer raised inside the GPU container; the text after the "
            "colon is the container's own failure payload."
        ),
        "action": "Re-run once. If it repeats on the same clip, the clip is the cause.",
    },
    {
        "key": "gpu_job_failed",
        "match": ("GPU job failed.",),
        "title": "The GPU job failed without saying why",
        "detail": (
            "The job reached a failed state carrying no message — nothing was "
            "recorded to explain it, which is itself the finding."
        ),
        "action": "Re-run; if it repeats, check the endpoint logs for that job id.",
    },
    {
        "key": "job_vanished",
        "match": ("no longer exists",),
        "title": "The step's analysis job is gone",
        "detail": "The job row was deleted while the run was still waiting on it.",
        "action": "Re-run — it will spawn a fresh job.",
    },
    {
        "key": "never_reached_gpu",
        "match": ("Never reached the GPU",),
        "title": "The request never reached the GPU",
        "detail": "The server restarted mid-submission and the retry also failed.",
        "action": "Safe to re-run.",
    },
    {
        "key": "timed_out",
        "match": ("Timed out: no result",),
        "title": "Timed out waiting for a result",
        "detail": (
            "Past the async platform's caps, so the request can never land. "
            "Usually a very long clip, or a queue that stayed full."
        ),
        "action": "Re-run; if it recurs, the clip may need chunking.",
    },
    {
        "key": "video_missing",
        "match": ("not found, or not yours", "Selected video not found"),
        "title": "The clip could not be read",
        "detail": "It was deleted, or it belongs to someone whose device is not shared with you.",
        "action": "Re-running will not help until the clip is reachable.",
        "retryable": False,
    },
    {
        "key": "out_of_memory",
        "match": ("CUDA out of memory", "OutOfMemoryError", "Killed"),
        "title": "Ran out of GPU memory",
        "detail": "The model and the clip together exceeded the card.",
        "action": "Re-run on a larger instance, or with fewer jobs per instance.",
    },
    {
        "key": "upstream",
        "match": ("Upstream step failed",),
        "title": "A previous step failed",
        "detail": "This step never ran; fix the step above it.",
        "action": "Re-running the run fixes this once the real cause is fixed.",
        "retryable": False,
    },
]

UNKNOWN = {
    "key": "unknown",
    "title": "Failed for another reason",
    "detail": "No cause matched this message, so it is shown as recorded:",
    "action": "",
    # The panel renders `sample` for this cause only. It used to promise the
    # recorded text and show nothing, which is how a whole class of failure
    # ("No detections to track" — 18 of 19 on one batch) stayed invisible for as
    # long as it did: the only place the text appeared was a row's hover title.
    "show_sample": True,
}


def classify(message: str) -> dict:
    """The cause for one error message. Never returns None."""
    text = message or ""
    for cause in CAUSES:
        if any(needle in text for needle in cause["match"]):
            return cause
    return UNKNOWN


def is_fixed_for(cause: dict, when) -> bool:
    """True when a fix for this cause shipped AFTER the run happened.

    A claim about chronology only: it says the run predates the fix, not that
    the cause is impossible now.
    """
    fixed_at = cause.get("fixed_at")
    return bool(fixed_at and when and when < fixed_at)


def group(failures) -> list:
    """``[(video, message, when), ...]`` -> one entry per distinct cause.

    Ordered by count, worst first, so the biggest problem is the first thing
    read. Each entry carries the video ids, which is what a per-cause re-run
    needs.
    """
    buckets = {}
    for video_id, message, when in failures:
        cause = classify(message)
        b = buckets.setdefault(cause["key"], {
            "cause": cause, "video_ids": [], "count": 0,
            "sample": message, "fixed": True,
        })
        b["count"] += 1
        if video_id is not None:
            b["video_ids"].append(video_id)
        # "fixed" only if EVERY run in the bucket predates the fix.
        b["fixed"] = b["fixed"] and is_fixed_for(cause, when)
    return sorted(buckets.values(), key=lambda b: -b["count"])
