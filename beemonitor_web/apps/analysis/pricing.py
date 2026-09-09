"""What a GPU job actually cost, priced from what actually ran.

The single source of truth for job cost. It exists because the arithmetic was
copy-pasted into ``analysis/views.py`` and ``api/views.py``, so fixing one left
the other reporting the old number, and a third consumer (credits) derived its
own figure from the same seconds.

What was wrong with the old basis, and what this fixes:

* **The tier was fiction.** ``Job.gpu_tier`` was a user-facing dropdown that was
  stored, priced against, and *never sent to SageMaker*. Every analysis job runs
  on whatever the endpoint is pinned to. A user picking "A100 (Fastest)" got a
  T4 and was billed 2.85x the T4 rate. The selector is gone; the field is now
  set from the hardware the run reports.
* **The prices were Modal's**, carried over from before the AWS migration.
  These are SageMaker on-demand, us-east-1.

What is deliberately NOT priced: cold start and the scale-in cooldown. The
instance is up longer than any one job's handler window — roughly 10 minutes of
instance for 4 minutes of work on a scale-from-zero — but that time belongs to
no single job, and the scale-to-zero design is intentionally out of scope. Cost
here is the handler window at the true instance rate; it is a floor on the AWS
bill, not the whole of it.
"""

from __future__ import annotations

# SageMaker on-demand, us-east-1, USD per instance-hour. Update alongside
# infra/aws-sagemaker/Pulumi.<stack>.yaml when an endpoint's instance changes.
INSTANCE_HOURLY_USD = {
    "ml.g4dn.xlarge": 0.7364,   # T4 16 GB   — the video/tracking endpoint
    "ml.g5.xlarge": 1.408,      # A10G 24 GB — the SAM 3 endpoint
    "ml.g6.xlarge": 0.9765,     # L4 24 GB
    "ml.g6e.xlarge": 1.861,     # L40S 48 GB
    "ml.p4d.24xlarge": 37.688,  # A100 x8    — training only
}

# The container reports its own GPU (`_detect_device`, sagemaker_backend/
# inference.py), e.g. "cuda:Tesla T4". That string is the honest key: it comes
# from the run itself, so it cannot drift from the endpoint config the way a
# stored tier did. Matched case-insensitively on substring.
GPU_NAME_INSTANCE = (
    ("t4", "ml.g4dn.xlarge"),
    ("a10g", "ml.g5.xlarge"),
    ("l40s", "ml.g6e.xlarge"),
    ("l4", "ml.g6.xlarge"),
    ("a100", "ml.p4d.24xlarge"),
)

# Instance -> the Job.GpuTier label, so the stored tier describes the hardware
# that ran rather than a dropdown choice.
INSTANCE_TIER = {
    "ml.g4dn.xlarge": "T4",
    "ml.g5.xlarge": "A10G",
    "ml.g6.xlarge": "L4",
    "ml.g6e.xlarge": "L40S",
    "ml.p4d.24xlarge": "A100",
}

# What the tracking endpoint runs when a job didn't report a device — an older
# image, or a failure before the GPU was touched. Mirrors
# infra/aws-sagemaker/Pulumi.dev.yaml:instance-type.
DEFAULT_INSTANCE = "ml.g4dn.xlarge"

# Credits are a product unit, not a cost: one per GPU second, so the meaning
# does not shift when a price does.
CREDITS_PER_GPU_SECOND = 1


def instance_for_device(device: str | None) -> str:
    """Map a reported device string to a SageMaker instance type.

    ``device`` is what the container saw ("cuda:Tesla T4"). Unknown or missing
    falls back to the endpoint's configured instance rather than guessing high.
    """
    haystack = (device or "").lower()
    for needle, instance in GPU_NAME_INSTANCE:
        if needle in haystack:
            return instance
    return DEFAULT_INSTANCE


def rate_per_second(instance: str) -> float:
    """USD per second for an instance type."""
    return INSTANCE_HOURLY_USD.get(instance, INSTANCE_HOURLY_USD[DEFAULT_INSTANCE]) / 3600.0


def tier_for_instance(instance: str) -> str:
    return INSTANCE_TIER.get(instance, INSTANCE_TIER[DEFAULT_INSTANCE])


def price_run(result: dict) -> dict:
    """Cost a completed job from its own result payload.

    Returns the fields a caller writes back to ``Job``, plus ``credits`` for
    ``UserProfile.charge``. One function so the three call sites cannot drift.

    ``execution_seconds`` (the whole handler) is what the instance was busy for,
    so it is what gets billed. ``gpu_seconds`` — inference only — is recorded
    alongside it: it is the diagnostic, not the meter.
    """
    exec_seconds = float(result.get("execution_seconds") or 0)
    gpu_seconds = float(result.get("gpu_seconds") or 0)
    instance = instance_for_device(result.get("device"))

    return {
        "execution_seconds": exec_seconds,
        "gpu_seconds": gpu_seconds,
        "stage_seconds": result.get("stage_seconds") or {},
        "gpu_tier": tier_for_instance(instance),
        "compute_cost_usd": round(exec_seconds * rate_per_second(instance), 4),
        "credits": int(exec_seconds * CREDITS_PER_GPU_SECOND),
        "instance_type": instance,
    }


def estimate_per_video(user, device_ids=None, sample=40) -> dict:
    """What one more clip is likely to cost, from what past clips actually cost.

    The number this replaces was ``est_credits_per_video = 349``, a hardcoded
    constant carrying the retired A10G tier — wrong hardware and a made-up
    average. Completed jobs record ``execution_seconds``, so the estimate can be
    measured instead: the median of recent runs, with the 25th-75th percentile
    as the spread. A range, because the spread is real and one number would be
    false precision.

    Scoped to the same hotels where possible, since a device's clip length and
    activity drive the cost far more than anything else. Returns ``sample: 0``
    when there is no history to go on — the caller should say the estimate is
    provisional rather than print a confident figure.
    """
    from .models import Job

    qs = (Job.objects
          .filter(user=user, status="completed", execution_seconds__gt=0)
          .order_by("-id"))
    if device_ids:
        scoped = qs.filter(video__device_id__in=device_ids)
        # Fall back to all runs rather than reporting nothing for a new hotel.
        qs = scoped if scoped.exists() else qs

    seconds = sorted(qs.values_list("execution_seconds", flat=True)[:sample])
    if not seconds:
        return {"sample": 0, "seconds": 0.0, "low": 0.0, "high": 0.0,
                "cost": 0.0, "cost_low": 0.0, "cost_high": 0.0,
                "instance": DEFAULT_INSTANCE, "tier": tier_for_instance(DEFAULT_INSTANCE)}

    def pct(p):
        return seconds[min(len(seconds) - 1, int(len(seconds) * p))]

    median, low, high = pct(0.5), pct(0.25), pct(0.75)
    # Priced at the endpoint's instance: analysis jobs all land on the same one.
    rate = rate_per_second(DEFAULT_INSTANCE)
    return {
        "sample": len(seconds),
        "seconds": round(median, 1),
        "low": round(low, 1),
        "high": round(high, 1),
        "cost": round(median * rate, 4),
        "cost_low": round(low * rate, 4),
        "cost_high": round(high * rate, 4),
        "instance": DEFAULT_INSTANCE,
        "tier": tier_for_instance(DEFAULT_INSTANCE),
    }
