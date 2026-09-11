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


# There is deliberately no estimate function here any more. A projection shown
# before a run reads as a promise, and the figure it would replace — a hardcoded
# 349 credits/clip — had been wrong for months without anyone noticing, because
# nothing ever checked it against reality. What a run cost is recorded after it
# finishes, from the hardware it actually used, and finished runs report GPU
# TIME rather than money in the UI: seconds are a fact about the work, where a
# dollar figure is a claim about a price list that drifts.
#
# price_run above still records compute_cost_usd, which the account usage page
# aggregates — that is billing, and it is the one place money belongs.
