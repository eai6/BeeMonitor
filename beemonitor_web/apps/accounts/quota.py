"""GPU credits: what work costs, whether a user can start it, and charging it.

One gate for every place GPU work starts (memory/50 §5). The rules:

* **1 credit = 1 GPU-hour on the video machine** (T4). Dearer machines cost
  more credits per hour (``CREDITS_PER_GPU_HOUR``), rounded so it is easy to
  say: SAM 3, sampling and training run on the A10G at 2 an hour.
* **Checked before anything starts.** ``check`` refuses a launch whose estimate
  does not fit what is left, counting work already queued or running. A launch
  that was let through always finishes; if it overruns its estimate the balance
  goes a little negative and the next launch is refused.
* **Charged once, when it finishes**, from the seconds the instance was busy
  (``charge``). Each charge is a ``UsageEvent`` keyed on its source, so a job
  polled twice is charged once. Failed work is never charged.
* **One balance.** A new account starts with ``STARTING_CREDITS`` once; more
  come from approved requests (``AccountRequest``) and promo codes. Nothing
  renews.
* **Pro and superusers are never blocked**, but are still charged, so usage
  stays visible.

Estimates come from recent real runs (``UsageEvent``) rather than constants,
so they track the platform as it changes. The fallbacks below only apply until
a kind of work has history; each is labelled with where its number came from.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass
from datetime import timedelta
from decimal import ROUND_HALF_UP, Decimal

from django.db import IntegrityError, transaction
from django.db.models import Sum
from django.utils import timezone

logger = logging.getLogger(__name__)

VIDEO_INSTANCE = "ml.g4dn.xlarge"   # T4 — YOLO tracking, YOLO auto-label
SAM3_INSTANCE = "ml.g5.xlarge"      # A10G — SAM 3, sampling, training

# Credits per GPU-hour, by SageMaker instance. Proportional to the AWS price
# (apps/analysis/pricing.py) and rounded by hand; edit both together.
CREDITS_PER_GPU_HOUR = {
    "ml.g4dn.xlarge": Decimal("1"),     # T4    $0.736/h
    "ml.g6.xlarge": Decimal("1.5"),     # L4    $0.977/h
    "ml.g5.xlarge": Decimal("2"),       # A10G  $1.408/h
    "ml.g6e.xlarge": Decimal("2.5"),    # L40S  $1.861/h
    "ml.p4d.24xlarge": Decimal("50"),   # A100x8 $37.69/h — training only
}

# What one unit of each kind of work is.
UNITS = {
    "analysis": "second of footage",
    "photo": "photo",
    "annotated_video": "second of footage",
    "auto_label": "frame",
    "sampling": "clip",
    "training": "image-epoch",
}

# GPU-seconds per unit, used only until UsageEvent has history for a kind.
FALLBACK_SECONDS_PER_UNIT = {
    # memory/48: median 644 s for a 600 s clip on the T4.
    ("analysis", VIDEO_INSTANCE): 1.07,
    # SAM 3 tracks every frame with a transformer; chunked at 60 s of video per
    # invocation to stay under the 1 h cap (analysis.views._CHUNK_LIMIT_SECONDS),
    # so assume well under 60x. Replaced by history after the first runs.
    ("analysis", SAM3_INSTANCE): 6.0,
    ("photo", VIDEO_INSTANCE): 10.0,
    ("annotated_video", VIDEO_INSTANCE): 0.5,
    ("auto_label", VIDEO_INSTANCE): 0.3,
    ("auto_label", SAM3_INSTANCE): 1.0,
    # memory/38: ~10-13 s per average clip on the g5 with 4 workers.
    ("sampling", SAM3_INSTANCE): 13.0,
    ("training", SAM3_INSTANCE): 0.05,
}
# A clip with no duration recorded is assumed to be a full 10-minute recording.
DEFAULT_CLIP_SECONDS = 600.0
# Credits every new account starts with, once (memory/50 §3).
STARTING_CREDITS = Decimal(25)

# How long an approved launch holds its credits while it creates its jobs.
HOLD_SECONDS = 60
HISTORY = 200          # recent events an estimate is the median of
MIN_HISTORY = 5        # fewer than this and the fallback is used

FOUR = Decimal("0.0001")


class QuotaExceeded(Exception):
    """A launch does not fit the user's remaining credits."""

    def __init__(self, needed, available):
        self.needed = needed
        self.available = available
        super().__init__(self.message)

    @property
    def message(self) -> str:
        return (
            f"Not enough credits: this needs about {fmt(self.needed)} and you have "
            f"{fmt(max(self.available, Decimal(0)))}. Nothing was started. Ask for more "
            "from Usage & credits.")


def fmt(credits) -> str:
    return f"{Decimal(credits).quantize(Decimal('0.1'), ROUND_HALF_UP)} credits"


def credits_for(gpu_seconds: float, instance: str) -> Decimal:
    rate = CREDITS_PER_GPU_HOUR.get(instance, CREDITS_PER_GPU_HOUR[VIDEO_INSTANCE])
    return (Decimal(str(max(gpu_seconds, 0.0))) / 3600 * rate).quantize(FOUR)


# ── Balance ──────────────────────────────────────────────────────────────────

def is_unlimited(user) -> bool:
    if user.is_superuser:
        return True
    return bool(_profile(user).tier_limits.get("unlimited_gpu"))


def _profile(user):
    from .models import UserProfile
    profile, _ = UserProfile.objects.get_or_create(user=user)
    expire_tier(profile)
    return profile


def expire_tier(profile) -> bool:
    """Drop a granted tier whose end date has passed back to Standard. Granted
    credits are kept. Returns True if it changed anything."""
    from .models import AdminAction, UserProfile

    if not profile.tier_expires_at or profile.tier_expires_at > timezone.now():
        return False
    old = profile.get_tier_display()
    profile.tier = UserProfile.Tier.STANDARD
    profile.tier_expires_at = None
    profile.save(update_fields=["tier", "tier_expires_at"])
    AdminAction.record(None, "tier_expired", f"{profile.user.username}'s {old} plan ended; now Standard",
                       target=profile.user, previous=old)
    return True


def available(user) -> Decimal:
    """Credits left."""
    return _profile(user).granted_gpu_credits


# ── Estimates ────────────────────────────────────────────────────────────────

def seconds_per_unit(kind: str, instance: str) -> float:
    """Median GPU-seconds per unit over recent runs of this kind on this machine."""
    from .models import UsageEvent
    rows = (UsageEvent.objects.filter(kind=kind, instance_type=instance, units__gt=0)
            .order_by("-created_at").values_list("gpu_seconds", "units")[:HISTORY])
    ratios = [s / u for s, u in rows if u]
    if len(ratios) >= MIN_HISTORY:
        return statistics.median(ratios)
    return FALLBACK_SECONDS_PER_UNIT.get((kind, instance), FALLBACK_SECONDS_PER_UNIT[("analysis", VIDEO_INSTANCE)])


@dataclass
class Work:
    """A piece of GPU work to estimate: ``units`` of ``kind`` on ``instance``."""
    kind: str
    instance: str
    units: float


def estimate(items) -> Decimal:
    """Credits a list of ``Work`` is expected to cost."""
    rates = {}
    total = Decimal(0)
    for w in items:
        key = (w.kind, w.instance)
        if key not in rates:
            rates[key] = seconds_per_unit(*key)
        total += credits_for(rates[key] * w.units, w.instance)
    return total


def clip_seconds(video) -> float:
    return float(getattr(video, "duration_seconds", 0) or 0) or DEFAULT_CLIP_SECONDS


def video_work(video, detector_kind: str = "yolo", task: str = "", frames: int = 0) -> Work:
    """One analysis job on one clip (or photo). ``task`` is the Job's config task:
    a photo is one unit, sampled detection (pre_annotate) is its frames."""
    instance = SAM3_INSTANCE if detector_kind == "sam3" else VIDEO_INSTANCE
    if task == "detect_photo" or getattr(video, "kind", "") == "photo":
        return Work("photo", VIDEO_INSTANCE, 1)
    if task == "pre_annotate":
        return Work("auto_label", instance, frames or 20)
    return Work("analysis", instance, clip_seconds(video))


def auto_label_work(labeler: str, frames: int) -> Work:
    return Work("auto_label", SAM3_INSTANCE if labeler == "sam3" else VIDEO_INSTANCE, frames)


def sampling_work(clips: int) -> Work:
    return Work("sampling", SAM3_INSTANCE, clips)


def training_work(images: int, epochs: int, instance: str = SAM3_INSTANCE) -> Work:
    return Work("training", instance, max(images, 1) * max(epochs, 1))


def pipeline_work(pipeline_steps, videos) -> list:
    """What launching a pipeline over ``videos`` queues: one analysis job per clip
    per detector the pipeline uses (Detect nodes on the same detector ride one
    job — executors.build_detect_and_track_config). Clips whose GPU result is
    already cached cost nothing, so a re-run is over-estimated, never under."""
    from apps.pipelines.registry import get_block

    kinds = set()
    for step in pipeline_steps or []:
        if (get_block(step.get("block_type", "")) or {}).get("backend") == "gpu":
            cfg = step.get("config") or {}
            family = str(cfg.get("model_family") or cfg.get("detector") or "").lower()
            kinds.add("sam3" if family == "sam3" else "yolo")
    return [video_work(v, k) for v in videos for k in sorted(kinds)]


def pending(user) -> Decimal:
    """Credits already promised to work that is queued or running and not yet charged."""
    from apps.analysis.models import Job
    from apps.annotations.models import FrameSamplingTask, PreAnnotationTask
    from apps.training.models import TrainingJob

    from apps.annotations import sampling_remote

    items = []
    active_jobs = Job.objects.filter(
        user=user, status__in=[Job.Status.QUEUED, Job.Status.INGESTING,
                               Job.Status.PROCESSING, Job.Status.POST_PROCESSING],
    ).select_related("video")
    for job in active_jobs:
        cfg = job.config or {}
        items.append(video_work(job.video, cfg.get("detector_kind", "yolo"), cfg.get("task", ""),
                                int(cfg.get("max_frames") or 0)))
    for labeler, params in PreAnnotationTask.objects.filter(
            user=user, status__in=[PreAnnotationTask.Status.QUEUED, PreAnnotationTask.Status.PROCESSING],
    ).values_list("labeler", "params"):
        items.append(auto_label_work(labeler, int((params or {}).get("max_frames") or 20)))
    gpu_sampling = FrameSamplingTask.objects.filter(
        user=user, params__method=sampling_remote.METHOD,
        status__in=[FrameSamplingTask.Status.QUEUED, FrameSamplingTask.Status.PROCESSING],
    ).count()
    if gpu_sampling:
        items.append(sampling_work(gpu_sampling))
    for job in TrainingJob.objects.filter(
            user=user, status__in=[TrainingJob.Status.QUEUED, TrainingJob.Status.PREPARING,
                                   TrainingJob.Status.TRAINING]).select_related("project"):
        items.append(training_work(_project_images(job.project), job.epochs))
    return estimate(items)


def _project_images(project) -> int:
    if project is None:
        return 0
    from apps.annotations.models import Annotation
    return Annotation.objects.filter(project=project).count()


# ── The gate ─────────────────────────────────────────────────────────────────

def check(user, items) -> Decimal:
    """Refuse a launch that doesn't fit. Returns its estimate in credits.

    ``items`` is a list of ``Work`` (or a precomputed Decimal). Call it before
    creating any job, so nothing is half-started when it raises.
    """
    from .models import CreditHold, UserProfile

    needed = items if isinstance(items, Decimal) else estimate(items)
    if is_unlimited(user) or needed <= 0:
        return needed
    now = timezone.now()
    with transaction.atomic():
        # One check at a time per user, and what it approves is held until the
        # jobs exist: otherwise launches sent together each see the whole
        # balance and together spend several times it.
        UserProfile.objects.select_for_update().filter(user=user).first()
        CreditHold.objects.filter(user=user, expires_at__lte=now).delete()
        held = CreditHold.objects.filter(user=user).aggregate(s=Sum("credits"))["s"] or Decimal(0)
        left = available(user) - pending(user) - held
        if needed > left:
            raise QuotaExceeded(needed, left)
        CreditHold.objects.create(user=user, credits=needed,
                                  expires_at=now + timedelta(seconds=HOLD_SECONDS))
    return needed


def how_many_fit(user, per_video) -> int:
    """How many of the leading launches fit the user's credits now.

    ``per_video`` is one list of ``Work`` per clip, in the order they would
    run; a refused batch can then offer to run just the first that many."""
    from .models import CreditHold

    if is_unlimited(user):
        return len(per_video)
    held = (CreditHold.objects.filter(user=user, expires_at__gt=timezone.now())
            .aggregate(s=Sum("credits"))["s"] or Decimal(0))
    left = available(user) - pending(user) - held
    n = 0
    for items in per_video:
        left -= estimate(items)
        if left < 0:
            break
        n += 1
    return n


# ── Charging ─────────────────────────────────────────────────────────────────

def charge(user, kind, gpu_seconds, instance, source_type, source_id, units=0, cost_usd=None):
    """Record finished GPU work and take it off the user's credits. Idempotent.

    Returns the ``UsageEvent``, or None when nothing ran (0 seconds). A second
    call for the same (source_type, source_id) changes nothing.
    """
    from apps.analysis.pricing import rate_per_second

    from .models import UsageEvent, UserProfile

    gpu_seconds = float(gpu_seconds or 0)
    if gpu_seconds <= 0:
        return None
    user_id = getattr(user, "pk", user)
    credits = credits_for(gpu_seconds, instance)
    if cost_usd is None:
        cost_usd = gpu_seconds * rate_per_second(instance)
    try:
        with transaction.atomic():
            event = UsageEvent.objects.create(
                user_id=user_id, kind=kind, gpu_seconds=gpu_seconds, instance_type=instance,
                credits=credits, cost_usd=Decimal(str(round(cost_usd, 4))), units=units or 0,
                source_type=source_type, source_id=str(source_id),
            )
            UserProfile.objects.get_or_create(user_id=user_id)
            profile = UserProfile.objects.select_for_update().get(user_id=user_id)
            if not (profile.user.is_superuser or profile.tier_limits.get("unlimited_gpu")):
                profile.granted_gpu_credits -= credits
            # Legacy lifetime counters the usage page still shows.
            profile.total_gpu_seconds += gpu_seconds
            profile.total_jobs_submitted += 1
            profile.save(update_fields=["granted_gpu_credits", "total_gpu_seconds",
                                        "total_jobs_submitted"])
    except IntegrityError:
        return UsageEvent.objects.filter(source_type=source_type, source_id=str(source_id)).first()
    return event
