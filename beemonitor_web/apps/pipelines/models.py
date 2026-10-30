"""
Data model for the BeeMonitor visual pipeline builder.

A ``Pipeline`` is a saved graph of steps (``steps`` JSON, the same shape the
Workshop builder used, extended with an optional ``inputs`` map per step for
multi-input ecological analyses). A ``PipelineRun`` is one execution: it advances
as a **resumable state machine** — local steps run inline, GPU steps spawn an
``analysis.Job`` and the run parks until the poller reports the job finished
(see ``engine.advance_run`` / ``engine.on_job_finished``).

Design: ``memory/23_pipeline_builder_port_design.md``.
"""

import uuid

from django.conf import settings
from django.db import models


class Pipeline(models.Model):
    """A visual pipeline composed of ordered (optionally branched) steps."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="pipelines",
    )
    title = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    steps = models.JSONField(
        default=list,
        blank=True,
        help_text="Ordered list of step dicts: "
                  "[{id, block_type, config, inputs?}, ...]",
    )
    graph = models.JSONField(
        default=dict,
        blank=True,
        help_text="Raw Drawflow node-graph export (canvas layout). Derived into "
                  "`steps` on save; empty for pipelines built before the canvas.",
    )
    is_template = models.BooleanField(default=False)

    class Visibility(models.TextChoices):
        PRIVATE = "private", "Private"
        PUBLIC = "public", "Public — anyone can view, run and copy it"

    # Published like a model or a dataset (memory/59): readable, runnable and
    # copyable by anyone signed in, never editable by them.
    visibility = models.CharField(
        max_length=10, choices=Visibility.choices, default=Visibility.PRIVATE,
    )
    published_at = models.DateTimeField(null=True, blank=True)
    cloned_from = models.ForeignKey(
        "self",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="clones",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]

    def __str__(self):
        return self.title

    @property
    def is_public(self):
        return self.visibility == self.Visibility.PUBLIC

    @classmethod
    def public(cls):
        return cls.objects.filter(visibility=cls.Visibility.PUBLIC).order_by("-published_at")

    @classmethod
    def usable(cls, user):
        """Pipelines this user may view, run and copy: their own, the
        templates, and anyone's published ones."""
        return cls.objects.filter(
            models.Q(user=user) | models.Q(is_template=True)
            | models.Q(visibility=cls.Visibility.PUBLIC))

    def usable_by(self, user):
        return self.user_id == user.id or self.is_template or self.is_public


class PipelineRun(models.Model):
    """A single execution of a pipeline (resumable state machine)."""

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        RUNNING = "running", "Running"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"

    # Per-step lifecycle stored inside ``step_status`` (step_id -> one of these).
    STEP_PENDING = "pending"
    STEP_RUNNING = "running"       # local step executing / GPU job in flight
    STEP_DONE = "done"
    STEP_FAILED = "failed"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    pipeline = models.ForeignKey(
        Pipeline, on_delete=models.CASCADE, related_name="runs"
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="pipeline_runs",
    )
    status = models.CharField(
        max_length=20, choices=Status.choices, default=Status.PENDING
    )
    # Runs launched together (one pipeline over many videos) share a batch_id so
    # results can be aggregated across videos (combined CSVs, cross-video trips).
    batch_id = models.UUIDField(null=True, blank=True, db_index=True)
    # Frozen copy of the pipeline steps at launch time (so edits mid-run don't corrupt it).
    steps = models.JSONField(default=list, blank=True)
    # step_id -> artifact/output dict produced by that step.
    context = models.JSONField(default=dict, blank=True)
    # step_id -> STEP_* status.
    step_status = models.JSONField(default=dict, blank=True)
    error_message = models.TextField(blank=True)
    # Launched with the GPU result cache disabled. Set when a batch is re-run to
    # measure a change: the cache is keyed on the clip and the job config, not
    # on the analyzer build, so a plain re-run after a GPU-side fix returns the
    # OLD result and reports success.
    fresh = models.BooleanField(default=False)
    # Which version of the event/interaction rules this run's tables were
    # counted with. Older runs are recounted from their saved tracks by the
    # reconcile loop (apps/pipelines/recount.py), no GPU needed.
    counting_rules = models.PositiveSmallIntegerField(default=4)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    execution_time_ms = models.IntegerField(null=True, blank=True)

    class Meta:
        ordering = ["-started_at", "-id"]

    def __str__(self):
        return f"Run {self.id} — {self.pipeline.title} [{self.status}]"

    # ── Convenience accessors used by the engine + templates ──────────────────
    def step_state(self, step_id):
        return (self.step_status or {}).get(step_id, self.STEP_PENDING)

    @property
    def is_terminal(self):
        return self.status in (self.Status.COMPLETED, self.Status.FAILED)


class StepResult(models.Model):
    """Content-addressed cache of a completed step's output, for cross-run reuse.

    Keyed by ``cache_key`` = hash(user, block_type, effective config, upstream cache
    keys). When a step in a new run has a matching key, its output is reused instantly
    instead of recomputing — so re-running an edited pipeline only recomputes the
    steps that actually changed (and their descendants). Only *successful* outputs are
    cached, so a failed GPU step (e.g. endpoint misconfigured) re-runs next time.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="step_results"
    )
    cache_key = models.CharField(max_length=64, db_index=True)
    block_type = models.CharField(max_length=64)
    output = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["user", "cache_key"], name="uniq_user_cache_key"),
        ]

    def __str__(self):
        return f"{self.block_type} [{self.cache_key[:8]}]"


class BatchShare(models.Model):
    """A public, read-only link to one batch's results (memory/47).

    Anyone holding ``token`` can view the batch without signing in: its summary,
    contact time per reference, clip list with tracks over the video, and the
    combined CSVs. Nothing else — no re-run, no GPU time, no error text, and no
    site or location unless the owner turns that on.

    The token is kept in the clear because the owner must be able to copy the
    link again; it is a capability URL, like a document share link. Turning a
    link off stamps ``revoked_at``; turning sharing on again makes a NEW token,
    so a link that was passed around stays dead.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    batch_id = models.UUIDField(db_index=True)
    token = models.CharField(max_length=64, unique=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="batch_shares"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    revoked_at = models.DateTimeField(null=True, blank=True)
    show_videos = models.BooleanField(default=True)
    show_locations = models.BooleanField(default=False)
    view_count = models.PositiveIntegerField(default=0)
    last_viewed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        constraints = [
            # At most one live link per batch.
            models.UniqueConstraint(fields=["batch_id"], condition=models.Q(revoked_at=None),
                                    name="uniq_live_share_per_batch"),
        ]

    def __str__(self):
        return f"share {str(self.batch_id)[:8]} ({'off' if self.revoked_at else 'on'})"


class BatchMember(models.Model):
    """Someone the batch's launcher invited (memory/62), like an annotation
    project's ``ProjectShare``. A reviewer watches, votes on events and trips,
    and downloads; a viewer watches and downloads. Neither can re-run, cancel
    or share the batch. The public link (``BatchShare``) stays view-only."""

    class Role(models.TextChoices):
        REVIEWER = "reviewer", "Reviewer — watch, vote on events and trips, download"
        VIEWER = "viewer", "Viewer — watch and download"

    batch_id = models.UUIDField(db_index=True)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
                             related_name="batch_memberships")
    role = models.CharField(max_length=16, choices=Role.choices, default=Role.REVIEWER)
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
                                   null=True, blank=True, related_name="+")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("batch_id", "user")
        ordering = ["user__username"]

    def __str__(self):
        return f"{self.user} on {str(self.batch_id)[:8]} ({self.role})"


class EventReview(models.Model):
    """One reviewer's call on one detected event (memory/62).

    The event is identified by itself — action, reference, track, frame — not
    by a row number, and its fields are copied here, so a recount that removes
    the event keeps the vote (shown as "no longer detected"). A review never
    changes the detected events; it sits beside them.
    """

    class Verdict(models.TextChoices):
        ACCEPTED = "accepted", "Accepted"
        REJECTED = "rejected", "Rejected"

    run = models.ForeignKey(PipelineRun, on_delete=models.CASCADE, related_name="event_reviews")
    event_key = models.CharField(max_length=200)
    action = models.CharField(max_length=16)
    reference = models.CharField(max_length=100, blank=True)
    track_id = models.CharField(max_length=32, blank=True)
    frame = models.IntegerField(null=True, blank=True)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
                             related_name="event_reviews")
    verdict = models.CharField(max_length=16, choices=Verdict.choices)
    reason = models.CharField(max_length=40, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("run", "event_key", "user")
        indexes = [models.Index(fields=["run", "event_key"], name="evrev_run_key")]

    def __str__(self):
        return f"{self.user} {self.verdict} {self.event_key}"


class TripReview(models.Model):
    """One reviewer's call on a suggested trip: an exit and the entry the
    same bee came back on (memory/62 §4b).

    A trip is data only when a person says so: accepted (the suggested pair or
    one the reviewer re-linked to), rejected, "didn't come back in this period"
    (no entry) or "can't tell". The exit and the entry may be in different
    clips, so each side names its own run.
    """

    class Verdict(models.TextChoices):
        ACCEPTED = "accepted", "Same bee, a trip"
        REJECTED = "rejected", "Not this pair"

    exit_run = models.ForeignKey(PipelineRun, on_delete=models.CASCADE, related_name="trip_exits")
    exit_key = models.CharField(max_length=200)
    entry_run = models.ForeignKey(PipelineRun, on_delete=models.SET_NULL, null=True, blank=True,
                                  related_name="trip_entries")
    entry_key = models.CharField(max_length=200, blank=True)
    reference = models.CharField(max_length=100, blank=True)
    exit_track = models.CharField(max_length=32, blank=True)
    exit_frame = models.IntegerField(null=True, blank=True)
    entry_track = models.CharField(max_length=32, blank=True)
    entry_frame = models.IntegerField(null=True, blank=True)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
                             related_name="trip_reviews")
    verdict = models.CharField(max_length=16, choices=Verdict.choices)
    # "relinked", "didn't come back", "can't tell", or blank.
    reason = models.CharField(max_length=40, blank=True)
    relinked = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("exit_run", "exit_key", "user")

    def __str__(self):
        return f"{self.user} {self.verdict} trip from {self.exit_key}"
