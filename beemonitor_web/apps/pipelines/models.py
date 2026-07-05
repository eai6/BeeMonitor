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
