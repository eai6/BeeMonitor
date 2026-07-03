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
