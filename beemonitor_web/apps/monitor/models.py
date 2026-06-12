"""Taxonomic monitoring models.

An **Activity** is one motion event on a device — the unit of analysis. The
recorder samples a few **ActivityFrame** crops of the mover and ships them over
cellular (the WiFi-gated video clip uploads later and is linked back by
``activity_uid``). A perception pipeline (Phase 1) runs BioCLIP on each frame,
writing **Detection** rows that reference a **Taxon**, and aggregates them per
activity into an **Observation** ("2 individuals of species X"). See
``memory/15_monitoring_agent_design.md``.

Phase 0 (this change) populates Activity + ActivityFrame only; the perception
tables exist so the activity page and the next phase have a stable schema.
"""

from __future__ import annotations

from django.conf import settings
from django.db import models

from apps.devices.models import Device


class Taxon(models.Model):
    """A node in the biological taxonomy, cached from BioCLIP/GBIF output."""

    class Rank(models.TextChoices):
        KINGDOM = "kingdom", "Kingdom"
        PHYLUM = "phylum", "Phylum"
        CLASS = "class", "Class"
        ORDER = "order", "Order"
        FAMILY = "family", "Family"
        GENUS = "genus", "Genus"
        SPECIES = "species", "Species"

    rank = models.CharField(max_length=20, choices=Rank.choices)
    name = models.CharField(max_length=200, help_text="Scientific name at this rank.")
    common_name = models.CharField(max_length=200, blank=True, default="")
    gbif_id = models.CharField(max_length=32, blank=True, default="", db_index=True)
    parent = models.ForeignKey(
        "self", on_delete=models.SET_NULL, null=True, blank=True, related_name="children",
    )

    class Meta:
        unique_together = ("rank", "name")
        ordering = ["rank", "name"]

    def __str__(self) -> str:
        return f"{self.name} ({self.rank})"


class Activity(models.Model):
    """A single motion event on a device — the unit of taxonomic analysis."""

    class Status(models.TextChoices):
        PENDING = "pending", "Pending analysis"
        ANALYZING = "analyzing", "Analyzing"
        ANALYZED = "analyzed", "Analyzed"
        NO_DETECTION = "no_detection", "No organism detected"
        FAILED = "failed", "Analysis failed"

    device = models.ForeignKey(Device, on_delete=models.CASCADE, related_name="activities")
    # Device-generated stable id so the sampled frames and the (later) video clip
    # reconcile to the same event even though they arrive on different transports.
    activity_uid = models.CharField(max_length=64, db_index=True)
    # Hive wall-clock instant of the event (device-reported, stored UTC-aware).
    started_at = models.DateTimeField(db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)

    # GPS snapshot at the event (device fix, else the Device's set coordinates).
    lat = models.FloatField(null=True, blank=True)
    lon = models.FloatField(null=True, blank=True)
    peak_motion = models.FloatField(null=True, blank=True)

    # The uploaded clip, linked when it arrives over WiFi (may never, if pruned).
    video = models.ForeignKey(
        "videos.Video", on_delete=models.SET_NULL, null=True, blank=True,
        related_name="activities",
    )

    status = models.CharField(max_length=16, choices=Status.choices, default=Status.PENDING)
    # Denormalized best result for list display (filled by the perception pipeline).
    best_taxon = models.ForeignKey(
        Taxon, on_delete=models.SET_NULL, null=True, blank=True, related_name="+",
    )
    best_confidence = models.FloatField(null=True, blank=True)
    individual_count = models.PositiveIntegerField(null=True, blank=True)

    class Meta:
        verbose_name_plural = "Activities"
        unique_together = ("device", "activity_uid")
        ordering = ["-started_at"]
        indexes = [models.Index(fields=["device", "-started_at"])]

    def __str__(self) -> str:
        return f"activity {self.activity_uid} on {self.device.name}"

    @property
    def frame_count(self) -> int:
        return self.frames.count()


class ActivityFrame(models.Model):
    """One sampled frame (a crop of the mover) for an activity."""

    class Kind(models.TextChoices):
        CROP = "crop", "Mover crop"
        WIDE = "wide", "Wide context"  # reserved for the deferred plant-ID phase

    activity = models.ForeignKey(Activity, on_delete=models.CASCADE, related_name="frames")
    kind = models.CharField(max_length=8, choices=Kind.choices, default=Kind.CROP)
    storage_key = models.CharField(max_length=500, help_text="S3 key in the raw-videos bucket.")
    # [x1, y1, x2, y2] normalized (0..1) in the source frame the crop came from.
    bbox = models.JSONField(null=True, blank=True)
    motion_score = models.FloatField(null=True, blank=True)
    width = models.PositiveIntegerField(null=True, blank=True)
    height = models.PositiveIntegerField(null=True, blank=True)
    captured_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["captured_at", "id"]

    def __str__(self) -> str:
        return f"frame {self.id} ({self.kind}) of {self.activity_id}"


class Detection(models.Model):
    """One model's identification output for one frame (Phase 1)."""

    class Model(models.TextChoices):
        BIOCLIP = "bioclip", "BioCLIP (insect)"
        PLANTNET = "plantnet", "Pl@ntNet (plant)"  # deferred (Phase 5)

    frame = models.ForeignKey(ActivityFrame, on_delete=models.CASCADE, related_name="detections")
    model = models.CharField(max_length=16, choices=Model.choices, default=Model.BIOCLIP)
    taxon = models.ForeignKey(
        Taxon, on_delete=models.SET_NULL, null=True, blank=True, related_name="detections",
    )
    confidence = models.FloatField(null=True, blank=True)
    raw = models.JSONField(default=dict, blank=True, help_text="Full ranked model output.")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-confidence"]
        # One result per (frame, model): re-running classification updates the
        # row in place instead of piling up duplicates that skew aggregation.
        constraints = [
            models.UniqueConstraint(fields=["frame", "model"],
                                    name="uniq_detection_per_frame_model"),
        ]

    def __str__(self) -> str:
        return f"{self.model} -> {self.taxon} ({self.confidence})"


class Observation(models.Model):
    """Per-activity aggregate identity + individual count (Phase 1)."""

    class Status(models.TextChoices):
        AUTO = "auto", "Auto (pipeline)"
        AGENT_REVIEWED = "agent_reviewed", "Agent-reviewed"
        HUMAN_CONFIRMED = "human_confirmed", "Human-confirmed"

    activity = models.ForeignKey(Activity, on_delete=models.CASCADE, related_name="observations")
    taxon = models.ForeignKey(
        Taxon, on_delete=models.SET_NULL, null=True, blank=True, related_name="observations",
    )
    confidence = models.FloatField(null=True, blank=True)
    individual_count = models.PositiveIntegerField(default=1)
    representative_frame = models.ForeignKey(
        ActivityFrame, on_delete=models.SET_NULL, null=True, blank=True, related_name="+",
    )
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.AUTO)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        # One observation per (activity, taxon): an activity can record more than
        # one taxon (co-occurrence), but not duplicate the same one — makes
        # update_or_create(activity, taxon) race-safe.
        constraints = [
            models.UniqueConstraint(fields=["activity", "taxon"],
                                    name="uniq_observation_per_activity_taxon"),
        ]

    def __str__(self) -> str:
        return f"observation {self.taxon} x{self.individual_count}"


class AgentConversation(models.Model):
    """A chat thread with the taxonomic monitoring agent (Phase 3)."""

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
        related_name="monitor_conversations",
    )
    device = models.ForeignKey(
        Device, on_delete=models.SET_NULL, null=True, blank=True, related_name="+",
        help_text="Device the conversation is scoped to, if any.",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"agent convo {self.pk} ({self.user})"


class AgentMessage(models.Model):
    """One turn in an AgentConversation. Content is secret-redacted before save."""

    conversation = models.ForeignKey(
        AgentConversation, on_delete=models.CASCADE, related_name="messages",
    )
    role = models.CharField(max_length=12)  # "user" | "assistant"
    content = models.TextField()
    meta = models.JSONField(default=dict, blank=True)  # e.g. {"tools": [...]}
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self) -> str:
        return f"{self.role} @ {self.created_at:%Y-%m-%d %H:%M}"


class DailyDigest(models.Model):
    """A per-device, per-day summary written by the monitoring agent (Phase 3).

    ``stats`` holds the structured data the summary was written from (taxa seen,
    new-for-site taxa, activity count, telemetry anomalies); ``summary`` is the
    narrated prose. One row per (device, date) — regenerating updates in place.
    """

    device = models.ForeignKey(Device, on_delete=models.CASCADE, related_name="digests")
    date = models.DateField(help_text="The UTC day summarized.")
    summary = models.TextField()
    stats = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-date", "device__name"]
        constraints = [
            models.UniqueConstraint(fields=["device", "date"], name="uniq_digest_per_device_day"),
        ]

    def __str__(self) -> str:
        return f"digest {self.device.name} {self.date}"
