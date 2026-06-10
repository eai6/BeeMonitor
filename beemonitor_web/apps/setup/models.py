"""Persistent state for the guided walkthrough and the AI assistant.

The walkthrough is treated as a stateful, resumable process (field setup gets
interrupted), and the assistant keeps multi-turn conversation history so we can
resend it to the Claude API each turn (the API is stateless).
"""

from django.conf import settings
from django.db import models

from apps.devices.models import Device


class SetupSession(models.Model):
    """One person's progress building one device.

    Keyed by (user, device) so it resumes across browsers/phones. ``device`` is
    nullable for a generic browse-only run started before a device is created.
    """

    class UnitType(models.TextChoices):
        UNSET = "unset", "Not chosen yet"
        WIFI = "wifi", "WiFi / bench unit"
        CELLULAR = "cellular", "Cellular field unit"

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
        related_name="setup_sessions",
    )
    device = models.ForeignKey(
        Device, on_delete=models.CASCADE, null=True, blank=True,
        related_name="setup_sessions",
    )
    unit_type = models.CharField(
        max_length=12, choices=UnitType.choices, default=UnitType.UNSET,
    )
    # The step the user is currently on (a content.STEPS id); "" before start.
    current_step = models.CharField(max_length=64, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("user", "device")
        ordering = ["-updated_at"]

    def __str__(self) -> str:
        who = self.device.name if self.device else "(no device)"
        return f"setup {who} for {self.user} [{self.unit_type}]"


class SetupStepState(models.Model):
    """Per-step status within a session (drives resume + the stepper UI)."""

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        ACTIVE = "active", "Active"
        PASSED = "passed", "Passed"
        FAILED = "failed", "Failed"
        SKIPPED = "skipped", "Skipped"

    session = models.ForeignKey(
        SetupSession, on_delete=models.CASCADE, related_name="step_states",
    )
    step_id = models.CharField(max_length=64)
    status = models.CharField(
        max_length=12, choices=Status.choices, default=Status.PENDING,
    )
    last_checked_at = models.DateTimeField(null=True, blank=True)
    detail = models.CharField(max_length=300, blank=True, default="")

    class Meta:
        unique_together = ("session", "step_id")
        ordering = ["id"]

    def __str__(self) -> str:
        return f"{self.step_id}={self.status}"


class AssistantConversation(models.Model):
    """A chat thread between a user and the setup assistant.

    Optionally bound to a device + the setup step it started on, so the
    assistant can be fed the user's real context. History is reconstructed from
    AssistantMessage rows on every turn.
    """

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
        related_name="assistant_conversations",
    )
    device = models.ForeignKey(
        Device, on_delete=models.SET_NULL, null=True, blank=True,
        related_name="assistant_conversations",
    )
    step_id = models.CharField(max_length=64, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]

    def __str__(self) -> str:
        return f"conversation #{self.pk} ({self.user})"


class AssistantMessage(models.Model):
    """One turn in a conversation. Secrets are redacted before persistence."""

    class Role(models.TextChoices):
        USER = "user", "User"
        ASSISTANT = "assistant", "Assistant"

    conversation = models.ForeignKey(
        AssistantConversation, on_delete=models.CASCADE, related_name="messages",
    )
    role = models.CharField(max_length=12, choices=Role.choices)
    content = models.TextField()
    # Optional structured trace (tools called, citations) for the UI/debugging.
    meta = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["id"]

    def __str__(self) -> str:
        return f"{self.role}: {self.content[:40]}"
