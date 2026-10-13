"""The software update a newly enrolled card gets on its first check-in.

Every card is flashed from a golden image, and the image ages: a new unit used
to run the image's code until someone pressed Update. On an image older than
camera detection that meant a 64MP Arducam never came up and the recorder
crash-looped (IndexError from Picamera2 with no camera). Enrollment now sets
Device.update_after_enroll, and the first heartbeat queues the update here.

A unit says how it is installed by what it reports as its version: a git
clone reports a short commit SHA, a release (signed artifact) install reports
the release version. Each gets the update its updater can apply.
"""
from __future__ import annotations

import re

SHA = re.compile(r"^[0-9a-f]{7,40}$")


def params_for(reported: str) -> dict | None:
    """The update command's params for a unit reporting ``reported``, or None
    when there is nothing to do (a release install already on the latest)."""
    reported = (reported or "").strip()
    if not reported or SHA.match(reported):
        # A git clone (or a unit too old to say): latest main.
        return {"ref": "origin/main"}
    from apps.devices.views import _resolve_edge_descriptor
    descriptor = _resolve_edge_descriptor()
    if not descriptor or descriptor["version"] == reported:
        return None
    return descriptor


def queue_if_due(device, metrics: dict) -> bool:
    """On a newly enrolled unit's first beat, queue its update. One-shot: the
    flag is cleared whether or not an update was needed. Waits while another
    command is queued, so it never replaces one."""
    if not device.update_after_enroll or device.pending_command:
        return False
    from django.utils import timezone

    params = params_for(metrics.get("code_commit"))
    device.update_after_enroll = False
    fields = ["update_after_enroll"]
    if params:
        device.pending_command = "update"
        device.command_params = params
        fields += ["pending_command", "command_params"]
        if params.get("version"):
            # A release install shows "Updating → <version>" until it lands.
            device.update_target = params["version"]
            device.update_requested_at = timezone.now()
            fields += ["update_target", "update_requested_at"]
    device.save(update_fields=fields)
    return bool(params)
