---
name: video-device-deletion-two-key
description: "A clip is deleted from a field device ONLY after it's uploaded AND a human clicks \"Delete from device\" on the dashboard. Two-key gate, never auto. Server delete is separate."
metadata: 
  node_type: memory
  type: project
  originSessionId: 416c68fc-5615-46ab-98ac-9e38009bdaf2
---

**Hard safety rule (user requirement):** the device may free a local clip from its SD card ONLY when BOTH keys are present:
1. it uploaded successfully — proven by the `<clip>.mp4.uploaded` sidecar (records `video_id=`) and a `Video` row, AND
2. a human explicitly clicked **"Delete from device"** on the dashboard (sets `Video.device_delete_requested`).
The device NEVER self-prunes (no auto-delete on disk pressure / age). A clip with no `.uploaded` sidecar is never eligible. (Unit-tested.)

**Flow / where things live** (shipped a6bcb15 device+API, ed856f3 dashboard):
- Model `apps/videos/models.py`: `device_delete_requested` (human gate) + `device_deleted_at` (stamped on device confirmation). Migration 0006.
- API `apps/api/cleanup.py` (`GET/POST /api/v1/devices/cleanup`, device-auth): GET lists this device's cleared-but-not-yet-deleted video_ids; POST records the ids the device deleted. Scoped to the calling device.
- Device `hardware/telemetry.py` `_run_cleanup()`: runs over cellular (telemetry cgroup) every `BEEMONITOR_CLEANUP_INTERVAL` (600s) — GET cleared ids, match via sidecar `video_id`, delete `.mp4` + sidecar, POST confirm, log freed bytes.
- Dashboard: video detail "Device storage" card (Delete from device / Cancel / Freed) + list-page bulk "Delete from Device". `VideoDeviceDeleteView` + `VideoBatchDeviceDeleteView`.

**Distinct from server deletion:** "Delete from device" keeps the S3 + DB copy. Deleting the server copy is the separate, pre-existing `VideoDeleteView`/`VideoBatchDeleteView` (removes S3 objects + row).

Web changes need an App Runner deploy + `migrate` to go live (the Pi pulls code itself; the web service does not). Rides the same command/auth pattern as [[dashboard-wifi-needs-nmcli-sudoers]] and the cellular firewall ([[cellular-firewall-two-phase]]).
