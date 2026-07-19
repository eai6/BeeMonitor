# 29 — Device update: cancel action (now) + minisign artifact fix (parked)

## Context

Field devices (Raspberry-Pi-class) update two ways:
- **git-update** (`git fetch origin`) — needs GitHub credentials the field units don't have → dead end in the field.
- **artifact-update** — device downloads a signed hardware-only bundle from S3 via a **presigned URL** (no device credentials), verifies **sha256 + minisign signature**, unpacks to `releases/<version>/`.

Symptom that prompted this: units (e.g. Ethan, Adeg) stuck showing "Updating → <version>" because their artifact updates fail on the device at the minisign step, and there is currently **no way to cancel** a queued/stuck update. The publish side is healthy (the v0.1.6 bundle is signed and in `s3://…/edge/`), so this is NOT an S3/access problem.

Two parts:
1. **Implement now:** a manager "Cancel update" action (per-device + fleet) to clear stuck "Updating →" pills.
2. **Parked (needs a new golden image):** the minisign fix — documented below for later.

---

# PART 1 — Cancel update (implement now)

## Scope of "cancel"
The update command is **one-shot**: `apps/api/heartbeat.py:180-186` clears `pending_command` the moment it's handed to the device. The persistent "Updating →" pill is driven purely by `update_target` (`apps/devices/views.py:279-280`; heartbeat stale-clear at `apps/api/heartbeat.py:137-145`). There is **no device-side abort** in `hardware/update.sh` / `hardware/telemetry.py`.

So cancel = clear the web-side state. It fully cancels a **still-queued** update (device hasn't checked in since queuing) and **removes the stuck pill** in all cases. It does NOT stop an update already downloading/applying on the device (that runs to completion and auto-rolls-back if unhealthy). **UI copy must say this** — e.g. "Cancel queued update — won't stop an update already downloading on the device."

## Fields to clear (the cancel write)
On the Device model (`apps/devices/models.py:255-264`):
- `update_target = ""`
- `update_requested_at = None`
- `command_params = {}`
- `pending_command = ""` — **only if** it currently equals `"update"` (guard so we don't clobber another queued command)

## Changes

### 1. Views — `apps/devices/views.py`
- **`DeviceCancelUpdateView`** (per-device). Mirror `DeviceWifiScanView` (971-980) / `DeviceUpdateView` (1128-1173). `_device_or_403(request.user, pk, "manager")`, apply the cancel write, `save(update_fields=[...])`. Dual response like `DeviceTelemetryRateView` (line 1224): AJAX (`X-Requested-With == XMLHttpRequest` → `JsonResponse({"ok": True})`) and form-post (`messages.success` + `redirect("devices:detail", pk=pk)`).
- **`DeviceFleetCancelUpdateView`** (fleet). Mirror `DeviceFleetUpdateView` (1078-1125): read `device_ids` from POST/JSON body, filter to `Device.accessible(request.user)` where `is_active and can(user,"manager")`, `bulk_update` the four fields, return `JsonResponse({"ok": True, "cancelled": N, "skipped": M})`.

### 2. URLs — `apps/devices/urls.py`
- `path("<int:pk>/update/cancel/", views.DeviceCancelUpdateView.as_view(), name="cancel_update")`
- `path("fleet/update/cancel/", views.DeviceFleetCancelUpdateView.as_view(), name="fleet_cancel_update")`

### 3. List template — `apps/devices/templates/devices/list.html`
- **Per-row Cancel:** inside the `{% if d.updating %}` block (155-160), after line 160, a small "Cancel" button next to the "Updating →" pill (AJAX POST to `devices:cancel_update` with the row pk, then `location.reload()`).
- **Fleet "Cancel updates" button:** in the action row near `#fleet-update-btn` (92-101). Clone the fleet-update click handler (261-297) — confirm → POST selected `device_ids` (JSON) to `devices:fleet_cancel_update` with the same CSRF/`X-Requested-With` headers → on `ok`, `setTimeout(location.reload, 1200)`. Reuse existing `selectedIds()`/checkbox helpers. Operates on **selected** rows (user checks the stuck ones).

### 4. Detail template — `apps/devices/templates/devices/detail.html`
- In the Software card `{% if can_manage %}` button row (924-945), after line 943, a Cancel `<form method="post" action="{% url 'devices:cancel_update' device.pk %}">` wrapped in `{% if device.update_target %}` so it only shows while an update is pending. (`metrics.update` already shows device-side status at 905-915.)

## Verification
- Local Django not bootable in the dev sandbox (missing `rest_framework`); `ast.parse` the edited `.py` and rely on deploy-preview.
- Functional (after deploy): queue an artifact update on a test device → "Updating →" appears → click **Cancel** → pill clears (per-device and fleet). Non-manager gets 403. Cancel with no pending update = harmless no-op.
- Real stuck units: fleet Cancel on Ethan + Adeg to clear their pills.

---

# PART 2 — Minisign artifact fix (PARKED — needs a golden image)

**Do not implement until a new golden image is being cut.**

## Root cause (confirmed)
The vendored static `minisign` verifier binary **was never built or committed**. `.github/workflows/build-minisign.yml` is a manual `workflow_dispatch` that has never run, so `hardware/provision/minisign-<arch>` does not exist. On a cellular field unit `hardware/update.sh` finds no verifier:
- system `minisign` (`update.sh:58`) → not installed
- vendored `hardware/provision/minisign-$(uname -m)` (`update.sh:59-62`) → file missing
- `apt install minisign` (`provision.sh:56-57`) → blocked by the cellular firewall

→ `cmd_fetch_artifact` bails at `update.sh:181` ("minisign unavailable") before it even downloads. The pubkey (`hardware/provision/minisign.pub`, key `53844D71B2989AE1`) is present and fine; the **binary** is missing. The bundle can't carry its own verifier (excluded at `build-edge-artifact.sh:36` + the hardware-only guardrail). See also `memory/18_edge_artifact_delivery_design.md`.

## Fix (proposed, for later)
1. **Build + commit the static binaries** — run/adjust `build-minisign.yml` to produce `hardware/provision/minisign-aarch64` (plus a 32-bit `armv7l`/`armv6l` binary if any 32-bit units exist — fleet ARCH TBD). This is the actual missing artifact.
2. **Bake verifier + pubkey into the golden image** — `prepare-card.sh` / `generalize.sh` currently have zero minisign references. Install `/usr/local/bin/minisign` + `/home/beemonitor/minisign.pub` at image time so **new flashes verify standalone**. (See `memory/14_golden_image_provisioning_design.md`.)
3. **(Optional, fully standalone) hash-pinned S3 bootstrap** — teach `update.sh` to fetch the static verifier over the same cellular-allowed S3 channel as the bundle, pinned by a sha256 committed in `update.sh`. Only path that lets an *already-deployed* unit self-heal without git/apt/WiFi — but takes effect only once the new `update.sh` is on the device.

## Consequence for currently-stuck units
Until Part 2 ships and reaches them, field units **cannot complete an artifact update**. Each needs a **one-time touch** — WiFi + `apt install minisign`, or open the cellular firewall (`cellular_open`) + rpi-connect SSH to drop the binary — after which artifact updates work. For now, just **Cancel** their stuck updates (Part 1).

## Open decisions (when unparking)
- Device architectures in the fleet (which binaries to build).
- Whether to include the S3 hash-pinned bootstrap (fully standalone) or stop at binary-commit + golden-image.
