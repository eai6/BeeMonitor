# BeeMonitor — Remote Wake Scheduling Design

**Status:** Proposal for review (not yet implemented)
**Author:** Drafted with Claude Code, 2026-06-12
**Goal:** Set a field unit's **WittyPi wake schedule from the dashboard** — including
an **"always-on"** mode for solar+battery/mains units — applied on the device's next
check-in. Safe by construction: a bad schedule can never permanently strand a unit.

---

## 1. Motivation & the hard constraint

A powered-off Pi is unreachable (no OS, and the cellular modem is powered from the
Pi, so it's off too). The WittyPi 4 has **no network** — it only wakes the Pi on its
RTC schedule, the button, or an external pin. So you cannot *push* a wake to an off
device. What you *can* do is **reprogram when it will be on**: the server stores a
desired schedule; the device, while on, writes it into the WittyPi; future power
cycles follow it. "Remote wake" → "remote re-schedule." (True instant on-demand wake
needs an always-on cellular listener wired to the WittyPi wake pin — out of scope
here, a separate hardware path.)

Responsiveness is bounded by how often the device wakes to poll — the **check-in
cadence is the dial** between battery life and how fast you can re-task it.

Relates to [[10_cellular_telemetry_design]] (the device already reports its schedule
window; "off as planned" ≠ "died") and [[12_device_dashboard_telemetry_v2]] (the
command channel + reconcile patterns).

---

## 2. How it fits the existing system

- **Config push via the heartbeat response.** The heartbeat/command response in
  `apps/api/heartbeat.py` already carries device config the Pi reconciles to each
  beat — `telemetry_interval`, `motion_tuning`, `roi_override`, `nest_layout`. Add
  `wake_schedule` the same way: the desired spec rides every response; the device
  applies it on boot/change and self-heals. (Cleaner than a one-shot
  `pending_command`, which clears after pickup.)
- **Reconcile pattern.** Desired stored on `Device`; device reports the *active*
  schedule in telemetry; the dashboard shows desired-vs-active — exactly how
  `motion_tuning` / `telemetry_interval` already work (`DeviceTelemetryRateView`,
  `_apply_motion_tuning`).
- **Device-side handlers.** `telemetry.py` already has `_apply_interval`,
  `_apply_motion_tuning`, `_apply_override_file`. Add `_apply_schedule`.
- **WittyPi.** The schedule lives on the WittyPi (RTC alarms) and survives the Pi
  being off; the Pi writes it while on via UUGear's `wittyPi.sh` / `runScript.sh`
  (a `.wpi` ON/OFF-block script, or direct next startup/shutdown alarms). The Pi
  already reports `schedule_window` in metrics.

---

## 3. Schedule modes (the dashboard menu)

A single `wake_schedule` JSON: `{mode, ...params}`. Times are in the device's local
tz (it knows tz from GPS).

| Mode | Params | Use |
|---|---|---|
| `daylight` (default) | — | On during the GPS/sun window (current behaviour). Bees are diurnal, so this captures ~all activity at min power. |
| `window` | `on`, `off` (clock), optional `night_checkins:[{at,minutes}]` | Explicit on/off; the brief night check-ins are the cheap reachability knob. |
| `interval` | `wake_every_min`, `on_minutes` | Wake every N min for M — the responsiveness dial. |
| `always_on` | — | **No scheduled off** (solar+battery once the power budget allows, or mains). Dissolves the remote-wake problem entirely. WittyPi low-voltage cutoff stays active. |
| `one_shot` (later) | `at` | Boot once at a time, then resume the base schedule. |

---

## 4. Data model & flow

- **`Device.wake_schedule`** (JSONField) — the *desired* spec. Default
  `{"mode":"daylight"}`.
- **Heartbeat response** includes `wake_schedule` (desired). Device applies it.
- **Telemetry metrics** report `active_schedule` (what's actually programmed on the
  WittyPi) + next wake time, so the server confirms desired==active.
- **`DeviceScheduleView`** (POST, manager+) sets the schedule from the device page —
  mirror `DeviceTelemetryRateView`. UI: mode presets + params + a desired-vs-active
  "confirmed/pending" badge.

Flow: dashboard sets `wake_schedule` → next beat's response carries it → device
validates + writes it to the WittyPi → next cycle follows it → device reports
`active_schedule` → dashboard shows confirmed.

---

## 5. Safety — the central risk

A bad schedule can leave a unit off forever, and you **can't fix it remotely** (only
the WittyPi's own recovery or a physical button in the field). Guardrails are
enforced **on the device**, not trusted from the server:

- **Validate + clamp locally.** Reject malformed specs; clamp any off-duration to a
  hard `MAX_OFF_HOURS` floor (e.g. 24h) so a fat-fingered schedule can't strand it.
- **Guaranteed wake floor.** The device always programs at least one wake within
  `MAX_OFF_HOURS`, regardless of the spec.
- **Confirm-or-revert watchdog.** Keep the last known-good schedule; apply the new
  one; if the server hasn't confirmed receipt by the next guaranteed wake, revert.
- **Keep WittyPi low-voltage protection ON in every mode** — "always_on" means "no
  *scheduled* off," not "no power protection." Lean on WittyPi's low-voltage
  recovery startup as the ultimate backstop.
- **Never fully disable the WittyPi** (keep RTC + cutoff).

---

## 6. Phased plan

- **Phase 1 — server (safe, testable now).** `Device.wake_schedule` + migration;
  `DeviceScheduleView` + dashboard schedule editor (mode presets); include
  `wake_schedule` in the heartbeat/command response; desired-vs-active reconcile UI
  (device reports `active_schedule`). The device ignores the unknown field for now,
  so this ships with zero field risk. Tests.
- **Phase 2 — device (needs on-Pi validation).** `telemetry.py` `_apply_schedule`:
  translate the spec → WittyPi `.wpi`/alarms, apply via `wittyPi.sh`, with all of
  §5's guardrails; report `active_schedule`. This is the part that genuinely needs
  hardware testing (WittyPi schedule format, the wake-floor logic).
- **Phase 3 — polish.** `one_shot` wake, `night_checkins` pulses, the
  confirm-or-revert watchdog, and the `always_on` mode wired end-to-end (for the
  solar+battery / mains future).

Build Phase 1 first — it's safe, fully testable in Django, and lets the dashboard
drive a schedule the moment the Pi side lands.

---

## 7. Risks / open questions

- **Stranding** is the dominant risk → §5 guardrails are the whole point; the
  wake-floor + confirm-or-revert + WittyPi recovery are non-negotiable.
- **Power** for `always_on` — needs the panel/battery budget (the operator's call;
  the future solar+battery use case). The device can't enforce that; surface a
  warning on the dashboard when selecting it.
- **WittyPi schedule format** — must be validated on real WittyPi 4 hardware
  (`.wpi` script vs. direct alarm-setting; how `runScript.sh` reschedules).
- **Timezone** — schedule clock times are device-local; the device already resolves
  tz from GPS.
- A future **Option B** (always-on cellular listener → WittyPi wake pin) is the only
  path to *instant* on-demand wake; this design deliberately covers the
  no-new-hardware 95%.
