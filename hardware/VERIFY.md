# BeeMonitor — verification runbook (device dashboard)

Step-by-step verification for the device dashboard: telemetry @60s + online
status, activity card/graph, on-demand photo, Sources-nav cleanup, and WittyPi
auto power-on.

Most checks are a **Pi command** plus **what to look for on the dashboard**.
Do **Step 0** first, then each feature in order.

The mechanical Pi-side steps (health curl, venv import check, service restart)
are also bundled in `hardware/verify.sh` — run that on the Pi instead of copying
commands by hand:

```bash
cd ~/BeeMonitor && git pull
./hardware/verify.sh           # runs Step 0 + the probeable checks
```

Two caveats up front:
- The dashboard camera card only works once the Pi is on the new code (Step 0).
- Video upload verification needs the Pi on **WiFi**. Telemetry and photos work on cellular.

---

## Step 0 — Deploy

Backend auto-deploys via CI (migrations run on container start). Confirm it's up:

```bash
curl -s -o /dev/null -w "%{http_code}\n" \
  https://mqnafc3ejc.us-east-1.awsapprunner.com/api/v1/health/   # expect 200
```

Pi — pull, sanity-check the venv, restart services:

```bash
cd ~/BeeMonitor && git pull
~/BeeMonitor/hardware/venv/bin/python -c "import picamera2, cv2, requests; print('venv ok')"
sudo systemctl restart cellular beemonitor-recorder beemonitor-telemetry beemonitor-uploader
journalctl -u beemonitor-telemetry -f        # leave open — your main verification window
```

---

## 1 — Telemetry JSON @60s + online status

- **Pi:** the telemetry log shows `beat: …` then `heartbeat ok` every ~60s, with no `image=`.
- **Dashboard** `/devices/` → device badge = **Online** (green).
- Stop telemetry (`sudo systemctl stop beemonitor-telemetry`), wait ~3 min → badge flips to
  **Offline**. Then restart it.

## 2 — Activity card + graph

- Device page → **Activity** card shows `N snippets / 1h` (the activity window is decoupled
  from the 60s beat).
- **Activity over time** graph fills in as beats accumulate (range buttons 24h/7d/30d/90d).
  It only has data from after this deploy.

## 3 — On-demand photo (fast path)

- Device page → **Camera** card → **Take photo**. The image appears within **~10–15s**
  (a lightweight command poll runs every ~8s, then the recorder captures + uploads — no
  need to wait for the 60s beat).
- **Pi log:** `command: capture_image`, the recorder writes a still, then `heartbeat ok`
  with `image=True`.
- If it doesn't appear: the device is offline, or the recorder isn't running (the recorder
  is what grabs the frame). Tune cadence with `BEEMONITOR_COMMAND_POLL_SECONDS`.

## 4 — Sources nav removed

- Top nav: **Sources** is gone; **Devices** present. ✅

## 5 — Health cards

- Device page shows **Storage**, **Uptime**, **CPU temp**, **Activity**, **Videos uploaded**,
  and a **Services** row with Recorder / Uploader / Cellular dots (green = the matching
  systemd service is active on the Pi).

## 6 — WittyPi auto power-on

- Set it (README §8.4: *Default state when powered = ON*), then pull power and reconnect →
  the Pi boots **without** pressing the button → services come up on their own.

---

> **Not in this build (removed for simplicity):** live view, WiFi LAN streaming, and the
> cellular-signal + GPS/Location widgets. On-demand **photo** is the camera feature; the
> Cellular **dot** under Services only reflects whether `cellular.service` is running, not
> signal strength.
