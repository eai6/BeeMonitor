# BeeMonitor — verification runbook (device dashboard v2)

Step-by-step verification for everything in the device dashboard v2 work
(telemetry @60s, cellular signal + GPS, activity card/graph, on-demand
photo, live view, WiFi LAN MJPEG stream, nav cleanup, WittyPi auto power-on).

Most checks are a **Pi command** plus **what to look for on the dashboard**.
Do **Step 0** first, then each feature in order.

The mechanical Pi-side steps (health curl, venv import check, service
restart, `:8090` stream probe) are also bundled in `hardware/verify.sh` —
run that on the Pi instead of copying commands by hand:

```bash
cd ~/BeeMonitor && git pull
./hardware/verify.sh           # runs Step 0 + the probeable checks
```

Two caveats up front:
- The dashboard image cards/links only work once the Pi is on the new code (Step 0).
- Video upload verification needs the Pi on **WiFi**. Telemetry and photos work on cellular.

---

## Step 0 — Deploy

Backend auto-deploys via CI (migrations 0003–0005 run on container start).
Confirm it's up:

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

## 2 — Cellular signal + GPS

```bash
cat /run/beemonitor/modem-status.json
# expect: {"ts":…, "rssi_dbm":-71, "lat":…, "lon":…, "fix":true}
```

- If `rssi_dbm` missing → qmicli issue.
- If `lat`/`lon` missing → no GNSS fix yet (go outside, wait minutes) **or** wrong AT port →
  set `BEEMONITOR_AT_PORT=/dev/ttyUSB2` in `/etc/beemonitor/uploader.env`, restart `cellular`.
- **Dashboard** device page → **Cellular** card shows dBm; **Location** card shows coords + "View on map".

## 3 — Activity card + graph

- Device page → **Activity** card shows `N snippets / 1h` (not `/60s`).
- **Activity over time** graph fills in as beats accumulate (range buttons 24h/7d/30d/90d).
  It only has data from after this deploy.

## 4 — On-demand photo

- Device page → **Camera** card → **Take photo**. Within ~60s the image appears.
- **Pi log:** `command: capture_image`, then recorder writes a still, then `heartbeat ok` with `image=True`.

## 5 — On-demand live view

- Camera → **Live view** → image refreshes (~1.5 fps) for ~60s.
- **Pi log:** repeated `command: stream` captures.

## 6 — WiFi live stream (5c) — the experimental one

- Camera → **WiFi stream** → wait ~1 beat → a "● Live LAN stream — open ↗" link appears.
- **Pi log:** `command: wifi_stream`, `recorder: mjpeg stream server listening on :8090`.
- Quick local check on the Pi:

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8090/stream.mjpg   # expect 200
```

- From a laptop on the same WiFi: open `http://<pi-lan-ip>:8090/` → live video.
  (If the link's IP is wrong, `_lan_ip()` picked the wrong interface — flag it.)

## 7 — Sources nav removed

- Top nav: **Sources** is gone; **Devices** present. ✅

## 8 — WittyPi auto power-on

- Set it (README §8.4), then pull power and reconnect → the Pi boots without pressing the
  button → services come up on their own.
