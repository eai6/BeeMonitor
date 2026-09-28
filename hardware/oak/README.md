# Luxonis OAK (USB camera)

The recorder uses a Luxonis OAK whenever one is plugged in, and the ribbon
camera otherwise. Tested on an **OAK-1-AF** (12 MP IMX378, autofocus, RVC2)
with depthai 3.10.0 on a Pi 4.

## Setup (once per unit, over WiFi)

```bash
# 1. let the recorder's user open the device (provision.sh also does this)
sudo cp hardware/oak/80-movidius.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger --attr-match=idVendor=03e7

# 2. depthai into the recorder's venv (~150 MB installed)
hardware/venv/bin/pip install depthai==3.10.0

# 3. check it
hardware/venv/bin/python hardware/oak/oak_probe.py --out /tmp/oak
sudo systemctl restart beemonitor-recorder
journalctl -u beemonitor-recorder -n 20 | grep -i oak   # "camera: OAK OAK-1-AF (imx378) over USB SUPER"
```

depthai is deliberately **not** in `hardware/requirements.txt`: `update.sh`
reinstalls that file over cellular, and depthai is too big to push to every
unit that way. Install it like torch — once, over WiFi.

Use a **blue (USB 3) port**. It works over USB 2 too, since the video arrives
already encoded, but the probe will report `HIGH` instead of `SUPER`.

## How it records

The OAK encodes H.264 on its own chip; the Pi only receives the bitstream, a
640x480 motion stream and, when asked, a full-size JPEG. Pre-roll,
the motion gate, remux, crops, dashboard pictures and hotel detection all work
as they do on the ribbon cameras. See `hardware/motion/oak.py`.

Not supported on the OAK:
- **flips** — the OAK records as mounted; mount it the right way up. A flip in
  `camera.json` is logged and ignored.
- **timestamp burn-in** — the clip's filename carries its start time.
- **runFocus.py** — Pi-camera only. The OAK autofocuses once at startup and
  holds; to pin the lens, set `"oak_lens": 0..255` in `camera.json` or
  `BEEMONITOR_OAK_LENS`.

## Resolution

By default the OAK records the **whole 12 MP sensor, 4032x3040 H.264 at 20 fps**
(~13.6 Mbit/s, ~95 MB per minute of activity — about twice a 1080p25 clip).
Measured on the OAK-1-AF, what its encoder can and cannot do:

| Recording | Result |
|---|---|
| 4056x3040 (native) @ 25, H.264 or H.265 | encoder "out of resources" |
| 4032x3040 @ 25 | encoder "out of resources" |
| **4032x3040 @ 20** | **sustained — the default** |
| 3840x2160 @ 25 | sustained, but a 16:9 crop: loses the top and bottom |
| 1920x1080 @ 25 | sustained, ~7 Mbit/s |

## Photos

Photos — periodic, "Take one now", and the burst on each motion trigger — are
the **same 4032x3040 as the video**, ~4 MB each. The OAK JPEG-encodes a frame
of the running video on request, so unlike the 64 MP ribbon camera **the video
never pauses**: measured, H.264 holds 20.0 fps through a 5-photo burst, which
takes ~0.3 s, and the clip after a burst keeps its pre-roll. Crops, dashboard
pictures and hotel detection use the same full-size frames.

## Settings (env, e.g. /etc/beemonitor/uploader.env)

| Variable | Default | |
|---|---|---|
| `BEEMONITOR_CAMERA` | `auto` | `auto` / `oak` / `picamera2` |
| `BEEMONITOR_OAK_MAIN_W` / `_H` | `4032` x `3040` | the whole 12 MP sensor — the largest the encoder takes |
| `BEEMONITOR_OAK_FPS` | `20` | the fastest 12 MP sustains; the motion stream runs at it too |
| `BEEMONITOR_OAK_BITRATE_KBPS` | encoder's choice | |
| `BEEMONITOR_OAK_LENS` | autofocus | 0..255 |

## Boot-time detection

`hardware/camera-autodetect.sh` reports the OAK. When one is on USB and no
ribbon camera enumerates, it no longer edits `config.txt` or reboots to go
looking for one — the unit already has its camera.
