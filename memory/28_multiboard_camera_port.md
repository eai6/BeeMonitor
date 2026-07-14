# Multi-board Support: Camera Abstraction + Orange Pi Zero 2W Port

**Status:** PLANNED (2026-07-14). Do later.
**Goal:** Let the hardware stack run on non-Raspberry-Pi boards (first target:
Orange Pi Zero 2W, Allwinner H618, ARM64 quad Cortex-A53, 1.5–4 GB) without
forking the recorder, by putting the camera behind a backend interface.

---

## 1. Why this is needed

The field stack is Raspberry-Pi-coupled in exactly two subsystems; everything
else already ports to Debian/Ubuntu ARM64:

| Subsystem | Pi-coupling | Ports as-is? |
|---|---|---|
| **Camera capture** (`hardware/motion/recorder.py`) | `picamera2` + `libcamera`, dual main+lores stream, `H264Encoder` + `CircularOutput` pre-roll ring | ❌ Pi-only — the blocker |
| **Power / battery** (`hardware/telemetry.py:179+`) | UUGear **WittyPi** HAT over i2c (`utilities.sh`) | ❌ Pi HAT; degrades gracefully (read path returns nothing, apply gated off) |
| Uploader, telemetry, WiFi (`nmcli`), cellular (qmi/udhcpc), YOLO bee-confirm (CPU), serial (`/proc/device-tree/serial-number`), Tailscale remote access | distro-level | ✅ works |

Orange Pi Zero 2W is **ARM64** (unlike the ARMv6 RPi Zero/Zero W we don't
support), so Python + ultralytics wheels install fine. The only real work is the
camera; power management is drop-or-replace. NOTE: this is a DIFFERENT board from
`project_pi_zero2w_lite` (that's the Raspberry Pi Zero 2 W / Broadcom) — do not
conflate.

## 2. Camera touchpoints to abstract (from recorder.py)

The recorder uses picamera2 in ~6 places, all in `record()`:
- dual-stream config: `main` (record res) + `lores` (motion-detect res, YUV420)
- `H264Encoder(repeat=True, iperiod=FPS)` → `CircularOutput(buffersize=PRE_ROLL*fps)`
  — the ring buffer is what gives **pre-roll** (footage before the trigger)
- `cam.capture_buffer("lores")` each loop → grayscale motion frame (paced at fps)
- `cam.capture_array("main")` → full-res BGR still (crops + telemetry image)
- `cam.pre_callback` → timestamp burn-in overlay
- `circ.fileoutput = path; circ.start()/.stop()` → open/close each clip (h264 → remux)

Everything else in `record()` (motion gate, bee confirmation, crop sampling,
clip state machine, activity-frame archive, telemetry stills, hot-reload of all
the dashboard settings) is board-agnostic and MUST stay untouched.

## 3. Design: `CameraBackend` interface

One interface the recorder talks to; pick the backend by board at startup.

```python
class CameraBackend(Protocol):
    def start(self) -> None: ...
    def read_lores_gray(self) -> "np.ndarray":   # blocks ~1 frame → paces loop at fps
    def capture_main_bgr(self) -> "np.ndarray":  # full-res still, on demand
    def open_clip(self, h264_path: Path) -> None:  # begin recording, incl. pre-roll
    def close_clip(self) -> None:                  # stop; file ready to remux
    def set_overlay(self, fn) -> None: ...         # optional timestamp burn-in
    def stop(self) -> None: ...
```

- **`PiCamera2Backend`** — wraps today's code verbatim. **Zero behaviour change on
  real Pis** (the whole point: refactor is safe to ship to the fleet first).
- **`V4L2Backend`** (Orange Pi / generic Linux) — GStreamer (preferred) or ffmpeg:
  - `v4l2src` → `tee`: one branch a low-res `appsink` for `read_lores_gray`
    (or downscale in-pipeline), one branch H.264 encode for clips.
  - **Pre-roll** = a bounded `queue`/ring ahead of the muxer that `open_clip`
    flushes into the file (the hard part — see §5). Alt: ffmpeg segment muxing
    stitched on close.
  - `capture_main_bgr` = a second full-res `appsink` grab (or a snapshot valve).
  - Encoder: try Allwinner H.618 hardware H.264 via V4L2 M2M; fall back to
    software x264 (fine for 640×480 lores + moderate main on a quad A53 — MUST
    be load-tested).

Backend selection: `BEEMONITOR_CAMERA_BACKEND=picamera2|v4l2` env, default
auto-detect (picamera2 import succeeds → Pi; else v4l2).

## 4. Power management on non-Pi

WittyPi is a Pi HAT. Options (out of scope for the camera work, decide later):
- **Drop scheduled sleep/wake + battery telemetry** — telemetry already no-ops
  when no WittyPi is found (read path returns {}, apply gated off), so nothing
  crashes; you lose solar duty-cycling.
- **Replace** with an Orange-Pi-compatible RTC + power board behind a similar
  small abstraction (a `PowerBackend` returning voltage/current + arming
  wake/shutdown). Lower priority than the camera.

## 5. Risks / unknowns to validate on real hardware

1. **Pre-roll ring on GStreamer** — reproducing `CircularOutput`'s N-second
   look-back cleanly. Prototype first; it decides the V4L2Backend shape.
2. **H.618 H.264 encode path** — hardware M2M availability + quality; software
   x264 CPU headroom on the A53 at the target res/fps.
3. **libcamera on Allwinner** — immature; that's WHY we go V4L2/GStreamer, not
   picamera2, on the Orange Pi.
4. **Boot/provisioning paths** — `/boot/firmware/beemonitor.conf`,
   `prepare-card.sh`, golden image are Pi-layout; enrollment needs path fixups.

## 6. Rollout plan

1. **Refactor only (safe, ship to fleet):** extract `CameraBackend`, move
   today's picamera2 code into `PiCamera2Backend`, make `record()` use the
   interface. Verify on real Pis — behaviour identical. No V4L2 yet.
2. **V4L2Backend prototype:** stand up the GStreamer pipeline on an Orange Pi;
   nail pre-roll + encoder first in isolation before wiring into `record()`.
3. **Port fixups:** boot-config paths, serial format, drop/replace WittyPi.
4. **Field test** one Orange Pi unit end-to-end (record → motion → upload →
   telemetry) before any lite-profile / golden-image work.

**First step is low-risk and independently valuable:** the interface refactor
(step 1) improves the codebase and is testable on existing Pis, with no V4L2
dependency. Do that whenever we pick this up; steps 2–4 are the Orange Pi bring-up.
