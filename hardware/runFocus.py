#!/usr/bin/env python3
"""
BeeMonitor — camera aim & focus tool for the Arducam 64MP OwlSight (OV64A40).

WHAT THIS IS: a desktop (Tk) window showing live video from the camera next to a
live focus readout, with buttons and a lens slider. Run it on the Pi's own screen
(or any X/VNC session) while you point the camera at the bee hotel.

WHY A GUI AND NOT A RING: the OwlSight has no focus ring — it focuses
electronically via the LensPosition control (libcamera's rpi.af algorithm, which
ov64a40.json ships). So aiming means watching the picture, and focusing means
driving a number. This window does both at once.

CONTROLS
  Autofocus now (f)      one-shot AF, reports where it landed
  Continuous AF (c)      let the AF algorithm keep refocusing
  lens slider / ← →      manual LensPosition (switches AF to manual)
  Run sweep (s)          rack the lens across a range, report the sharpest spot
  Zoom 1:1 (z)           show the centre of the frame at sensor pixels — the only
                         honest way to judge fine focus on a scaled-down preview
  turn 90° (o)           rotate the preview, if the picture is not upright
  Save for recorder (w)  write the focus + orientation into camera.json, which
                         the recorder applies at startup — do this when you are
                         happy, or the service keeps its old focus
  Snapshot (space)       full-resolution still into --outdir
  Record (r)             mp4 of --record-seconds into --outdir
  Reset peak (p)         forget the highest sharpness seen so far
  Quit (q / Esc)

ORIENTATION: the camera is mounted upside down, so the picture needs a 180°
turn. The ISP does that as hflip+vflip, which is free and applies to everything
— the recorder's video included. Press 'o' if the picture still isn't upright;
a 90/270 turn can only be done in software here (the ISP cannot transpose), so
the recorder cannot follow and will tell you so. Both come from the shared
camera profile, motion/camera.py.

FOCUS THAT STICKS: 'Save for recorder' (w) writes the lens position into
camera.json, and motion/recorder.py applies it at startup. Without a saved
value the recorder autofocuses once when it starts and holds that.

FOCUS METRIC: variance of the Laplacian over the lores luma plane. Higher is
sharper. The absolute value is scene-dependent (a blank wall scores low however
well focused), so only compare readings of the *same* scene — which is exactly
what the sweep does.

LENS POSITION is in dioptres = 1 / distance_in_metres:
    0.0 = infinity    1.0 = 1 m    4.0 = 25 cm    12.0 = 8.3 cm
ov64a40.json tunes 'normal' to 0-12 and 'macro' to 3-15. A bee hotel filling the
frame usually lands in the 3-10 range.

Usage (must be the venv interpreter — it has Pillow's ImageTk):
  hardware/venv/bin/python hardware/runFocus.py              # GUI
  hardware/venv/bin/python hardware/runFocus.py --sweep      # GUI, sweep on start
  hardware/venv/bin/python hardware/runFocus.py --lens 5.5   # GUI, pinned lens
  hardware/venv/bin/python hardware/runFocus.py --no-gui     # terminal only (SSH)

Over SSH there is no window to draw into, so use --no-gui: same focus numbers,
same --sweep, printed to the terminal.

The recorder holds the camera exclusively, so stop it first:
    sudo systemctl stop beemonitor-recorder
    ...
    sudo systemctl start beemonitor-recorder
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from libcamera import controls
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput

# Match the recorder so what you focus and frame here is what actually gets
# recorded. Sizes come from the shared config; the flip mirrors recorder.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from motion import camera as camera_profile
from motion.config import MAIN_W, MAIN_H, LORES_W, LORES_H, FPS, CAMERA_FILE

PROFILE = camera_profile.load_profile()
ROTATE = PROFILE["rotate"]
# Same flips the recorder applies — what you frame here is what it records.
TRANSFORM = camera_profile.transform(PROFILE)
OUT_DIR = Path("/home/beemonitor/Desktop/cameraOutput/cameraTesting")

# --rotate is applied in software, on top of TRANSFORM's flips. It has to be:
# the Pi's ISP can flip but not transpose, so libcamera cannot give us a 90/270
# rotation at capture time however we configure it.
ROTATIONS = {0: None, 90: cv2.ROTATE_90_CLOCKWISE,
             180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}

AF_RANGES = {
    "normal": controls.AfRangeEnum.Normal,
    "macro": controls.AfRangeEnum.Macro,
    "full": controls.AfRangeEnum.Full,
}

LENS_MIN, LENS_MAX = 0.0, 15.0     # ov64a40.json's widest tuned range
SWEEP_SETTLE_FRAMES = 4            # the lens is an actuator: let it arrive
AF_TIMEOUT = 12.0


# --- state shared between the camera thread and the GUI ----------------------

class State:
    """Latest frame + readings. The camera thread writes, the GUI reads."""

    def __init__(self, echo: bool = False) -> None:
        self._lock = threading.Lock()
        self.echo = echo                        # terminal mode prints instead
        self.logs: queue.Queue[str] = queue.Queue()
        self.frame: np.ndarray | None = None   # display-ready RGB
        self.seq = 0                           # bumped per new frame
        self.fatal: str | None = None
        self.readings = {"lens": None, "sharpness": 0.0, "peak": 0.0,
                         "brightness": 0.0, "fps": 0.0, "phase": "starting",
                         "zoom": False, "rotate": 0}

    def publish(self, frame: np.ndarray | None = None, **kw) -> None:
        with self._lock:
            if frame is not None:
                self.frame = frame
                self.seq += 1
            self.readings.update(kw)
            if "sharpness" in kw:
                self.readings["peak"] = max(float(self.readings["peak"]),
                                            float(kw["sharpness"]))

    def snapshot(self) -> tuple[np.ndarray | None, int, dict]:
        with self._lock:
            return self.frame, self.seq, dict(self.readings)

    def reset_peak(self) -> None:
        with self._lock:
            self.readings["peak"] = 0.0

    def log(self, msg: str) -> None:
        if self.echo:
            # Wipe the meter's carriage-return line before writing over it —
            # but not when the output is a pipe or a log file.
            wipe = "\r" + " " * 110 + "\r" if sys.stdout.isatty() else ""
            print(wipe + msg, flush=True)
        else:
            self.logs.put(msg)


def orientation_label(rotate: int) -> str:
    """How far the picture has actually been turned, flips included.

    The ISP's hflip+vflip IS the 180-degree turn this camera needs, so a bare
    'rotate 0' would read as 'not turned at all' when the picture is in fact the
    right way up. Spell out both halves.
    """
    hflip, vflip = PROFILE["hflip"], PROFILE["vflip"]
    base = 180 if (hflip and vflip) else 0
    parts = []
    if hflip and vflip:
        parts.append("h+v flip")
    elif hflip:
        parts.append("h mirror")
    elif vflip:
        parts.append("v mirror")
    if rotate:
        parts.append(f"+{rotate}° software")
    total = (base + rotate) % 360
    return f"{total}°" + (f" ({', '.join(parts)})" if parts else " (none)")


def save_for_recorder(state: State) -> None:
    """Write the current focus + orientation into camera.json, which the recorder
    reads at startup. This is the whole point of the tool: focus once, and the
    service uses it.

    A 180-degree software rotation is folded into the flips on the way out. The
    ISP can do that one, so the recorder can honour it on the video too — saving
    it as 'rotate: 180' instead would leave the recorder unable to comply.
    """
    rd = state.snapshot()[2]
    lens, rotate = rd["lens"], int(rd["rotate"]) % 360
    hflip, vflip = PROFILE["hflip"], PROFILE["vflip"]
    if rotate == 180:
        hflip, vflip, rotate = not hflip, not vflip, 0
    fields = {"rotate": rotate, "hflip": hflip, "vflip": vflip,
              "lens": round(float(lens), 3) if isinstance(lens, float) else None}
    try:
        camera_profile.save_profile(**fields)
    except OSError as exc:
        state.log(f"could not write {CAMERA_FILE}: {exc}")
        return
    state.log(f"saved for the recorder -> {CAMERA_FILE}")
    state.log(f"  focus {'autofocus at startup' if fields['lens'] is None else str(fields['lens']) + ' D'}"
              f", hflip={hflip} vflip={vflip} rotate={rotate}°")
    if rotate:
        state.log(f"  NOTE: the recorder cannot rotate by {rotate}° (the ISP flips "
                  f"but cannot transpose) — it will record unrotated. Turn the "
                  f"camera in its mount instead.")
    state.log("  restart the recorder to pick it up: "
              "sudo systemctl restart beemonitor-recorder")


def describe_lens(pos: float | None) -> str:
    """'4.20 D  (~24 cm)' — dioptres are unintuitive, distances are not."""
    if pos is None:
        return "n/a"
    if pos <= 0.01:
        return f"{pos:.2f} D  (infinity)"
    metres = 1.0 / pos
    dist = f"{metres * 100:.0f} cm" if metres < 1.0 else f"{metres:.2f} m"
    return f"{pos:.2f} D  (~{dist})"


# --- the camera thread -------------------------------------------------------

class Camera(threading.Thread):
    """Owns the Picamera2 object. Every camera call happens on this thread.

    The GUI never touches the camera; it posts closures onto .commands, which
    are drained once per captured frame. Long jobs (autofocus, sweep, record)
    are step machines advanced by that same loop, so the live view keeps
    running while they work — you can watch the lens rack through a sweep.
    """

    daemon = True

    def __init__(self, args: argparse.Namespace, state: State) -> None:
        super().__init__(name="camera")
        self.args = args
        self.state = state
        self.commands: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.ready = threading.Event()

        self.cam: Picamera2 | None = None
        self.lores_size = (LORES_W, LORES_H)
        self.main_size = (MAIN_W, MAIN_H)
        self.display_size = (960, 540)
        self.roi_spec: str | None = args.roi
        self.roi: tuple[int, int, int, int] | None = None
        self.zoom = False
        self.rotate = args.rotate

        self._af_deadline: float | None = None
        self._sweep: dict | None = None
        self._rec: tuple[H264Encoder, Path, float] | None = None
        self._snap_path: Path | None = None
        self._last_tick = 0.0

    # -- public API, callable from any thread ------------------------------

    def submit(self, fn) -> None:
        self.commands.put(fn)

    def stop(self) -> None:
        self.stop_event.set()

    def autofocus(self) -> None:
        self.submit(lambda c: c._start_autofocus())

    def continuous(self) -> None:
        self.submit(lambda c: c._start_continuous())

    def set_lens(self, pos: float) -> None:
        self.submit(lambda c: c._set_lens(pos))

    def nudge_lens(self, delta: float) -> None:
        self.submit(lambda c: c._set_lens((c.state.readings["lens"] or 0.0) + delta))

    def sweep(self, lo: float, hi: float, steps: int) -> None:
        self.submit(lambda c: c._start_sweep(lo, hi, steps))

    def snapshot(self) -> None:
        self.submit(lambda c: setattr(
            c, "_snap_path", c.args.outdir / f"focus-{time.strftime('%Y%m%d-%H%M%S')}.jpg"))

    def record(self, seconds: float) -> None:
        self.submit(lambda c: c._start_record(seconds))

    def set_zoom(self, on: bool) -> None:
        self.submit(lambda c: c._set_zoom(on))

    def set_roi(self, spec: str | None) -> None:
        """spec is fractional 'x,y,w,h' of the *displayed* frame, or None."""
        self.submit(lambda c: c._set_roi(spec))

    def set_rotation(self, degrees: int) -> None:
        self.submit(lambda c: c._set_rotation(degrees))

    # -- thread body -------------------------------------------------------

    def run(self) -> None:
        try:
            self._open()
            # Before ready.set(), so a caller that waits on busy() cannot race
            # past a startup sweep that has not been queued yet.
            self._apply_initial_mode()
        except Exception as exc:
            self.state.fatal = str(exc)
            self.state.log(f"camera error: {exc}")
            self.ready.set()
            return
        self.ready.set()
        try:
            while not self.stop_event.is_set():
                self._drain_commands()
                self._one_frame()
        except Exception as exc:  # a dead camera must not hang the GUI
            self.state.fatal = str(exc)
            self.state.log(f"camera loop stopped: {exc}")
        finally:
            self._close()

    def _open(self) -> None:
        cam = Picamera2()
        config = cam.create_video_configuration(
            main={"size": (MAIN_W, MAIN_H)},
            lores={"size": (LORES_W, LORES_H), "format": "YUV420"},
            controls={"FrameRate": FPS},
            transform=TRANSFORM,
        )
        cam.configure(config)
        cam.start()
        self.cam = cam
        self.lores_size = tuple(cam.camera_configuration()["lores"]["size"])
        self.main_size = tuple(cam.camera_configuration()["main"]["size"])
        self._resize_display()
        self._apply_roi()
        time.sleep(1.0)  # let AE/AWB settle before anything is measured
        self.state.publish(rotate=self.rotate)
        self.state.log(f"camera up: main={self.main_size[0]}x{self.main_size[1]} "
                       f"lores={self.lores_size[0]}x{self.lores_size[1]}")
        # Say which of the two orientation mechanisms is doing what, so a picture
        # that is already upright doesn't look like nothing was applied.
        self.state.log(f"orientation: {orientation_label(self.rotate)} — the flips "
                       f"come from the ISP and the recorder shares them")
        if self.rotate:
            self.state.log(f"  the {self.rotate}° part is done in software here, and "
                           f"the recorder CANNOT follow it (the ISP cannot transpose)")
        self.state.log(f"preview {self.display_size[0]}x{self.display_size[1]}"
                       f"; metering {'roi ' + str(self.roi) if self.roi else 'full frame'}")
        if getattr(self.args, "focus_from_profile", False):
            self.state.log(f"starting at the focus saved for the recorder "
                           f"({self.args.lens:.2f} D, from {CAMERA_FILE.name})")

    def metric_size(self) -> tuple[int, int]:
        """lores dimensions as displayed — swapped for a quarter turn, so --roi
        fractions mean the same thing on screen as in the numbers."""
        lw, lh = self.lores_size
        return (lh, lw) if self.rotate % 180 else (lw, lh)

    def _resize_display(self) -> None:
        """Largest preview of the rotated frame that fits the requested box."""
        mw, mh = self.main_size
        if self.rotate % 180:
            mw, mh = mh, mw
        scale = min(self.args.display_width / mw, self.args.display_height / mh, 1.0)
        self.display_size = (max(1, round(mw * scale)), max(1, round(mh * scale)))

    def _rotated(self, a: np.ndarray) -> np.ndarray:
        code = ROTATIONS[self.rotate]
        return a if code is None else cv2.rotate(a, code)

    def _apply_roi(self) -> None:
        self.roi = parse_roi(self.roi_spec, self.metric_size())

    def _set_roi(self, spec: str | None) -> None:
        self.roi_spec = spec
        self._apply_roi()

    def _set_rotation(self, degrees: int) -> None:
        self.rotate = degrees % 360
        self._resize_display()
        # The roi is fractions of the *displayed* frame, so a quarter turn has
        # to re-derive it against the new orientation.
        self._apply_roi()
        self.state.publish(rotate=self.rotate)
        self.state.log(f"preview rotation {self.rotate}° "
                       f"({self.display_size[0]}x{self.display_size[1]})")

    def _close(self) -> None:
        if self.cam is None:
            return
        try:
            if self._rec is not None:
                self.cam.stop_encoder(self._rec[0])
            self.cam.stop()
        except Exception:
            pass
        finally:
            self.cam.close()
            self.cam = None

    def _drain_commands(self) -> None:
        while True:
            try:
                fn = self.commands.get_nowait()
            except queue.Empty:
                return
            try:
                fn(self)
            except Exception as exc:
                self.state.log(f"command failed: {exc}")

    def _one_frame(self) -> None:
        cam = self.cam
        assert cam is not None
        req = cam.capture_request()
        try:
            md = req.get_metadata()
            lw, lh = self.lores_size
            yuv = req.make_array("lores")           # (lh*3//2, lw) uint8
            # Rotate before cropping so the roi means the same thing here as it
            # does on screen. Laplacian variance itself is orientation-blind, so
            # the whole-frame number is unaffected either way.
            gray = self._rotated(np.ascontiguousarray(yuv[:lh, :lw]))
            if self.roi:
                x, y, w, h = self.roi
                # cv2 needs a contiguous buffer, and an roi slice is not one.
                gray = np.ascontiguousarray(gray[y:y + h, x:x + w])
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            brightness = float(gray.mean())
            frame = self._display_frame(req) if self.args.gui else None
            if self._snap_path is not None:
                path, self._snap_path = self._snap_path, None
                self._save_still(req, path)
        finally:
            req.release()

        now = time.monotonic()
        fps = 1.0 / (now - self._last_tick) if self._last_tick else 0.0
        self._last_tick = now
        smoothed = 0.7 * self.state.readings["fps"] + 0.3 * fps if fps else 0.0
        self.state.publish(frame=frame, lens=md.get("LensPosition"),
                           sharpness=sharpness, brightness=brightness,
                           fps=smoothed)

        self._step_autofocus(md)
        self._step_sweep(sharpness)
        self._step_record()

    def _display_frame(self, req) -> np.ndarray:
        """Main stream -> RGB array sized for the window.

        The main stream is XBGR8888, whose numpy channels are R,G,B,X (see
        picamera2's _get_pil_mode), so the first three are already RGB. Crop or
        scale while it is still four channels — dropping X first would leave a
        non-contiguous view, which cv2 refuses and PIL would have to copy.
        """
        arr = self._rotated(req.make_array("main"))   # (H, W, 4)
        dw, dh = self.display_size
        if self.zoom:
            # No resampling: sensor pixels straight to screen pixels. Softness
            # you see here is real softness, not the preview downscale.
            h, w = arr.shape[:2]
            cw, ch = min(dw, w), min(dh, h)
            x, y = (w - cw) // 2, (h - ch) // 2
            out = arr[y:y + ch, x:x + cw]
        else:
            out = cv2.resize(arr, (dw, dh), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(out[:, :, :3])

    def _save_still(self, req, path: Path) -> None:
        """Full-resolution JPEG, upright. picamera2's own req.save() would write
        the sensor orientation, so a rotated unit has to go via an array."""
        if not self.rotate:
            req.save("main", str(path))
        else:
            arr = self._rotated(req.make_array("main"))
            bgr = np.ascontiguousarray(arr[:, :, 2::-1])   # RGBX -> BGR for cv2
            cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        self.state.log(f"still saved: {path}")

    # -- focus modes -------------------------------------------------------

    def _apply_initial_mode(self) -> None:
        if self.args.sweep:
            lo, hi = self.args.sweep_bounds
            self._start_sweep(lo, hi, self.args.sweep_steps)
        elif self.args.lens is not None:
            self._set_lens(self.args.lens)
        elif self.args.af == "continuous":
            self._start_continuous()
        elif self.args.af == "manual":
            self._set_lens(1.0)
        else:
            self._start_autofocus()
        if self.args.snapshot:
            self.snapshot()
        if self.args.record > 0:
            self.record(self.args.record)

    def _set_zoom(self, on: bool) -> None:
        self.zoom = on
        self.state.publish(zoom=on)

    def _set_lens(self, pos: float) -> None:
        pos = max(LENS_MIN, min(LENS_MAX, float(pos)))
        self._sweep = None
        self._af_deadline = None
        self.cam.set_controls({"AfMode": controls.AfModeEnum.Manual,
                               "LensPosition": pos})
        self.state.publish(phase=f"manual {pos:.2f} D")

    def _start_continuous(self) -> None:
        self._sweep = None
        self._af_deadline = None
        self.cam.set_controls({"AfMode": controls.AfModeEnum.Continuous,
                               "AfRange": AF_RANGES[self.args.range]})
        self.state.publish(phase="continuous AF")
        self.state.log("continuous autofocus on")

    def _start_autofocus(self) -> None:
        self._sweep = None
        self.cam.set_controls({"AfMode": controls.AfModeEnum.Auto,
                               "AfRange": AF_RANGES[self.args.range],
                               "AfSpeed": controls.AfSpeedEnum.Normal})
        self.cam.set_controls({"AfTrigger": controls.AfTriggerEnum.Start})
        self._af_deadline = time.monotonic() + AF_TIMEOUT
        self.state.publish(phase="autofocus")
        self.state.log(f"autofocus (range={self.args.range})...")

    def _step_autofocus(self, md: dict) -> None:
        if self._af_deadline is None:
            return
        state = md.get("AfState")
        pos = md.get("LensPosition")
        if state == controls.AfStateEnum.Focused:
            self._af_deadline = None
            self.state.publish(phase="focused")
            self.state.log(f"  focused at {describe_lens(pos)}")
        elif state == controls.AfStateEnum.Failed:
            self._af_deadline = None
            self.state.publish(phase="AF failed")
            self.state.log(f"  autofocus FAILED (lens left at {describe_lens(pos)})")
            self.state.log("  Low contrast or a featureless scene defeats contrast AF.")
            self.state.log("  Try the sweep, or aim at something with texture.")
        elif time.monotonic() > self._af_deadline:
            self._af_deadline = None
            self.state.publish(phase="AF timed out")
            self.state.log(f"  autofocus timed out after {AF_TIMEOUT:.0f}s "
                           f"(state={state}, lens={describe_lens(pos)})")

    # -- sweep -------------------------------------------------------------

    def _start_sweep(self, lo: float, hi: float, steps: int) -> None:
        lo = max(LENS_MIN, min(LENS_MAX, lo))
        hi = max(LENS_MIN, min(LENS_MAX, hi))
        if hi <= lo or steps < 2:
            self.state.log(f"bad sweep {lo}:{hi} in {steps} steps — ignored")
            return
        self._af_deadline = None
        self.cam.set_controls({"AfMode": controls.AfModeEnum.Manual})
        positions = np.linspace(lo, hi, steps)
        self.cam.set_controls({"LensPosition": float(positions[0])})
        self._sweep = {"pos": positions, "i": 0, "settle": SWEEP_SETTLE_FRAMES,
                       "scores": []}
        self.state.reset_peak()
        self.state.publish(phase="sweeping")
        self.state.log(f"\n== Focus sweep: {lo:g} -> {hi:g} dioptres in {steps} steps ==")

    def _step_sweep(self, sharpness: float) -> None:
        s = self._sweep
        if s is None:
            return
        if s["settle"] > 0:
            s["settle"] -= 1
            return
        pos = float(s["pos"][s["i"]])
        s["scores"].append(sharpness)
        self.state.log(f"  {pos:5.2f} D  sharpness {sharpness:9.1f}")
        s["i"] += 1
        if s["i"] < len(s["pos"]):
            self.cam.set_controls({"LensPosition": float(s["pos"][s["i"]])})
            s["settle"] = SWEEP_SETTLE_FRAMES
            return
        self._finish_sweep(s)

    def _finish_sweep(self, s: dict) -> None:
        self._sweep = None
        positions, scores = s["pos"], s["scores"]
        best_i = int(np.argmax(scores))
        best = float(positions[best_i])

        # A bar chart makes the peak obvious, in the log pane or over SSH.
        peak = max(scores) or 1.0
        self.state.log("\n  profile (# = sharpness, * = best):")
        for pos, sc in zip(positions, scores):
            bar = "#" * int(round(30 * sc / peak))
            mark = " *" if abs(pos - best) < 1e-9 else ""
            self.state.log(f"  {pos:5.2f} |{bar}{mark}")
        self.state.log(f"\n  sharpest at {describe_lens(best)}, "
                       f"score {scores[best_i]:.1f}")
        if best_i in (0, len(positions) - 1):
            self.state.log("  NOTE: the peak is at the edge of the swept range — "
                           "the true focus may lie outside it. Widen the bounds.")
        self._set_lens(best)

    # -- recording ---------------------------------------------------------

    def _start_record(self, seconds: float) -> None:
        if self._rec is not None:
            self.state.log("already recording")
            return
        path = self.args.outdir / f"focus-{time.strftime('%Y%m%d-%H%M%S')}.mp4"
        encoder = H264Encoder()
        self.cam.start_encoder(encoder, FfmpegOutput(str(path)), name="main")
        self._rec = (encoder, path, time.monotonic() + seconds)
        self.state.log(f"recording {seconds:.0f}s -> {path}")
        if self.rotate:
            # The hardware encoder is fed by the ISP directly, so nothing in
            # Python can rotate these frames without re-encoding the clip.
            self.state.log(f"  NOTE: the clip is in sensor orientation, not "
                           f"rotated {self.rotate}°. Play it with "
                           f"'ffplay -vf transpose=1' or rotate it afterwards.")

    def _step_record(self) -> None:
        if self._rec is None:
            return
        encoder, path, until = self._rec
        if time.monotonic() < until:
            return
        self.cam.stop_encoder(encoder)
        self._rec = None
        self.state.log(f"  clip saved: {path}")

    def recording_left(self) -> float:
        rec = self._rec
        return max(0.0, rec[2] - time.monotonic()) if rec else 0.0

    def busy(self) -> bool:
        """True while a sweep or a one-shot autofocus is still running."""
        return self._sweep is not None or self._af_deadline is not None


# --- GUI ---------------------------------------------------------------------

def run_gui(cam: Camera, state: State, args: argparse.Namespace) -> int:
    import tkinter as tk
    from tkinter import ttk
    try:
        from PIL import Image, ImageTk
    except ImportError:
        sys.exit(
            "the GUI needs Pillow's ImageTk, which the system PIL does not ship.\n"
            "Run it with the project venv, which has it:\n"
            "    hardware/venv/bin/python hardware/runFocus.py\n"
            "or install it system-wide:  sudo apt install python3-pil.imagetk\n"
            "or skip the window entirely: --no-gui"
        )

    MONO = ("monospace", 11)
    BIG = ("monospace", 13, "bold")

    root = tk.Tk()
    root.title("BeeMonitor — aim & focus")
    root.configure(bg="#111")

    body = tk.Frame(root, bg="#111")
    body.pack(fill="both", expand=True, padx=8, pady=8)

    video = tk.Label(body, bg="#000", text="waiting for the camera...",
                     fg="#888", font=MONO, width=60, height=20)
    video.grid(row=0, column=0, sticky="nw")

    panel = tk.Frame(body, bg="#111")
    panel.grid(row=0, column=1, sticky="nw", padx=(10, 0))

    # -- readouts
    readout = tk.Frame(panel, bg="#111")
    readout.pack(fill="x")
    values: dict[str, tk.Label] = {}
    for i, (key, label) in enumerate([("phase", "phase"), ("lens", "lens"),
                                      ("sharpness", "sharp"), ("peak", "peak"),
                                      ("brightness", "bright"), ("fps", "fps"),
                                      ("orient", "orient")]):
        tk.Label(readout, text=label, bg="#111", fg="#888", font=MONO,
                 anchor="w", width=7).grid(row=i, column=0, sticky="w")
        values[key] = tk.Label(readout, text="-", bg="#111", fg="#7ee787",
                               font=BIG, anchor="w", width=22)
        values[key].grid(row=i, column=1, sticky="w")

    bar = tk.Canvas(panel, width=230, height=10, bg="#333", highlightthickness=0)
    bar.pack(pady=(6, 10), anchor="w")
    fill = bar.create_rectangle(0, 0, 0, 10, fill="#7ee787", width=0)

    def row() -> tk.Frame:
        f = tk.Frame(panel, bg="#111")
        f.pack(fill="x", pady=2)
        return f

    # -- focus controls
    r = row()
    ttk.Button(r, text="Autofocus now (f)", command=cam.autofocus).pack(side="left")
    cont = tk.BooleanVar(value=args.af == "continuous")

    def toggle_continuous() -> None:
        if cont.get():
            cam.continuous()
        else:
            cam.set_lens(state.readings["lens"] or 1.0)

    ttk.Checkbutton(r, text="continuous (c)", variable=cont,
                    command=toggle_continuous).pack(side="left", padx=6)

    r = row()
    tk.Label(r, text="lens", bg="#111", fg="#888", font=MONO).pack(side="left")
    lens_var = tk.DoubleVar(value=args.lens if args.lens is not None else 1.0)
    pending_lens: list[float] = []

    def flush_lens() -> None:
        if pending_lens:
            cam.set_lens(pending_lens.pop())
            pending_lens.clear()
            cont.set(False)
        root.after(80, flush_lens)

    def on_slider(_v) -> None:
        # ttk.Scale fires continuously while dragging; coalesce so we don't
        # flood the camera thread with control changes.
        pending_lens.append(lens_var.get())

    scale = ttk.Scale(r, from_=LENS_MIN, to=LENS_MAX, variable=lens_var,
                      command=on_slider, length=230)
    scale.pack(side="left", padx=6)

    # -- sweep
    r = row()
    tk.Label(r, text="sweep", bg="#111", fg="#888", font=MONO).pack(side="left")
    lo_e = ttk.Entry(r, width=4)
    lo_e.insert(0, f"{args.sweep_bounds[0]:g}")
    lo_e.pack(side="left", padx=2)
    hi_e = ttk.Entry(r, width=4)
    hi_e.insert(0, f"{args.sweep_bounds[1]:g}")
    hi_e.pack(side="left", padx=2)
    steps_e = ttk.Entry(r, width=4)
    steps_e.insert(0, str(args.sweep_steps))
    steps_e.pack(side="left", padx=2)

    def do_sweep() -> None:
        try:
            cam.sweep(float(lo_e.get()), float(hi_e.get()), int(steps_e.get()))
        except ValueError:
            state.log("sweep needs numbers in min / max / steps")

    ttk.Button(r, text="run (s)", command=do_sweep).pack(side="left", padx=4)

    # -- capture
    r = row()
    ttk.Button(r, text="Snapshot (space)", command=cam.snapshot).pack(side="left")
    rec_e = ttk.Entry(r, width=4)
    rec_e.insert(0, f"{args.record_seconds:g}")
    rec_e.pack(side="left", padx=(8, 2))

    def do_record() -> None:
        try:
            cam.record(float(rec_e.get()))
        except ValueError:
            state.log("record needs a number of seconds")

    ttk.Button(r, text="Record (r)", command=do_record).pack(side="left")

    # -- view options
    r = row()
    zoom = tk.BooleanVar(value=False)
    ttk.Checkbutton(r, text="zoom 1:1 (z)", variable=zoom,
                    command=lambda: cam.set_zoom(zoom.get())).pack(side="left")
    centre = tk.BooleanVar(value=bool(args.roi))

    def toggle_centre() -> None:
        cam.set_roi("0.3,0.3,0.4,0.4" if centre.get() else args.roi)
        state.reset_peak()

    ttk.Checkbutton(r, text="meter centre only", variable=centre,
                    command=toggle_centre).pack(side="left", padx=6)

    r = row()

    def cycle_rotation() -> None:
        cam.set_rotation((state.readings["rotate"] + 90) % 360)

    ttk.Button(r, text="turn 90° (o)", command=cycle_rotation).pack(side="left")
    tk.Label(r, text="if it isn't upright", bg="#111", fg="#666",
             font=("monospace", 9)).pack(side="left", padx=6)

    r = row()
    ttk.Button(r, text="Save for recorder (w)",
               command=lambda: save_for_recorder(state)).pack(side="left")

    r = row()
    ttk.Button(r, text="Reset peak (p)", command=state.reset_peak).pack(side="left")
    ttk.Button(r, text="Quit (q)", command=lambda: shutdown()).pack(side="left", padx=6)

    # -- log pane
    log = tk.Text(body, height=9, bg="#0a0a0a", fg="#ccc", font=("monospace", 9),
                  insertbackground="#ccc", relief="flat", wrap="none")
    log.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(8, 0))
    body.rowconfigure(1, weight=1)
    body.columnconfigure(0, weight=1)

    # -- render loop
    photo: dict[str, object] = {"img": None, "size": None}
    last_seq = [-1]
    closing = [False]

    def shutdown() -> None:
        if closing[0]:
            return
        closing[0] = True
        state.log("shutting down")
        cam.stop()
        cam.join(timeout=5)
        root.destroy()

    def tick() -> None:
        if closing[0]:
            return
        while True:
            try:
                log.insert("end", state.logs.get_nowait() + "\n")
            except queue.Empty:
                break
            log.see("end")

        frame, seq, rd = state.snapshot()
        if state.fatal and not closing[0]:
            video.configure(image="", text=f"camera stopped:\n{state.fatal}",
                            fg="#ff7b72")

        if frame is not None and seq != last_seq[0]:
            last_seq[0] = seq
            h, w = frame.shape[:2]
            img = Image.frombuffer("RGB", (w, h), frame, "raw", "RGB", 0, 1)
            if photo["size"] != (w, h):
                photo["img"] = ImageTk.PhotoImage(img)
                photo["size"] = (w, h)
                video.configure(image=photo["img"], text="", width=w, height=h)
            else:
                photo["img"].paste(img)   # in place: much cheaper than a new one
            video.image = photo["img"]

        values["phase"].configure(text=str(rd["phase"]))
        values["lens"].configure(text=describe_lens(rd["lens"]))
        values["sharpness"].configure(text=f"{rd['sharpness']:.0f}")
        values["peak"].configure(text=f"{rd['peak']:.0f}")
        values["brightness"].configure(text=f"{rd['brightness']:.0f} / 255")
        left = cam.recording_left()
        values["fps"].configure(
            text=f"{rd['fps']:.1f}" + (f"   REC {left:.0f}s" if left else ""))
        values["orient"].configure(text=orientation_label(int(rd["rotate"])))
        frac = rd["sharpness"] / rd["peak"] if rd["peak"] else 0.0
        bar.coords(fill, 0, 0, 230 * min(1.0, frac), 10)

        root.after(30, tick)

    def hotkey(fn):
        """Letter shortcuts must not fire while you're typing in an entry box."""
        def handler(_event):
            if isinstance(root.focus_get(), (tk.Entry, ttk.Entry)):
                return
            fn()
        return handler

    def toggle_zoom() -> None:
        zoom.set(not zoom.get())
        cam.set_zoom(zoom.get())

    def toggle_cont() -> None:
        cont.set(not cont.get())
        toggle_continuous()

    root.bind("<KeyPress-f>", hotkey(cam.autofocus))
    root.bind("<KeyPress-s>", hotkey(do_sweep))
    root.bind("<KeyPress-p>", hotkey(state.reset_peak))
    root.bind("<KeyPress-r>", hotkey(do_record))
    root.bind("<space>", hotkey(cam.snapshot))
    root.bind("<KeyPress-c>", hotkey(toggle_cont))
    root.bind("<KeyPress-z>", hotkey(toggle_zoom))
    root.bind("<KeyPress-o>", hotkey(cycle_rotation))
    root.bind("<KeyPress-w>", hotkey(lambda: save_for_recorder(state)))
    def nudge(delta: float) -> None:
        cam.nudge_lens(delta)
        cont.set(False)          # nudging drops the camera into manual focus
        lens_var.set(max(LENS_MIN, min(LENS_MAX,
                                       (state.readings["lens"] or 0.0) + delta)))

    root.bind("<Left>", lambda e: nudge(-0.1))
    root.bind("<Right>", lambda e: nudge(+0.1))
    root.bind("<KeyPress-q>", hotkey(shutdown))
    root.bind("<Escape>", lambda e: shutdown())
    root.protocol("WM_DELETE_WINDOW", shutdown)

    state.log("window open — f autofocus, s sweep, z zoom 1:1, o turn 90°, "
              "space snapshot, q quit")
    root.after(30, tick)
    root.after(80, flush_lens)
    root.mainloop()

    final = state.readings["lens"]
    if isinstance(final, float):
        print(f"\nFinal LensPosition: {final:.2f} dioptres")
        print(f"Press 'Save for recorder' (w) to store it in {CAMERA_FILE}; the "
              f"recorder applies it at startup.")
    return 0


# --- terminal mode (SSH: no window to draw into) -----------------------------

def run_terminal(cam: Camera, state: State, args: argparse.Namespace) -> int:
    # A startup sweep or autofocus has to finish before the meter means anything,
    # and --meter must not cut it short.
    deadline = time.monotonic() + 120
    try:
        while cam.is_alive() and cam.busy() and time.monotonic() < deadline:
            time.sleep(0.2)
    except KeyboardInterrupt:
        cam.stop()
        cam.join(timeout=5)
        return 0
    if args.record > 0:
        while cam.is_alive() and cam.recording_left():
            time.sleep(0.2)

    forever = args.meter <= 0
    end = time.monotonic() + args.meter
    print(f"\n== Focus meter ({'until Ctrl-C' if forever else f'{args.meter:.0f}s'}) ==")
    print("  Higher sharpness = better focus. Aim the camera and watch it move.")
    try:
        while (forever or time.monotonic() < end) and cam.is_alive():
            # state.log() prints straight to stdout in this mode (State.echo),
            # so the meter line just has to keep redrawing itself.
            rd = state.snapshot()[2]
            bar = "#" * min(40, int(round(40 * rd["sharpness"] / rd["peak"]))) \
                if rd["peak"] else ""
            print(f"  lens {describe_lens(rd['lens']):22} bright {rd['brightness']:5.1f}"
                  f"  sharp {rd['sharpness']:8.1f} |{bar:<40}", end="\r", flush=True)
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    print()
    rd = state.snapshot()[2]
    print(f"  peak sharpness seen: {rd['peak']:.1f}")
    if isinstance(rd["lens"], float):
        print(f"  final LensPosition: {rd['lens']:.2f} dioptres")
    if args.save:
        save_for_recorder(state)
    elif isinstance(rd["lens"], float):
        print(f"  add --save to store it in {CAMERA_FILE} for the recorder")
    cam.stop()
    cam.join(timeout=5)
    return 0


# --- args --------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aim and focus the OwlSight camera in a live Tk window.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--af", choices=["auto", "continuous", "manual"], default="auto",
                   help="focus mode at startup: auto = one-shot autofocus "
                        "(default); continuous = keep refocusing; manual = --lens")
    p.add_argument("--lens", type=float, metavar="DIOPTRES",
                   help="manual lens position at startup (implies --af manual)")
    p.add_argument("--range", choices=list(AF_RANGES), default="normal",
                   help="autofocus search range (default: normal)")
    p.add_argument("--sweep", action="store_true",
                   help="run a focus sweep at startup")
    p.add_argument("--sweep-range", default="0:12", metavar="MIN:MAX",
                   help="sweep bounds in dioptres (default: 0:12)")
    p.add_argument("--sweep-steps", type=int, default=25,
                   help="number of sweep samples (default: 25)")
    p.add_argument("--gui", action="store_true", default=True,
                   help="show the live video window (default)")
    p.add_argument("--no-gui", dest="gui", action="store_false",
                   help="terminal readout only — for SSH sessions")
    p.add_argument("--rotate", type=int, default=ROTATE, metavar="DEGREES",
                   help=f"rotate the preview clockwise, on top of the recorder's "
                        f"180° flip: 0, 90, 180 or 270 (default: {ROTATE}, from "
                        f"BEEMONITOR_ROTATE). Cycle it live in the window with 'o'")
    p.add_argument("--display-width", type=int, default=960, metavar="PX",
                   help="preview box width in the window (default: 960; lower it "
                        "if the frame rate feels sluggish)")
    p.add_argument("--display-height", type=int, default=620, metavar="PX",
                   help="preview box height (default: 620). The rotated frame is "
                        "scaled to fit inside width x height")
    p.add_argument("--meter", type=float, default=0.0, metavar="SECONDS",
                   help="--no-gui only: how long to hold the meter, 0 = until "
                        "Ctrl-C (default: 0)")
    p.add_argument("--record", type=float, default=0.0, metavar="SECONDS",
                   help="record a clip of this length at startup")
    p.add_argument("--record-seconds", type=float, default=40.0, metavar="SECONDS",
                   help="length the GUI's Record button uses (default: 40)")
    p.add_argument("--snapshot", action="store_true",
                   help="save a full-resolution still at startup")
    p.add_argument("--save", action="store_true",
                   help=f"on exit, store the focus + orientation in {CAMERA_FILE} "
                        f"for the recorder (the GUI's 'Save for recorder' button)")
    p.add_argument("--roi", metavar="X,Y,W,H",
                   help="fractional region to measure sharpness over, e.g. "
                        "0.3,0.3,0.4,0.4 for the centre")
    p.add_argument("--outdir", type=Path, default=OUT_DIR,
                   help=f"where stills/clips land (default: {OUT_DIR})")
    args = p.parse_args()
    args.sweep_bounds = parse_sweep_range(args.sweep_range)
    # With a focus already saved for the recorder, start there: the point of
    # opening this tool on a configured unit is to see what the recorder sees.
    args.focus_from_profile = (args.lens is None and not args.sweep
                               and args.af == "auto" and PROFILE["lens"] is not None)
    if args.focus_from_profile:
        args.lens = PROFILE["lens"]
    args.rotate %= 360
    if args.rotate not in ROTATIONS:
        sys.exit(f"bad --rotate {args.rotate}: expected one of "
                 f"{', '.join(str(r) for r in sorted(ROTATIONS))}")
    parse_roi(args.roi, (LORES_W, LORES_H))  # reject bad --roi before opening the camera
    if args.lens is not None:
        args.af = "manual"
    return args


def parse_roi(spec: str | None, lores: tuple[int, int]) -> tuple[int, int, int, int] | None:
    """Fractional 'x,y,w,h' -> pixel box on the lores frame."""
    if not spec:
        return None
    lw, lh = lores
    try:
        x, y, w, h = (float(v) for v in spec.split(","))
    except ValueError:
        sys.exit(f"bad --roi {spec!r}: expected four comma-separated fractions")
    if not all(0.0 <= v <= 1.0 for v in (x, y, w, h)) or w <= 0 or h <= 0:
        sys.exit(f"bad --roi {spec!r}: fractions must be within 0..1 and w,h > 0")
    return (int(x * lw), int(y * lh), max(1, int(w * lw)), max(1, int(h * lh)))


def parse_sweep_range(spec: str) -> tuple[float, float]:
    """'MIN:MAX' -> bounds. Validated before the camera is opened."""
    try:
        lo, hi = (float(v) for v in spec.split(":"))
    except ValueError:
        sys.exit(f"bad --sweep-range {spec!r}: expected MIN:MAX")
    if hi <= lo:
        sys.exit(f"bad --sweep-range {spec!r}: MAX must exceed MIN")
    return lo, hi


def main() -> int:
    args = parse_args()
    if args.gui and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        sys.exit("no display found (DISPLAY/WAYLAND_DISPLAY unset) — this is an SSH\n"
                 "session or a headless boot. Use --no-gui for the terminal meter,\n"
                 "or run it from the Pi's own desktop.")
    args.outdir.mkdir(parents=True, exist_ok=True)

    state = State(echo=not args.gui)
    cam = Camera(args, state)
    cam.start()
    cam.ready.wait(timeout=20)
    if state.fatal:
        sys.exit(
            f"could not open the camera: {state.fatal}\n\n"
            "If the recorder is running it owns the camera exclusively:\n"
            "    sudo systemctl stop beemonitor-recorder\n"
            "and start it again when you're done:\n"
            "    sudo systemctl start beemonitor-recorder\n"
            "If instead no camera enumerates at all, run:\n"
            "    ./hardware/setup-camera.sh --probe-only"
        )

    try:
        return run_gui(cam, state, args) if args.gui else run_terminal(cam, state, args)
    finally:
        cam.stop()
        cam.join(timeout=5)


if __name__ == "__main__":
    sys.exit(main())
