"""Bringing up the camera: orientation, focus and the per-unit camera profile.

Everything that opens the camera goes through here so the recorder, the focus
tool and the stills all agree on which way up the picture is and where the lens
is focused. Two knobs, and they are not equivalent:

    orientation — hflip/vflip, done by the ISP. Free, and every consumer sees
                  the same upright frame. Both together are the 180-degree
                  turn, and whether a unit needs it is detected from the
                  sensor model (see FLIPPED_MODELS) unless the profile says.
    focus       — LensPosition, in dioptres (1/metres). Fixed-focus modules
                  ignore it; the Arducam OwlSight (OV64A40) needs it, and
                  nothing in the recorder used to set it at all, so a unit
                  recorded at whatever position the lens happened to power up
                  in until somebody ran the focus tool by hand.

The profile lives in camera.json beside calibration.json — written by
runFocus.py when you press "save for recorder", read here at startup. Env
defaults (BEEMONITOR_HFLIP / VFLIP / ROTATE / LENS_POSITION / AF_RANGE) apply
when the file has nothing to say, matching how the other JSON contracts in
motion/ layer over config.py. The flips are the one knob with no fixed default:
left unset they follow the sensor model (see FLIPPED_MODELS), because whether
the picture needs turning over is a property of which module is fitted.
"""

from __future__ import annotations

import json
import os
import time

from motion.config import (
    log, CAMERA_FILE, HFLIP, VFLIP, ROTATE, LENS_POSITION, AF_RANGE,
    CAMERA_BACKEND, OAK_LENS,
)

# Lens modules we know need a LensPosition. Anything else is fixed focus, where
# asking for autofocus is not an error to report — there is simply nothing to do.
FOCUSABLE_MODELS = ("ov64a40",)
AF_TIMEOUT = 12.0

# Modules whose native readout is upside down in our enclosure and so need the
# ISP's 180 (hflip+vflip). The OV5647 night-vision module does, and so does
# the Pi HQ camera (IMX477), which recorded upside down at flip=none when first
# fitted. The Arducam OwlSight (OV64A40) does not, and turning it anyway
# records everything inverted — which is what happened when the OwlSight first
# inherited the OV5647's default.
#
# Measured, not assumed: with the camera to itself, a still captured under
# Transform(hflip=1, vflip=1) scores ncc +0.995 against rot180 of the
# unflipped still, and it is the flipped one that comes out upright — desk
# along the bottom, cables hanging down. scripts/camera-flip-test.sh re-runs
# that check on any unit and prints which way up each variant lands.
#
# Orientation is decided by what detect_model() finds rather than by a
# baked-in default, so a module that turns out to need the turn is listed here
# once and every unit fitted with it picks it up on the next update. A single
# oddly-mounted unit overrides hflip/vflip in camera.json instead of having it
# encoded as a property of its sensor.
FLIPPED_MODELS = ("ov5647", "imx477")


def select_backend() -> str:
    """Which camera the recorder opens: 'oak' or 'picamera2'.

    BEEMONITOR_CAMERA=auto (the default) takes a Luxonis OAK whenever one is on
    USB — it is the better camera, and a unit only has one plugged in because
    somebody wants it used — and falls back to the ribbon camera otherwise.
    An OAK on USB without depthai installed is said out loud: it is one pip
    install away from being used and nothing else would tell you.
    """
    if CAMERA_BACKEND in ("oak", "picamera2"):
        return CAMERA_BACKEND
    from motion import oak
    if oak.available():
        return "oak"
    if _oak_on_usb():
        log.warning("an OAK is plugged in but depthai cannot use it (not installed, "
                    "or no udev rule — see hardware/oak/README.md); using the "
                    "ribbon camera")
    return "picamera2"


def _oak_on_usb() -> bool:
    """Movidius vendor id on the USB bus, without depthai."""
    import glob
    for path in glob.glob("/sys/bus/usb/devices/*/idVendor"):
        try:
            if open(path).read().strip() == "03e7":
                return True
        except OSError:
            pass
    return False


def detect_model() -> str:
    """Sensor model ('ov64a40'), lowercased, WITHOUT opening the camera.

    global_camera_info() only enumerates, so this is safe at import time and
    while the recorder already holds the camera — which is what lets runFocus.py
    and the recorder arrive at the same default. '' when picamera2 is missing
    (off-device) or no camera is attached.
    """
    try:
        from picamera2 import Picamera2
        info = Picamera2.global_camera_info()
    except Exception:
        return ""
    if not info:
        return ""
    return str(info[0].get("Model", "")).lower()


def default_flip(model=None) -> bool:
    """Whether this module needs the ISP's 180, decided from the sensor model.

    False unless the model is listed in FLIPPED_MODELS. An unrecognised or
    unreadable model lands on False too — the modules we ship disagree with
    each other, so there is no majority to fall back on and no turn is the
    safer guess: it leaves the picture as the sensor read it out rather than
    inverting it on a guess. Such a unit wants an explicit hflip/vflip in
    camera.json. Override per unit there.
    """
    if model is None:
        model = detect_model()
    return model in FLIPPED_MODELS


def load_profile(model=None) -> dict:
    """Merged camera profile: camera.json over the env defaults.

    Keys: hflip, vflip (bool), rotate (0/90/180/270), lens (float | None,
    None = autofocus at startup), af_range ('normal'|'macro'|'full'),
    oak_lens (int 0..255 | None — the OAK's own scale, None = autofocus).
    `model` decides the default flip; None = ask libcamera (the ribbon camera).
    Never raises — an unreadable or half-written profile falls back to env.
    """
    flip = default_flip(model)
    profile = {
        "hflip": flip if HFLIP is None else bool(HFLIP),
        "vflip": flip if VFLIP is None else bool(VFLIP),
        "rotate": ROTATE % 360,
        "lens": float(LENS_POSITION) if _is_number(LENS_POSITION) else None,
        "af_range": AF_RANGE if AF_RANGE in ("normal", "macro", "full") else "normal",
        "oak_lens": _oak_lens(OAK_LENS),
    }
    try:
        saved = json.loads(CAMERA_FILE.read_text())
    except (OSError, ValueError):
        return profile
    if not isinstance(saved, dict):
        return profile
    for key in ("hflip", "vflip"):
        if isinstance(saved.get(key), bool):
            profile[key] = saved[key]
    if saved.get("rotate") in (0, 90, 180, 270):
        profile["rotate"] = saved["rotate"]
    if _is_number(saved.get("lens")):
        profile["lens"] = float(saved["lens"])
    elif "lens" in saved and saved["lens"] is None:
        profile["lens"] = None          # explicit "autofocus, please"
    if saved.get("af_range") in ("normal", "macro", "full"):
        profile["af_range"] = saved["af_range"]
    if "oak_lens" in saved:
        profile["oak_lens"] = _oak_lens(saved["oak_lens"])
    return profile


def _oak_lens(v) -> int | None:
    if not _is_number(v) or isinstance(v, bool):
        return None
    return max(0, min(255, int(float(v))))


def save_profile(**fields) -> None:
    """Merge fields into camera.json (atomic). Raises on a write failure — the
    caller is a human pressing a button and wants to be told."""
    profile = {}
    try:
        existing = json.loads(CAMERA_FILE.read_text())
        if isinstance(existing, dict):
            profile = existing
    except (OSError, ValueError):
        pass
    profile.update(fields)
    profile["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    CAMERA_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CAMERA_FILE.with_name(CAMERA_FILE.name + ".part")
    tmp.write_text(json.dumps(profile, indent=2))
    os.replace(tmp, CAMERA_FILE)


def transform(profile: dict):
    """libcamera Transform for a profile. Imported lazily so this module stays
    importable off-device (the recorder does the same for picamera2)."""
    import libcamera
    return libcamera.Transform(hflip=int(bool(profile["hflip"])),
                               vflip=int(bool(profile["vflip"])))


def describe(profile: dict, oak: bool = False) -> str:
    flips = "+".join([n for n, on in (("hflip", profile["hflip"]),
                                      ("vflip", profile["vflip"])) if on]) or "none"
    if oak:
        lens = ("autofocus" if profile.get("oak_lens") is None
                else f"{profile['oak_lens']}/255")
    else:
        lens = "autofocus" if profile["lens"] is None else f"{profile['lens']:.2f}D"
    return f"flip={flips} rotate={profile['rotate']}° focus={lens}"


def model_of(cam) -> str:
    """Sensor model as libcamera reports it ('ov64a40'), lowercased; '' if unknown."""
    try:
        return str(cam.camera_properties.get("Model", "")).lower()
    except Exception:
        return ""


def has_focus(cam) -> bool:
    """True if this module has a movable lens we should be driving."""
    model = model_of(cam)
    if any(m in model for m in FOCUSABLE_MODELS):
        return True
    try:  # a module we have not listed but which advertises the control
        return "LensPosition" in cam.camera_controls
    except Exception:
        return False


# Where autofocus looks when there is no ROI: the middle of the frame. Edges are
# where sky, sun flare and near plants are; the hotel is usually in the middle.
CENTER_REGION = (0.25, 0.25, 0.75, 0.75)


def focus_region(gate_roi, lores_wh, drawn: bool):
    """(region, source) for autofocus: the drawn ROI, else the hotel the
    recorder detected, else the centre. ``region`` is normalised x1,y1,x2,y2
    of the frame; ``gate_roi`` is the motion gate's lores-pixel box or None."""
    if gate_roi is not None:
        w, h = lores_wh
        x1, y1, x2, y2 = gate_roi
        r = (max(0.0, x1 / w), max(0.0, y1 / h), min(1.0, x2 / w), min(1.0, y2 / h))
        if r[2] - r[0] > 0.02 and r[3] - r[1] > 0.02:
            return r, ("roi" if drawn else "hotel")
    return CENTER_REGION, "center"


def autofocus(cam, region=CENTER_REGION, af_range: str = "normal") -> dict:
    """Autofocus once on ``region`` (normalised), hold the lens where it lands,
    and say how it went: {"ok", "lens", "settled"}. Retries once if the first
    try does not settle. Both cameras: the OAK drives its own lens."""
    if hasattr(cam, "autofocus"):        # the OAK (motion/oak.py)
        return cam.autofocus(region)
    if not has_focus(cam):
        return {"ok": False, "lens": None, "settled": False, "error": "fixed-focus camera"}

    from libcamera import controls

    af_range = {"normal": controls.AfRangeEnum.Normal,
                "macro": controls.AfRangeEnum.Macro,
                "full": controls.AfRangeEnum.Full}.get(af_range, controls.AfRangeEnum.Normal)
    windows = []
    try:  # AfWindows are in ScalerCrop coordinates (the sensor's active area)
        cx, cy, cw, ch = cam.camera_properties["ScalerCropMaximum"]
        x1, y1, x2, y2 = region
        windows = [(int(cx + x1 * cw), int(cy + y1 * ch),
                    max(1, int((x2 - x1) * cw)), max(1, int((y2 - y1) * ch)))]
    except Exception as e:
        log.warning("autofocus: no focus window (%s) — using the whole frame", e)

    state = pos = None
    for attempt in (1, 2):
        ctl = {"AfMode": controls.AfModeEnum.Auto, "AfRange": af_range,
               "AfSpeed": controls.AfSpeedEnum.Normal}
        if windows:
            ctl.update(AfMetering=controls.AfMeteringEnum.Windows, AfWindows=windows)
        cam.set_controls(ctl)
        cam.set_controls({"AfTrigger": controls.AfTriggerEnum.Start})
        deadline = time.monotonic() + AF_TIMEOUT
        while time.monotonic() < deadline:
            md = cam.capture_metadata()
            state, pos = md.get("AfState"), md.get("LensPosition")
            if state in (controls.AfStateEnum.Focused, controls.AfStateEnum.Failed):
                break
        if state == controls.AfStateEnum.Focused:
            break
        log.warning("autofocus attempt %d did not converge (state=%s lens=%s)",
                    attempt, state, pos)
    settled = state == controls.AfStateEnum.Focused
    if pos is not None:
        # Hold it: leaving AfMode in Auto lets a later trigger (or a mode
        # change) move the lens mid-recording.
        cam.set_controls({"AfMode": controls.AfModeEnum.Manual,
                          "LensPosition": float(pos)})
    log.info("autofocus %s at %s dioptres, holding",
             "settled" if settled else "did NOT settle", pos)
    return {"ok": settled, "lens": None if pos is None else float(pos), "settled": settled}


def apply_focus(cam, profile: dict, region=CENTER_REGION) -> float | None:
    """Put the lens where the profile says, on a started camera.

    A saved position is applied and held. Without one we autofocus once and hold
    wherever that landed — a bee hotel does not move, and continuous AF would
    hunt on every passing bee. Returns the resulting LensPosition, or None for a
    fixed-focus module (or if the lens never reported back).
    """
    if hasattr(cam, "apply_focus"):      # the OAK drives its own lens (motion/oak.py)
        if profile.get("oak_lens") is not None:
            return cam.apply_focus(profile.get("oak_lens"))
        return autofocus(cam, region).get("lens")

    from libcamera import controls

    if not has_focus(cam):
        log.info("camera %s is fixed focus — nothing to set", model_of(cam) or "?")
        return None

    if profile["lens"] is not None:
        cam.set_controls({"AfMode": controls.AfModeEnum.Manual,
                          "LensPosition": float(profile["lens"])})
        for _ in range(4):          # let the actuator arrive before we move on
            cam.capture_metadata()
        pos = cam.capture_metadata().get("LensPosition")
        log.info("lens set to %.2f dioptres (profile)", float(profile["lens"]))
        return pos

    log.info("no saved focus — autofocusing once on %s (range=%s)",
             "the centre" if region == CENTER_REGION else "the ROI", profile["af_range"])
    return autofocus(cam, region, profile["af_range"]).get("lens")


def warn_if_unrotatable(profile: dict) -> None:
    """A quarter turn cannot reach the recorded video, so say so loudly rather
    than quietly writing sideways clips that only look right in the focus tool."""
    if profile["rotate"] % 360:
        log.warning(
            "camera profile asks for %d° of rotation, which the recorder cannot "
            "do: the Pi's ISP flips but does not transpose, and the hardware "
            "encoder is fed straight from it. Recording UNROTATED. Turn the "
            "camera in its mount, or use hflip/vflip for 180°.",
            profile["rotate"] % 360)


def _is_number(v) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False
