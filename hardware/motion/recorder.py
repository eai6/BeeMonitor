"""Motion-gated recorder loop for BeeMonitor (cellular-friendly).

Records **only activity snippets** instead of fixed 10-minute chunks, so a
cellular uplink doesn't have to push hours of empty footage.

How it works
------------
``picamera2`` runs two streams at once (same trick ``main.py`` already uses):

    * ``main``  — full-res H.264, fed into a ``CircularOutput`` ring buffer that
      is *always* encoding but only flushed to disk when there's activity. The
      ring buffer is what gives us **pre-roll** (the seconds *before* motion).
    * ``lores`` — a small grayscale stream we run MOG2 background subtraction on
      to decide "is something moving?". This is the cheap half of BeeMonitor's
      detection stack (see ``motion.gate``); YOLO/tracking stay in the cloud,
      run later on the uploaded snippets.

Segment lifecycle:
    motion starts  -> open a new .h264 segment (ring buffer flushes pre-roll)
    motion ongoing -> keep writing
    motion stops   -> after POST_ROLL seconds, close + remux to .mp4
    (safety)       -> a segment is force-rotated after MAX_SEGMENT seconds

Snippets are written as ``RECORD_DIR/YYYY-MM-DD/YYYY-MM-DD_HH_MM_SS.mp4`` — the
exact convention ``uploader.py`` already scans for, so the existing uploader
service ships them unchanged.

IMPORTANT — on-device gating means a missed detection is permanent data loss.
Defaults favour *over*-triggering. Validate with the logged stats (triggers/hour,
snippet length) before trusting it in the field, then tighten thresholds.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import cv2

from motion.config import (
    log, RECORD_DIR, WORK_DIR, MAIN_W, MAIN_H, LORES_W, LORES_H, FPS,
    PRE_ROLL, POST_ROLL, MAX_SEGMENT, WARMUP_SECONDS, TIMESTAMP_OVERLAY,
    DETECT_EVERY_N, BG_RESET_INTERVAL,
    CALIB_FILE, TUNING_FILE, ROI_OVERRIDE_FILE,
    CALIB_RELOAD_SECONDS, OVERRIDE_RELOAD_SECONDS,
    TELEMETRY_IMAGE_INTERVAL, TELEMETRY_QUEUE,
    FRAME_MAX_CANDIDATES, FRAME_CAPTURE_INTERVAL, BEE_CONFIRM_MODE_FILE,
    ACTIVITY_FRAMES_FILE, SAVE_ACTIVITY_FRAMES,
    RECORD_SETTINGS_FILE, CONTINUOUS_SEGMENT,
)
from motion.camera import (
    load_profile as load_camera_profile, transform as camera_transform,
    apply_focus, warn_if_unrotatable, describe as describe_camera, model_of,
)
from motion.frames import _main_array_to_bgr
from motion.roi import _resolve_record_roi
from motion.overrides import (
    _build_gate, load_calibration, _apply_calibration,
    load_tuning, _apply_tuning, load_roi_override_lores, load_bee_confirm_mode,
    load_activity_crops_mode, load_record_settings,
)
from motion.remux import _remux, _snippet_paths
from motion.telemetry_still import _save_telemetry_still
from motion.activity_frames import (
    _largest_blob, _mover_crop, _flush_activity_frames,
    _encode_source, _save_activity_archive,
)
from motion.confirm import BeeConfirmer, CONFIRMED, UNCONFIRMED, DISABLED

# picamera2 only exists on the Pi. Keep the import soft so the module can be
# imported off-device (e.g. for linting); record() requires it.
try:
    import libcamera
    from picamera2 import MappedArray, Picamera2
    from picamera2.encoders import H264Encoder
    from picamera2.outputs import CircularOutput
    HAVE_PICAMERA2 = True
except ImportError:  # pragma: no cover - not on a Pi
    HAVE_PICAMERA2 = False


_running = True


def _handle_signal(signum, frame):  # noqa: ARG001
    global _running
    log.info("signal %s received, finalising current segment then exiting", signum)
    _running = False


def _should_send_crops(crops_mode, status) -> bool:
    """Whether to ship a finished activity's crops, given the dashboard crop mode
    and the bee-confirmation verdict:
      all       — every activity
      confirmed — only confirmed bees; DISABLED means no confirmer ran (nothing to
                  reject), so send. UNCONFIRMED/PENDING are dropped.
      off       — never (and sampling is already gated off upstream)
    """
    if crops_mode == "off":
        return False
    if crops_mode == "all":
        return True
    return status == CONFIRMED or status == DISABLED  # "confirmed"


def _write_clip_tag(mp4_path, status, confidence, taxon, runs, mode) -> None:
    """Write the per-clip bee-confirmation tag next to the .mp4 (atomic).

    ``<stem>.bee.json`` rides with the uploaded clip so telemetry counts only
    confirmed activities and the cloud can flag/route. Best-effort; never raises.
    """
    if mp4_path is None:
        return
    tag = {
        "confirm_status": status,
        "bee_confidence": round(float(confidence), 4),
        "taxon": taxon,
        "confirm_runs": int(runs),
        "mode": mode,
    }
    try:
        # `<clip>.mp4.bee.json` — sits beside the clip (uploader idiom, like
        # `.uploaded`/`.usb`) so it ships with the upload and the cloud can flag it.
        out = mp4_path.with_suffix(mp4_path.suffix + ".bee.json")
        tmp = out.with_name(out.name + ".part")
        tmp.write_text(json.dumps(tag))
        os.replace(tmp, out)
    except OSError as e:  # pragma: no cover - disk hiccup mustn't crash recording
        log.warning("could not write clip tag for %s: %s", getattr(mp4_path, "name", "?"), e)
    # Cheap O(1) markers for telemetry's per-hour histograms (no JSON read per
    # clip every beat). In `gate` mode: a rejected clip gets `.unconfirmed` (still
    # recorded + uploaded, not counted as bee activity); a positively-confirmed
    # bee gets `.confirmed` (a strict subset, so the dashboard's "Confirmed" filter
    # can show on-card bees before upload without counting untagged clips). Off/
    # disabled clips get neither (untagged). Keep the two markers mutually exclusive.
    try:
        unconf = mp4_path.with_suffix(mp4_path.suffix + ".unconfirmed")
        conf = mp4_path.with_suffix(mp4_path.suffix + ".confirmed")
        if mode == "gate" and status == UNCONFIRMED:
            unconf.touch()
            if conf.exists():
                conf.unlink()
        elif mode == "gate" and status == CONFIRMED:
            conf.touch()
            if unconf.exists():
                unconf.unlink()  # late confirm — promote it
        else:
            # off / disabled / pending → untagged: clear any stale markers.
            if unconf.exists():
                unconf.unlink()
            if conf.exists():
                conf.unlink()
    except OSError:
        pass


def _in_record_window(window) -> bool:
    """True when the current DEVICE-LOCAL hour is inside the window.

    window = (start_hour, end_hour) or None (all day). start > end wraps past
    midnight (e.g. (20, 6) = evenings + nights). The system timezone is kept
    correct remotely (GPS-derived tz via the heartbeat), so local hours are
    meaningful even on field units."""
    if not window:
        return True
    start, end = window
    h = datetime.now().hour
    if start < end:
        return start <= h < end
    return h >= start or h < end


def record() -> None:
    """Main capture loop. Blocks until SIGTERM/SIGINT."""
    if not HAVE_PICAMERA2:
        raise RuntimeError("record() needs picamera2 — run this on the Pi")
    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    # Orientation and focus come from the per-unit camera profile (camera.json,
    # written by runFocus.py) over the env defaults — see motion/camera.py.
    cam_profile = load_camera_profile()
    warn_if_unrotatable(cam_profile)

    cam = Picamera2()
    config = cam.create_video_configuration(
        main={"size": (MAIN_W, MAIN_H)},
        lores={"size": (LORES_W, LORES_H), "format": "YUV420"},
        controls={"FrameRate": FPS},
        transform=camera_transform(cam_profile),
    )
    cam.configure(config)

    if TIMESTAMP_OVERLAY:
        font = cv2.FONT_HERSHEY_SIMPLEX
        def _apply_timestamp(request):
            ts = time.strftime("%Y-%m-%d %X")
            with MappedArray(request, "main") as m:
                cv2.putText(m.array, ts, (10, 30), font, 0.8, (0, 255, 0), 2)
        cam.pre_callback = _apply_timestamp

    # repeat=True re-emits SPS/PPS so each flushed segment is independently
    # decodable; a ~1s keyframe interval lets segments start cleanly.
    encoder = H264Encoder(repeat=True, iperiod=FPS)
    circ = CircularOutput(buffersize=max(1, int(PRE_ROLL * FPS)))
    encoder.output = circ

    cam.start_encoder(encoder)
    cam.start()

    # Focus BEFORE detecting the hotel: on a lens module the frame is only as
    # sharp as the last thing that set LensPosition, and YOLO on a blurred frame
    # is exactly how hotel detection ends up falling back to the whole frame.
    # apply_focus logs what it actually did with the lens.
    apply_focus(cam, cam_profile)

    # Cloud-faithful step 1: detect the hotel and confine detection to it before
    # we start recording. Falls back to the whole frame if detection fails.
    roi = _resolve_record_roi(cam)

    log.info(
        "recorder up: %s main=%dx%d lores=%dx%d @ %dfps | %s | pre=%.1fs "
        "post=%.1fs max=%.0fs roi=%s",
        model_of(cam) or "camera", MAIN_W, MAIN_H, LORES_W, LORES_H, FPS,
        describe_camera(cam_profile), PRE_ROLL, POST_ROLL, MAX_SEGMENT,
        roi or "full",
    )

    gate = _build_gate(roi)
    # Async YOLO bee-confirmation. Initial mode = dashboard push (if any) else env;
    # switchable at runtime via the override-reload tick below.
    confirmer = BeeConfirmer(mode=load_bee_confirm_mode())
    remux_pool = []  # list of threading.Thread for in-flight remuxes

    def _spawn_remux(h264_path: Path, mp4_path: Path):
        t = threading.Thread(target=_remux, args=(h264_path, mp4_path), daemon=True)
        t.start()
        remux_pool.append(t)
        remux_pool[:] = [t for t in remux_pool if t.is_alive()]

    # State machine.
    encoding = False
    seg_start = 0.0
    last_motion = 0.0
    cur_h264: Path | None = None
    cur_mp4: Path | None = None

    # Activity-frame sampling state — one activity == one open segment.
    act_uid: "str | None" = None
    act_started = 0.0
    act_cands: list = []
    act_last_cap = 0.0
    act_confirm_submits = 0   # full frames handed to the confirmer this activity

    # Per-segment / rolling stats for tuning.
    triggers = 0
    total_clip_time = 0.0
    warmup_deadline = time.monotonic() + WARMUP_SECONDS
    frame_i = 0
    last_stats_log = time.monotonic()

    # Periodic background-model rebuild so the gate tracks slow scene changes.
    # The model is already fresh here (gate just built + warmup below).
    next_bg_reset = (
        time.monotonic() + BG_RESET_INTERVAL if BG_RESET_INTERVAL > 0 else float("inf"))

    # Hot-reload of calibration.json written by the scheduled --calibrate job.
    calib_mtime = CALIB_FILE.stat().st_mtime if CALIB_FILE.exists() else 0.0
    tuning_mtime = TUNING_FILE.stat().st_mtime if TUNING_FILE.exists() else 0.0
    roi_ov_mtime = ROI_OVERRIDE_FILE.stat().st_mtime if ROI_OVERRIDE_FILE.exists() else 0.0
    bee_mode_mtime = BEE_CONFIRM_MODE_FILE.stat().st_mtime if BEE_CONFIRM_MODE_FILE.exists() else 0.0
    # Dashboard-pushed crop mode: all|confirmed|off (env ACTIVITY_CROPS_MODE is the
    # fallback). act_frames_on gates sampling (mode != off); crops_mode also decides,
    # per verdict, whether a finished activity's crops are sent (see _should_send_crops).
    crops_mode = load_activity_crops_mode()
    act_frames_on = crops_mode != "off"
    act_frames_mtime = ACTIVITY_FRAMES_FILE.stat().st_mtime if ACTIVITY_FRAMES_FILE.exists() else 0.0
    # Recording mode (motion|continuous) + daily hour window, dashboard-pushed
    # via record_settings.json (env fallback) and hot-reloaded below.
    rec_mode, rec_window, rec_post_roll, rec_max_segment = load_record_settings()
    rec_settings_mtime = (RECORD_SETTINGS_FILE.stat().st_mtime
                          if RECORD_SETTINGS_FILE.exists() else 0.0)
    log.info("record settings: mode=%s window=%s post_roll=%.0fs max_seg=%.0fs",
             rec_mode, rec_window or "all-day", rec_post_roll, rec_max_segment)
    last_calib_check = time.monotonic()
    last_override_check = time.monotonic()

    # First telemetry still shortly after warmup, then every interval.
    next_telemetry_image = (
        warmup_deadline if TELEMETRY_IMAGE_INTERVAL > 0 else float("inf"))

    def _open_segment(now_mono: float, reason: str):
        nonlocal encoding, seg_start, cur_h264, cur_mp4, triggers
        nonlocal act_uid, act_started, act_cands, act_last_cap, act_confirm_submits
        cur_h264, cur_mp4 = _snippet_paths(datetime.now())
        circ.fileoutput = str(cur_h264)
        circ.start()
        encoding = True
        seg_start = now_mono
        triggers += 1
        # Start a fresh activity (used by both crop sampling and bee confirmation).
        # The short random suffix keeps the uid unique even if two clips open in
        # the same wall-clock second (e.g. a max-len rotation that reopens).
        act_uid = f"{cur_mp4.stem}-{uuid.uuid4().hex[:4]}"
        act_started = time.time()
        act_cands = []
        act_last_cap = 0.0
        act_confirm_submits = 0
        log.info("clip START (%s) -> %s", reason, cur_mp4.name)

    def _close_segment(now_mono: float, reason: str):
        nonlocal encoding, cur_h264, cur_mp4, total_clip_time
        nonlocal act_uid, act_cands
        circ.stop()
        encoding = False
        dur = now_mono - seg_start
        total_clip_time += dur
        log.info("clip STOP (%s) len=%.1fs -> remux %s", reason, dur, cur_mp4.name)
        mp4 = cur_mp4
        _spawn_remux(cur_h264, cur_mp4)
        cur_h264 = cur_mp4 = None
        if confirmer.active and act_uid:
            # Defer: the clip is already recorded; the verdict (async) decides the
            # tag + whether to count it + whether to send its crops. Finalised in
            # the drain handler in the main loop.
            confirmer.register_close(
                act_uid, {"mp4": mp4, "started": act_started, "crops": act_cands},
                now_mono)
        elif act_frames_on and act_uid:
            # No confirmation (mode=off or inert). Archive durably (crop + source
            # frame, WiFi-uploaded) and send all crops over cellular.
            _save_activity_archive(mp4, act_uid, act_started, act_cands)
            _flush_activity_frames(act_uid, act_started, act_cands)
        act_uid = None
        act_cands = []

    try:
        while _running:
            # Blocks until the next lores frame ~= paces the loop at FPS.
            buf = cam.capture_buffer("lores")
            now_mono = time.monotonic()
            frame_i += 1

            # Y-plane of YUV420 == grayscale, first W*H bytes (no padding at
            # 32-aligned widths).
            gray = buf[:LORES_W * LORES_H].reshape(LORES_H, LORES_W)

            # Warmup: build the background model, never trigger.
            if now_mono < warmup_deadline:
                gate.warm(gray)
                continue

            motion = False
            if frame_i % DETECT_EVERY_N == 0:
                motion, n_blobs, area = gate.update(gray)
                if motion:
                    last_motion = now_mono
            else:
                # Still keep the bg model current on skipped frames.
                gate.warm(gray)

            # Open decisions, gated by recording being ON and the daily hour
            # window (both modes): continuous keeps a clip open through the whole
            # window; motion opens only on a trigger. mode "off" never opens
            # (recording disabled from the dashboard).
            in_window = _in_record_window(rec_window)
            if not encoding and rec_mode != "off" and in_window:
                if rec_mode == "continuous":
                    _open_segment(now_mono, "continuous")
                elif motion:
                    _open_segment(now_mono, "motion")

            # While a clip is open, grab a full main still (the costly bit) at most
            # once every FRAME_CAPTURE_INTERVAL, capped per clip — and feed that ONE
            # capture to BOTH consumers: crop sampling (cloud BioCLIP) and bee
            # confirmation (YOLO). The confirmer submit is fire-and-forget (async),
            # so a 1–3 s inference never delays capture or drops a fast bee.
            want_crop = act_frames_on and len(act_cands) < FRAME_MAX_CANDIDATES
            want_confirm = (confirmer.active and act_uid is not None
                            and act_confirm_submits < FRAME_MAX_CANDIDATES
                            and not confirmer.is_confirmed(act_uid))
            if (encoding and motion and gate.last_blobs
                    and (want_crop or want_confirm)
                    and (now_mono - act_last_cap) >= FRAME_CAPTURE_INTERVAL):
                try:
                    main_bgr = _main_array_to_bgr(cam.capture_array("main"))
                    if want_confirm:
                        confirmer.submit(act_uid, main_bgr, gate.last_blobs, gate.roi)
                        act_confirm_submits += 1
                    if want_crop:
                        blob = _largest_blob(gate.last_blobs)
                        if blob is not None:
                            sample = _mover_crop(main_bgr, blob, gate.roi)
                            if sample is not None:
                                jpg, bbox, wh = sample
                                cand = {
                                    "jpg": jpg, "bbox": bbox, "wh": wh,
                                    "area": float(blob[4]), "captured_at": time.time(),
                                }
                                # Keep the full source frame the crop came from, so
                                # the durable archive can save it (uploaded over WiFi).
                                if SAVE_ACTIVITY_FRAMES:
                                    src = _encode_source(main_bgr)
                                    if src is not None:
                                        cand["src_jpg"], cand["src_wh"] = src
                                act_cands.append(cand)
                    act_last_cap = now_mono
                except Exception as e:  # never crash recording over a sample
                    log.warning("activity frame capture failed: %s", e)

            if encoding:
                # Recording turned off (dashboard) closes the open clip now.
                if rec_mode == "off":
                    _close_segment(now_mono, "recording-off")
                # The window closing ends the clip in EITHER mode.
                elif not in_window:
                    _close_segment(now_mono, "window-end")
                elif rec_mode == "continuous":
                    # Continuous: rotate on the segment length; never close on
                    # idle (the whole window is recorded).
                    if now_mono - seg_start >= CONTINUOUS_SEGMENT:
                        _close_segment(now_mono, "segment")
                        _open_segment(now_mono, "continuous")
                # Force-rotate over-long segments (stuck scene / wind in frame).
                elif now_mono - seg_start >= rec_max_segment:
                    _close_segment(now_mono, "max-len")
                    if motion:  # still active -> immediately start a fresh clip
                        _open_segment(now_mono, "motion-continued")
                # Normal close: no motion for the (dashboard-tunable) clip tail.
                elif (now_mono - last_motion) >= rec_post_roll:
                    _close_segment(now_mono, "idle")

            # Resolve finished bee-confirmation verdicts (async, off the hot path):
            # finalise each closed clip's tag and, per mode, send or suppress its
            # crops. The video is already recorded + remuxed regardless.
            if confirmer.active:
                for r in confirmer.drain_resolved(now_mono):
                    p = r["payload"] or {}
                    _write_clip_tag(p.get("mp4"), r["status"], r["confidence"],
                                    r["taxon"], r["runs"], confirmer.mode)
                    # Durable archive for EVERY sampled activity (crop + source
                    # frame, WiFi-uploaded) so nothing is lost regardless of the
                    # cellular send decision below.
                    _save_activity_archive(p.get("mp4"), r["uid"],
                                           p.get("started", 0.0), p.get("crops"))
                    # The dashboard crop mode decides which verdicts get their crops
                    # shipped over cellular (all | confirmed | off).
                    if _should_send_crops(crops_mode, r["status"]) and p.get("crops"):
                        _flush_activity_frames(r["uid"], p.get("started", 0.0), p["crops"])

            # Periodic background-model rebuild. Deferred while a clip is open so
            # an active capture isn't cut short; fires as soon as the scene idles.
            if now_mono >= next_bg_reset and not encoding:
                gate.reset()
                warmup_deadline = now_mono + WARMUP_SECONDS  # re-learn before trusting motion
                next_bg_reset = now_mono + BG_RESET_INTERVAL
                log.info("background model rebuilt (every %.0fs); re-warming %.1fs",
                         BG_RESET_INTERVAL, WARMUP_SECONDS)

            # Background calibration.json (the scheduled --calibrate job) — slow
            # reload; tuning is re-applied on top so it always wins.
            if now_mono - last_calib_check >= CALIB_RELOAD_SECONDS:
                last_calib_check = now_mono
                try:
                    cm = CALIB_FILE.stat().st_mtime if CALIB_FILE.exists() else 0.0
                except OSError:
                    cm = calib_mtime
                if cm != calib_mtime:
                    _apply_calibration(gate, load_calibration())
                    _apply_tuning(gate, load_tuning())
                    log.info("reloaded calibration: var=%.0f area=[%.0f, %.0f] "
                             "min_blobs=%d", gate.var_threshold, gate.min_area,
                             gate.max_area, gate.min_blobs)
                    calib_mtime = cm

            # Interactive dashboard edits — motion tuning + hotel-ROI override.
            # Fast reload so a change made on the dashboard shows within seconds,
            # not minutes (these only do work when the file's mtime changes).
            if now_mono - last_override_check >= OVERRIDE_RELOAD_SECONDS:
                last_override_check = now_mono
                try:
                    tm = TUNING_FILE.stat().st_mtime if TUNING_FILE.exists() else 0.0
                except OSError:
                    tm = tuning_mtime
                if tm != tuning_mtime:
                    # Re-establish the calibration base, then overlay tuning so a
                    # field cleared back to "auto" falls back to the calibrated value.
                    _apply_calibration(gate, load_calibration())
                    _apply_tuning(gate, load_tuning())
                    log.info("reloaded motion tuning: var=%.0f area=[%.0f, %.0f] "
                             "min_blobs=%d", gate.var_threshold, gate.min_area,
                             gate.max_area, gate.min_blobs)
                    tuning_mtime = tm
                # Live hotel-ROI override edits: re-gate without a restart. A new
                # crop size needs a fresh background, so reset + re-warm.
                try:
                    rm = ROI_OVERRIDE_FILE.stat().st_mtime if ROI_OVERRIDE_FILE.exists() else 0.0
                except OSError:
                    rm = roi_ov_mtime
                if rm != roi_ov_mtime:
                    new_roi = load_roi_override_lores()
                    if new_roi is not None:
                        gate.roi = new_roi
                        gate.reset()
                        warmup_deadline = now_mono + WARMUP_SECONDS
                        log.info("applied ROI override %s — re-warming %.1fs",
                                 new_roi, WARMUP_SECONDS)
                    roi_ov_mtime = rm
                # Live bee-confirmation mode switch (off|tag|gate) from the dashboard.
                try:
                    bm = (BEE_CONFIRM_MODE_FILE.stat().st_mtime
                          if BEE_CONFIRM_MODE_FILE.exists() else 0.0)
                except OSError:
                    bm = bee_mode_mtime
                if bm != bee_mode_mtime:
                    confirmer.set_mode(load_bee_confirm_mode())
                    bee_mode_mtime = bm
                # Live crop mode (all|confirmed|off) from the dashboard.
                try:
                    am = (ACTIVITY_FRAMES_FILE.stat().st_mtime
                          if ACTIVITY_FRAMES_FILE.exists() else 0.0)
                except OSError:
                    am = act_frames_mtime
                if am != act_frames_mtime:
                    new_mode = load_activity_crops_mode()
                    if new_mode != crops_mode:
                        crops_mode = new_mode
                        act_frames_on = new_mode != "off"
                        log.info("activity-crop mode %s (dashboard)", new_mode)
                    act_frames_mtime = am
                # Live recording mode + hour window from the dashboard. An open
                # clip isn't cut here — the state machine above closes/rotates
                # it naturally on the next frame under the new rules.
                try:
                    rsm = (RECORD_SETTINGS_FILE.stat().st_mtime
                           if RECORD_SETTINGS_FILE.exists() else 0.0)
                except OSError:
                    rsm = rec_settings_mtime
                if rsm != rec_settings_mtime:
                    new = load_record_settings()
                    if new != (rec_mode, rec_window, rec_post_roll, rec_max_segment):
                        rec_mode, rec_window, rec_post_roll, rec_max_segment = new
                        log.info("record settings -> mode=%s window=%s post_roll=%.0fs "
                                 "max_seg=%.0fs (dashboard)", rec_mode,
                                 rec_window or "all-day", rec_post_roll, rec_max_segment)
                    rec_settings_mtime = rsm

            # On-demand still requested by telemetry (picture / live view / ROI
            # editor): the telemetry service drops capture.request. We send a CLEAN
            # frame — the dashboard ROI editor overlays the hotel ROI + nest tubes
            # as editable boxes from the device's stored layout.
            req_file = TELEMETRY_QUEUE / "capture.request"
            if req_file.exists():
                _save_telemetry_still(cam)
                try:
                    req_file.unlink()
                except OSError:
                    pass

            # Optional periodic still (off by default; TELEMETRY_IMAGE_INTERVAL=0).
            if now_mono >= next_telemetry_image:
                _save_telemetry_still(cam)
                next_telemetry_image = now_mono + TELEMETRY_IMAGE_INTERVAL

            # Periodic stats line for field tuning.
            if now_mono - last_stats_log >= 300:
                mins = (now_mono - last_stats_log) / 60.0
                msg = ("stats: %d clips in last %.0f min, %.1fs recorded (%.1f%% duty)"
                       % (triggers, mins, total_clip_time,
                          100.0 * total_clip_time / (now_mono - last_stats_log)))
                if confirmer.active:
                    cs = confirmer.stats()
                    msg += (" | confirm: %d ok %d no, %d inferences @ %.0fms avg"
                            % (cs["confirmed"], cs["unconfirmed"], cs["inferences"],
                               cs["mean_ms"]))
                log.info("%s", msg)
                triggers = 0
                total_clip_time = 0.0
                last_stats_log = now_mono

    finally:
        if encoding:
            _close_segment(time.monotonic(), "shutdown")
        # Resolve any verdicts that already settled; pending ones fail closed
        # (the clip is recorded + uploaded, just untagged-confirmed).
        if confirmer.active:
            for r in confirmer.drain_resolved(time.monotonic()):
                p = r["payload"] or {}
                _write_clip_tag(p.get("mp4"), r["status"], r["confidence"],
                                r["taxon"], r["runs"], confirmer.mode)
                _save_activity_archive(p.get("mp4"), r["uid"],
                                       p.get("started", 0.0), p.get("crops"))
                if _should_send_crops(crops_mode, r["status"]) and p.get("crops"):
                    _flush_activity_frames(r["uid"], p.get("started", 0.0), p["crops"])
        confirmer.stop()
        try:
            cam.stop()
            cam.stop_encoder()
        except Exception:  # pragma: no cover - best-effort teardown
            pass
        for t in remux_pool:
            t.join(timeout=30)
        log.info("recorder stopped")
