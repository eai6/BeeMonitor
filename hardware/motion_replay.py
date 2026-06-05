#!/usr/bin/env python3
"""Offline replay of the motion-gated snippet pipeline against a video file.

The live recorder (`main_motion.record`) reads the Pi camera's lores stream and
uses the hardware H264 encoder, so it can't ingest an mp4. This tool exercises
the *same* motion gate (`main_motion.MotionGate`, same thresholds/constants) and
the *same* segmentation state machine (warmup -> detect-every-N -> pre-roll /
post-roll / max-segment), but sourced from a video file and writing snippets
with ffmpeg. Use it to sanity-check what clips the gate would cut from a sample.

    python3 hardware/motion_replay.py short_videos/clip.mp4
    python3 hardware/motion_replay.py short_videos/clip.mp4 --out /tmp/snips --warmup 1.0
    python3 hardware/motion_replay.py short_videos/clip.mp4 --roi 120,40,520,300
    python3 hardware/motion_replay.py short_videos/clip.mp4 --full-frame

Like production, it first runs the hotel detector (`nest_detection.pt`) on the
clip's first frame, scales that hotel box into lores coords, and confines the
motion gate to it — so a replay uses the *exact same ROI* the field unit would.
`--roi` overrides with a manual lores box (== BEEMONITOR_ROI); `--full-frame`
skips detection. If detection is unavailable/fails it falls back to the whole
frame, same as production.

Detection runs on each frame downscaled to the recorder's lores size (so the
blob-area thresholds stay valid). Snippets are re-encoded (frame-accurate) from
the original-resolution file. This validates motion gating + segmentation, not
the hardware encoder path.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import main_motion as mm  # noqa: E402  (reuse the exact gate + constants)


def _pct(vals, q):
    return float(np.percentile(vals, q)) if len(vals) else 0.0


def _print_sensitivity(stats: dict) -> None:
    """How trigger-happy was the gate? Shows the % of evaluated frames that
    fired and the size distribution of every blob it kept — so you can tell
    whether triggers are bee-sized or noise, and where to set --min-area."""
    ev = stats["evaluated"]
    if not ev:
        return
    mf = stats["motion_frames"]
    areas = stats["areas"]
    counts = stats["blob_counts"]
    print("sensitivity:")
    print(f"  motion fired on {mf}/{ev} evaluated frames ({100*mf/ev:.0f}%)")
    if areas:
        print(f"  kept blobs: n={len(areas)} area px "
              f"min={min(areas):.0f} p50={_pct(areas,50):.0f} "
              f"p90={_pct(areas,90):.0f} p99={_pct(areas,99):.0f} max={max(areas):.0f}")
    print(f"  blobs/frame: p50={_pct(counts,50):.0f} p90={_pct(counts,90):.0f} "
          f"max={max(counts) if counts else 0}")


def _resolve_roi(video: Path, manual: str | None, full_frame: bool):
    """Decide the lores-coord detection ROI, mirroring production priority:
    explicit --roi (lores coords) > hotel auto-detection > whole frame.

    Production grabs a main-stream frame and runs nest_detection.pt to find the
    hotel, then scales that box into lores coords. We do the same here off the
    video's first frame so a replay uses the *exact same ROI* as the field unit.
    Returns (x1, y1, x2, y2) in lores coords, or None for the whole frame.
    """
    if manual:
        try:
            x1, y1, x2, y2 = (int(v) for v in manual.split(","))
        except ValueError:
            raise SystemExit(f"bad --roi {manual!r} (want x1,y1,x2,y2 in lores px)")
        print(f"  roi : manual {(x1, y1, x2, y2)} (lores coords)")
        return (x1, y1, x2, y2)
    if full_frame:
        print("  roi : full frame (--full-frame)")
        return None

    # Hotel auto-detection on the first frame — the production step 1.
    cap = cv2.VideoCapture(str(video))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("  roi : full frame (could not read first frame)")
        return None
    vh, vw = frame.shape[:2]
    roi_native = mm.detect_hotel_roi(frame)  # nest_detection.pt; None on failure
    if roi_native is None:
        print("  roi : full frame (hotel detection unavailable/failed — same "
              "fallback as production)")
        return None
    roi_lores = mm._scale_roi(roi_native, (vw, vh), (mm.LORES_W, mm.LORES_H))
    print(f"  roi : hotel {tuple(roi_native)} @ {vw}x{vh} -> {roi_lores} (lores)")
    return roi_lores


def _segments_for(video: Path, roi, warmup_s: float, detect_every: int,
                  pre_roll: float, post_roll: float, max_seg: float,
                  min_area: float, max_area: float, min_blobs: int,
                  var_threshold: float):
    """Replay the gate over `video`; return (fps, n_frames, segments, stats).

    `segments` is [(start_f, end_f, reason)]. `stats` quantifies sensitivity:
    how many evaluated frames registered motion and the size of every blob the
    gate *kept* (so we can see whether triggers are bee-sized or noise).
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or mm.FPS
    warmup_frames = int(warmup_s * fps)
    post_frames = int(post_roll * fps)
    pre_frames = int(pre_roll * fps)
    max_frames = int(max_seg * fps)

    gate = mm.MotionGate(roi=roi, var_threshold=var_threshold, min_area=min_area,
                         max_area=max_area, min_blobs=min_blobs)
    segments = []
    stats = {"evaluated": 0, "motion_frames": 0, "blob_counts": [], "areas": []}
    encoding = False
    seg_start = last_motion = 0
    i = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        gray = cv2.cvtColor(
            cv2.resize(frame, (mm.LORES_W, mm.LORES_H)), cv2.COLOR_BGR2GRAY)

        if i < warmup_frames:        # warmup: learn bg, never trigger
            gate.warm(gray)
            continue

        motion = False
        if i % detect_every == 0:
            motion, n_blobs, area = gate.update(gray)
            stats["evaluated"] += 1
            stats["blob_counts"].append(n_blobs)
            stats["areas"].extend(b[4] for b in gate.last_blobs)
            if motion:
                last_motion = i
                stats["motion_frames"] += 1
        else:
            gate.warm(gray)

        if motion and not encoding:
            encoding = True
            seg_start = i
            start_f = max(0, i - pre_frames)   # pre-roll
            if segments:
                # The live recorder's CircularOutput never double-records; here we
                # compute pre-roll from the source, so a new clip's pre-roll could
                # reach back into the previous clip. Clamp so snippets don't overlap.
                start_f = max(start_f, segments[-1][1])
            print(f"  clip START (motion) @ {i/fps:5.2f}s "
                  f"(blobs={n_blobs}, area={area:.0f})")
            segments.append([start_f, i, "motion"])
        elif encoding:
            if i - seg_start >= max_frames:
                segments[-1][1] = i
                segments[-1][2] = "max-len"
                print(f"  clip STOP  (max-len) @ {i/fps:5.2f}s")
                encoding = False
            elif (i - last_motion) >= post_frames:
                end_f = last_motion + post_frames     # post-roll
                segments[-1][1] = end_f
                print(f"  clip STOP  (idle)    @ {end_f/fps:5.2f}s "
                      f"(last motion {last_motion/fps:.2f}s)")
                encoding = False

    if encoding:
        segments[-1][1] = i
        segments[-1][2] = "eof"
        print(f"  clip STOP  (eof)     @ {i/fps:5.2f}s")
    cap.release()
    return fps, i + 1, segments, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--out", type=Path, default=None,
                    help="snippet output dir (default: <video_dir>/replay_snippets)")
    ap.add_argument("--warmup", type=float, default=1.0,
                    help="MOG2 warmup seconds (production=%.1f; lower for short "
                         "clips) [default: 1.0]" % mm.WARMUP_SECONDS)
    ap.add_argument("--detect-every", type=int, default=mm.DETECT_EVERY_N)
    ap.add_argument("--pre-roll", type=float, default=mm.PRE_ROLL)
    ap.add_argument("--post-roll", type=float, default=mm.POST_ROLL)
    ap.add_argument("--max-seg", type=float, default=mm.MAX_SEGMENT)
    ap.add_argument("--roi", default=None,
                    help="manual ROI 'x1,y1,x2,y2' in lores px (overrides hotel "
                         "detection; same as BEEMONITOR_ROI)")
    ap.add_argument("--full-frame", action="store_true",
                    help="skip hotel detection and gate on the whole frame")
    # Sensitivity knobs — defaults are the production thresholds; override to sweep.
    ap.add_argument("--min-area", type=float, default=mm.MIN_BLOB_AREA,
                    help="min blob area in lores px to count (production=%.0f)" % mm.MIN_BLOB_AREA)
    ap.add_argument("--max-area", type=float, default=mm.MAX_BLOB_AREA,
                    help="max blob area in lores px (production=%.0f)" % mm.MAX_BLOB_AREA)
    ap.add_argument("--min-blobs", type=int, default=mm.MIN_MOTION_BLOBS,
                    help="qualifying blobs needed to call it motion (production=%d)" % mm.MIN_MOTION_BLOBS)
    ap.add_argument("--var", type=float, default=mm.MOG2_VAR_THRESHOLD,
                    help="MOG2 var threshold; higher = less sensitive (production=%.0f)" % mm.MOG2_VAR_THRESHOLD)
    ap.add_argument("--no-cut", action="store_true",
                    help="don't write snippet files (fast sensitivity sweeps)")
    args = ap.parse_args()

    if not args.video.exists():
        raise SystemExit(f"no such file: {args.video}")
    out_dir = args.out or (args.video.parent / "replay_snippets")
    out_dir.mkdir(parents=True, exist_ok=True)

    star = lambda v, d: "" if v == d else " *"  # noqa: E731 - mark overridden knobs
    print(f"replaying {args.video.name} through MotionGate")
    print(f"  gate: lores={mm.LORES_W}x{mm.LORES_H} detect_every={args.detect_every} "
          f"area=[{args.min_area:.0f},{args.max_area:.0f}]{star(args.min_area, mm.MIN_BLOB_AREA)} "
          f"min_blobs={args.min_blobs}{star(args.min_blobs, mm.MIN_MOTION_BLOBS)} "
          f"var={args.var:.0f}{star(args.var, mm.MOG2_VAR_THRESHOLD)}  (* = overridden)")
    print(f"  seg : warmup={args.warmup}s pre={args.pre_roll}s post={args.post_roll}s "
          f"max={args.max_seg}s")

    roi = _resolve_roi(args.video, args.roi, args.full_frame)

    fps, n_frames, segments, stats = _segments_for(
        args.video, roi, args.warmup, args.detect_every,
        args.pre_roll, args.post_roll, args.max_seg,
        args.min_area, args.max_area, args.min_blobs, args.var)

    dur = n_frames / fps
    print(f"\nsource: {n_frames} frames @ {fps:.2f}fps = {dur:.2f}s")
    _print_sensitivity(stats)
    if not segments:
        print("=> NO motion snippets (try a lower --warmup, or the clip has no "
              "motion the gate considers bee-sized).")
        return 0

    # Cut each [start,end] from the original file (frame-accurate re-encode).
    total_clip = 0.0
    dest = "(--no-cut: not written)" if args.no_cut else f"-> {out_dir}/"
    print(f"\n=> {len(segments)} snippet(s) {dest}")
    for n, (s, e, reason) in enumerate(segments, 1):
        start_s, end_s = s / fps, e / fps
        clip_len = end_s - start_s
        total_clip += clip_len
        name = f"snippet_{n:02d}_{start_s:05.1f}-{end_s:05.1f}s.mp4"
        flag = ""
        if not args.no_cut:
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(args.video),
                   "-ss", f"{start_s:.3f}", "-to", f"{end_s:.3f}", "-c:v", "libx264",
                   "-preset", "veryfast", "-an", str(out_dir / name)]
            if subprocess.run(cmd).returncode != 0:
                flag = "  [ffmpeg FAILED]"
        print(f"  {name}  ({clip_len:.2f}s, {reason}){flag}")

    print(f"\nrecorded {total_clip:.2f}s of {dur:.2f}s "
          f"({100*total_clip/dur:.0f}% duty)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
