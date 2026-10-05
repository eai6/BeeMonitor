#!/usr/bin/env python3
"""How much detail can this unit's OAK keep? (memory/42 part A)

Records a short clip at each codec x bitrate, then reports, per setting:
frames delivered vs expected (dropped frames = the encoder can't hold it),
achieved fps, actual Mbit/s, MB per minute, and saves one frame of each clip
as a lossless PNG so the detail can be compared side by side.

The recorder holds the OAK, so stop it first:

    sudo systemctl stop beemonitor-recorder
    cd /home/beemonitor/BeeMonitor/hardware      # wherever the repo lives
    python3 oak_quality_test.py                  # default sweep, ~4 min
    python3 oak_quality_test.py --codecs h265 --bitrates 60000,100000 \\
        --isp 0,0,0 --max-exposure-us 1000 --seconds 30
    sudo systemctl start beemonitor-recorder

Point the camera at something with fine texture (the hotel, a printed page),
in the light it will record in. Results go to ./oak_quality/<timestamp>/.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from motion.config import LORES_W, LORES_H, OAK_MAIN_W, OAK_MAIN_H, OAK_FPS  # noqa: E402
from motion.oak import OakCamera  # noqa: E402
from motion.remux import _remux  # noqa: E402


def _ints(text):
    return [int(v) for v in str(text).split(",") if str(v).strip() != ""]


def _probe_frames(mp4: Path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(mp4)],
        capture_output=True, text=True, check=False).stdout.strip()
    try:
        return int(out.split(",")[0])
    except ValueError:
        return 0


def _png_frame(mp4: Path, png: Path, frame: int) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4),
                    "-vf", f"select=eq(n\\,{frame})", "-vframes", "1", str(png)],
                   check=False)


def run_one(out: Path, codec: str, kbps: int, seconds: float, fps: int, isp: dict) -> dict:
    name = f"{codec}_{kbps // 1000 if kbps else 'auto'}{'M' if kbps else ''}"
    raw = out / f"{name}.{'h265' if codec == 'h265' else 'h264'}"
    mp4 = out / f"{name}.mp4"
    result = {"setting": name, "codec": codec, "bitrate_kbps": kbps}
    cam = None
    try:
        cam = OakCamera((OAK_MAIN_W, OAK_MAIN_H), (LORES_W, LORES_H), fps, 2.0,
                        bitrate_kbps=kbps, codec=codec, isp=isp)
        cam.start()
        time.sleep(3)                     # let AE / the encoder settle
        cam.clip_output.fileoutput = str(raw)
        cam.clip_output.start()
        t0 = time.monotonic()
        while time.monotonic() - t0 < seconds:
            cam.capture_buffer("lores")   # keep the motion stream drained, as the recorder does
        cam.clip_output.stop()
        elapsed = time.monotonic() - t0
    except Exception as e:  # e.g. "out of resources" at a rate the encoder can't hold
        result["error"] = str(e)[:300]
        return result
    finally:
        if cam is not None:
            cam.stop()

    size = raw.stat().st_size if raw.exists() else 0
    _remux(raw, mp4, fps)
    frames = _probe_frames(mp4) if mp4.exists() else 0
    expected = int(round(elapsed * fps))
    _png_frame(mp4, out / f"{name}.png", max(0, frames // 2))
    result.update({
        "seconds": round(elapsed, 1),
        "frames": frames,
        "expected_frames": expected,
        "fps": round(frames / elapsed, 2) if elapsed else 0,
        "dropped_pct": round(100 * max(0, expected - frames) / max(1, expected), 1),
        "mbit_s": round(size * 8 / elapsed / 1e6, 1) if elapsed else 0,
        "mb_per_min": round(size / elapsed * 60 / 1e6, 1) if elapsed else 0,
        "ok": frames >= 0.97 * expected,
    })
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--codecs", default="h264,h265")
    ap.add_argument("--bitrates", default="0,30000,60000,100000,150000",
                    help="kbit/s, comma separated; 0 = encoder's auto")
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--fps", type=int, default=OAK_FPS)
    ap.add_argument("--isp", default="-1,-1,-1",
                    help="luma denoise, chroma denoise, sharpness (0..4, -1 = default)")
    ap.add_argument("--max-exposure-us", type=int, default=0)
    args = ap.parse_args()

    luma, chroma, sharp = (_ints(args.isp) + [-1, -1, -1])[:3]
    isp = {"luma_denoise": luma, "chroma_denoise": chroma, "sharpness": sharp,
           "max_exposure_us": args.max_exposure_us}
    out = Path("oak_quality") / datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    out.mkdir(parents=True, exist_ok=True)
    print(f"{OAK_MAIN_W}x{OAK_MAIN_H} @ {args.fps} fps, ISP {isp} -> {out}/")

    results = []
    for codec in [c.strip() for c in args.codecs.split(",") if c.strip()]:
        for kbps in _ints(args.bitrates):
            r = run_one(out, codec, kbps, args.seconds, args.fps, isp)
            results.append(r)
            if "error" in r:
                print(f"  {r['setting']:>12}  FAILED: {r['error']}")
            else:
                print(f"  {r['setting']:>12}  {'OK ' if r['ok'] else 'DROP'}  "
                      f"{r['fps']:5.2f} fps  dropped {r['dropped_pct']:4.1f}%  "
                      f"{r['mbit_s']:6.1f} Mbit/s  {r['mb_per_min']:6.1f} MB/min")

    summary = {"resolution": [OAK_MAIN_W, OAK_MAIN_H], "fps": args.fps, "isp": isp,
               "results": results}
    (out / "results.json").write_text(json.dumps(summary, indent=2))
    best = [r for r in results if r.get("ok")]
    if best:
        top = max(best, key=lambda r: (r["bitrate_kbps"] or 0, r["codec"] == "h265"))
        print(f"\nHighest setting held without dropping frames: {top['setting']}")
    print(f"Compare detail: {out}/*.png   Full results: {out}/results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
