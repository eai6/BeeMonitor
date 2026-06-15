"""Command-line entrypoint: record (default) or run the --calibrate pass."""

from __future__ import annotations

import argparse
import signal
from pathlib import Path

from motion.config import YOLO_MODEL, CALIB_MAX_CLIPS
from motion.calibrate import calibrate, _find_snippets
from motion.recorder import record, _handle_signal


def main() -> int:
    parser = argparse.ArgumentParser(description="BeeMonitor motion-gated recorder")
    parser.add_argument("--calibrate", action="store_true",
                        help="learn the bee blob-area window from recorded "
                             "snippets, then exit (scheduled job; no live bee needed)")
    parser.add_argument("--calibrate-from", nargs="+", metavar="MP4",
                        help="calibrate from these specific snippet(s) instead "
                             "of auto-scanning the recordings")
    parser.add_argument("--model", default=YOLO_MODEL,
                        help=f"YOLO model for calibration (default {YOLO_MODEL})")
    parser.add_argument("--force", action="store_true",
                        help="recalibrate even if calibration.json is recent")
    args = parser.parse_args()

    if args.calibrate or args.calibrate_from:
        paths = ([Path(p) for p in args.calibrate_from]
                 if args.calibrate_from else _find_snippets(CALIB_MAX_CLIPS))
        return calibrate(paths, args.model, force=args.force or bool(args.calibrate_from))

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    record()
    return 0
