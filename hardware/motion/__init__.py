"""motion — the BeeMonitor motion-gated recorder, split by concern.

Public entrypoints: ``record()`` (the capture loop), ``calibrate()`` (offline
blob-window learning), and ``main()`` (the CLI). ``MotionGate`` and the config
constants are re-exported for tools like ``motion_replay.py`` that drive the gate
directly.

Layout:
    config            — every BEEMONITOR_* constant + paths + the logger
    frames            — tiny shared frame/geometry helpers
    gate              — MotionGate (MOG2 blob detector)
    roi               — hotel/nest detection + recording-ROI resolution
    overrides         — calibration.json / motion_tuning.json / ROI override loaders
    calibrate         — the scheduled, offline YOLO calibration pass
    remux             — .h264 -> .mp4 stream copy
    telemetry_still   — on-demand still capture
    activity_frames   — mover-crop sampling for cloud taxonomic ID
    recorder          — the capture loop that ties it together
    cli               — argparse entrypoint
"""

from motion.config import *  # noqa: F401,F403 - re-export BEEMONITOR_* constants + log
from motion.gate import MotionGate  # noqa: F401
from motion.frames import _main_array_to_bgr, _scale_roi  # noqa: F401
from motion.roi import (  # noqa: F401
    detect_hotel_roi, detect_nest_boxes, _order_nest_boxes,
    _resolve_record_roi, _parse_roi,
)
from motion.overrides import (  # noqa: F401
    load_calibration, _apply_calibration, load_tuning, _apply_tuning,
    load_roi_override_lores, load_roi_polygon_lores, load_nest_layout, _build_gate,
)
from motion.remux import _remux, _snippet_paths  # noqa: F401
from motion.telemetry_still import _save_telemetry_still  # noqa: F401
from motion.activity_frames import (  # noqa: F401
    _largest_blob, _mover_crop, _flush_activity_frames,
)
from motion.calibrate import (  # noqa: F401
    calibrate, _find_snippets, _calibration_fresh, _iter_video_frames,
    _lores_from_bgr, _bbox_overlap,
)
from motion.confirm import BeeConfirmer  # noqa: F401
from motion.recorder import record, _handle_signal, HAVE_PICAMERA2  # noqa: F401
from motion.cli import main  # noqa: F401
