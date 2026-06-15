"""Back-compat shim for the motion-gated recorder.

The recorder used to live entirely in this file; it now lives in the ``motion``
package (see ``motion/__init__.py`` for the layout). This shim stays because:

  * systemd runs it directly — ``beemonitor-recorder.service`` execs
    ``python .../hardware/main_motion.py`` and ``beemonitor-calibrate.service``
    adds ``--calibrate``;
  * ``motion_replay.py`` does ``import main_motion as mm`` and reaches into the
    gate (``mm.MotionGate``), the hotel detector (``mm.detect_hotel_roi``) and a
    handful of constants (``mm.LORES_W`` …).

So we re-export the package's public surface here and keep ``python
main_motion.py [--calibrate]`` working unchanged. New code should import from
``motion`` directly.
"""

import sys
from pathlib import Path

# Make the `motion` package importable when this file is run as a script from
# any cwd (systemd) or imported by a sibling tool (motion_replay.py).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from motion.config import *  # noqa: E402,F401,F403 - re-export BEEMONITOR_* constants + log
from motion.gate import MotionGate  # noqa: E402,F401
from motion.frames import _main_array_to_bgr, _scale_roi  # noqa: E402,F401
from motion.roi import (  # noqa: E402,F401
    detect_hotel_roi, detect_nest_boxes, _order_nest_boxes,
    _resolve_record_roi, _parse_roi,
)
from motion.overrides import (  # noqa: E402,F401
    load_calibration, _apply_calibration, load_tuning, _apply_tuning,
    load_roi_override_lores, load_nest_layout, _build_gate,
)
from motion.remux import _remux, _snippet_paths  # noqa: E402,F401
from motion.telemetry_still import _save_telemetry_still  # noqa: E402,F401
from motion.activity_frames import (  # noqa: E402,F401
    _largest_blob, _mover_crop, _flush_activity_frames,
)
from motion.calibrate import (  # noqa: E402,F401
    calibrate, _find_snippets, _calibration_fresh, _iter_video_frames,
    _lores_from_bgr, _bbox_overlap,
)
from motion.confirm import BeeConfirmer  # noqa: E402,F401
from motion.recorder import record, _handle_signal, HAVE_PICAMERA2  # noqa: E402,F401
from motion.cli import main  # noqa: E402


if __name__ == "__main__":
    sys.exit(main())
