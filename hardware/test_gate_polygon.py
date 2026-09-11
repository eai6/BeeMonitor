#!/usr/bin/env python3
"""Polygon ROI masking in the motion gate — run it anywhere, including on the Pi.

    python3 hardware/test_gate_polygon.py

The dashboard's ROI editor can save the hotel ROI as a traced outline instead of
a dragged rectangle. The crop stays the outline's bounding box; the polygon then
masks the background that box unavoidably contains, so grass beside a round trap
never triggers a recording. These checks pin exactly that: same synthetic mover,
inside vs outside the outline, with and without a crop.

Needs only cv2 + numpy (already on the recorder), no camera. Exits non-zero on
failure so it can gate a deploy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from motion.gate import MotionGate  # noqa: E402

H, W = 120, 160
# A diamond in the middle of the frame: its bounding box is nearly the whole
# frame, but the corners of that box are background.
DIAMOND = [(80, 10), (150, 60), (80, 110), (10, 60)]


def _frame(level: int = 10) -> np.ndarray:
    return np.full((H, W), level, np.uint8)


def _fires(gate: MotionGate, at: tuple[int, int]) -> bool:
    """Warm the gate on a static scene, then show it one mover at `at`."""
    for _ in range(30):
        gate.warm(_frame())
    f = _frame()
    x, y = at
    f[y - 6:y + 6, x - 6:x + 6] = 255
    motion, _n, _area = gate.update(f)
    return motion


def _gate(**kw) -> MotionGate:
    return MotionGate(min_area=10, max_area=1e9, min_blobs=1, **kw)


CASES = [
    ("mover inside the outline fires",
     lambda: _fires(_gate(polygon=DIAMOND), (80, 60)) is True),
    ("mover in the box corner (outside the outline) does not",
     lambda: _fires(_gate(polygon=DIAMOND), (12, 12)) is False),
    ("without a polygon that same corner still fires",
     lambda: _fires(_gate(), (12, 12)) is True),
    ("masking survives the ROI crop",
     lambda: _fires(_gate(roi=(40, 20, 160, 120), polygon=DIAMOND), (80, 60)) is True),
    ("outside the outline, inside the crop, does not fire",
     lambda: _fires(_gate(roi=(40, 20, 160, 120), polygon=DIAMOND), (145, 110)) is False),
    ("a live re-gate re-rasterises the mask",
     lambda: _fires(_regated(), (80, 60)) is True),
    ("fewer than 3 points is not a polygon",
     lambda: _gate(polygon=[(0, 0), (10, 10)]).polygon is None),
    ("clearing the polygon restores the plain rectangle gate",
     lambda: _fires(_cleared(), (12, 12)) is True),
]


def _regated() -> MotionGate:
    """A gate whose crop changed after construction (dashboard ROI edit)."""
    g = _gate(roi=(0, 0, 100, 100), polygon=DIAMOND)
    g.roi = (40, 20, 140, 120)
    return g


def _cleared() -> MotionGate:
    g = _gate(polygon=DIAMOND)
    g.set_polygon(None)
    return g


def main() -> int:
    failed = 0
    for name, check in CASES:
        try:
            ok = check()
        except Exception as e:  # a raising check is a failing check
            ok, name = False, f"{name} (raised {e!r})"
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
        failed += not ok
    print(f"\n{len(CASES) - failed}/{len(CASES)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
