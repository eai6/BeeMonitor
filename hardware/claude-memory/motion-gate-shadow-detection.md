---
name: motion-gate-shadow-detection
description: "The MOG2 motion gate uses detectShadows=True + thresholds out shadow pixels to reject cast shadows / background-light triggers. Don't revert to detectShadows=False."
metadata: 
  node_type: memory
  type: project
  originSessionId: 416c68fc-5615-46ab-98ac-9e38009bdaf2
---

The recorder's MOG2 motion gate (`MotionGate` in `hardware/main_motion.py`) deliberately runs with **`detectShadows=True`** (via `BEEMONITOR_DETECT_SHADOWS`, default on). MOG2 marks shadow pixels (darker but same chromaticity as background — moving shadows, the hotel's own shadow drifting with the sun, passing clouds) as `127`, and `update()` thresholds them out (`cv2.threshold(fg, 200, 255, ...)`) before counting blobs. **Do not revert to `detectShadows=False`** — that makes cast shadows and soft light changes count as motion and trigger false clips (the original bug).

- `BEEMONITOR_SHADOW_THRESHOLD` (MOG2 tau, default 0.5): **lower = more aggressive** shadow rejection (wider shadow band); **raise toward 0.7** if real bees get misread as shadow and missed.
- BG creation lives in `MotionGate._make_bg()` so `reset()` (periodic rebuild via `BG_RESET_INTERVAL`) keeps the shadow setting — don't re-hardcode the subtractor in `reset()`.

**Verified** offline with `motion_replay.py` (A/B `BEEMONITOR_DETECT_SHADOWS=false` vs `true`) on the two afternoon sample clips: motion-positive frames dropped 57%→35% and 77%→53% while the real bee event was still captured. That's the way to test motion-gate changes: `motion_replay.py <clip> --full-frame --min-area 62 --max-area 554` and compare the "motion fired on N frames" line. Implemented in commit b740ac3.

Related: [[pi-torch-must-be-cpu-wheel]] (the calibrate path that feeds the gate's area window).
