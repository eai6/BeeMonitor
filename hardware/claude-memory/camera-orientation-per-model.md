---
name: camera-orientation-per-model
description: User swaps three camera modules; upside-down footage = model missing from FLIPPED_MODELS; HQ fix not yet confirmed upright on a real clip
metadata:
  node_type: memory
  type: project
  originSessionId: fa9e86bd-41ae-4282-80fc-0e0732432307
  modified: 2026-09-23T14:47:46.331Z
---

The user has three modules they swap on the Pi: the Pi HQ camera (imx477), the Arducam 64MP OwlSight (ov64a40) and the night-vision module (ov5647). Only one is attached at a time, and autodetect finds each one.

If a module records upside down, the usual cause is that its model is missing from `FLIPPED_MODELS` in hardware/motion/camera.py. Fix it per model there. Use camera.json only for a single unit that is mounted oddly. As of 2026-09-23 the list is ov5647 + imx477 (commit b0c3c3b); ov64a40 must stay unturned.

**Open as of 2026-09-23:** the user restarted the recorder after the imx477 fix, but nobody has confirmed that a new HQ clip actually comes out upright. scripts/camera-flip-test.sh documents a past case where the recorder silently dropped the transform. If the HQ is still inverted, run that script (needs sudo, see [[pi-no-passwordless-sudo]]) rather than reasoning from log lines. A past commit (efb3311) got the direction backwards by trusting the log.
