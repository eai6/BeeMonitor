# BeeMonitor — Pi Zero 2 W "lite" profile

**Status:** Sketch / planned (not implemented). RAM headroom must be validated on
real Zero 2 W hardware before any fleet use.
**Author:** Drafted with Claude Code, 2026-06-17
**Goal:** Make the golden image deployable on a **Pi Zero 2 W** (512 MB RAM, A53,
~3–4× slower than the Pi 4 it's tuned for) by running only the motion-gated
recorder + telemetry — no on-device inference.

---

## Hardware reality

| Board | Verdict |
|---|---|
| Pi Zero / Zero W | **Not supported** — ARMv6, 32-bit only; the arm64 golden image won't boot, and 512 MB + single slow core can't run the stack regardless. |
| Pi Zero 2 W | **Boots** (Cortex-A53 / arm64, same family as Pi 3), but **512 MB RAM** and the slow CPU make the on-device YOLO bee-confirmation (CPU PyTorch + ultralytics) and crop sampling impractical → OOM. Viable only with the lite profile, and only after on-hardware RAM testing. |
| Pi 3B+ / Pi 4 / Pi 5 | Full image as-is (Pi 4 is the tuned target). |

Other Zero 2 W caveats: narrow **CSI camera connector** (needs an adapter ribbon),
**WiFi-only** (cellular HAT options limited by the single micro-USB OTG), and the
image's `cellular-up.sh` assumes a specific USB modem.

## The profile

The biggest saver is `BEE_CONFIRM_MODE=off` — with confirmation off the recorder
never imports torch/ultralytics or loads a model (the bulk of the RAM/CPU).

**`hardware/profiles/zero2w.env`** (operator appends to `/etc/beemonitor/uploader.env`,
which every service reads via `EnvironmentFile=-/etc/beemonitor/uploader.env`):
```
BEEMONITOR_BEE_CONFIRM_MODE=off       # no YOLO/torch load (the big saver)
BEEMONITOR_ACTIVITY_FRAMES=false      # no BioCLIP crop sampling (CPU/RAM/cellular)
BEEMONITOR_MAIN_W=1280                 # 720p H.264 instead of 1080p
BEEMONITOR_MAIN_H=720
BEEMONITOR_FPS=15
BEEMONITOR_TELEMETRY_IMAGE_HEIGHT=480
# LORES stays 640x480 for the MOG2 motion gate
```
Knobs live in `hardware/motion/config.py` (`MAIN_W/H`, `LORES_W/H`, `FPS`,
`BEE_CONFIRM_MODE`, `ACTIVITY_FRAMES`). The two inference toggles
(`bee_confirm_mode`, `send_activity_crops`) are **also dashboard-pushable**, so a
Zero 2 W unit can be flipped from the dashboard too.

## Validate (on real hardware, before fleet use)
Flash a Zero 2 W with the preset; confirm recorder + telemetry stay up, `free -m`
stays under budget, motion clips record, and no torch import appears in the logs.

See [[14_golden_image_provisioning_design]] (same image), [[17_bee_confirmation_design]]
(the inference this disables), [[18_edge_artifact_delivery_design]].
