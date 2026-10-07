---
name: pi-torch-must-be-cpu-wheel
description: "On the Pi, torch/torchvision must be the +cpu wheels or YOLO inference SIGILLs"
metadata: 
  node_type: memory
  type: project
  originSessionId: 87980b96-0146-4a50-ac80-9e5c978a75ec
---

On the Raspberry Pi 4 (aarch64, Cortex-A72 = ARMv8.0), the recorder venv is
`~/BeeMonitor/hardware/venv` (Python 3.13). **torch and torchvision MUST be the
CPU-only wheels** or YOLO inference crashes the whole process with **SIGILL
(illegal instruction, systemd `status=4/ILL`, exit 132)** — uncatchable by
try/except.

- Cause: `pip install torch` on aarch64 pulls the **CUDA/ARMv8.2+ wheel**
  (`2.12.0+cu130`, drags in ~2.9 GB of `nvidia-*` + `triton`). Its kernels use
  instructions the A72 lacks → SIGILL on the first `model.predict()`. Import and
  model-load succeed; only inference crashes.
- Fix: install from the CPU index —
  `hardware/venv/bin/pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cpu`
  (gives `2.12.0+cpu` / `0.27.0+cpu`), then remove orphaned `nvidia-*`/`cuda-*`/
  `triton`. Verify with a real inference, not just import:
  `python -c "import numpy,ultralytics; ultralytics.YOLO('models/nest_detection.pt').predict(__import__('numpy').zeros((1080,1920,3),'uint8'))"`.
- This affects BOTH the recorder's hotel-ROI detection (loads `nest_detection.pt`
  at startup, `main_motion.py` `_resolve_record_roi`/`detect_hotel_roi`) and the
  scheduled `--calibrate` job.
- Emergency stop-gap (no torch fix): `BEEMONITOR_HOTEL_ROI_DETECT=false` in
  `/etc/beemonitor/uploader.env` (root, 600) → recorder records full-frame and
  skips all YOLO. Restart `beemonitor-recorder`.

Fixed & verified on-Pi 2026-06-05. Related: [[cellular-modem-is-telit]].
