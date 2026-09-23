---
name: pi-no-passwordless-sudo
description: Claude has no sudo on the BeeMonitor Pi — the user must restart services (e.g. beemonitor-recorder) themselves
metadata:
  node_type: memory
  type: reference
  originSessionId: fa9e86bd-41ae-4282-80fc-0e0732432307
  modified: 2026-09-23T14:47:41.836Z
---

`sudo -n` fails (password required). Anything needing root — `systemctl restart beemonitor-recorder`, `scripts/camera-flip-test.sh`, config.txt edits — has to be handed to the user as `! sudo ...`.

The recorder holds the camera exclusively, so a live camera test means stopping it (needs sudo). Without sudo, you can still read the current camera and profile with `hardware/venv/bin/python` (`Picamera2.global_camera_info()`, `motion.camera.load_profile()`). `journalctl -u beemonitor-recorder` is readable, and its "recorder up: ... flip=..." line shows the orientation actually in use. Related: [[camera-orientation-per-model]].
