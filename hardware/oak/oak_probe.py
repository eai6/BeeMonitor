#!/usr/bin/env python3
"""Identify a Luxonis OAK on USB and grab one frame from each of its sensors.

    python3 hardware/oak/oak_probe.py [--out DIR] [--no-frames]

Prints the model, the sensors on each socket (with their native resolutions
and frame rates) and the USB link speed, then saves a full-resolution still
per sensor to DIR (default: the current directory). Exits 1 if no OAK is
found, 2 if one is found but cannot be opened (usually the udev rule —
hardware/oak/80-movidius.rules).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("."))
    ap.add_argument("--no-frames", action="store_true")
    args = ap.parse_args()

    try:
        import depthai as dai
    except ImportError:
        print("depthai is not installed (pip install depthai)", file=sys.stderr)
        return 1

    print(f"depthai {dai.__version__}")
    found = dai.Device.getAllAvailableDevices()
    if not found:
        print("no OAK found (lsusb should show 03e7:xxxx; if it does, check the udev rule)")
        return 1
    for info in found:
        print(f"  found {info.name} state={info.state.name} mxid={info.getDeviceId()}")

    try:
        device = dai.Device()
    except Exception as e:
        print(f"could not open the OAK: {e}", file=sys.stderr)
        return 2

    with device:
        print(f"product:  {device.getProductName() or '?'}")
        print(f"name:     {device.getDeviceName() or '?'}")
        print(f"platform: {device.getPlatformAsString()}")
        print(f"usb:      {device.getUsbSpeed().name}")
        try:
            print(f"bootloader: {device.getBootloaderVersion()}")
        except Exception:
            pass
        feats = device.getConnectedCameraFeatures()
        for f in feats:
            configs = sorted({(c.width, c.height, c.maxFps) for c in f.configs},
                             reverse=True)
            modes = ", ".join(f"{w}x{h}@{fps:.0f}" for w, h, fps in configs[:6])
            print(f"  {f.socket.name}: {f.sensorName} {f.width}x{f.height} "
                  f"types={[t.name for t in f.supportedTypes]} "
                  f"af={'yes' if f.hasAutofocus else 'no'} modes: {modes}")
        try:
            t = device.getChipTemperature().average
            print(f"chip temp: {t:.1f} C")
        except Exception:
            pass

        if args.no_frames:
            return 0

        import cv2
        args.out.mkdir(parents=True, exist_ok=True)
        with dai.Pipeline(device) as pipeline:
            queues = {}
            for f in feats:
                cam = pipeline.create(dai.node.Camera).build(f.socket)
                queues[f.socket.name] = cam.requestFullResolutionOutput().createOutputQueue()
            pipeline.start()
            time.sleep(2)  # let AE/AWB settle
            for name, q in queues.items():
                frame = None
                for _ in range(10):
                    frame = q.get().getCvFrame()
                path = args.out / f"oak_{name}.jpg"
                cv2.imwrite(str(path), frame)
                print(f"  saved {path} {frame.shape[1]}x{frame.shape[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
