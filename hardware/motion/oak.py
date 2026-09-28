"""Luxonis OAK (USB, RVC2) as a drop-in for the recorder's Picamera2.

The recorder was written against picamera2, and rather than put an interface
in front of every call it makes, OakCamera answers the same handful of calls
the recorder and its helpers use:

    capture_buffer("lores")  -> flat uint8 Y plane, blocks ~1 frame (paces the loop)
    capture_array("main")    -> RGB array, a full-size recorded frame
    capture_jpegs(n)         -> n consecutive full-size frames as JPEG bytes (photos)
    camera_properties        -> {"Model": "imx378"}
    start() / stop() / stop_encoder()

and ``clip_output`` stands in for picamera2's CircularOutput: set
``.fileoutput`` to a path, ``.start()`` writes the pre-roll and keeps writing,
``.stop()`` closes the file, ready for the usual remux.

What runs where. The OAK does the heavy lifting on its own chip: ISP, H.264
encode of the main stream, and — only when a frame is asked for — a JPEG of
that same full-size frame. A small script on the OAK holds the newest frame
and passes one to the JPEG encoder per request, so stills and photos are the
video's own resolution (4032x3040 by default), come back in ~70 ms, and never
pause the video: measured, H.264 holds 20.0 fps at 12 MP through single photos
and a 5-frame burst (0.3 s). Encoding JPEGs continuously instead does not fit
beside 12 MP H.264 — both encoders stall.

Three RVC2 facts this is built around, all measured on an OAK-1-AF:
  * every output of the Camera node must run at the SAME fps. Ask for one at
    5 fps next to others at 25 and the whole pipeline stalls to ~0.2 fps.
  * every H.264 keyframe carries its own SPS/PPS, so a clip that starts on a
    keyframe decodes on its own. The pre-roll ring is kept in whole GOPs for
    exactly that reason.
  * the JPEG encoder's default output pool (4 x 18.7 MB at 12 MP) runs the
    OAK out of memory beside the H.264 encoder; 2 x 8 MB is plenty (a 12 MP
    JPEG at q95 is ~4 MB).
  * there is no free flip. Camera.setImageOrientation is accepted and ignored
    (depthai 3.10), and an ImageManip flip in front of the encoder costs frames:
    ~23 fps at 1080p, ~6 at 4K, and it drags the motion stream down with it.
    So the OAK records as mounted — mount it the right way up.
"""

from __future__ import annotations

import collections
import threading
import time
from datetime import timedelta

import cv2
import numpy as np

from motion.config import log

FRAME_TIMEOUT = timedelta(seconds=5)
AF_SETTLE = 6.0      # seconds to let a one-shot autofocus land
AF_STABLE_FRAMES = 15
MAX_BURST = 10       # photos per request; the JPEG queue holds this many


def available() -> bool:
    """True if depthai is installed and can see an OAK. Never raises."""
    try:
        import depthai as dai
        return bool(dai.Device.getAllAvailableDevices())
    except Exception:
        return False


class ClipOutput:
    """Pre-roll ring of encoded packets + the clip file, CircularOutput-style.

    Kept in whole GOPs so the file always begins on a keyframe, and trimmed so
    at least ``pre_roll`` seconds are buffered (up to one GOP more, ~1 s — the
    same slack picamera2's CircularOutput has).
    """

    def __init__(self, pre_roll: float):
        self.pre_roll = pre_roll
        self.fileoutput: str | None = None
        self._gops: collections.deque = collections.deque()  # [t0, [bytes, ...]]
        self._file = None
        self._synced = False
        self._lock = threading.Lock()

    def push(self, data: bytes, keyframe: bool, t: float) -> None:
        with self._lock:
            if self._file is not None:
                # Recording: a clip opened with no buffered pre-roll waits for
                # its first keyframe rather than starting on undecodable P-frames.
                self._synced = self._synced or keyframe
                if self._synced:
                    self._file.write(data)
                return
            if keyframe:
                self._gops.append([t, [data]])
            elif self._gops:
                self._gops[-1][1].append(data)
            while len(self._gops) > 1 and self._gops[1][0] <= t - self.pre_roll:
                self._gops.popleft()

    def start(self) -> None:
        with self._lock:
            self._file = open(self.fileoutput, "wb")
            self._synced = bool(self._gops)
            for _, packets in self._gops:
                for data in packets:
                    self._file.write(data)
            self._gops.clear()

    def stop(self) -> None:
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None


class OakCamera:
    def __init__(self, main_wh, lores_wh, fps: int, pre_roll: float, *,
                 bitrate_kbps: int = 0):
        import depthai as dai
        self._dai = dai
        self.main_wh = tuple(main_wh)
        self.lores_wh = tuple(lores_wh)
        self.fps = fps
        self.clip_output = ClipOutput(pre_roll)
        self._error: Exception | None = None
        self._running = False

        self.pipeline = dai.Pipeline()
        cam = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)

        # All outputs at `fps` — see the module docstring for why.
        main = cam.requestOutput(self.main_wh, dai.ImgFrame.Type.NV12, fps=fps)
        h264 = self.pipeline.create(dai.node.VideoEncoder).build(
            main, frameRate=fps, profile=dai.VideoEncoderProperties.Profile.H264_MAIN,
            keyframeFrequency=fps)
        if bitrate_kbps > 0:
            h264.setBitrateKbps(bitrate_kbps)
        # Stills on request: the script forwards the newest main frame to the
        # JPEG encoder once per trigger, so the encoder is idle between asks.
        gate = self.pipeline.create(dai.node.Script)
        main.link(gate.inputs["frames"])
        gate.inputs["frames"].setBlocking(False)
        gate.inputs["frames"].setMaxSize(1)
        gate.setScript(
            "while True:\n"
            "    node.inputs['trigger'].get()\n"
            "    node.outputs['out'].send(node.inputs['frames'].get())\n")
        mjpeg = self.pipeline.create(dai.node.VideoEncoder).build(
            gate.outputs["out"], frameRate=fps,
            profile=dai.VideoEncoderProperties.Profile.MJPEG, quality=95)
        mjpeg.setNumFramesPool(2)
        mjpeg.setMaxOutputFrameSize(8 * 1024 * 1024)
        # STRETCH, not the default CROP: the ROI code maps lores coordinates onto
        # the main frame by scaling alone, which assumes both see the same field
        # of view — as picamera2's lores does.
        lores = cam.requestOutput(self.lores_wh, dai.ImgFrame.Type.NV12,
                                  dai.ImgResizeMode.STRETCH, fps=fps)

        # ~2 s of H.264 slack in case the pump thread is starved for a moment.
        self._h264_q = h264.out.createOutputQueue(maxSize=fps * 2, blocking=False)
        self._jpeg_q = mjpeg.out.createOutputQueue(maxSize=MAX_BURST, blocking=False)
        self._trigger_q = gate.inputs["trigger"].createInputQueue()
        self._jpeg_lock = threading.Lock()
        self._lores_q = lores.createOutputQueue(maxSize=2, blocking=False)
        self._ctrl_q = cam.inputControl.createInputQueue()

        device = self.pipeline.getDefaultDevice()
        sensor = next((f.sensorName for f in device.getConnectedCameraFeatures()
                       if f.socket == dai.CameraBoardSocket.CAM_A), "oak")
        self.camera_properties = {
            "Model": str(sensor).lower(),
            "Product": device.getProductName() or device.getDeviceName(),
            "Usb": device.getUsbSpeed().name,
        }
        self._pump = threading.Thread(target=self._pump_h264, name="oak-h264", daemon=True)

    # --- picamera2-shaped surface ------------------------------------------
    def start(self) -> None:
        self.pipeline.start()
        self._running = True
        self._pump.start()

    def stop(self) -> None:
        self._running = False
        try:
            self.pipeline.stop()
        except Exception:
            pass
        self._pump.join(timeout=5)
        self.clip_output.stop()

    def stop_encoder(self) -> None:
        pass  # the encoder lives in the pipeline; stop() takes it down

    def capture_buffer(self, name: str = "lores"):
        if name != "lores":
            raise ValueError(f"OakCamera has no {name!r} buffer")
        frame = self._get(self._lores_q, "motion stream")
        w, h = self.lores_wh
        stride = frame.getStride() or w
        y = np.asarray(frame.getData())[:stride * h].reshape(h, stride)[:, :w]
        return np.ascontiguousarray(y).ravel()

    def capture_array(self, name: str = "main"):
        if name != "main":
            raise ValueError(f"OakCamera has no {name!r} stream")
        _, jpeg = self.capture_jpegs(1)[0]
        bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("OAK sent an undecodable JPEG")
        # picamera2's main stream is RGB; _main_array_to_bgr turns it back.
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def capture_jpegs(self, n: int = 1) -> list:
        """n consecutive full-size frames, as [(captured_at, jpeg_bytes)].
        The video keeps running throughout."""
        from motion.stills import _now
        n = max(1, min(int(n), MAX_BURST))
        with self._jpeg_lock:
            self._jpeg_q.tryGetAll()          # nothing stale from a timed-out ask
            for _ in range(n):
                self._trigger_q.send(self._dai.Buffer())
            out = []
            for _ in range(n):
                pkt = self._get(self._jpeg_q, "still")
                out.append((_now(), bytes(pkt.getData())))
            return out

    # --- focus -------------------------------------------------------------
    def apply_focus(self, lens: int | None) -> int | None:
        """Hold the lens at `lens` (0..255), or autofocus once and hold that."""
        dai = self._dai
        ctrl = dai.CameraControl()
        if lens is not None:
            ctrl.setManualFocus(int(lens))
            self._ctrl_q.send(ctrl)
            log.info("OAK lens set to %d (profile)", int(lens))
            return int(lens)

        log.info("OAK: no saved focus — autofocusing once")
        ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.AUTO)
        ctrl.setAutoFocusTrigger()
        self._ctrl_q.send(ctrl)
        # The lens position rides on every frame's metadata; call it landed once
        # it has stopped moving for half a second.
        deadline = time.monotonic() + AF_SETTLE
        last, same, pos = None, 0, None
        time.sleep(0.5)  # let the trigger reach the device before we judge
        while time.monotonic() < deadline:
            pos = self._get(self._lores_q, "motion stream").getLensPosition()
            same = same + 1 if pos == last else 0
            last = pos
            if same >= AF_STABLE_FRAMES:
                break
        if pos is None:
            log.warning("OAK autofocus: no lens position reported — leaving it be")
            return None
        # Hold it: continuous AF would hunt on every passing bee.
        hold = dai.CameraControl()
        hold.setManualFocus(int(pos))
        self._ctrl_q.send(hold)
        log.info("OAK autofocused at %d/255, holding%s", pos,
                 "" if same >= AF_STABLE_FRAMES else " (had not settled)")
        return int(pos)

    # --- internals ---------------------------------------------------------
    def _get(self, q, what: str):
        if self._error is not None:
            raise RuntimeError(f"OAK failed: {self._error}")
        msg = q.get(FRAME_TIMEOUT)
        if msg is None:
            raise RuntimeError(f"OAK {what} delivered nothing for "
                               f"{FRAME_TIMEOUT.total_seconds():.0f}s — unplugged?")
        return msg

    def _pump_h264(self) -> None:
        I = self._dai.EncodedFrame.FrameType.I
        try:
            while self._running:
                pkt = self._h264_q.get(timedelta(seconds=1))
                if pkt is None:
                    continue
                self.clip_output.push(bytes(pkt.getData()), pkt.getFrameType() == I,
                                      time.monotonic())
        except Exception as e:
            if self._running:
                log.error("OAK H.264 stream died: %s", e)
                self._error = e
