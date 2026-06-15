"""Bee confirmation — low-DL YOLO filter for the motion-gated recorder.

Confirms that a recorded activity is actually a **bee** by running the whole-frame
bee detector (``bee_tracking.pt``) on a few of the full main-stream frames the
recorder already captures per activity — **asynchronously, off the capture hot
path**. The capture loop is never blocked or delayed; the verdict only decides
the activity's *downstream* treatment (telemetry count + cellular crop send +
clip tag). See ``memory/17_bee_confirmation_design.md``.

Cost is kept tiny by **count**, not by shrinking the input:
  * per-activity, not per-frame;
  * **confirm-then-coast** — one bee detection settles the activity, YOLO stops;
  * a bounded **negative budget** (a shadow costs ≤ N inferences, then $0);
  * a single CPU-throttled worker thread; idle ⇒ no inference.

A YOLO box only counts if it **overlaps the MOG2 mover blob** (same pairing the
calibration pass uses), so a bee resting elsewhere in frame can't confirm a
shadow's motion. Graceful + inert on any model/torch failure — never crashes
recording (same contract as the startup hotel detector).

Threading model: the worker thread does *only* inference + verdict bookkeeping.
All file I/O (clip tag, crop flush) stays on the recorder's main thread, which
calls :meth:`drain_resolved` each loop and acts on settled activities.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Optional

import numpy as np

from motion.config import (
    log, BEE_CONFIRM_MODE, BEE_CONFIRM_MODEL, BEE_CONFIRM_CONF, BEE_CONFIRM_IMGSZ,
    BEE_CONFIRM_MIN_CONFIRMATIONS, BEE_CONFIRM_MAX_RUNS, BEE_CONFIRM_TORCH_THREADS,
    BEE_CONFIRM_QUEUE_MAX, BEE_CONFIRM_VERDICT_TIMEOUT, BEE_CONFIRM_OVERLAP_PAD,
    LORES_W, LORES_H,
)

# Verdict status strings (also written into the clip tag for telemetry/cloud).
PENDING = "pending"
CONFIRMED = "confirmed"
UNCONFIRMED = "unconfirmed"
DISABLED = "disabled"   # confirmer inert/off → downstream behaves as 'off'


def _bbox_overlap(a, b) -> bool:
    """Do (x1,y1,x2,y2) boxes a and b intersect at all?"""
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


class _Activity:
    """Per-activity verdict state (shared between worker + main thread)."""
    __slots__ = ("status", "confidence", "taxon", "runs", "negatives",
                 "closed_at", "payload")

    def __init__(self):
        self.status = PENDING
        self.confidence = 0.0
        self.taxon = None
        self.runs = 0
        self.negatives = 0
        self.closed_at = None   # monotonic time the clip closed; None while open
        self.payload = None     # opaque recorder payload handed at close


class BeeConfirmer:
    """Async whole-frame bee confirmation with confirm-then-coast + bounded budget."""

    def __init__(
        self,
        mode: str = BEE_CONFIRM_MODE,
        model_path: str = BEE_CONFIRM_MODEL,
        conf: float = BEE_CONFIRM_CONF,
        imgsz: int = BEE_CONFIRM_IMGSZ,
        min_confirmations: int = BEE_CONFIRM_MIN_CONFIRMATIONS,
        max_runs: int = BEE_CONFIRM_MAX_RUNS,
        torch_threads: int = BEE_CONFIRM_TORCH_THREADS,
        queue_max: int = BEE_CONFIRM_QUEUE_MAX,
        verdict_timeout: float = BEE_CONFIRM_VERDICT_TIMEOUT,
        overlap_pad: int = BEE_CONFIRM_OVERLAP_PAD,
    ):
        self.mode = (mode or "off").strip().lower()
        self.model_path = model_path
        self.conf = conf
        self.imgsz = imgsz
        self.min_confirmations = max(1, min_confirmations)
        self.overlap_pad = max(0, overlap_pad)
        self.max_runs = max(1, max_runs)
        self.torch_threads = max(1, torch_threads)
        self.verdict_timeout = verdict_timeout

        self.disabled = self.mode not in ("tag", "gate")
        self.inert = False          # set if the model/torch can't load
        self._loaded = False
        self._model = None

        self._acts: dict[str, _Activity] = {}
        self._lock = threading.Lock()
        self._q: "queue.Queue" = queue.Queue(maxsize=queue_max)
        self._stop = threading.Event()

        # Rolling stats for the recorder's periodic log line.
        self._stat_runs = 0
        self._stat_ms = 0.0
        self._stat_conf = 0
        self._stat_unconf = 0

        # The worker always runs (even in `off`), so the mode can be switched to
        # `tag`/`gate` from the dashboard at runtime; the model loads lazily on the
        # first active job, so `off` costs nothing.
        self._worker = threading.Thread(target=self._run_worker, name="bee-confirm",
                                        daemon=True)
        self._worker.start()
        log.info("bee confirmation: mode=%s model=%s conf=%.2f imgsz=%d "
                 "min_confirm=%d max_runs=%d (model loads lazily when active)",
                 self.mode, model_path, conf, imgsz, self.min_confirmations,
                 self.max_runs)

    def set_mode(self, mode: str) -> None:
        """Switch mode at runtime (dashboard push). Cheap: flips gating; the worker
        keeps running and the model (once loaded) stays warm."""
        mode = (mode or "off").strip().lower()
        if mode not in ("off", "tag", "gate"):
            return
        with self._lock:
            if mode == self.mode:
                return
            self.mode = mode
            self.disabled = mode == "off"
        log.info("bee confirmation: mode -> %s", mode)

    # -- public API (main thread) -------------------------------------------

    @property
    def active(self) -> bool:
        """True when confirmation should drive downstream gating. False when off
        or inert — in which case the recorder behaves exactly as ``mode=off``."""
        return (not self.disabled) and (not self.inert)

    def is_confirmed(self, uid: str) -> bool:
        with self._lock:
            a = self._acts.get(uid)
            return a is not None and a.status == CONFIRMED

    def submit(self, uid: str, frame_bgr: np.ndarray, blobs, roi) -> None:
        """Queue one full frame + the mover blobs for confirmation (non-blocking).

        No-op when off/inert, or once the activity is already confirmed
        (confirm-then-coast). Drops the oldest job on overflow rather than ever
        back-pressuring the recorder.
        """
        if not self.active:
            return
        with self._lock:
            a = self._acts.get(uid)
            if a is None:
                a = self._acts[uid] = _Activity()
            elif a.status == CONFIRMED or a.runs >= self.max_runs:
                return
        job = (uid, frame_bgr, blobs, roi)
        try:
            self._q.put_nowait(job)
        except queue.Full:
            try:
                self._q.get_nowait()      # drop oldest, favour recency
                self._q.put_nowait(job)
            except queue.Empty:
                pass

    def register_close(self, uid: str, payload, now_mono: float) -> None:
        """Mark an activity's clip closed, attaching the recorder payload (crops,
        tag path, …) to surface back via :meth:`drain_resolved` once the verdict
        settles or times out."""
        if not self.active:
            return
        with self._lock:
            a = self._acts.get(uid)
            if a is None:
                a = self._acts[uid] = _Activity()
            a.closed_at = now_mono
            a.payload = payload

    def drain_resolved(self, now_mono: float) -> list:
        """Return + remove closed activities whose verdict has settled (confirmed /
        unconfirmed / budget-exhausted) or timed out (fail-closed → unconfirmed).
        Each item: ``{uid, status, confidence, taxon, runs, payload}``."""
        out = []
        with self._lock:
            for uid, a in list(self._acts.items()):
                if a.closed_at is None:
                    continue  # clip still open
                if self.inert:
                    status = DISABLED
                elif a.status in (CONFIRMED, UNCONFIRMED):
                    status = a.status
                elif a.runs >= self.max_runs:
                    status = UNCONFIRMED
                elif (now_mono - a.closed_at) >= self.verdict_timeout:
                    status = UNCONFIRMED        # fail-closed; clip already up + tagged
                    log.debug("bee confirm: %s timed out -> unconfirmed", uid)
                else:
                    continue  # still pending, within timeout
                out.append({"uid": uid, "status": status, "confidence": a.confidence,
                            "taxon": a.taxon, "runs": a.runs, "payload": a.payload})
                if status == CONFIRMED:
                    self._stat_conf += 1
                elif status == UNCONFIRMED:
                    self._stat_unconf += 1
                del self._acts[uid]
        return out

    def stats(self) -> dict:
        """Snapshot + reset rolling stats for the recorder's periodic log line."""
        with self._lock:
            d = {"inferences": self._stat_runs,
                 "mean_ms": (self._stat_ms / self._stat_runs) if self._stat_runs else 0.0,
                 "confirmed": self._stat_conf, "unconfirmed": self._stat_unconf,
                 "pending": sum(1 for a in self._acts.values() if a.status == PENDING)}
            self._stat_runs = 0
            self._stat_ms = 0.0
            self._stat_conf = 0
            self._stat_unconf = 0
        return d

    def stop(self) -> None:
        self._stop.set()

    # -- worker thread ------------------------------------------------------

    def _load(self) -> bool:
        try:
            from ultralytics import YOLO  # noqa: PLC0415 - heavy, lazy
            import torch  # noqa: PLC0415
            torch.set_num_threads(self.torch_threads)
            self._model = YOLO(self.model_path)
            self._loaded = True
            log.info("bee confirm: model loaded (%s), torch_threads=%d",
                     self.model_path, self.torch_threads)
            return True
        except Exception as e:  # pragma: no cover - missing dep / bad model
            self.inert = True
            log.warning("bee confirm: model load failed (%s) — confirmation INERT "
                        "(behaves as mode=off)", e)
            return False

    def _run_worker(self) -> None:
        while not self._stop.is_set():
            try:
                uid, frame_bgr, blobs, roi = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if self.inert:
                continue
            if not self._loaded and not self._load():
                continue  # load failed -> inert; discard the job
            # Skip if this activity already settled while queued.
            with self._lock:
                a = self._acts.get(uid)
                if a is None or a.status == CONFIRMED or a.runs >= self.max_runs:
                    continue
            try:
                conf, taxon = self._infer(frame_bgr, blobs, roi)
            except Exception as e:  # never let one bad frame kill the worker
                log.warning("bee confirm: inference error: %s", e)
                continue
            with self._lock:
                a = self._acts.get(uid)
                if a is None:
                    continue
                a.runs += 1
                if conf is not None:
                    a.status = CONFIRMED
                    a.confidence = max(a.confidence, conf)
                    a.taxon = taxon or a.taxon
                else:
                    a.negatives += 1
                    if a.negatives >= self.max_runs and a.status == PENDING:
                        a.status = UNCONFIRMED

    def _infer(self, frame_bgr, blobs, roi):
        """Run YOLO on the full frame; return (confidence, taxon) of the best bee
        box overlapping a mover blob, else (None, None)."""
        t0 = time.monotonic()
        results = self._model.predict(frame_bgr, conf=self.conf, imgsz=self.imgsz,
                                      verbose=False)
        dt = (time.monotonic() - t0) * 1000.0
        with self._lock:
            self._stat_runs += 1
            self._stat_ms += dt

        h, w = frame_bgr.shape[:2]
        sx, sy = LORES_W / w, LORES_H / h
        # Mover blobs -> full-lores coords (gate.last_blobs are ROI-relative).
        ox, oy = (roi[0], roi[1]) if roi is not None else (0, 0)
        # Inflate each mover blob by the tolerance pad so a bee box that is *near*
        # the motion (body vs wing-motion offset, slight lag) still matches; a
        # large gap (resting bee + unrelated motion elsewhere) stays unmatched.
        pad = self.overlap_pad
        mover_boxes = [(ox + bx - pad, oy + by - pad,
                        ox + bx + bw + pad, oy + by + bh + pad)
                       for (bx, by, bw, bh, _area) in (blobs or [])]
        if not mover_boxes:
            return None, None

        best_conf = None
        best_taxon = None
        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            names = getattr(r, "names", {}) or {}
            for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
                yb = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)  # YOLO box in lores coords
                if any(_bbox_overlap(yb, mb) for mb in mover_boxes):
                    if best_conf is None or c > best_conf:
                        best_conf = float(c)
                        best_taxon = names.get(int(k), str(int(k)))
        return best_conf, best_taxon
