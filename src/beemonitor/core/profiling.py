"""Stage timing for one analysis run.

Answers "where did the wall clock go" for a single video, so the cost shown to
a user is derived from what the run actually did rather than a flat rate times
handler time. Three consumers:

* the SageMaker handler, which returns ``stage_seconds`` alongside the result;
* the web app, which bills GPU wall time at the real instance rate;
* anyone asking whether the GPU is the bottleneck — ``inference`` against
  ``decode`` is exactly that question.

``inference`` is wall time around synchronous detector calls, not pure kernel
time. Ultralytics copies results back to the host before returning, so the call
already blocks on the GPU; measuring around it costs nothing and is within
noise of CUDA-event timing at these batch sizes. It does include host-side
pre/post-processing, which is the honest number for "how long was the GPU
step", just not for "how long were kernels resident".

Thread-safe because the reader thread records ``decode`` while the main thread
records ``inference``.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Dict


class StageProfiler:
    """Accumulates elapsed seconds and call counts per named stage."""

    def __init__(self):
        self._lock = threading.Lock()
        self._seconds: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}

    def reset(self) -> None:
        with self._lock:
            self._seconds.clear()
            self._counts.clear()

    def record(self, stage: str, seconds: float, count: int = 1) -> None:
        with self._lock:
            self._seconds[stage] = self._seconds.get(stage, 0.0) + seconds
            self._counts[stage] = self._counts.get(stage, 0) + count

    @contextmanager
    def stage(self, name: str, count: int = 1):
        """Time a block and add it to ``name``."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.record(name, time.perf_counter() - t0, count)

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        """``{stage: {"seconds": float, "calls": int}}`` — a copy, safe to ship."""
        with self._lock:
            return {
                name: {"seconds": round(secs, 3), "calls": self._counts.get(name, 0)}
                for name, secs in sorted(self._seconds.items())
            }

    def seconds(self, stage: str) -> float:
        with self._lock:
            return round(self._seconds.get(stage, 0.0), 3)


# One profiler per process. A SageMaker container serves one invocation at a
# time (async inference, MaxConcurrentInvocationsPerInstance=1), so a module
# global is per-run — but reset() at the start of a run makes that explicit
# rather than assumed.
PROFILER = StageProfiler()
