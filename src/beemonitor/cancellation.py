"""Stopping a clip that its owner cancelled (memory/49).

SageMaker async inference has no way to abort a request, so a cancelled clip
would otherwise run to the end and bill for it. The web app leaves a marker in
S3 when a job is cancelled; the worker watches for it and long loops call
``check()``, which raises ``JobCancelled`` once the marker has been seen.

Library code only reads a flag here. Who sets it (the SageMaker worker's
watcher thread) is the caller's business, so the analysis library never
touches S3 for this. Per thread, because a container can run more than one
clip at once.
"""
from __future__ import annotations

import logging
import re
import threading

logger = logging.getLogger(__name__)

_local = threading.local()

# Where the web app leaves the marker: the "processed" bucket, keyed by the
# job's id without a chunk suffix, so cancelling a job stops every chunk of it.
MARKER_PREFIX = "cancel/"


class JobCancelled(Exception):
    """The owner cancelled this clip while it was being processed."""


def marker_key(job_id: str) -> str:
    return MARKER_PREFIX + re.sub(r"-c\d+$", "", str(job_id))


def watch(event: threading.Event | None) -> None:
    """Make ``event`` this thread's cancel flag (None clears it)."""
    _local.event = event


def requested() -> bool:
    event = getattr(_local, "event", None)
    return bool(event is not None and event.is_set())


def check() -> None:
    """Raise ``JobCancelled`` if this thread's clip has been cancelled."""
    if requested():
        raise JobCancelled("Cancelled by user.")


class Watcher:
    """Polls for a job's cancel marker in the background while a clip runs.

    ``exists`` is any callable answering "is the marker there?"; it is called
    once up front (``cancelled_already``) and then every ``interval`` seconds.
    A failing lookup is logged and treated as "not cancelled": a flaky S3 call
    must never stop a clip nobody cancelled.
    """

    def __init__(self, job_id: str, exists, interval: float = 15.0):
        self.key = marker_key(job_id)
        self._exists = exists
        self._interval = interval
        self.event = threading.Event()
        self._stop = threading.Event()
        self._thread = None

    def _look(self) -> bool:
        try:
            return bool(self._exists(self.key))
        except Exception:  # noqa: BLE001 - see the class docstring
            logger.warning("cancel: could not check %s", self.key, exc_info=True)
            return False

    def cancelled_already(self) -> bool:
        if self._look():
            self.event.set()
        return self.event.is_set()

    def _run(self):
        while not self._stop.wait(self._interval):
            if self._look():
                logger.info("cancel: %s found; stopping this clip", self.key)
                self.event.set()
                return

    def __enter__(self):
        watch(self.event)
        self._thread = threading.Thread(target=self._run, name="cancel-watch", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        watch(None)
        return False
