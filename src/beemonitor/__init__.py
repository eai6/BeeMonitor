"""
BeeMonitor: Computer vision system for monitoring solitary bee activity.

This package provides tools for detecting, tracking, and analyzing bee behavior
in bee hotel videos using YOLO-based object detection and custom tracking algorithms.

``BeeMonitor``/``AnalysisResults``/``Config`` are resolved lazily (PEP 562). The
top-level analyzer pulls in ultralytics and torch, which only the GPU worker
image carries — importing a light submodule such as
:mod:`beemonitor.identification` from the web app must not drag those in.
``from beemonitor import BeeMonitor`` still works exactly as before.
"""

__version__ = "1.0.0"
__author__ = "Edward Amoah"
__email__ = "eai6@psu.edu"

__all__ = [
    "BeeMonitor",
    "AnalysisResults",
    "Config",
]

_LAZY = {
    "BeeMonitor": ("beemonitor.core.video_analyzer", "BeeMonitor"),
    "AnalysisResults": ("beemonitor.core.video_analyzer", "AnalysisResults"),
    "Config": ("beemonitor.core.config", "Config"),
}


def __getattr__(name):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(target[0]), target[1])
    globals()[name] = value  # resolve once
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
