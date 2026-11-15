"""The YOLO models a pipeline node can pick, with the classes each one detects
(memory/82).

A Detect node's class was typed by hand, and the worker keeps a detection
only when its class name matches the model's exactly: a typo found nothing,
silently. Models trained here know their classes (``CustomModel.classes``,
from the project, in the training data's order); the built-in ones are
listed below. A model whose classes aren't known (an upload left blank) keeps
the text box.
"""

from __future__ import annotations

# The worker's built-in models, by the S3 key in the models bucket
# (cloud/wrapper/model_manager.py MODEL_FILES under MODEL_VERSION).
BUILTIN_NEST = "builtin:nest"
BUILTIN_NEST_KEY = "v1/nest_detection.pt"
BUILTIN_BEE_KEY = "v1/bee_tracking.pt"
# The names in models/nest_detection.pt (read 2026-11-15): YOLOv12m.
BUILTIN_NEST_CLASSES = ["hotel", "nest"]


def builtin_bee_classes() -> list[str]:
    from apps.annotations.labelers import builtin_classes

    return builtin_classes()


def _builtin() -> list[dict]:
    return [{"id": "", "name": "BeeMonitor default bee model", "classes": builtin_bee_classes(),
             "key": BUILTIN_BEE_KEY},
            {"id": BUILTIN_NEST, "name": "BeeMonitor default nest model",
             "classes": list(BUILTIN_NEST_CLASSES), "key": BUILTIN_NEST_KEY}]


def yolo_models(user) -> list[dict]:
    """``[{id, name, classes, key}]`` for a YOLO model picker: the built-in
    bee model (id ""), the built-in nest model, then the user's usable models."""
    from .models import CustomModel

    out = _builtin()
    for m in (CustomModel.usable(user).exclude(model_type=CustomModel.ModelType.EVENT_CLASSIFIER)
              .order_by("-pk")):
        out.append({"id": str(m.pk), "name": m.name, "classes": list(m.classes or []),
                    "key": m.storage_key})
    return out


def resolve(value, user) -> dict | None:
    """``{id, name, classes, key}`` of a picker value, or None when it names
    no model this user may run. ``""`` is the built-in bee model."""
    from .models import CustomModel

    value = str(value or "").strip()
    if value == "":
        return _builtin()[0]
    if value == BUILTIN_NEST:
        return _builtin()[1]
    try:
        pk = int(value)
    except (TypeError, ValueError):
        return None
    m = CustomModel.usable(user).filter(pk=pk).first()
    if m is None:
        return None
    return {"id": str(m.pk), "name": m.name, "classes": list(m.classes or []),
            "key": m.storage_key, "model": m}
