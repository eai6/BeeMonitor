"""Who draws the pre-label boxes: SAM 3, or a YOLO model (memory/57).

SAM 3 is prompted with the project's class names. A YOLO model only knows its
own classes, so each of them is mapped onto a project class or skipped: a bee
model can pre-label a project whose class is "insect". The map travels with the
task; the GPU returns boxes under the model's names and the web app renames
them when it writes the frames.
"""

from django.conf import settings

SAM3 = "sam3"
BUILTIN = "builtin"
SKIP = ""

YOLO_CONFIDENCE = 0.25
SAM3_CONFIDENCE = 0.3


class LabelerError(ValueError):
    pass


def builtin_classes():
    # bee_tracking.pt knows one class, "bee".
    return list(getattr(settings, "BEEMONITOR_DEFAULT_BEE_MODEL_CLASSES", None) or ["bee"])


def choices(user):
    """``[{value, name, classes, model_key}]`` the user may pre-label with."""
    from apps.training.models import CustomModel

    out = [{"value": SAM3, "name": "SAM 3", "classes": [], "model_key": ""},
           {"value": BUILTIN, "name": "Built-in bee model (YOLO)",
            "classes": builtin_classes(), "model_key": ""}]
    for cm in CustomModel.usable(user).order_by("-created_at"):
        owner = "" if cm.user_id == user.pk else " (shared)"
        out.append({"value": f"cm:{cm.pk}", "name": f"{cm.name}{owner}",
                    "classes": [c for c in (cm.classes or []) if c],
                    "model_key": cm.storage_key})
    return out


def default_map(model_classes, project_classes):
    """Each model class → the project class of the same name, else the
    project's only class, else skipped."""
    by_lower = {c.lower(): c for c in project_classes}
    only = project_classes[0] if len(project_classes) == 1 else SKIP
    return {m: by_lower.get(m.lower(), only) for m in model_classes}


def with_rows(user, project):
    """``choices`` plus mapping rows for the form: each YOLO choice gets
    ``rows = [{model, field, picked}]``."""
    project_classes = project.classes or []
    out = []
    for c in choices(user):
        if c["value"] != SAM3:
            picks = default_map(c["classes"], project_classes)
            c = {**c, "rows": [{"model": m, "field": field_name(c["value"], m), "picked": picks[m]}
                               for m in c["classes"]]}
        out.append(c)
    return out


def field_name(value, model_class, prefix="s_"):
    return f"{prefix}map:{value}:{model_class}"


def from_post(post, user, project, prefix="s_"):
    """The labeler picked on a form: ``{"labeler": "sam3"}``, or
    ``{"labeler": "yolo", "model_key", "model_name", "class_map"}``.
    Raises LabelerError for a model the user can't use or a map that keeps
    nothing."""
    value = (post.get(prefix + "labeler") or SAM3).strip()
    if value == SAM3:
        return {"labeler": SAM3}
    picked = next((c for c in choices(user) if c["value"] == value), None)
    if picked is None:
        raise LabelerError("That model isn't available to you.")
    if not picked["classes"]:
        raise LabelerError(f"{picked['name']} doesn't list its classes, so its boxes can't be mapped.")
    project_classes = project.classes or []
    class_map = {}
    for m in picked["classes"]:
        target = (post.get(field_name(value, m, prefix)) or "").strip()
        if target and target in project_classes:
            class_map[m] = target
    if not class_map:
        raise LabelerError(
            f"Map at least one of {picked['name']}'s classes ({', '.join(picked['classes'])}) "
            f"to a project class ({', '.join(project_classes)}).")
    return {"labeler": "yolo", "model_key": picked["model_key"],
            "model_name": picked["name"], "class_map": class_map}


def map_boxes(boxes, class_map):
    """Rename boxes from the model's classes to the project's; boxes of
    unmapped classes are dropped. No map: unchanged."""
    if not class_map:
        return list(boxes or [])
    lower = {k.lower(): v for k, v in class_map.items()}
    out = []
    for b in boxes or []:
        target = lower.get((b.get("class") or "").lower())
        if target:
            out.append({**b, "class": target})
    return out


def describe(spec):
    if spec.get("labeler") != "yolo":
        return "SAM 3"
    pairs = ", ".join(f"{m} → {p}" for m, p in spec["class_map"].items())
    return f"{spec['model_name']} ({pairs})"
