"""LLM vision review of annotations (design in repo memory/26).

A vision model looks at a pre-annotated frame with its boxes drawn + NUMBERED, and
reports which numbers are real insects (keep) vs false positives (remove) — cutting
the detector's over-detection on bee-hotel scenes. Filter-only: it never invents
coordinates (LLMs are unreliable at precise boxes), it only selects among existing
boxes by index. Fails safe: on any error it returns the input boxes unchanged.

Per-frame review (editor) uses the accurate model; batch uses the cheap one.
"""

import base64
import io
import json
import logging

from django.conf import settings

logger = logging.getLogger(__name__)

# Distinct outline colors for numbered boxes (RGB).
_COLORS = [(239, 68, 68), (59, 130, 246), (34, 197, 94), (234, 179, 8),
           (168, 85, 247), (236, 72, 153), (20, 184, 166), (249, 115, 22)]

_TOOL = {
    "name": "report_review",
    "description": "Report which numbered boxes are real insects (bees/wasps).",
    "input_schema": {
        "type": "object",
        "properties": {
            "keep": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Indices of the numbered boxes that are ACTUALLY insects. "
                               "Omit every false positive (nest holes, wood knots, shadows, leaves, grass).",
            },
            "notes": {"type": "string", "description": "One short line on what was removed."},
        },
        "required": ["keep"],
    },
}


_DETECT_TOOL = {
    "name": "report_detections",
    "description": "Report a bounding box for every insect visible in the frame.",
    "input_schema": {
        "type": "object",
        "properties": {
            "boxes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number", "description": "left edge, pixels from left"},
                        "y": {"type": "number", "description": "top edge, pixels from top"},
                        "w": {"type": "number", "description": "width in pixels"},
                        "h": {"type": "number", "description": "height in pixels"},
                        "class": {"type": "string", "description": "one of the allowed class labels"},
                    },
                    "required": ["x", "y", "w", "h"],
                },
            },
            "notes": {"type": "string", "description": "One short line (e.g. 'no insects found')."},
        },
        "required": ["boxes"],
    },
}


def available():
    return bool(getattr(settings, "ANTHROPIC_API_KEY", ""))


def detect_boxes(jpeg_bytes, class_names, model=None):
    """Ask the vision model to propose bounding boxes for insects in one frame.

    Returns (boxes, notes). boxes are pixel dicts {x,y,w,h,class,class_id,confidence}
    clamped to the image. Fails safe → ([], reason). Intended as a per-frame
    annotation *assist*: the user reviews/adjusts before saving (LLM boxes are
    approximate, so treat them as suggestions)."""
    if not jpeg_bytes:
        return [], "frame image unavailable"
    if not available():
        return [], "LLM not configured"
    try:
        import anthropic
        from PIL import Image
    except ImportError:
        return [], "dependencies not installed"

    try:
        img = Image.open(io.BytesIO(jpeg_bytes))
        W, H = img.size
        classes = [c for c in (class_names or []) if c]
        class_list = ", ".join(classes) or "insect"
        default_class = classes[0] if classes else "bee"
        b64 = base64.b64encode(jpeg_bytes).decode()
        prompt = (
            f"This is a {W}x{H} px frame from a camera monitoring a solitary-bee hotel "
            "(a wooden block with rows of round nest holes). Find EVERY insect actually "
            f"present (allowed labels: {class_list}). Ignore nest holes, wood knots, "
            "shadows, leaves and grass — those are NOT insects. For each real insect call "
            "report_detections with a tight bounding box in PIXELS (origin top-left: x,y = "
            "top-left corner, w,h = size). If there are no insects, return an empty list."
        )
        client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        msg = client.messages.create(
            model=model or settings.ASSISTANT_VISION_MODEL,
            max_tokens=2048,
            tools=[_DETECT_TOOL],
            tool_choice={"type": "tool", "name": "report_detections"},
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64",
                                             "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": prompt},
            ]}],
        )
        raw, notes = None, ""
        for block in msg.content:
            if getattr(block, "type", "") == "tool_use" and block.name == "report_detections":
                raw = block.input.get("boxes", [])
                notes = block.input.get("notes", "")
        if raw is None:
            return [], "no tool output"

        class_id = {c: i for i, c in enumerate(classes)}
        out = []
        for b in raw:
            try:
                x, y, w, h = float(b["x"]), float(b["y"]), float(b["w"]), float(b["h"])
            except (KeyError, TypeError, ValueError):
                continue
            # clamp to the frame
            x = max(0, min(x, W - 1)); y = max(0, min(y, H - 1))
            w = max(1, min(w, W - x)); h = max(1, min(h, H - y))
            name = b.get("class") if b.get("class") in class_id else default_class
            out.append({
                "x": round(x), "y": round(y), "w": round(w), "h": round(h),
                "class": name, "class_id": class_id.get(name, 0),
                "confidence": 0.5, "source": "llm",
            })
        return out, notes or f"found {len(out)} insect(s)"
    except Exception as e:  # noqa: BLE001
        logger.error("llm_review: detect_boxes failed: %s", e)
        return [], f"error: {e}"


def _frame_jpeg(annotation):
    """Bytes of the frame JPEG (processed bucket), or None."""
    key = annotation.frame_image_path
    if not key:
        return None
    try:
        import urllib.request

        from config.storage import presigned_get
        url = presigned_get(key, container="processed", expiry_hours=1)
        if not url:
            return None
        with urllib.request.urlopen(url, timeout=20) as r:
            return r.read()
    except Exception as e:  # noqa: BLE001
        logger.warning("llm_review: frame fetch failed for ann %s: %s", annotation.pk, e)
        return None


def _numbered_overlay(jpeg_bytes, boxes):
    """Draw each box + its index number on the frame → new JPEG bytes (Pillow)."""
    from PIL import Image, ImageDraw
    img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)
    for i, b in enumerate(boxes):
        c = _COLORS[i % len(_COLORS)]
        x, y, w, h = int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"])
        draw.rectangle([x, y, x + w, y + h], outline=c, width=2)
        draw.text((x + 2, max(0, y - 12)), str(i), fill=c)
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=85)
    return out.getvalue()


def review_boxes(annotation, boxes, model=None):
    """Return (kept_boxes, notes): the subset the LLM judges real. Filter-only.

    Never raises — on any failure returns (boxes, reason) so the caller can decide
    (a review that can't run leaves the boxes untouched).
    """
    if not boxes:
        return boxes, "no boxes"
    if not available():
        return boxes, "LLM not configured"
    try:
        import anthropic
    except ImportError:
        return boxes, "anthropic not installed"

    jpeg = _frame_jpeg(annotation)
    if not jpeg:
        return boxes, "frame image unavailable"

    try:
        overlay = _numbered_overlay(jpeg, boxes)
        b64 = base64.b64encode(overlay).decode()
        listing = [{"i": i, "class": b.get("class"), "confidence": b.get("confidence")}
                   for i, b in enumerate(boxes)]
        prompt = (
            "This frame is from a camera monitoring a solitary-bee hotel (a wooden block "
            "with rows of round nest holes). An automatic detector drew the NUMBERED boxes "
            "shown on the image. Many are false positives on nest holes, wood knots, "
            "shadows, leaves, or grass. Proposed boxes:\n"
            + json.dumps(listing)
            + "\n\nLook closely at the image. Call report_review with the indices of the "
            "boxes that are ACTUALLY insects (bees/wasps) on or near the hotel. Omit every "
            "false positive. When you are unsure, omit it."
        )
        client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        msg = client.messages.create(
            model=model or settings.ASSISTANT_MODEL,
            max_tokens=1024,
            tools=[_TOOL],
            tool_choice={"type": "tool", "name": "report_review"},
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64",
                                             "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": prompt},
            ]}],
        )
        keep, notes = None, ""
        for block in msg.content:
            if getattr(block, "type", "") == "tool_use" and block.name == "report_review":
                keep = block.input.get("keep", [])
                notes = block.input.get("notes", "")
        if keep is None:
            return boxes, "no tool output"
        kept = [boxes[i] for i in keep if isinstance(i, int) and 0 <= i < len(boxes)]
        return kept, notes or f"kept {len(kept)}/{len(boxes)}"
    except Exception as e:  # noqa: BLE001
        logger.error("llm_review: review_boxes failed for ann %s: %s", annotation.pk, e)
        return boxes, f"error: {e}"


def review_annotation(annotation_id, model=None):
    """Batch path: load an annotation, LLM-filter its boxes, persist as llm-reviewed.
    Skips frames a human already reviewed. Runs in a daemon thread."""
    from django.db import connection
    from django.utils import timezone

    from .models import Annotation
    try:
        ann = Annotation.objects.select_related("video").get(pk=annotation_id)
        if ann.review_source == Annotation.ReviewSource.HUMAN:
            return  # never override a human
        kept, notes = review_boxes(ann, ann.boxes or [], model=model)
        ann.boxes = kept
        ann.reviewed = True
        ann.review_source = Annotation.ReviewSource.LLM
        ann.reviewed_at = timezone.now()
        ann.save()
        logger.info("llm_review: ann %s -> %s", annotation_id, notes)
    except Exception as e:  # noqa: BLE001
        logger.error("llm_review: review_annotation %s failed: %s", annotation_id, e, exc_info=True)
    finally:
        connection.close()
