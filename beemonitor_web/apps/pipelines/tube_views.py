"""A batch's references a model found, from its page (memory/83): how many
clips were numbered, borrowed a nearby clip's grid or couldn't be placed, and
how many recounts are left. No day by day check: the References page is gone;
renumbering is Update this batch."""

from __future__ import annotations

from django.contrib.auth.decorators import login_required
from django.http import Http404, JsonResponse

from .models import PipelineRun

# tubes.state of a run's reference -> its line on the batch page. "detected"
# is memory/82's: numbered from the site's drawing.
STATES = {"placed": "placed", "borrowed": "borrowed", "unplaced": "unplaced",
          "pending": "waiting", "error": "errors", "detected": "old", "drawn": "old"}


def summary(rows) -> dict | None:
    """``{placed, borrowed, unplaced, waiting, errors, old, label}`` over the
    batch's finished runs counted with a model's references, from their
    listings; None when none were."""
    out = dict.fromkeys(("placed", "borrowed", "unplaced", "waiting", "errors", "old"), 0)
    label, seen = "", False
    for r in rows:
        tubes = (r.get("layout") or {}).get("tubes") or {}
        key = STATES.get(tubes.get("state"))
        if key is None or r["run"].status != PipelineRun.Status.COMPLETED:
            continue
        seen = True
        out[key] += 1
        label = label or (r.get("layout") or {}).get("label") or ""
    if not seen:
        return None
    out["label"] = label
    return out


def pending(batch_id) -> int:
    """The batch's finished runs no recount worker has taken yet."""
    from .recount import CURRENT_RULES

    return PipelineRun.objects.filter(batch_id=batch_id, status=PipelineRun.Status.COMPLETED,
                                      counting_rules__lt=CURRENT_RULES).count()


@login_required
def batch_recount_progress(request, batch_id):
    from . import sharing

    if not sharing.owns_batch(request.user, batch_id):
        raise Http404("No such batch.")
    return JsonResponse({"pending": pending(batch_id)})
