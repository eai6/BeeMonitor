"""A batch's name (memory/64). A batch is the runs sharing a ``batch_id``;
``Batch`` holds a name only once someone gives it one, and a batch with none
shows its pipeline's title."""

from __future__ import annotations

NAME_MAX = 120


def names(batch_ids):
    """``{batch_id: name}`` for the batches that have one."""
    from .models import Batch

    return dict(Batch.objects.filter(batch_id__in=list(batch_ids)).exclude(name="")
                .values_list("batch_id", "name"))


def title(batch_id, pipeline=None):
    """What a batch is called: its name, else its pipeline's title."""
    return (names([batch_id]).get(batch_id)
            or (pipeline.title if pipeline is not None else "") or "Batch")


def rename(batch_id, user, name):
    """Set (or, blank, clear) a batch's name. Returns the stored name."""
    from .models import Batch

    name = " ".join((name or "").split())[:NAME_MAX]
    Batch.objects.update_or_create(batch_id=batch_id,
                                   defaults={"name": name, "created_by": user})
    return name
