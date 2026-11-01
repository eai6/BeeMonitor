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


# The event actions review metrics can count (memory/70).
ACTIONS = ("exit", "enter", "stay")


def review_actions(batch_id):
    """The actions the batch's review metrics count; all of them by default."""
    from .models import Batch

    chosen = (Batch.objects.filter(batch_id=batch_id).values_list("review_actions", flat=True)
              .first()) or []
    chosen = [a for a in chosen if a in ACTIONS]
    return chosen or list(ACTIONS)


def set_review_actions(batch_id, user, actions):
    """Save which actions count; at least one. Returns what was saved."""
    from .models import Batch

    chosen = [a for a in ACTIONS if a in set(actions or [])] or list(ACTIONS)
    row, _ = Batch.objects.get_or_create(batch_id=batch_id, defaults={"created_by": user})
    row.review_actions = [] if len(chosen) == len(ACTIONS) else chosen
    row.save(update_fields=["review_actions", "updated_at"])
    return chosen

