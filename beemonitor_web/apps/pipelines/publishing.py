"""Publishing a pipeline (memory/59).

A published pipeline is the method, not the results: anyone signed in can
view it, run it on their own clips and copy it to change, but never edit the
original. Batch share links (sharing.py) are what share a run's results.

A step can name a custom model by pk. The executor resolves it through
``CustomModel.usable(run.user)``, so a private model fails for everyone but
its owner — publishing waits until every model the pipeline uses is public.
"""

from django.utils import timezone

from .models import Pipeline
from .registry import get_block

# Step config keys that hold a CustomModel pk (executors._resolve_custom_models).
MODEL_KEYS = ("object_model", "bee_model", "reference_model", "nest_model")


def models_used(pipeline):
    """The custom models the steps name, in step order, without repeats."""
    from apps.training.models import CustomModel

    pks = []
    for step in pipeline.steps or []:
        cfg = step.get("config") or {}
        for key in MODEL_KEYS:
            try:
                pk = int(cfg.get(key) or 0)
            except (TypeError, ValueError):
                continue
            if pk and pk not in pks:
                pks.append(pk)
    found = CustomModel.objects.in_bulk(pks)
    return [found[pk] for pk in pks if pk in found]


def blockers(pipeline):
    """Models that would stop the pipeline running for anyone else."""
    return [m for m in models_used(pipeline) if m.visibility != "public"]


def publish(pipeline):
    """Publish, or return the models in the way (nothing is changed then)."""
    in_the_way = blockers(pipeline)
    if in_the_way:
        return in_the_way
    pipeline.visibility = Pipeline.Visibility.PUBLIC
    pipeline.published_at = pipeline.published_at or timezone.now()
    pipeline.save(update_fields=["visibility", "published_at"])
    return []


def unpublish(pipeline):
    """Stop listing it and stop new runs by others. Copies and past runs stay."""
    pipeline.visibility = Pipeline.Visibility.PRIVATE
    pipeline.save(update_fields=["visibility"])


def summary(pipeline):
    """What a Browse card says: the steps in order and the models they use."""
    steps = []
    for step in pipeline.steps or []:
        block = get_block(step.get("block_type") or "")
        steps.append({"name": block["display_name"] if block else step.get("block_type") or "step",
                      "icon": block.get("icon", "") if block else ""})
    return {
        "steps": steps,
        "models": [m.name for m in models_used(pipeline)],
        "copies": pipeline.clones.count(),
    }
