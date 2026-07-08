"""Closed auto-fine-tuning loop (memory/25, P3).

State machine that chains the existing pieces into "the model adapts itself":

    RELABELING  → SAM 3 diverse pre-annotation writes Annotation rows
    TRAINING    → a TrainingJob fine-tunes YOLO on those labels (existing pipeline)
    EVALUATING  → read the candidate's val mAP50 (produced by train.py)
    AWAITING    → mAP50 ≥ min_map50 → wait for the user to approve promotion
    HELD        → below the bar → not promoted
    PROMOTED    → candidate CustomModel set active (user-approved)

advance_run() is pure DB-state (no SageMaker calls): SAM 3 pre-annotation and the
training poll update the underlying rows; this just moves the run forward. Per the
chosen posture, promotion is a user click (promote_run), gated by the mAP50 bar.
"""

import logging
import threading

logger = logging.getLogger(__name__)

DEFAULT_MIN_MAP50 = 0.5


def _metric(metrics, *keys):
    for k in keys:
        if metrics and metrics.get(k) is not None:
            return float(metrics[k])
    return 0.0


def start_run(user, video_ids, scope="default", min_map50=DEFAULT_MIN_MAP50):
    """Create an AnnotationProject from the given videos, kick off SAM 3 diverse
    relabeling, and open an AdaptationRun in RELABELING."""
    from apps.annotations.models import AnnotationProject, PreAnnotationTask
    from apps.annotations.views import spawn_preannotation_async
    from apps.videos.models import Video

    from .models import AdaptationRun

    videos = list(Video.objects.filter(user=user, pk__in=video_ids))
    videos = [v for v in videos if v.storage_key]
    if not videos:
        return None

    project = AnnotationProject.objects.create(
        user=user, name=f"Auto-adapt · {scope}", classes=["bee", "wasp", "nest"],
    )
    project.videos.set(videos)
    run = AdaptationRun.objects.create(
        user=user, scope=scope, project=project,
        status=AdaptationRun.Status.RELABELING, min_map50=min_map50,
        detail=f"SAM 3 relabeling {len(videos)} video(s)",
    )
    # Durable SAM 3 diverse pre-annotation tasks (reconciler finalizes them).
    for v in videos:
        task = PreAnnotationTask.objects.create(
            user=user, project=project, video=v, labeler="sam3",
            params={"sample_interval": 10, "max_frames": 300, "confidence": 0.3,
                    "selection": "diverse", "nms_iou": 0.5, "max_detections": 100},
        )
        spawn_preannotation_async(task.pk)
    logger.info("adapt: run %s started (scope=%s, %d videos)", run.pk, scope, len(videos))
    return run


def advance_run(run):
    """Advance one run based on current DB state. Idempotent."""
    from apps.annotations.models import Annotation, PreAnnotationTask

    from .models import AdaptationRun, CustomModel, TrainingJob

    S = AdaptationRun.Status

    if run.status == S.RELABELING:
        # Wait until ALL of the project's SAM3 pre-annotation tasks have
        # finished (previously fired training the instant *any* annotation row
        # existed — a race that trained on a half-labeled project). Durable
        # PreAnnotationTask state makes this reliable.
        if not run.project:
            return run
        pending = PreAnnotationTask.objects.filter(
            project=run.project,
            status__in=[PreAnnotationTask.Status.QUEUED, PreAnnotationTask.Status.PROCESSING],
        ).exists()
        has_labels = Annotation.objects.filter(project=run.project).exists()
        if pending:
            run.detail = "SAM 3 relabeling in progress…"
            run.save(update_fields=["detail", "updated_at"])
        elif has_labels:
            job = TrainingJob.objects.create(
                user=run.user, project=run.project,
                name=f"Auto-adapt #{run.pk} · {run.scope}",
                base_model=TrainingJob.BaseModel.YOLOV8N,
            )
            run.training_job = job
            run.status = S.TRAINING
            run.detail = "training on SAM 3 labels"
            run.save(update_fields=["training_job", "status", "detail", "updated_at"])
            from .views import _spawn_training_job
            threading.Thread(target=_spawn_training_job, args=(job.pk,), daemon=True).start()
        else:
            # All relabel tasks finished but produced nothing usable.
            run.status = S.FAILED
            run.detail = "SAM 3 relabeling produced no annotations"
            run.save(update_fields=["status", "detail", "updated_at"])

    elif run.status == S.TRAINING:
        job = run.training_job
        if job:
            job.refresh_from_db()
            if job.status == TrainingJob.Status.FAILED:
                run.status = S.FAILED
                run.detail = job.error_message[:255] or "training failed"
                run.save(update_fields=["status", "detail", "updated_at"])
            else:
                cm = CustomModel.objects.filter(training_job=job).first()
                if cm:
                    run.candidate_model = cm
                    run.status = S.EVALUATING
                    run.detail = "reading candidate metrics"
                    run.save(update_fields=["candidate_model", "status", "detail", "updated_at"])

    elif run.status == S.EVALUATING:
        cm = run.candidate_model
        map50 = _metric(cm.metrics if cm else {}, "mAP50", "map50", "metrics/mAP50(B)")
        if cm and map50 >= run.min_map50:
            run.status = S.AWAITING
            run.detail = f"candidate mAP50={map50:.3f} ≥ {run.min_map50:.2f} — promote?"
        else:
            run.status = S.HELD
            run.detail = f"candidate mAP50={map50:.3f} < {run.min_map50:.2f} — not promoted"
        run.save(update_fields=["status", "detail", "updated_at"])

    return run


def advance_user_runs(user):
    """Advance all of a user's in-flight runs (called on each dashboard load)."""
    from .models import AdaptationRun

    active = [AdaptationRun.Status.RELABELING, AdaptationRun.Status.TRAINING,
              AdaptationRun.Status.EVALUATING]
    for run in AdaptationRun.objects.filter(user=user, status__in=active):
        try:
            advance_run(run)
        except Exception as e:  # noqa: BLE001
            logger.error("adapt: advance run %s failed: %s", run.pk, e, exc_info=True)


def promote_run(run):
    """User-approved promotion: activate the candidate, deactivate siblings of the
    same model_type. Guarded — only from AWAITING."""
    from .models import AdaptationRun, CustomModel

    if run.status != AdaptationRun.Status.AWAITING or not run.candidate_model:
        return run
    cm = run.candidate_model
    CustomModel.objects.filter(
        user=run.user, model_type=cm.model_type, is_active=True,
    ).exclude(pk=cm.pk).update(is_active=False)
    cm.is_active = True
    cm.save(update_fields=["is_active"])
    run.status = AdaptationRun.Status.PROMOTED
    run.detail = f"promoted {cm.name}"
    run.save(update_fields=["status", "detail", "updated_at"])
    logger.info("adapt: run %s promoted model %s", run.pk, cm.pk)
    return run
