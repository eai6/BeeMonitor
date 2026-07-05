"""
Pipeline execution engine — a resumable state machine.

``advance_run`` runs every step that has become *ready* (all its inputs are done):
local steps execute inline; GPU steps spawn an ``analysis.Job`` and leave the run
parked. When the analysis poller marks a tagged Job finished it calls
``on_job_finished``, which records the step's output and calls ``advance_run`` again.
The same readiness rule (all inputs done) drives both linear pipelines and — later —
the DAG canvas, so the scheduler never needs rewriting for Phase 2.

Design: ``memory/23_pipeline_builder_port_design.md``.
"""

import copy
import hashlib
import json
import logging

from django.db import transaction
from django.utils import timezone

from .models import PipelineRun, StepResult
from .registry import get_block
from . import executors

logger = logging.getLogger(__name__)


def _gpu_cache_key(run, step, context, index):
    """Content hash of a GPU step's effective job config (video + ROI + flags).

    Two runs whose track/detect step resolves to the *same* video, ROI/nest layout
    and analysis flags share a key — so the expensive SageMaker result is reused.
    A changed ROI or video yields a different key and re-runs. Returns None when the
    step can't be keyed yet (no upstream video) — then it simply isn't cached.
    """
    built, _err = executors.build_detect_and_track_config(step, run, context, index)
    if not built:
        return None
    eff = {"video": built.get("video_id"), **(built.get("config") or {})}
    payload = json.dumps(
        {"u": run.user_id, "b": step.get("block_type", ""), "c": eff},
        sort_keys=True, default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _persist_step_result(user, cache_key, block_type, output):
    """Cache a successful step output for cross-run reuse (strips _-prefixed meta)."""
    if not cache_key:
        return
    clean = {k: v for k, v in output.items() if not k.startswith("_")}
    StepResult.objects.update_or_create(
        user=user, cache_key=cache_key,
        defaults={"block_type": block_type, "output": clean},
    )


def steps_with_video(pipeline, video_id):
    """A copy of a pipeline's steps with every ``input.video`` step bound to a video.

    Lets one pipeline run across many videos (e.g. from the Processing hub) by
    injecting each video into the input step at launch time.
    """
    steps = copy.deepcopy(pipeline.steps or [])
    for step in steps:
        if step.get("block_type") == "input.video":
            step.setdefault("config", {})["video_id"] = str(video_id)
    return steps


def start_run(run, steps=None):
    """Kick a freshly-created run: freeze steps, init status, advance.

    ``steps`` overrides the pipeline's steps for this run (e.g. with a specific
    video injected); defaults to the pipeline's saved steps.
    """
    run.steps = steps if steps is not None else (run.pipeline.steps or [])
    run.step_status = {s["id"]: PipelineRun.STEP_PENDING for s in run.steps if s.get("id")}
    run.context = {}
    run.status = PipelineRun.Status.RUNNING
    run.started_at = timezone.now()
    run.save(update_fields=["steps", "step_status", "context", "status", "started_at"])
    advance_run(run.pk)


def advance_run(run_pk):
    """Advance a run as far as it can go right now (transactional + idempotent)."""
    with transaction.atomic():
        try:
            run = PipelineRun.objects.select_for_update().get(pk=run_pk)
        except PipelineRun.DoesNotExist:
            return
        if run.is_terminal:
            return

        steps = run.steps or []
        status = dict(run.step_status or {})
        context = dict(run.context or {})
        by_index = list(enumerate(steps))

        progressed = True
        while progressed:
            progressed = False
            for index, step in by_index:
                sid = step.get("id")
                if not sid:
                    continue
                state = status.get(sid, PipelineRun.STEP_PENDING)
                if state in (PipelineRun.STEP_DONE, PipelineRun.STEP_FAILED, PipelineRun.STEP_RUNNING):
                    continue

                ups = executors.upstream_ids(step, steps, index)
                # Cascade failure: a failed dependency fails this step.
                if any(status.get(u) == PipelineRun.STEP_FAILED for u in ups):
                    status[sid] = PipelineRun.STEP_FAILED
                    context[sid] = {"error": "Upstream step failed."}
                    progressed = True
                    continue
                # Not ready until every dependency is done.
                if not all(status.get(u) == PipelineRun.STEP_DONE for u in ups):
                    continue

                block = get_block(step.get("block_type", "")) or {}
                backend = block.get("backend", "local")

                if backend == "gpu":
                    # Reuse an identical GPU step's cached output (skips SageMaker).
                    key = _gpu_cache_key(run, step, context, index)
                    cached = (StepResult.objects.filter(user=run.user, cache_key=key).first()
                              if key else None)
                    if cached:
                        out = dict(cached.output)
                        out["_cache_key"] = key
                        out["_cached"] = True
                        context[sid] = out
                        status[sid] = PipelineRun.STEP_DONE
                        progressed = True
                        continue

                    result_state, output = executors.submit_gpu_step(run, step, context, index)
                    if key:
                        output["_cache_key"] = key   # so on_job_finished can persist it
                    context[sid] = output
                    if result_state == "submitted":
                        status[sid] = PipelineRun.STEP_RUNNING   # parks the run
                    elif result_state == "done":
                        status[sid] = PipelineRun.STEP_DONE
                        _persist_step_result(run.user, key, step.get("block_type", ""), output)
                    else:
                        status[sid] = PipelineRun.STEP_FAILED
                    progressed = True
                else:
                    output = executors.run_local_step(step, run, context, index)
                    context[sid] = output
                    status[sid] = (
                        PipelineRun.STEP_FAILED if output.get("error") else PipelineRun.STEP_DONE
                    )
                    progressed = True

        run.step_status = status
        run.context = context
        _finalize_if_terminal(run, steps, status)
        run.save()


def _finalize_if_terminal(run, steps, status):
    """Mark the run completed/failed once every step reaches a terminal state."""
    states = [status.get(s.get("id"), PipelineRun.STEP_PENDING) for s in steps if s.get("id")]
    if not states:
        run.status = PipelineRun.Status.FAILED
        run.error_message = "Pipeline has no steps."
    elif all(st in (PipelineRun.STEP_DONE, PipelineRun.STEP_FAILED) for st in states):
        any_failed = any(st == PipelineRun.STEP_FAILED for st in states)
        run.status = PipelineRun.Status.FAILED if any_failed else PipelineRun.Status.COMPLETED
    else:
        return  # still has RUNNING/PENDING work — stay RUNNING

    run.completed_at = timezone.now()
    if run.started_at:
        run.execution_time_ms = int(
            (run.completed_at - run.started_at).total_seconds() * 1000
        )


def on_job_finished(job):
    """Poller hook: an ``analysis.Job`` tagged with a pipeline run has finished.

    Records the step's output from the job result and advances the run. Safe to call
    for every finished job — it no-ops for non-pipeline jobs. Never raises into the
    poller.
    """
    try:
        config = getattr(job, "config", None) or {}
        run_id = config.get("pipeline_run_id")
        step_id = config.get("pipeline_step_id")
        if not run_id or not step_id:
            return

        with transaction.atomic():
            try:
                run = PipelineRun.objects.select_for_update().get(pk=run_id)
            except PipelineRun.DoesNotExist:
                return

            status = dict(run.step_status or {})
            context = dict(run.context or {})
            job_status = getattr(job, "status", "")

            if job_status == "completed":
                result = _job_result_summary(job)
                out = dict(context.get(step_id, {}))
                out.update({"result": result, "pending": False, "job_id": job.pk})
                context[step_id] = out
                status[step_id] = PipelineRun.STEP_DONE
                # Cache the (expensive) GPU output for reuse by future runs.
                block_type = next((s.get("block_type", "") for s in (run.steps or [])
                                   if s.get("id") == step_id), "")
                _persist_step_result(run.user, out.get("_cache_key"), block_type, out)
            elif job_status in ("failed", "cancelled"):
                context[step_id] = {
                    "error": getattr(job, "error_message", "") or "GPU job failed.",
                    "job_id": job.pk,
                }
                status[step_id] = PipelineRun.STEP_FAILED
            else:
                return  # not a terminal job status; ignore

            run.step_status = status
            run.context = context
            run.save(update_fields=["step_status", "context"])

        # Continue the state machine outside the first transaction.
        advance_run(run.pk)
    except Exception:  # the poller must never break on a pipeline hook
        logger.exception("on_job_finished hook failed for job %s", getattr(job, "pk", "?"))


def _job_result_summary(job):
    """Pull the JobResult fields a downstream step might read into a plain dict."""
    from apps.analysis.models import JobResult

    result = JobResult.objects.filter(job_id=job.pk).first()
    if result is None:
        return {}
    fields = (
        "foraging_trip_count", "avg_trip_duration_sec", "foraging_trips_csv_path",
        "tracking_csv_path", "events_csv_path", "unique_tracks", "entry_count",
        "exit_count", "nest_count", "total_events", "interaction_count",
        "annotated_video_path", "summary_stats",
    )
    return {f: getattr(result, f, None) for f in fields}
