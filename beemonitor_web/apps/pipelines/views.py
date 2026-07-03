"""
Views for the BeeMonitor pipeline builder (Phase 1 — linear builder).

Function-based + HTMX, following the Workshop builder's proven endpoint contract
(add / remove / move / configure / save / run) but re-themed onto BeeMonitor's
Tailwind base template and backed by the ``analysis.Job`` machinery.
"""

import copy
import json
import uuid

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from apps.videos.models import Video

from . import engine
from .models import Pipeline, PipelineRun
from .registry import BLOCK_REGISTRY, get_block, get_categories, validate_steps


# ── Helpers ──────────────────────────────────────────────────────────────────

def _is_htmx(request):
    return request.headers.get("HX-Request") == "true"


def _own_pipeline(request, pk):
    return get_object_or_404(Pipeline, pk=pk, user=request.user)


def _enrich_steps(steps):
    """Attach block metadata + config fields (with current values) for rendering."""
    enriched = []
    for step in steps:
        block = get_block(step.get("block_type", "")) or {}
        config = step.get("config", {})
        fields = []
        for field in block.get("config_fields", []):
            fields.append({
                **field,
                "current_value": config.get(field["name"], field.get("default", "")),
            })
        enriched.append({
            **step,
            "block": block,
            "display_name": block.get("display_name", step.get("block_type", "Unknown")),
            "icon": block.get("icon", "🔧"),
            "input_type": block.get("input_type", "none"),
            "output_type": block.get("output_type", "none"),
            "config_fields": fields,
        })
    return enriched


def _user_videos(request):
    return Video.objects.filter(user=request.user).order_by("-id")[:200]


def _editor_context(request, pipeline, extra=None):
    ctx = {
        "pipeline": pipeline,
        "steps": _enrich_steps(pipeline.steps or []),
        "categories": get_categories(),
        "videos": _user_videos(request),
        "errors": validate_steps(pipeline.steps or []),
    }
    if extra:
        ctx.update(extra)
    return ctx


def _render_steps(request, pipeline):
    """Return the steps-column partial (used as the HTMX swap target)."""
    return render(request, "pipelines/_steps.html", _editor_context(request, pipeline))


# ── Pipeline CRUD ─────────────────────────────────────────────────────────────

@login_required
def pipeline_list(request):
    pipelines = Pipeline.objects.filter(user=request.user, is_template=False)
    templates = Pipeline.objects.filter(is_template=True)
    return render(request, "pipelines/list.html", {
        "pipelines": pipelines,
        "templates": templates,
    })


@login_required
def pipeline_create(request):
    if request.method == "POST":
        title = request.POST.get("title", "").strip() or "Untitled Pipeline"
        description = request.POST.get("description", "").strip()
        template_id = request.POST.get("template_id", "").strip()

        steps, cloned_from = [], None
        if template_id:
            template = Pipeline.objects.filter(pk=template_id, is_template=True).first()
            if template:
                steps = _reid(copy.deepcopy(template.steps or []))
                cloned_from = template
                if title == "Untitled Pipeline":
                    title = f"{template.title} (copy)"
                if not description:
                    description = template.description

        pipeline = Pipeline.objects.create(
            user=request.user,
            title=title,
            description=description,
            steps=steps,
            cloned_from=cloned_from,
        )
        return redirect("pipelines:editor", pk=pipeline.pk)

    return render(request, "pipelines/create.html", {
        "templates": Pipeline.objects.filter(is_template=True),
    })


def _reid(steps):
    """Give cloned steps fresh ids and rewrite any inputs references."""
    remap = {}
    for s in steps:
        old = s.get("id")
        new = uuid.uuid4().hex[:8]
        remap[old] = new
        s["id"] = new
    for s in steps:
        if s.get("inputs"):
            s["inputs"] = {p: remap.get(v, v) for p, v in s["inputs"].items()}
    return steps


@login_required
def pipeline_editor(request, pk):
    pipeline = _own_pipeline(request, pk)
    ctx = _editor_context(request, pipeline, {
        "recent_runs": pipeline.runs.all()[:5],
    })
    return render(request, "pipelines/editor.html", ctx)


@login_required
@require_POST
def pipeline_delete(request, pk):
    pipeline = _own_pipeline(request, pk)
    pipeline.delete()
    return redirect("pipelines:list")


@login_required
@require_POST
def pipeline_rename(request, pk):
    pipeline = _own_pipeline(request, pk)
    pipeline.title = request.POST.get("title", pipeline.title).strip() or pipeline.title
    pipeline.description = request.POST.get("description", pipeline.description).strip()
    pipeline.save(update_fields=["title", "description", "updated_at"])
    if _is_htmx(request):
        return HttpResponse('<span class="text-green-600 text-sm">Saved</span>')
    return redirect("pipelines:editor", pk=pk)


# ── Step editing (HTMX) ───────────────────────────────────────────────────────

@login_required
@require_POST
def add_step(request, pk):
    pipeline = _own_pipeline(request, pk)
    block_type = request.POST.get("block_type", "").strip()
    block = get_block(block_type)
    if not block:
        return HttpResponse("Unknown block type.", status=400)

    default_config = {f["name"]: f.get("default", "") for f in block.get("config_fields", [])}
    steps = pipeline.steps or []
    steps.append({
        "id": uuid.uuid4().hex[:8],
        "block_type": block_type,
        "config": default_config,
    })
    pipeline.steps = steps
    pipeline.save(update_fields=["steps", "updated_at"])
    return _render_steps(request, pipeline)


@login_required
@require_POST
def remove_step(request, pk, step_id):
    pipeline = _own_pipeline(request, pk)
    steps = [s for s in (pipeline.steps or []) if s.get("id") != step_id]
    # Drop dangling input references to the removed step.
    for s in steps:
        if s.get("inputs"):
            s["inputs"] = {p: v for p, v in s["inputs"].items() if v != step_id}
    pipeline.steps = steps
    pipeline.save(update_fields=["steps", "updated_at"])
    return _render_steps(request, pipeline)


@login_required
@require_POST
def move_step(request, pk, step_id, direction):
    pipeline = _own_pipeline(request, pk)
    steps = pipeline.steps or []
    idx = next((i for i, s in enumerate(steps) if s.get("id") == step_id), None)
    if idx is not None:
        if direction == "up" and idx > 0:
            steps[idx - 1], steps[idx] = steps[idx], steps[idx - 1]
        elif direction == "down" and idx < len(steps) - 1:
            steps[idx + 1], steps[idx] = steps[idx], steps[idx + 1]
        pipeline.steps = steps
        pipeline.save(update_fields=["steps", "updated_at"])
    return _render_steps(request, pipeline)


@login_required
@require_POST
def configure_step(request, pk, step_id):
    pipeline = _own_pipeline(request, pk)
    steps = pipeline.steps or []
    step = next((s for s in steps if s.get("id") == step_id), None)
    if step is None:
        return HttpResponse("Step not found.", status=404)

    block = get_block(step.get("block_type", "")) or {}
    config = step.get("config", {})
    for field in block.get("config_fields", []):
        name = field["name"]
        if field["field_type"] == "file":
            continue
        value = request.POST.get(name, config.get(name, ""))
        if field["field_type"] == "number" and value not in ("", None):
            try:
                value = float(value)
                if value == int(value):
                    value = int(value)
            except (ValueError, TypeError):
                pass
        config[name] = value
    step["config"] = config
    pipeline.steps = steps
    pipeline.save(update_fields=["steps", "updated_at"])
    return _render_steps(request, pipeline)


# ── Running ───────────────────────────────────────────────────────────────────

@login_required
@require_POST
def run_pipeline(request, pk):
    pipeline = _own_pipeline(request, pk)
    errors = validate_steps(pipeline.steps or [])
    if errors:
        for e in errors:
            messages.error(request, e)
        return redirect("pipelines:editor", pk=pk)
    if not pipeline.steps:
        messages.error(request, "Add at least one step before running.")
        return redirect("pipelines:editor", pk=pk)

    run = PipelineRun.objects.create(pipeline=pipeline, user=request.user)
    engine.start_run(run)
    return redirect("pipelines:run_detail", pk=pk, run_id=run.pk)


@login_required
def run_detail(request, pk, run_id):
    pipeline = _own_pipeline(request, pk)
    run = get_object_or_404(PipelineRun, pk=run_id, pipeline=pipeline)
    return render(request, "pipelines/run.html", {
        "pipeline": pipeline,
        "run": run,
        "steps": _run_steps(run),
    })


def _run_steps(run):
    """Enrich the run's frozen steps with per-step status + output for display."""
    enriched = []
    for step in run.steps or []:
        block = get_block(step.get("block_type", "")) or {}
        sid = step.get("id")
        enriched.append({
            **step,
            "display_name": block.get("display_name", step.get("block_type", "")),
            "icon": block.get("icon", "🔧"),
            "state": run.step_state(sid),
            "output": (run.context or {}).get(sid, {}),
        })
    return enriched


@login_required
def run_status(request, pk, run_id):
    """HTMX poll: nudge any in-flight GPU jobs, then report status."""
    pipeline = _own_pipeline(request, pk)
    run = get_object_or_404(PipelineRun, pk=run_id, pipeline=pipeline)

    if not run.is_terminal:
        _poll_run_jobs(run)
        run.refresh_from_db()

    if _is_htmx(request):
        if run.is_terminal:
            resp = HttpResponse(status=200)
            resp["HX-Refresh"] = "true"
            return resp
        return render(request, "pipelines/_run_status.html", {
            "pipeline": pipeline, "run": run, "steps": _run_steps(run),
        })

    return JsonResponse({
        "status": run.status,
        "steps": {s["id"]: s["state"] for s in _run_steps(run) if s.get("id")},
    })


def _poll_run_jobs(run):
    """Poll SageMaker for this run's in-flight jobs and advance the run.

    Self-contained so the run page drives its own completion even if nobody is on
    the Processing/analysis page. ``on_job_finished`` is idempotent, so it's safe to
    also fire from the analysis poller.
    """
    from apps.analysis.models import Job
    from apps.analysis.views import _poll_sagemaker_results

    jobs = list(
        Job.objects.filter(
            user=run.user,
            status=Job.Status.PROCESSING,
            config__pipeline_run_id=str(run.pk),
        ).exclude(modal_call_id="")
    )
    if jobs:
        _poll_sagemaker_results(jobs)
    # Advance for any of this run's jobs that reached a terminal state.
    for job in Job.objects.filter(config__pipeline_run_id=str(run.pk)).exclude(
        status=Job.Status.PROCESSING
    ):
        engine.on_job_finished(job)


# ── Templates / cloning ───────────────────────────────────────────────────────

@login_required
@require_POST
def clone_pipeline(request, pk):
    source = get_object_or_404(Pipeline, pk=pk)
    if not (source.is_template or source.user_id == request.user.id):
        return HttpResponse(status=403)
    new = Pipeline.objects.create(
        user=request.user,
        title=f"{source.title} (copy)",
        description=source.description,
        steps=_reid(copy.deepcopy(source.steps or [])),
        cloned_from=source,
    )
    return redirect("pipelines:editor", pk=new.pk)
