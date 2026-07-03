"""
Views for the BeeMonitor pipeline builder (Phase 1 — linear builder).

Function-based + HTMX, following the Workshop builder's proven endpoint contract
(add / remove / move / configure / save / run) but re-themed onto BeeMonitor's
Tailwind base template and backed by the ``analysis.Job`` machinery.
"""

import copy
import json
import re
import uuid

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import Http404, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from apps.videos.models import Video

from . import engine
from .graph import build_initial_steps, graph_to_steps
from .lessons import get_lesson, list_lessons
from .notebook import generate_notebook
from .models import Pipeline, PipelineRun
from .registry import (
    get_block, get_categories, serialize_blocks, validate_steps,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _is_htmx(request):
    return request.headers.get("HX-Request") == "true"


def _own_pipeline(request, pk):
    return get_object_or_404(Pipeline, pk=pk, user=request.user)


def _user_videos(request):
    return Video.objects.filter(user=request.user).order_by("-id")[:200]


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
    videos = [
        {"id": str(v.pk), "title": (getattr(v, "title", "") or f"Video {v.pk}")}
        for v in _user_videos(request)
    ]
    ctx = {
        "pipeline": pipeline,
        "categories": get_categories(),
        "blocks_json": json.dumps(serialize_blocks()),
        "initial_steps_json": json.dumps(build_initial_steps(pipeline)),
        "saved_graph_json": json.dumps(pipeline.graph or {}),
        "videos_json": json.dumps(videos),
        "errors_json": json.dumps(validate_steps(pipeline.steps or [])),
        "recent_runs": pipeline.runs.all()[:5],
    }
    return render(request, "pipelines/editor.html", ctx)


@login_required
def export_notebook(request, pk):
    """Download the pipeline as a Colab notebook (.ipynb).

    ?mode=api  → a live notebook that runs the pipeline via the public API (real
                 endpoints, no SDK — just requests).
    (default)  → the explainable/offline scaffold notebook (Phase 3b).
    """
    pipeline = _own_pipeline(request, pk)
    if request.GET.get("mode") == "api":
        from .notebook import generate_api_notebook
        base = request.build_absolute_uri("/").rstrip("/")
        nb = generate_api_notebook(pipeline, base)
        suffix = "_live"
    else:
        nb = generate_notebook(pipeline)
        suffix = ""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", pipeline.title).strip("_") or "pipeline"
    resp = HttpResponse(json.dumps(nb, indent=1), content_type="application/x-ipynb+json")
    resp["Content-Disposition"] = f'attachment; filename="{safe}{suffix}.ipynb"'
    return resp


@login_required
@require_POST
def save_graph(request, pk):
    """Persist the Drawflow canvas: convert graph → steps, store both, revalidate."""
    pipeline = _own_pipeline(request, pk)
    try:
        graph = json.loads(request.body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return JsonResponse({"ok": False, "error": "Invalid graph JSON."}, status=400)
    steps = graph_to_steps(graph)
    pipeline.graph = graph
    pipeline.steps = steps
    pipeline.save(update_fields=["graph", "steps", "updated_at"])
    return JsonResponse({
        "ok": True,
        "errors": validate_steps(steps),
        "step_count": len(steps),
    })


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
@require_POST
def run_on_videos(request):
    """Run a chosen pipeline once per selected video (from the Processing hub).

    Mirrors analysis ``BatchJobView``'s AJAX contract so the hub's live per-video
    status + poll keep working unchanged: each pipeline GPU step spawns a tagged
    ``analysis.Job`` for that video, which ``analysis:poll`` already surfaces.
    """
    from apps.videos.models import Video

    is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"

    def _fail(msg, status=400, level="warning"):
        if is_ajax:
            return JsonResponse({"error": msg}, status=status)
        getattr(messages, level)(request, msg)
        return redirect("analysis:processing")

    pipeline = Pipeline.objects.filter(pk=request.POST.get("pipeline")).first()
    if not pipeline or not (pipeline.is_template or pipeline.user_id == request.user.id):
        return _fail("Choose a pipeline to run.")
    if not any(s.get("block_type") == "input.video" for s in (pipeline.steps or [])):
        return _fail("That pipeline has no video input to run on.")

    video_ids = request.POST.getlist("video_ids")
    if not video_ids:
        return _fail("No videos selected.")

    videos = Video.manageable(request.user).filter(pk__in=video_ids)
    launched, invalid = [], 0
    for video in videos:
        steps = engine.steps_with_video(pipeline, video.pk)
        if validate_steps(steps):
            invalid += 1
            continue
        run = PipelineRun.objects.create(pipeline=pipeline, user=request.user)
        engine.start_run(run, steps=steps)
        launched.append(video.pk)

    if not launched:
        return _fail("Could not start — the pipeline isn't valid for the selected videos.")

    msg = f"Started “{pipeline.title}” on {len(launched)} video(s)."
    if invalid:
        msg += f" Skipped {invalid}."
    if is_ajax:
        return JsonResponse({"ok": True, "submitted": len(launched), "skipped": invalid,
                             "message": msg, "video_ids": launched})
    messages.success(request, msg)
    return redirect("analysis:processing")


@login_required
def run_list(request):
    """History of the current user's pipeline runs (across pipelines)."""
    runs = list(PipelineRun.objects.filter(user=request.user)
                .select_related("pipeline").order_by("-started_at", "-id")[:100])

    # Resolve each run's input video title(s) in one query.
    vid_ids = set()
    for r in runs:
        for s in (r.steps or []):
            if s.get("block_type") == "input.video":
                vid = (s.get("config") or {}).get("video_id")
                if str(vid).isdigit():
                    vid_ids.add(int(vid))
    titles = {}
    if vid_ids:
        titles = {str(v.pk): (getattr(v, "title", "") or f"Video {v.pk}")
                  for v in Video.objects.filter(pk__in=vid_ids)}

    rows = []
    for r in runs:
        inputs = [titles.get(str((s.get("config") or {}).get("video_id")))
                  for s in (r.steps or []) if s.get("block_type") == "input.video"]
        inputs = [i for i in inputs if i]
        status_vals = (r.step_status or {}).values()
        rows.append({
            "run": r,
            "input": ", ".join(inputs) or "—",
            "done": sum(1 for st in status_vals if st == PipelineRun.STEP_DONE),
            "total": len(r.steps or []),
        })
    return render(request, "pipelines/runs.html", {"rows": rows})


@login_required
@require_POST
def rerun(request, pk, run_id):
    """Re-run: a fresh run of the CURRENT pipeline on the old run's video.

    Unchanged GPU steps reuse their cached output (``StepResult``), so only steps
    that actually changed — or previously failed — recompute.
    """
    old = get_object_or_404(PipelineRun, pk=run_id, user=request.user)
    pipeline = old.pipeline
    video_id = next(((s.get("config") or {}).get("video_id")
                     for s in (old.steps or []) if s.get("block_type") == "input.video"), None)
    steps = engine.steps_with_video(pipeline, video_id) if video_id else (pipeline.steps or [])
    run = PipelineRun.objects.create(pipeline=pipeline, user=request.user)
    engine.start_run(run, steps=steps)
    return redirect("pipelines:run_detail", pk=pipeline.pk, run_id=run.pk)


@login_required
def run_detail(request, pk, run_id):
    run = get_object_or_404(PipelineRun, pk=run_id, user=request.user)
    return render(request, "pipelines/run.html", {
        "pipeline": run.pipeline,
        "run": run,
        "steps": _run_steps(run),
    })


@login_required
def run_output_csv(request, pk, run_id, step_id):
    """Download a step's tabular output (visitation / colony / marker rows) as CSV."""
    import csv
    import io

    run = get_object_or_404(PipelineRun, pk=run_id, user=request.user)
    out = (run.context or {}).get(step_id, {})
    rows = out.get("rows") or []
    buf = io.StringIO()
    if rows:
        writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    resp = HttpResponse(buf.getvalue(), content_type="text/csv")
    resp["Content-Disposition"] = f'attachment; filename="run-{run_id}-{step_id}.csv"'
    return resp


_DISPLAY_SKIP = {
    "rows", "from", "result", "note", "error", "artifact", "pending", "summary",
    "table_kind", "event_kind", "output_kind", "filtered_by",
}


def _step_display(output):
    """A review-friendly view of a step's output: scalar metrics + a rows table."""
    if not isinstance(output, dict):
        return {"fields": [], "rows": [], "columns": [], "row_total": 0}
    result = output.get("result") or {}
    fields = []
    seen = set()
    for src in (output, result):
        for k, v in src.items():
            if k in _DISPLAY_SKIP or k.startswith("_") or k in seen:
                continue
            if isinstance(v, (dict, list)) or v in (None, ""):
                continue
            fields.append((k.replace("_", " "), v))
            seen.add(k)
    rows = output.get("rows") or []
    columns = list(rows[0].keys()) if rows else []
    # Align each row into a list of cells matching `columns` (templates can't index
    # a dict by a variable key).
    cell_rows = [[r.get(c, "") for c in columns] for r in rows[:25]]
    return {
        "fields": fields[:24],
        "rows": cell_rows,
        "row_total": len(rows),
        "columns": columns,
        "annotated_video": result.get("annotated_video_path") or "",
        "summary": output.get("summary") or result.get("summary_stats") or {},
    }


def _run_steps(run):
    """Enrich the run's frozen steps with per-step status + output for display."""
    enriched = []
    for step in run.steps or []:
        block = get_block(step.get("block_type", "")) or {}
        sid = step.get("id")
        out = (run.context or {}).get(sid, {})
        enriched.append({
            **step,
            "display_name": block.get("display_name", step.get("block_type", "")),
            "icon": block.get("icon", "🔧"),
            "state": run.step_state(sid),
            "output": out,
            "cached": bool(out.get("_cached")),
            "display": _step_display(out),
        })
    return enriched


@login_required
def run_status(request, pk, run_id):
    """HTMX poll: nudge any in-flight GPU jobs, then report status."""
    run = get_object_or_404(PipelineRun, pk=run_id, user=request.user)

    if not run.is_terminal:
        _poll_run_jobs(run)
        run.refresh_from_db()

    if _is_htmx(request):
        if run.is_terminal:
            resp = HttpResponse(status=200)
            resp["HX-Refresh"] = "true"
            return resp
        return render(request, "pipelines/_run_status.html", {
            "pipeline": run.pipeline, "run": run, "steps": _run_steps(run),
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

# ── STEM lessons ──────────────────────────────────────────────────────────────

@login_required
def lesson_list(request):
    return render(request, "pipelines/lessons/list.html", {"lessons": list_lessons()})


@login_required
def lesson_detail(request, slug):
    lesson = get_lesson(slug)
    if not lesson:
        raise Http404("No such lesson.")
    template = Pipeline.objects.filter(is_template=True, title=lesson["template"]).first()
    steps = []
    if template:
        for step in template.steps or []:
            block = get_block(step.get("block_type", "")) or {}
            steps.append({
                "display_name": block.get("display_name", step.get("block_type", "")),
                "icon": block.get("icon", "🔧"),
                "description": block.get("description", ""),
            })
    return render(request, "pipelines/lessons/detail.html", {
        "lesson": lesson, "slug": slug, "template": template, "steps": steps,
    })


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
