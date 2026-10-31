"""
Views for the BeeMonitor pipeline builder (Phase 1 — linear builder).

Function-based + HTMX, following the Workshop builder's proven endpoint contract
(add / remove / move / configure / save / run) but re-themed onto BeeMonitor's
Tailwind base template and backed by the ``analysis.Job`` machinery.
"""

import copy
import json
import logging
import re
import uuid
from collections import Counter

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db.models import Q
from django.http import Http404, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.decorators.http import require_POST

from apps.accounts import quota
from apps.videos.models import Video
from apps.videos.timecode import fmt_clock, parse_clock

from . import engine, executors
from .graph import build_initial_steps, graph_to_steps
from .notebook import create_colab_gist, generate_api_notebook
from .models import Pipeline, PipelineRun
from .registry import (
    get_block, get_categories, serialize_blocks, validate_steps,
)

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _is_htmx(request):
    return request.headers.get("HX-Request") == "true"


def _own_pipeline(request, pk):
    return get_object_or_404(Pipeline, pk=pk, user=request.user)


def _usable_pipeline(request, pk):
    """Own, template or published: viewable, runnable and copyable."""
    return get_object_or_404(Pipeline.usable(request.user), pk=pk)


def _user_videos(request):
    return Video.objects.filter(user=request.user).order_by("-id")[:200]


# ── Pipeline CRUD ─────────────────────────────────────────────────────────────

@login_required
def pipeline_list(request):
    pipelines = Pipeline.objects.filter(user=request.user, is_template=False)
    return render(request, "pipelines/list.html", {"pipelines": pipelines})


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
    # Someone else's published pipeline opens read-only: the canvas is locked
    # and there is no save; copying it is how to change it.
    pipeline = get_object_or_404(
        Pipeline.objects.filter(Q(user=request.user) | Q(visibility=Pipeline.Visibility.PUBLIC)),
        pk=pk)
    readonly = pipeline.user_id != request.user.id
    videos = [
        {"id": str(v.pk), "title": (getattr(v, "title", "") or f"Video {v.pk}")}
        for v in _user_videos(request)
    ]
    from apps.training.models import CustomModel
    # Published models included: the weights are already in a bucket the
    # workers read, so publishing one is only meaningful if it can be selected.
    ready_models = CustomModel.usable(request.user)
    # Same type filters as the New Analysis form's model selects.
    bee_models = [
        {"id": str(m.pk), "name": m.name}
        for m in ready_models.filter(model_type__in=["bee_tracking", "custom"])
    ]
    nest_models = [
        {"id": str(m.pk), "name": m.name}
        for m in ready_models.filter(model_type__in=["nest_detection", "custom"])
    ]
    ctx = {
        "pipeline": pipeline,
        "categories": get_categories(),
        "blocks_json": json.dumps(serialize_blocks()),
        "initial_steps_json": json.dumps(build_initial_steps(pipeline)),
        "saved_graph_json": json.dumps(pipeline.graph or {}),
        "videos_json": json.dumps(videos),
        "bee_models_json": json.dumps(bee_models),
        "nest_models_json": json.dumps(nest_models),
        "errors_json": json.dumps(validate_steps(pipeline.steps or [])),
        "recent_runs": [] if readonly else pipeline.runs.all()[:5],
        "readonly": readonly,
    }
    if readonly:
        from . import publishing
        ctx["summary"] = publishing.summary(pipeline)
    elif request.GET.get("publish") == "blocked":
        from . import publishing
        ctx["publish_blockers"] = publishing.blockers(pipeline)
    return render(request, "pipelines/editor.html", ctx)


@login_required
@require_POST
def pipeline_publish(request, pk):
    """Publish or unpublish. Owner only."""
    from . import publishing

    pipeline = _own_pipeline(request, pk)
    if request.POST.get("visibility") == "public":
        if publishing.publish(pipeline):
            return redirect(reverse("pipelines:editor", args=[pk]) + "?publish=blocked")
        messages.success(request, "Published. Anyone signed in can now view, run and copy "
                                  "this pipeline from Browse; only you can change it.")
    else:
        publishing.unpublish(pipeline)
        messages.info(request, "No longer published. Copies people took and their runs are kept.")
    return redirect("pipelines:editor", pk=pk)


@login_required
def open_in_colab(request, pk):
    """Open the pipeline as a live Colab notebook in the browser.

    Pushes the generated (API-driven) notebook to a public GitHub gist and redirects
    to ``colab.research.google.com/gist/<id>`` — so the user does it all in the web.
    Falls back to downloading the ``.ipynb`` when no gist token is configured or the
    gist call fails (or when ``?download=1``).
    """
    from django.conf import settings

    pipeline = _own_pipeline(request, pk)
    base = request.build_absolute_uri("/").rstrip("/")
    nb = generate_api_notebook(pipeline, base)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", pipeline.title).strip("_") or "pipeline"

    token = getattr(settings, "GITHUB_GIST_TOKEN", "")
    if token and request.GET.get("download") != "1":
        try:
            return redirect(create_colab_gist(nb, f"{safe}.ipynb", token, f"BeeMonitor: {pipeline.title}"))
        except Exception:
            logger.exception("Colab gist export failed for pipeline %s; downloading instead", pk)

    resp = HttpResponse(json.dumps(nb, indent=1), content_type="application/x-ipynb+json")
    resp["Content-Disposition"] = f'attachment; filename="{safe}.ipynb"'
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
    # Runs belong to their pipeline (CASCADE): deleting a published pipeline
    # would take other people's runs with it.
    if pipeline.runs.exclude(user=request.user).exists():
        messages.error(request, "Other people have runs on this pipeline, and deleting it would "
                                "delete them too. Unpublish it instead.")
        return redirect("pipelines:list")
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

    # A pipeline is a template — the video is chosen at run time. If it has a
    # video input, send the user to Processing to pick which videos to run on.
    if any(s.get("block_type") == "input.video" for s in pipeline.steps):
        messages.info(request, "Pick the videos to run this pipeline on.")
        return redirect(f"{reverse('analysis:processing')}?pipeline={pipeline.pk}")
    if any(s.get("block_type") == "input.photo" for s in pipeline.steps):
        messages.info(request, "Pick the photos to run this pipeline on.")
        return redirect(f"{reverse('analysis:processing')}?kind=photo&pipeline={pipeline.pk}")

    run = PipelineRun.objects.create(pipeline=pipeline, user=request.user)
    try:
        engine.start_run(run)
    except quota.QuotaExceeded as e:
        messages.error(request, e.message)
        return redirect("pipelines:editor", pk=pk)
    return redirect("pipelines:run_detail", pk=pk, run_id=run.pk)


PART_LENGTHS = {"1": 60, "5": 300, "10": 600, "15": 900, "end": None}


def part_of_clip(data):
    """The run panel's "How much of each clip": None for the whole clip, else
    (start_seconds, length_seconds or None for "to the end")."""
    if data.get("part") != "part":
        return None
    length = data.get("part_minutes", "5")
    if length not in PART_LENGTHS:
        raise ValueError("Choose how long to analyse.")
    start = parse_clock(data.get("part_from") or "0")
    if start == 0 and PART_LENGTHS[length] is None:
        return None  # from the start to the end: the whole clip
    return (start, PART_LENGTHS[length])


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

    # Adding clips to an existing batch (memory/64): its pipeline, its own
    # frozen steps and its part of each clip, whatever the form says.
    add_batch, template, batch_window = None, None, None
    if request.POST.get("add_to"):
        from . import sharing
        try:
            add_batch = uuid.UUID(request.POST["add_to"])
        except ValueError:
            return _fail("No such batch.", status=404)
        if not sharing.owns_batch(request.user, add_batch):
            return _fail("Only the batch's launcher can add clips to it.", status=404)
        pipeline, template, batch_window = engine.batch_template(add_batch)
    else:
        pipeline = Pipeline.objects.filter(pk=request.POST.get("pipeline")).first()
        if not pipeline or not pipeline.usable_by(request.user):
            return _fail("Choose a pipeline to run.")
    work_steps = pipeline.steps if template is None else template
    from .registry import pipeline_input_kind
    photos = request.POST.get("kind") == "photo"
    if photos and pipeline_input_kind(work_steps) != "photo":
        return _fail("That pipeline starts from a video; choose one that starts from a Photo Input.")
    if not photos and not any(s.get("block_type") == "input.video" for s in (work_steps or [])):
        return _fail("That pipeline has no video input to run on.")

    def _source():
        # Photos are rows of the same table, kind=photo (memory/45).
        if photos:
            # The Photos tab's set: periodic and uploaded photos, not the burst
            # photos that belong to a clip.
            return Video.manageable(request.user, photos=True).filter(
                kind=Video.Kind.PHOTO, parent__isnull=True)
        return Video.manageable(request.user)

    # "Run on all filtered" — re-apply the hub's filter server-side so the whole
    # filtered set runs, not just the (capped) list rendered on the page. Capped
    # at MAX_BATCH so an over-broad filter can't launch tens of thousands of jobs.
    # Configurable via the PIPELINE_MAX_BATCH setting; set it to 0 (or None) to
    # disable the cap entirely.
    from django.conf import settings
    MAX_BATCH = getattr(settings, "PIPELINE_MAX_BATCH", 50000)
    over_cap = gone = 0
    if request.POST.get("all_filtered"):
        from django.db.models import Q

        from apps.analysis.views import apply_video_filters
        vqs = apply_video_filters(_source(), request.POST)
        # Clips picked under other filters (the page keeps picks across
        # filter changes) run too.
        picked = [v for v in request.POST.getlist("video_ids") if str(v).isdigit()]
        if picked:
            vqs = _source().filter(Q(pk__in=vqs.values("pk")) | Q(pk__in=picked))
        total = vqs.count()
        vqs = vqs.order_by("-recorded_at", "-uploaded_at")
        videos = list(vqs[:MAX_BATCH] if MAX_BATCH else vqs)
        over_cap = max(0, total - MAX_BATCH) if MAX_BATCH else 0
    else:
        video_ids = request.POST.getlist("video_ids")
        if not video_ids:
            return _fail("No videos selected.")
        vqs = _source().filter(pk__in=video_ids)
        videos = list(vqs[:MAX_BATCH] if MAX_BATCH else vqs)
        # Selected clips whose raw footage was deleted (manageable leaves them out).
        gone = Video.everything.filter(pk__in=video_ids, raw_deleted_at__isnull=False).count()
    if not videos:
        return _fail("The raw footage of these clips was deleted, so they can't be analysed again."
                     if gone else "No videos match the current filter.")
    # Groups this launch for aggregate results (combined CSVs, cross-video trips).
    # fresh: the cache is keyed on clip + config, not on the build, so running
    # the same pipeline again after a fix would hand back the old result.
    # Part of each clip (memory/53): "from m:ss for N minutes" (or to the end).
    already = 0
    if add_batch is not None:
        from .aggregate import run_video_id
        inside = {run_video_id(r) for r in PipelineRun.objects.filter(batch_id=add_batch)}
        already = sum(1 for v in videos if v.pk in inside)
        videos = [v for v in videos if v.pk not in inside]
        if not videos:
            return _fail("Every one of these clips is already in this batch.")
    try:
        window = batch_window if add_batch is not None else part_of_clip(request.POST)
    except ValueError as e:
        return _fail(str(e))
    too_short = 0
    if window:
        kept = [v for v in videos if engine.clip_window(v, window) is not None]
        too_short = len(videos) - len(kept)
        videos = kept
        if not videos:
            return _fail("Every selected clip ends before that start time.")
    # "Run the ones that fit": after a refusal, the hub resends with fit_only
    # and the batch is cut to the clips the credits cover, in list order.
    if request.POST.get("fit_only") == "1":
        videos = videos[:quota.how_many_fit(
            request.user, [quota.pipeline_work(work_steps, [v], window) for v in videos])]
        if not videos:
            return _fail("None of these clips fit your credits.", status=402)
    try:
        batch_id, launched, invalid = engine.launch_batch(
            pipeline, videos, request.user, fresh=request.POST.get("fresh") == "1", window=window,
            batch_id=add_batch, steps=template)
    except quota.QuotaExceeded as e:
        if is_ajax:
            fits = quota.how_many_fit(
                request.user, [quota.pipeline_work(work_steps, [v], window) for v in videos])
            return JsonResponse({"error": e.message, "fits": fits, "total": len(videos)}, status=402)
        return _fail(e.message, status=402)

    # The GPU steps were created QUEUED. Kick the queue once for the whole batch:
    # spawn up to the global SageMaker cap now, leave the rest QUEUED for the
    # reconciler to drain in waves. Best-effort — never break the launch response.
    if launched:
        try:
            from apps.analysis.views import _drain_queue
            _drain_queue()
        except Exception:
            logger.exception("inline drain after pipeline batch launch failed")

    if not launched:
        return _fail("Could not start — the pipeline isn't valid for the selected videos.")

    from django.urls import reverse

    from . import batches
    batch_url = reverse("pipelines:batch_detail", kwargs={"batch_id": batch_id})
    if add_batch is not None:
        msg = (f"Adding {len(launched)} clip{'s' if len(launched) != 1 else ''} to "
               f"“{batches.title(batch_id, pipeline)}”. They show as running below.")
        if already:
            msg += f" {already} were already in it."
    else:
        if (request.POST.get("batch_name") or "").strip():
            batches.rename(batch_id, request.user, request.POST["batch_name"])
        msg = f"Started “{batches.title(batch_id, pipeline)}” on {len(launched)} video(s)."
    if invalid:
        msg += f" Skipped {invalid}."
    if gone:
        msg += f" Left out {gone} whose raw footage was deleted."
    if window:
        start, length = window
        msg += (f" Analysing {fmt_clock(start)}–{fmt_clock(start + length) if length else 'the end'}"
                " of each clip.")
    if too_short:
        msg += f" Left out {too_short} that end before {fmt_clock(window[0])}."
    if over_cap:
        msg += f" ({over_cap} more matched but this run is capped at {MAX_BATCH} — narrow the filter and run again for the rest.)"
    if is_ajax:
        return JsonResponse({"ok": True, "submitted": len(launched), "skipped": invalid,
                             "message": msg, "video_ids": launched,
                             "batch_url": batch_url})
    messages.success(request, msg)
    return redirect(batch_url)


@login_required
def run_list(request):
    """History, by LAUNCH rather than by run.

    A launch is the unit a person thinks in: "the 12 clips I ran on Tuesday",
    not the 12 rows it became. Paginating runs meant page 3 of a 3,250-clip batch
    was still the same batch, and the launch before it was hundreds of pages
    away. Batches are paginated now; a batch's own page lists its runs.

    Runs launched on their own (one-click analysis, a single-clip pipeline run)
    have no batch id, so they are listed separately rather than being lost.
    """
    from django.core.paginator import Paginator
    from django.db.models import Count, F, Max, Q

    # Unstick runs whose jobs already resolved (missed completion notification).
    engine.reconcile_user_runs(request.user)

    mine = PipelineRun.objects.filter(user=request.user)

    batch_qs = (mine.exclude(batch_id=None)
                .values("batch_id")
                .annotate(
                    count=Count("id"),
                    done=Count("id", filter=Q(status=PipelineRun.Status.COMPLETED)),
                    failed=Count("id", filter=Q(status=PipelineRun.Status.FAILED)),
                    # Aliasing Max as `started_at` shadows the field, so a
                    # second aggregate on it cannot resolve. One is enough.
                    started_at=Max("started_at"),
                )
                .order_by("-started_at"))

    status_filter = request.GET.get("status", "")
    if status_filter == "failed":
        batch_qs = batch_qs.filter(failed__gt=0)
    elif status_filter == "running":
        batch_qs = batch_qs.filter(count__gt=F("done") + F("failed"))
    elif status_filter == "completed":
        batch_qs = batch_qs.filter(failed=0).filter(count=F("done"))

    paginator = Paginator(batch_qs, 20)
    page = paginator.get_page(request.GET.get("page"))
    batch_rows = list(page.object_list)

    # Each batch runs one pipeline; resolve them in a single query.
    pipe_by_batch = {}
    for r in (PipelineRun.objects.filter(batch_id__in=[b["batch_id"] for b in batch_rows])
              .values("batch_id", "pipeline_id").distinct()):
        pipe_by_batch.setdefault(r["batch_id"], r["pipeline_id"])
    pipes = {p.pk: p for p in Pipeline.objects.filter(pk__in=set(pipe_by_batch.values()))}
    from . import batches
    named = batches.names([b["batch_id"] for b in batch_rows])
    for b in batch_rows:
        b["pipeline"] = pipes.get(pipe_by_batch.get(b["batch_id"]))
        b["name"] = named.get(b["batch_id"], "")
        b["running"] = b["count"] - b["done"] - b["failed"]
        b["pct_done"] = round(100 * b["done"] / b["count"]) if b["count"] else 0

    # Runs with no batch — one-click analysis and single-clip pipeline runs.
    # Paged on their own (?solo_page=): they used to stop at the latest ten,
    # with no way back to anything older.
    solo_page = Paginator(mine.filter(batch_id=None).select_related("pipeline")
                          .order_by("-started_at", "-id"), 10
                          ).get_page(request.GET.get("solo_page"))
    solo = list(solo_page.object_list)

    # Batches someone invited this user to (memory/62).
    from .models import BatchMember
    shared = []
    for m in (BatchMember.objects.filter(user=request.user).order_by("-created_at")[:50]):
        first = (PipelineRun.objects.filter(batch_id=m.batch_id).select_related("pipeline", "user")
                 .order_by("started_at").first())
        if first is not None:
            shared.append({"batch_id": m.batch_id, "role": m.get_role_display().split(" —")[0],
                           "pipeline": first.pipeline, "owner": first.user,
                           "name": batches.names([m.batch_id]).get(m.batch_id, ""),
                           "count": PipelineRun.objects.filter(batch_id=m.batch_id).count()})

    return render(request, "pipelines/runs.html", {
        "shared": shared,
        "batches": batch_rows,
        "page": page,
        "solo": solo,
        "solo_page": solo_page,
        "status_filter": status_filter,
        "counts": {
            "all": mine.exclude(batch_id=None).values("batch_id").distinct().count(),
            "runs": mine.count(),
        },
    })


@login_required
@require_POST
def rerun(request, pk, run_id):
    """Re-run: a fresh run of the CURRENT pipeline on the old run's video.

    Unchanged GPU steps reuse their cached output (``StepResult``), so only steps
    that actually changed — or previously failed — recompute.
    """
    old = get_object_or_404(PipelineRun, pk=run_id, user=request.user)
    pipeline = old.pipeline
    if not pipeline.usable_by(request.user):
        messages.error(request, "That pipeline is no longer published; copy it to run it again.")
        return redirect("pipelines:run_detail", pk=pipeline.pk, run_id=old.pk)
    video_id = next(((s.get("config") or {}).get("video_id")
                     for s in (old.steps or [])
                     if s.get("block_type") in ("input.video", "input.photo")), None)
    steps = engine.steps_with_video(pipeline, video_id) if video_id else (pipeline.steps or [])
    run = PipelineRun.objects.create(pipeline=pipeline, user=request.user)
    try:
        engine.start_run(run, steps=steps)
    except quota.QuotaExceeded as e:
        messages.error(request, e.message)
        return redirect("pipelines:run_detail", pk=pipeline.pk, run_id=old.pk)
    return redirect("pipelines:run_detail", pk=pipeline.pk, run_id=run.pk)


def _descendant_step_ids(steps, target_id):
    """Step ids that (transitively) depend on target_id, plus target_id itself."""
    # Reverse adjacency: upstream_id -> [dependent step ids].
    children = {}
    for index, step in enumerate(steps):
        sid = step.get("id")
        for up in executors.upstream_ids(step, steps, index):
            children.setdefault(up, []).append(sid)
    seen = {target_id}
    stack = [target_id]
    while stack:
        cur = stack.pop()
        for child in children.get(cur, []):
            if child and child not in seen:
                seen.add(child)
                stack.append(child)
    return seen


@login_required
@require_POST
def retry_step(request, pk, run_id, step_id):
    """Retry a single failed step IN PLACE: reset it + everything downstream to
    pending, clear their outputs, and re-advance. Upstream results are kept, so
    only this step and its descendants recompute (a fresh GPU job for GPU steps).
    """
    from django.contrib import messages

    run = get_object_or_404(PipelineRun, pk=run_id, user=request.user)
    steps = run.steps or []
    if not any(s.get("id") == step_id for s in steps):
        messages.warning(request, "That step isn't part of this run.")
        return redirect("pipelines:run_detail", pk=pk, run_id=run_id)

    reset = _reset_steps(run, _descendant_step_ids(steps, step_id))
    engine.advance_run(run.pk)
    messages.info(request, f"Retrying — reset {len(reset)} step(s).")
    return redirect("pipelines:run_detail", pk=pk, run_id=run_id)


def _reset_steps(run, reset):
    """Set these steps back to pending, drop their outputs, and reopen the run."""
    status = dict(run.step_status or {})
    context = dict(run.context or {})
    for sid in reset:
        status[sid] = PipelineRun.STEP_PENDING
        context.pop(sid, None)
    run.step_status = status
    run.context = context
    run.status = PipelineRun.Status.RUNNING
    run.completed_at = None
    run.error_message = ""
    run.save(update_fields=["step_status", "context", "status", "completed_at", "error_message"])
    return reset


def _failed_step_ids(run):
    """The steps that failed on their own (not just because one above them did),
    plus everything downstream of them."""
    steps = run.steps or []
    context = run.context or {}
    reset = set()
    for step in steps:
        sid = step.get("id")
        if run.step_state(sid) != PipelineRun.STEP_FAILED:
            continue
        if "Upstream step failed" in str((context.get(sid) or {}).get("error") or ""):
            continue
        reset |= _descendant_step_ids(steps, sid)
    return reset


@login_required
@require_POST
def batch_retry_failed(request, batch_id):
    """Re-run a batch's failed clips IN PLACE, keeping them in this batch.

    Starting a new batch over every clip to recover one that the GPU dropped
    cost a whole batch of GPU time and split the results across two batches.
    This resets only the failed steps of only the failed runs (optionally just
    one cause's clips), so the batch fills in and its downloads stay whole.
    """
    from . import aggregate

    wanted = {int(v) for v in request.POST.getlist("video_ids") if str(v).isdigit()}
    runs = PipelineRun.objects.filter(batch_id=batch_id, user=request.user,
                                      status=PipelineRun.Status.FAILED)
    retried = 0
    for run in runs:
        if wanted and aggregate.run_video_id(run) not in wanted:
            continue
        reset = _failed_step_ids(run)
        if not reset:
            continue
        _reset_steps(run, reset)
        engine.advance_run(run.pk)
        retried += 1
    if retried:
        messages.success(request, f"Re-running {retried} failed clip{'s' if retried != 1 else ''} "
                                  "in this batch. The rest keep their results.")
    else:
        messages.info(request, "Nothing to re-run — no failed clips matched.")
    return redirect("pipelines:batch_detail", batch_id=batch_id)


def analyzer_options(run):
    """Analyzers this run could be re-analysed with, and the one it used.

    Every analyzer accepts tracks or detections, so a swap is nearly always
    valid — what matters is that the run HAS a cached GPU result to feed them.
    Hidden blocks are left out as swap *targets*, but the run's own analyzer is
    always listed even when it has been retired: a run built on Visitation
    Count still has to be able to say that is what it did, or the page offers
    three alternatives to nothing in particular.
    """
    from .registry import BLOCK_REGISTRY

    current = next((s.get("block_type") for s in (run.steps or [])
                    if str(s.get("block_type", "")).startswith("analyze.")), "")
    options = []
    for key, block in BLOCK_REGISTRY.items():
        if block.get("category") != "analyze":
            continue
        if block.get("hidden") and key != current:
            continue
        options.append({
            "key": key,
            "name": block.get("display_name", key),
            "description": block.get("description", ""),
            "icon": block.get("icon", ""),
            "current": key == current,
            # A retired analyzer can be kept but not chosen afresh.
            "retired": bool(block.get("hidden")),
        })
    return current, options


@login_required
@require_POST
def run_reanalyze(request, pk, run_id):
    """Re-run this clip with a different analyzer, reusing the tracking.

    Detection and tracking are the expensive part and they do not depend on
    which analyzer reads them. ``engine._gpu_cache_key`` hashes the clip and the
    GPU step's own config, so swapping a LOCAL analyzer leaves that key
    untouched and the cached result is served — the new run finishes in seconds
    and costs no GPU time.

    That is the same cache that makes a benchmark re-run dishonest unless it
    passes ``fresh`` (a GPU-side fix would return the old result). Here reuse is
    exactly what is wanted, so this deliberately does NOT pass it.

    A new run rather than a mutation: the original keeps its answer, so two
    analyses of one clip can be compared instead of one overwriting the other.
    """
    from .registry import BLOCK_REGISTRY

    old = get_object_or_404(PipelineRun, pk=run_id, user=request.user)
    new_type = request.POST.get("block_type", "")
    block = BLOCK_REGISTRY.get(new_type)
    if not block or block.get("category") != "analyze":
        messages.error(request, "That is not an analyzer.")
        return redirect("pipelines:run_detail", pk=pk, run_id=run_id)

    steps = copy.deepcopy(old.steps or [])
    swapped = False
    for step in steps:
        if str(step.get("block_type", "")).startswith("analyze."):
            step["block_type"] = new_type
            # The old analyzer's config does not transfer — a visitation gap
            # threshold means nothing to foraging trips. Seed the new block's
            # defaults so required fields (detection_count's `metric`) are set.
            step["config"] = {f["name"]: f.get("default")
                              for f in block.get("config_fields", [])
                              if f.get("default") is not None}
            swapped = True
    if not swapped:
        messages.error(request, "This run has no analyzer step to swap.")
        return redirect("pipelines:run_detail", pk=pk, run_id=run_id)

    errors = validate_steps(steps)
    if errors:
        messages.error(request, errors[0])
        return redirect("pipelines:run_detail", pk=pk, run_id=run_id)

    run = PipelineRun.objects.create(pipeline=old.pipeline, user=request.user)
    try:
        # Usually free (the clip's tracking is cached), but checked anyway:
        # with no cached result it would run the GPU.
        engine.start_run(run, steps=steps)      # NOT fresh: reuse is the point
    except quota.QuotaExceeded as e:
        messages.error(request, e.message)
        return redirect("pipelines:run_detail", pk=pk, run_id=run_id)
    messages.success(
        request,
        f"Re-analysing with {block.get('display_name', new_type)} — reuses this "
        "clip's tracking, so it costs no GPU time.")
    return redirect("pipelines:run_detail", pk=old.pipeline_id, run_id=run.pk)


def _viewable_run_or_404(request, run_id):
    """The launcher sees their run; device-share users (viewer/manager) see
    runs on videos of devices shared with them. Write paths (rerun, retry)
    stay launcher-only."""
    run = get_object_or_404(PipelineRun.objects.select_related("pipeline"), pk=run_id)
    if run.user_id == request.user.id:
        return run
    from apps.videos.models import Video
    from . import aggregate
    vid = aggregate.run_video_id(run)
    if vid is not None and Video.accessible(request.user, photos=True).filter(pk=vid).exists():
        return run
    raise Http404("No such run.")


@login_required
def run_detail(request, pk, run_id):
    run = _viewable_run_or_404(request, run_id)
    current_analyzer, analyzers = analyzer_options(run)
    return render(request, "pipelines/run.html", {
        "pipeline": run.pipeline,
        "run": run,
        "steps": _run_steps(run),
        "analyzers": analyzers,
        "current_analyzer": current_analyzer,
        # Re-analysing needs a cached GPU result to feed the new analyzer, and
        # only the owner may spend anything on this run's behalf.
        "can_reanalyze": run.user_id == request.user.id and run.status == "completed",
        "photo_view": photo_view(run),
        "part": _run_part_label(run),
    })


def _run_part_label(run):
    """'Analysed 0:00–5:00 of 34:32' when the run covered part of its clip."""
    from apps.videos.timecode import part_label

    cfg = next(((s.get("config") or {}) for s in run.steps or []
                if s.get("block_type") == "input.video"), {})
    if not cfg.get("video_id"):
        return ""
    return part_label(cfg, Video.everything.filter(pk=cfg["video_id"]).only("duration_seconds").first())


def photo_view(run):
    """What the run page shows for a photo run (memory/45): the preview with a
    box per insect (as % of the photo, so they scale with the image), and each
    insect's crop, class and species. None for clip runs."""
    from . import aggregate

    photo = aggregate.run_photo(run)
    if not photo:
        return None
    return photo_view_of(photo)


@login_required
def run_photo(request, pk, run_id):
    """A photo run's photo, boxes and species as a fragment: the batch page's
    viewer fetches it when a photo row is opened, so the page presigns nothing
    up front."""
    view = photo_view(_viewable_run_or_404(request, run_id))
    if view is None:
        raise Http404("This run has no photo.")
    return render(request, "pipelines/_photo_view.html", {"photo_view": view})


def photo_view_of(photo):
    """photo_view for one photo result (a job's summary_stats["photo"]). Each
    insect's species as the model called it, with its confidence (memory/60)."""
    from config.storage import presigned_get

    w, h = float(photo.get("width") or 1), float(photo.get("height") or 1)
    insects = []
    for d in photo.get("detections") or []:
        insects.append({
            "id": d.get("id"), "class": d.get("class"), "confidence": d.get("confidence"),
            "species": d.get("species") or "",
            "species_confidence": d.get("species_confidence"),
            "left": round(100 * d["x"] / w, 2), "top": round(100 * d["y"] / h, 2),
            "width": round(100 * d["w"] / w, 2), "height": round(100 * d["h"] / h, 2),
            "crop_url": presigned_get(d["crop_key"], container="processed") if d.get("crop_key") else None,
        })
    return {"preview_url": presigned_get(photo.get("preview_key"), container="processed")
            if photo.get("preview_key") else None,
            "width": int(w), "height": int(h), "tiles": photo.get("tiles"), "insects": insects,
            "named": len({i["species"] for i in insects if i["species"]})}


@login_required
def run_output_csv(request, pk, run_id, step_id):
    """Download a step's tabular output (visitation / colony / marker rows) as CSV."""
    import csv
    import io

    run = _viewable_run_or_404(request, run_id)
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


def _shows_nest_layout(block_type, config):
    """Whether a step's card should render the nest/hotel layout table.

    Every GPU job's summary carries ``nest_bboxes`` (the layout it tracked
    against), so without this scope the 60-row nest table renders on every
    detect/track step. Show it only where the reference *is* the point: the legacy
    nest node, or a Detector whose reference is the hotel/nest layout.
    """
    if block_type == "detect.nest":
        return True
    if block_type == "detect.objects":
        source = (config or {}).get("reference_source", "device_layout")
        return source in ("detect", "device_layout")
    return False


def _step_display(output, block_type="", config=None):
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

    # Nest/hotel layout — only where the reference object is the point of the step.
    stats = output.get("summary") or result.get("summary_stats") or {}
    nest_rows = []
    nest_preview_url = ""
    if _shows_nest_layout(block_type, config):
        nest_rows = [
            {"nest": k, "bbox": ", ".join(str(int(x)) for x in v)}
            for k, v in sorted((stats.get("nest_bboxes") or {}).items())
        ]
        if stats.get("nest_preview_path"):
            try:
                from config.storage import get_s3_client
                nest_preview_url = get_s3_client().generate_presigned_url(
                    "processed", stats["nest_preview_path"])
            except Exception:
                logger.warning("could not presign nest preview %s",
                               stats.get("nest_preview_path"))

    return {
        "fields": fields[:24],
        "rows": cell_rows,
        "row_total": len(rows),
        "columns": columns,
        "annotated_video": result.get("annotated_video_path") or "",
        "summary": stats,
        "nests": nest_rows,
        "nest_preview": nest_preview_url,
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
            "display": _step_display(
                out, step.get("block_type", ""), step.get("config") or {},
            ),
        })
    return enriched


@login_required
def run_status(request, pk, run_id):
    """HTMX poll: nudge any in-flight GPU jobs, then report status."""
    run = _viewable_run_or_404(request, run_id)

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

@login_required
@require_POST
def clone_pipeline(request, pk):
    source = _usable_pipeline(request, pk)
    new = Pipeline.objects.create(
        user=request.user,
        title=f"{source.title} (copy)",
        description=source.description,
        steps=_reid(copy.deepcopy(source.steps or [])),
        cloned_from=source,
    )
    return redirect("pipelines:editor", pk=new.pk)


# ── Batch (multi-video) aggregate results ────────────────────────────────────

def _batch_runs(request, batch_id):
    """The batch's runs, visible to the launcher — or to device-share users
    for the runs whose videos live on devices shared with them."""
    runs = list(PipelineRun.objects.filter(batch_id=batch_id)
                .select_related("pipeline").order_by("started_at", "id"))
    if not runs:
        raise Http404("No such batch.")
    if all(r.user_id == request.user.id for r in runs):
        return runs
    # Invited to it (memory/62): the whole batch, read-only.
    from .models import BatchMember
    if BatchMember.objects.filter(batch_id=batch_id, user=request.user).exists():
        return runs
    from apps.videos.models import Video
    from . import aggregate
    accessible = set(
        Video.accessible(request.user, photos=True)
        .filter(pk__in=[v for r in runs if (v := aggregate.run_video_id(r)) is not None])
        .values_list("pk", flat=True))
    runs = [r for r in runs
            if r.user_id == request.user.id or aggregate.run_video_id(r) in accessible]
    if not runs:
        raise Http404("No such batch.")
    return runs


# How many clips may be probed for their measured properties per page render.
PROBES_PER_RENDER = 8


@login_required
def batch_detail(request, batch_id):
    """Aggregate results for runs launched together: per-run status and
    combined CSV downloads."""
    from . import aggregate

    runs = _batch_runs(request, batch_id)
    sources, skipped = aggregate.collect_sources(runs)
    all_done = all(r.is_terminal for r in runs)

    # Per-run rows + failures grouped by cause. The page used to show a status
    # pill per run and nothing else, so nine failures of two kinds could only be
    # triaged by opening nine pages.
    from . import failures as failure_taxonomy
    rows = aggregate.batch_rows(runs)
    outcome = aggregate.batch_summary(rows)
    # Which units these clips came from. A batch can span devices, so this is a
    # list, not the first row's device wearing the whole batch's name.
    batch_devices = aggregate.batch_devices(rows)
    failure_groups = failure_taxonomy.group([
        (r["video"].pk if r["video"] else None, r["error"], r["run"].started_at)
        for r in rows if r["status"] == "failed"
    ])

    # The base tables this batch can hand back, combined across its clips. Only
    # the ones something actually wrote: a button that downloads nothing looks
    # like a bug rather than a missing pipeline step.
    _backfill_interactions_paths(sources)
    downloads = aggregate.available_downloads(sources, runs)

    # What the downloads leave out. collect_sources has always returned this and
    # the page has always thrown it away, so a clip that could not join an export
    # vanished from it with nothing on screen to say so — and a download that is
    # quietly missing clips is worse than one that says which.
    #
    # Counted by reason, not listed: a big batch printed every clip id here, a
    # wall of hundreds of hex strings above the clip list.
    skipped_sources = skipped

    # Whether a reference reached the analyzer decided every count on this page
    # and was visible nowhere.
    references = aggregate.reference_summary(runs, rows)

    # Running these same clips through a different pipeline. Offered instead of
    # an analyzer swap: a different pipeline may detect a different class,
    # prompt differently or use a different reference, and those are the
    # choices actually worth revisiting on a finished batch.
    rerun_videos = sorted({r["video"].pk for r in rows if r["video"]})

    # A clip's length comes from its own file, so "unknown" is a gap we can
    # close rather than a fact about the clip. Anything still missing is probed
    # in the background and fills in on a later load.
    #
    # Capped per render: a 3,000-clip batch would otherwise spawn 3,000 threads
    # that almost all immediately give up on the semaphore. A few per load
    # drains a backlog over a handful of visits without ever being a spike, and
    # `backfill_video_props` remains the way to do the whole library at once.
    from apps.videos.thumbnails import probe_on_demand
    probes = 0
    for row in rows:
        if probes >= PROBES_PER_RENDER:
            break
        # A photo has no length to measure.
        if (row["video"] and not row["video"].duration_seconds
                and row["video"].kind != row["video"].Kind.PHOTO):
            probe_on_demand(row["video"])
            probes += 1
    # A photo batch (memory/45) has no clips, tracks or tables: each row is a
    # photo, its insects and species, and its one download is the photos CSV.
    from .registry import pipeline_input_kind
    kind = pipeline_input_kind(runs[0].steps) if runs else "video"
    if kind == "photo":
        for row in rows:
            row["photo"] = aggregate.photo_counts(row["run"])
        skipped_sources = []
        downloads = [{"kind": "photos", "label": "Photos", "analyzed": False,
                      "hint": "One row per insect: photo, time taken, box, class, "
                              "species and its confidence."}]

    # Re-runs go through a pipeline that starts from the same kind of input,
    # this batch's own first — re-running it is the commonest choice.
    rerun_pipelines = [p for p in (Pipeline.objects
                                   .filter(Q(user=request.user) | Q(is_template=True))
                                   .exclude(pk=runs[0].pipeline_id if runs else None)
                                   .order_by("-is_template", "title"))
                       if pipeline_input_kind(p.steps) == kind]
    if runs and runs[0].pipeline.usable_by(request.user):
        rerun_pipelines.insert(0, runs[0].pipeline)

    # No cross-video aggregation here any more. The page used to pair trips and
    # build activity charts on every load — up to 300 clips' events CSVs read
    # from S3 — for panels that are now gone. Trips are a read over the events
    # table, and the table is a download.

    # The public link (memory/47): only the launcher can turn it on or off.
    from . import sharing
    from .models import BatchMember
    can_share = sharing.owns_batch(request.user, batch_id)
    share = sharing.live_share(batch_id) if can_share else None

    from . import review as review_mod
    photos_kind = kind == "photo"
    from . import batches
    started = [r["run"].started_at for r in rows if r["run"].started_at]
    return render(request, "pipelines/batch.html", {
        # Its own name, else its pipeline's (memory/64).
        "batch_name": batches.title(batch_id, runs[0].pipeline if runs else None),
        "named": bool(batches.names([batch_id])),
        "first_started": min(started) if started else None,
        "last_started": max(started) if started else None,
        "can_share": can_share,
        # People invited to review or view it (memory/62).
        "members": (list(BatchMember.objects.filter(batch_id=batch_id).select_related("user"))
                    if can_share else []),
        "my_role": (BatchMember.objects.filter(batch_id=batch_id, user=request.user)
                    .values_list("role", flat=True).first() if not can_share else None),
        "share": share,
        "share_url": (request.build_absolute_uri(reverse("public_batch", args=[share.token]))
                      if share else ""),
        "photos": kind == "photo",
        "rows": rows,
        # Events and trips review metrics (memory/62); None until something
        # in the batch has been reviewed.
        "review_summary": (review_mod.batch_review_summary(
            [r["run"] for r in rows], {r["run"].pk: r.get("review") for r in rows})
            if not photos_kind else None),
        # Apply the pipeline's insect groups here, without a re-run (memory/65).
        "pipeline_groups": (_pipeline_groups(runs[0].pipeline)
                            if can_share and runs and not photos_kind else None),
        # What its insect-group filters left out (memory/65).
        "group_summary": (review_mod.batch_group_summary(
            [r["run"] for r in rows], {r["run"].pk: r.get("review") for r in rows})
            if not photos_kind else None),
        "outcome": outcome,
        "failure_groups": failure_groups,
        "downloads": downloads,
        "references": references,
        "rerun_videos": rerun_videos,
        "rerun_pipelines": rerun_pipelines,
        "can_rerun": any(r.user_id == request.user.id for r in runs),
        "batch_id": batch_id,
        "pipeline": runs[0].pipeline,
        "running": sum(1 for r in runs if not r.is_terminal),
        "sources": sources,
        "skipped_sources": skipped_sources,
        "skipped_reasons": Counter(s["reason"] for s in skipped_sources).most_common(),
        "batch_devices": batch_devices,
        # Period review (memory/62): the batch's clips on one time axis, when
        # they are from one device or one site.
        "period_review_url": _period_review_url(rows, batch_id),
        "batch_days": aggregate.batch_days(rows),
    })


def _period_review_url(rows, batch_id):
    """The period review of a batch, or "" when its clips span several devices
    and several sites (a period is one device or one site)."""
    videos = [r["video"] for r in rows if r.get("video")]
    if not videos:
        return ""
    if len({v.device_id for v in videos}) == 1 and videos[0].device_id:
        pass
    elif len({v.site_id for v in videos}) == 1 and videos[0].site_id:
        pass
    else:
        return ""
    return reverse("analysis:period_review") + f"?batch={batch_id}"


@login_required
@require_POST
def batch_cancel(request, batch_id):
    """Stop a batch's unfinished clips (memory/49).

    Finished clips keep their results. Each unfinished clip's GPU job is
    cancelled — which now also tells the GPU, so a queued clip is skipped and a
    running one stops within about 20 seconds — and its run is closed as
    cancelled, so the in-place re-run can resume it later.
    """
    from apps.analysis.cancelling import ACTIVE, CANCELLED_MESSAGE, cancel_jobs
    from apps.analysis.models import Job

    from . import sharing

    if not sharing.owns_batch(request.user, batch_id):
        raise Http404("No such batch.")
    runs = list(PipelineRun.objects.filter(
        batch_id=batch_id, user=request.user,
        status__in=[PipelineRun.Status.PENDING, PipelineRun.Status.RUNNING]))
    job_ids = [out.get("job_id") for run in runs for out in (run.context or {}).values()
               if isinstance(out, dict) and out.get("job_id")]
    cancel_jobs(Job.objects.filter(pk__in=job_ids, status__in=ACTIVE))
    for run in runs:
        engine.cancel_run(run.pk, CANCELLED_MESSAGE)
    if runs:
        messages.success(request, f"Cancelled {len(runs)} clip{'s' if len(runs) != 1 else ''}. "
                                  "Running clips stop within about 20 seconds; waiting "
                                  "clips are skipped without using the GPU.")
    else:
        messages.info(request, "Nothing to cancel — every clip in this batch has finished.")
    return redirect("pipelines:batch_detail", batch_id=batch_id)


@login_required
@require_POST
def batch_rename(request, batch_id):
    """Name a batch, or (blank) go back to its pipeline's name (memory/64).
    Only the batch's launcher."""
    from . import batches, sharing

    if not sharing.owns_batch(request.user, batch_id):
        raise Http404("No such batch.")
    name = batches.rename(batch_id, request.user, request.POST.get("name", ""))
    messages.success(request, f"Renamed to “{name}”." if name
                     else "Name cleared: the batch shows its pipeline's name.")
    return redirect("pipelines:batch_detail", batch_id=batch_id)


def _pipeline_groups(pipeline):
    """What the pipeline's Identify Species node is set to now, for the batch
    page's Apply button; None when it has no such node."""
    from . import regroup, taxon_groups

    step = regroup.pipeline_group_step(pipeline)
    if step is None:
        return None
    cfg = taxon_groups.settings(step.get("config"))
    return {"on": cfg["on"], "leave_out": cfg["leave_out"],
            "in_line": regroup._feeds_analyzers(pipeline, step.get("id")),
            "rules": len(cfg["rules"])}


@login_required
@require_POST
def batch_apply_groups(request, batch_id):
    """Apply the pipeline's current Identify Species groups to this batch and
    recount it in place (memory/65): no re-run, no new batch, no GPU. Only
    the batch's launcher."""
    from . import regroup, sharing

    if not sharing.owns_batch(request.user, batch_id):
        raise Http404("No such batch.")
    done = regroup.apply_pipeline(batch_id, request.user)
    if not done["applied"]:
        messages.error(request, "No clip in this batch was tracked with Identify Species, so "
                                "there are no crop readings to group. Re-run it with the node.")
    else:
        msg = f"Applied the pipeline's insect groups to {done['applied']} clips"
        msg += (f": {done['recounted']} recounted, the other {done['queued']} in the next few "
                f"minutes." if done["queued"] else ", and recounted them.")
        if done["skipped"]:
            msg += (f" {done['skipped']} were tracked without Identify Species and keep "
                    f"every track.")
        messages.success(request, msg + " No GPU-hours.")
    return redirect("pipelines:batch_detail", batch_id=batch_id)


@login_required
@require_POST
def batch_members(request, batch_id):
    """Invite someone to review or view a batch, change their role, or remove
    them (memory/62). Only the batch's launcher."""
    from django.contrib.auth import get_user_model
    from django.db.models import Q

    from . import sharing
    from .models import BatchMember

    if not sharing.owns_batch(request.user, batch_id):
        raise Http404("No such batch.")
    back = redirect(reverse("pipelines:batch_detail", args=[batch_id]) + "#people")
    action = request.POST.get("action", "invite")
    if action == "remove":
        BatchMember.objects.filter(batch_id=batch_id, pk=request.POST.get("member")).delete()
        messages.info(request, "Removed from this batch.")
        return back
    role = request.POST.get("role", BatchMember.Role.REVIEWER)
    if role not in BatchMember.Role.values:
        messages.error(request, "Unknown role.")
        return back
    who = (request.POST.get("who") or "").strip()
    user = (get_user_model().objects.filter(Q(username__iexact=who) | Q(email__iexact=who))
            .first() if who else None)
    if user is None:
        messages.error(request, f"No BeeMonitor account matches “{who}”. They need to sign up first.")
        return back
    if user.pk == request.user.pk:
        messages.info(request, "That's you: you already own this batch.")
        return back
    _member, created = BatchMember.objects.update_or_create(
        batch_id=batch_id, user=user, defaults={"role": role, "created_by": request.user})
    messages.success(request, f"{user.username} {'can now' if created else 'now can'} "
                              f"{'review' if role == BatchMember.Role.REVIEWER else 'view'} this batch.")
    return back


@login_required
@require_POST
def batch_share(request, batch_id):
    """Turn a batch's public link on (or update what it shows), or off."""
    from . import sharing

    if not sharing.owns_batch(request.user, batch_id):
        raise Http404("No such batch.")
    if request.POST.get("action") == "off":
        sharing.turn_off(batch_id)
        messages.info(request, "Public link turned off. Anyone who opens it now sees that it isn't shared.")
    else:
        was_on = sharing.live_share(batch_id) is not None
        sharing.turn_on(request.user, batch_id,
                        show_videos=request.POST.get("show_videos") == "on",
                        show_locations=request.POST.get("show_locations") == "on")
        messages.success(request, "Sharing settings saved." if was_on else
                         "Public link is on. Anyone with the link can view these results.")
    return redirect(reverse("pipelines:batch_detail", args=[batch_id]) + "#share")


def _csv_response(filename, fieldnames, rows):
    import csv as _csv

    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = f'attachment; filename="{filename}"'
    writer = _csv.DictWriter(response, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return response


def _backfill_interactions_paths(sources):
    """Runs completed before interactions_csv_path joined the engine's result
    summary have contexts without it — enrich from the linked JobResult (each
    step's context entry stores its job_id)."""
    from apps.analysis.models import JobResult

    need = [s for s in sources if not s["result"].get("interactions_csv_path")]
    if not need:
        return
    job_by_run = {}
    for s in need:
        run = s.get("run")
        for out in (getattr(run, "context", None) or {}).values():
            r = (out or {}).get("result") or {}
            if out.get("job_id") and (r.get("events_csv_path") or r.get("tracking_csv_path")):
                job_by_run[run.pk] = out["job_id"]
                break
    if not job_by_run:
        return
    paths = dict(JobResult.objects.filter(job_id__in=job_by_run.values())
                 .exclude(interactions_csv_path="")
                 .values_list("job_id", "interactions_csv_path"))
    for s in need:
        jid = job_by_run.get(getattr(s.get("run"), "pk", None))
        if jid in paths:
            s["result"]["interactions_csv_path"] = paths[jid]


@login_required
def batch_combined_csv(request, batch_id, kind):
    from . import sharing

    runs = _batch_runs(request, batch_id)
    try:
        filename, fieldnames, rows = sharing.batch_csv(runs, batch_id, kind)
    except KeyError:
        raise Http404("Unknown CSV kind.")
    except sharing.EmptyExport as e:
        # Friendlier than a bare 404: back to the batch page with the reason.
        messages.warning(request, str(e))
        return redirect("pipelines:batch_detail", batch_id=batch_id)
    return _csv_response(filename, fieldnames, rows)
