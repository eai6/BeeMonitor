"""
Public pipeline API (P1) — build + run BeeMonitor pipelines from Colab / any client.

Thin DRF layer over ``apps.pipelines``. Auth is the project-wide default
(``APIKeyAuthentication`` — ``Authorization: Bearer bmk_…`` — plus session), so a
Colab client just needs an API key from the Developer page. See
``memory/24_pipeline_api_design.md``.

Endpoints (all under /api/v1/):
  GET  pipelines/blocks/                     block registry (typed ports + config)
  GET  pipelines/                            list my pipelines + templates
  POST pipelines/                            create {title, steps}
  GET/PUT/DELETE pipelines/{id}/             retrieve / update / delete
  POST pipelines/validate/                   validate a steps[] graph (no save)
  POST pipelines/{id}/clone/                 clone (e.g. a template) into mine
  POST pipelines/{id}/run/                   run on {video_ids:[...]}
  GET  pipeline-runs/                        list my runs
  GET  pipeline-runs/{id}/                   status + per-step status + outputs
  GET  pipeline-runs/{id}/steps/{sid}/output/  a step's output (?format=csv|json)
"""

import copy
import csv
import io
import uuid

from django.http import HttpResponse
from django.shortcuts import get_object_or_404
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.pipelines import engine
from apps.pipelines.models import Pipeline, PipelineRun
from apps.pipelines.registry import get_categories, serialize_blocks, validate_steps


# ── Serialization helpers ─────────────────────────────────────────────────────

def _pipeline_json(p):
    return {
        "id": str(p.id),
        "title": p.title,
        "description": p.description,
        "steps": p.steps,
        "is_template": p.is_template,
        "updated_at": p.updated_at.isoformat() if p.updated_at else None,
    }


def _sanitize(output):
    """Drop engine-internal (_-prefixed) keys from a step's output."""
    if not isinstance(output, dict):
        return output
    return {k: v for k, v in output.items() if not k.startswith("_")}


def _run_json(run, include_outputs=False):
    d = {
        "id": str(run.id),
        "pipeline_id": str(run.pipeline_id),
        "pipeline_title": run.pipeline.title,
        "status": run.status,
        "step_status": run.step_status or {},
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
    }
    if include_outputs:
        d["steps"] = run.steps or []
        d["outputs"] = {sid: _sanitize(o) for sid, o in (run.context or {}).items()}
    return d


def _reid(steps):
    """Fresh ids for cloned steps; rewrite inputs references."""
    remap = {}
    for s in steps:
        old = s.get("id")
        new = uuid.uuid4().hex[:8]
        remap[old] = new
        s["id"] = new
    for s in steps:
        if s.get("inputs"):
            s["inputs"] = {port: remap.get(v, v) for port, v in s["inputs"].items()}
    return steps


def _owned(request, pk):
    return get_object_or_404(Pipeline, pk=pk, user=request.user)


def _owned_or_template(request, pk):
    p = get_object_or_404(Pipeline, pk=pk)
    if p.user_id != request.user.id and not p.is_template:
        return None
    return p


# ── Introspection ─────────────────────────────────────────────────────────────

class BlocksView(APIView):
    """The block registry — types, categories, typed ports, config fields."""

    def get(self, request):
        return Response({"categories": get_categories(), "blocks": serialize_blocks()})


# ── Pipelines ─────────────────────────────────────────────────────────────────

class PipelineListCreateView(APIView):
    def get(self, request):
        mine = Pipeline.objects.filter(user=request.user, is_template=False)
        templates = Pipeline.objects.filter(is_template=True)
        return Response({
            "pipelines": [_pipeline_json(p) for p in mine],
            "templates": [_pipeline_json(p) for p in templates],
        })

    def post(self, request):
        steps = request.data.get("steps") or []
        if not isinstance(steps, list):
            return Response({"detail": "steps must be a list."}, status=400)
        p = Pipeline.objects.create(
            user=request.user,
            title=(request.data.get("title") or "").strip() or "Untitled Pipeline",
            description=(request.data.get("description") or "").strip(),
            steps=steps,
        )
        return Response({**_pipeline_json(p), "warnings": validate_steps(steps)}, status=201)


class PipelineDetailView(APIView):
    def get(self, request, pk):
        p = _owned_or_template(request, pk)
        if p is None:
            return Response({"detail": "Not found."}, status=404)
        return Response(_pipeline_json(p))

    def put(self, request, pk):
        p = _owned(request, pk)
        if "title" in request.data:
            p.title = (request.data.get("title") or "").strip() or p.title
        if "description" in request.data:
            p.description = (request.data.get("description") or "").strip()
        if "steps" in request.data:
            steps = request.data.get("steps")
            if not isinstance(steps, list):
                return Response({"detail": "steps must be a list."}, status=400)
            p.steps = steps
        p.save()
        return Response({**_pipeline_json(p), "warnings": validate_steps(p.steps or [])})

    def delete(self, request, pk):
        _owned(request, pk).delete()
        return Response(status=204)


class PipelineValidateView(APIView):
    def post(self, request):
        steps = request.data.get("steps") or []
        if not isinstance(steps, list):
            return Response({"detail": "steps must be a list."}, status=400)
        errors = validate_steps(steps)
        return Response({"valid": not errors, "errors": errors})


class PipelineCloneView(APIView):
    def post(self, request, pk):
        src = get_object_or_404(Pipeline, pk=pk)
        if not (src.is_template or src.user_id == request.user.id):
            return Response({"detail": "Not allowed."}, status=403)
        new = Pipeline.objects.create(
            user=request.user,
            title=f"{src.title} (copy)",
            description=src.description,
            steps=_reid(copy.deepcopy(src.steps or [])),
            cloned_from=src,
        )
        return Response(_pipeline_json(new), status=201)


class PipelineRunView(APIView):
    """Run a pipeline once per video (real SageMaker endpoints)."""

    def post(self, request, pk):
        from apps.videos.models import Video

        pipeline = get_object_or_404(Pipeline, pk=pk)
        if not (pipeline.is_template or pipeline.user_id == request.user.id):
            return Response({"detail": "Not allowed."}, status=403)
        if not any(s.get("block_type") == "input.video" for s in (pipeline.steps or [])):
            return Response({"detail": "Pipeline has no video input to run on."}, status=400)

        video_ids = request.data.get("video_ids")
        if not isinstance(video_ids, list) or not video_ids:
            return Response({"detail": "video_ids (non-empty list) is required."}, status=400)

        videos = Video.manageable(request.user).filter(pk__in=video_ids)
        launched, skipped = [], 0
        for video in videos:
            steps = engine.steps_with_video(pipeline, video.pk)
            if validate_steps(steps):
                skipped += 1
                continue
            run = PipelineRun.objects.create(pipeline=pipeline, user=request.user)
            engine.start_run(run, steps=steps)
            launched.append({"run_id": str(run.id), "video_id": video.pk, "status": run.status})

        if not launched:
            return Response(
                {"detail": "No valid videos to run.", "skipped": skipped}, status=400)
        return Response({"runs": launched, "skipped": skipped}, status=202)


# ── Runs ──────────────────────────────────────────────────────────────────────

class RunListView(APIView):
    def get(self, request):
        runs = (PipelineRun.objects.filter(user=request.user)
                .select_related("pipeline").order_by("-started_at", "-id")[:100])
        return Response({"runs": [_run_json(r) for r in runs]})


class RunDetailView(APIView):
    def get(self, request, run_id):
        run = get_object_or_404(PipelineRun, pk=run_id, user=request.user)
        if not run.is_terminal:
            # Nudge in-flight GPU jobs so a polling client sees fresh status.
            try:
                from apps.pipelines.views import _poll_run_jobs
                _poll_run_jobs(run)
                run.refresh_from_db()
            except Exception:
                pass
        return Response(_run_json(run, include_outputs=True))


class RunStepOutputView(APIView):
    def get(self, request, run_id, step_id):
        run = get_object_or_404(PipelineRun, pk=run_id, user=request.user)
        out = _sanitize((run.context or {}).get(step_id, {}))
        if request.query_params.get("format") == "csv":
            rows = out.get("rows") or []
            buf = io.StringIO()
            if rows:
                writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            resp = HttpResponse(buf.getvalue(), content_type="text/csv")
            resp["Content-Disposition"] = f'attachment; filename="run-{run_id}-{step_id}.csv"'
            return resp
        return Response({"step_id": step_id, "output": out})
