"""The insect grouping editor (memory/66).

An Identify Species node's grouping — which taxa make up each group — is too
much to edit inside a canvas node, so it has its own page: the groups and
their taxa on one side, every name the model gave in your runs on the other,
a preview of a grouping on a batch, and Apply. The grouping is saved in the
pipeline's step config, like every other setting.
"""

from __future__ import annotations

import json
import time

from django.contrib.auth.decorators import login_required
from django.http import Http404, JsonResponse
from django.shortcuts import get_object_or_404, render
from django.urls import reverse
from django.views.decorators.http import require_POST

from . import taxon_groups
from .models import Pipeline, PipelineRun

# Runs read for the names list, newest first.
NAMES_FROM_RUNS = 3000
# Seconds a preview may spend; the rest of a big batch is left out and said so.
PREVIEW_SECONDS = 40


def _step(pipeline, step_id):
    for s in pipeline.steps or []:
        if s.get("id") == step_id and s.get("block_type") == "identify.species":
            return s
    raise Http404("No Identify Species node with that id in this pipeline.")


def _name_counts(user):
    """``{name: tracks it was the most-voted name of}`` over the user's runs.

    On Postgres only the names leave the database (a JSON path over every
    step's output): a run's saved context also holds its event rows, and
    loading thousands of them to read a few names took minutes."""
    from django.db import connection

    counts = {}
    if connection.vendor == "postgresql":
        sql = (f"SELECT jsonb_path_query_array(context, '$.*.names_seen[*]'), "
               f"jsonb_path_query_array(context, '$.*.track_groups.*.top') "
               f"FROM {PipelineRun._meta.db_table} WHERE user_id = %s AND status = %s "
               f"ORDER BY started_at DESC NULLS LAST LIMIT %s")
        with connection.cursor() as cur:
            cur.execute(sql, [user.pk, PipelineRun.Status.COMPLETED, NAMES_FROM_RUNS])
            for seen, tops in cur.fetchall():
                seen = json.loads(seen) if isinstance(seen, str) else (seen or [])
                tops = json.loads(tops) if isinstance(tops, str) else (tops or [])
                for name in seen:
                    counts.setdefault(name, 0)
                for name in tops:
                    if name:
                        counts[name] = counts.get(name, 0) + 1
        return counts
    from .regroup import group_step

    runs = (PipelineRun.objects.filter(user=user, status=PipelineRun.Status.COMPLETED)
            .order_by("-started_at").only("pk", "steps", "context")[:NAMES_FROM_RUNS])
    for run in runs:
        _i, step = group_step(run)
        if step is None:
            continue
        out = (run.context or {}).get(step["id"]) or {}
        for name in out.get("names_seen") or []:
            counts.setdefault(name, 0)
        for g in (out.get("track_groups") or {}).values():
            if g.get("top"):
                counts[g["top"]] = counts.get(g["top"], 0) + 1
    return counts


# Names looked up in GBIF per request; the rest come on the next fetch.
LOOKUPS_PER_REQUEST = 40


def _names_seen(user):
    """``(rows, pending)``: every name the species model gave in the user's
    runs with its family and order, and how many names are still waiting for
    a GBIF lookup. Cached names are free; at most LOOKUPS_PER_REQUEST new ones
    are looked up each time, so the page never waits on hundreds of them."""
    from django.core.cache import cache

    key = f"grouping-names:{user.pk}"
    counts = cache.get(key)
    if counts is None:
        counts = _name_counts(user)
        cache.set(key, counts, 600)
    lineages, pending = taxon_groups.lineages(list(counts), budget=LOOKUPS_PER_REQUEST)
    rows = []
    for name, tracks in counts.items():
        lin = lineages.get(name)
        rows.append({"name": name, "tracks": tracks,
                     **{k: (lin or {}).get(k, "") for k in ("class", "order", "family", "genus")},
                     "bee": bool((lin or {}).get("bee")), "pending": lin is None})
    return sorted(rows, key=lambda r: (-r["tracks"], r["name"])), pending


def _batches(user, pipeline):
    from . import batches

    ids = []
    for bid in (PipelineRun.objects.filter(user=user, pipeline=pipeline, batch_id__isnull=False)
                .order_by("-started_at").values_list("batch_id", flat=True)):
        if bid not in ids:
            ids.append(bid)
    ids = ids[:30]
    names = batches.names(ids)
    from django.db.models import Count
    clips = dict(PipelineRun.objects.filter(batch_id__in=ids).values_list("batch_id")
                 .annotate(n=Count("pk")).values_list("batch_id", "n"))
    return [{"id": str(b), "name": names.get(b) or f"{pipeline.title} · batch {str(b)[:8]}",
             "clips": clips.get(b, 0)} for b in ids]


def _other_groupings(user, pipeline):
    out = []
    for p in Pipeline.objects.filter(user=user).exclude(pk=pipeline.pk).order_by("-updated_at"):
        for s in p.steps or []:
            if s.get("block_type") == "identify.species":
                out.append({"title": p.title,
                            "grouping": taxon_groups.grouping_of(s.get("config"))})
                break
    return out


@login_required
def grouping_editor(request, pk, step_id):
    """GET: the page. POST (owner, JSON ``{grouping}``): save it."""
    pipeline = get_object_or_404(Pipeline.usable(request.user), pk=pk)
    step = _step(pipeline, step_id)
    mine = pipeline.user_id == request.user.pk
    if request.method == "POST":
        if not mine:
            raise Http404("Only the pipeline's owner can change its grouping.")
        try:
            grouping = taxon_groups._valid(json.loads(request.body or b"{}")["grouping"])
        except (KeyError, ValueError, TypeError, AttributeError) as e:
            return JsonResponse({"ok": False, "error": str(e)}, status=400)
        steps = [dict(s) for s in pipeline.steps or []]
        for s in steps:
            if s.get("id") == step_id:
                cfg = dict(s.get("config") or {})
                cfg["grouping"] = json.dumps(grouping)
                cfg.pop("group_rules", None)
                s["config"] = cfg
        pipeline.steps = steps
        pipeline.save(update_fields=["steps", "updated_at"])
        return JsonResponse({"ok": True, "groups": len(grouping["groups"])})

    cfg = taxon_groups.settings(step.get("config"))
    page = {
        "grouping": cfg["grouping"],
        "default": taxon_groups.default_grouping(),
        "leave_out": cfg["leave_out"] if cfg["on"] else [],
        # Loaded after the page (grouping_names): it can take a few seconds.
        "names": None,
        "names_url": reverse("pipelines:grouping_names", args=[pipeline.pk, step_id]),
        "batches": _batches(request.user, pipeline) if mine else [],
        "others": _other_groupings(request.user, pipeline) if mine else [],
        "readonly": not mine,
        "save_url": request.path,
        "preview_url": reverse("pipelines:grouping_preview", args=[pipeline.pk, step_id]),
        "apply_url": reverse("pipelines:batch_apply_groups",
                             args=["00000000-0000-0000-0000-000000000000"]),
        "editor_url": reverse("pipelines:editor", args=[pipeline.pk]),
    }
    return render(request, "pipelines/grouping.html", {
        "pipeline": pipeline, "page_json": json.dumps(page), "readonly": not mine,
    })


@login_required
def grouping_names(request, pk, step_id):
    """The names list for the grouping page, fetched after it loads."""
    pipeline = get_object_or_404(Pipeline.usable(request.user), pk=pk)
    _step(pipeline, step_id)
    rows, pending = _names_seen(request.user)
    return JsonResponse({"names": rows, "pending": pending})


def _readings_of(out):
    """``{track: [name, ...]}`` from a run's saved groups (memory/66), or None
    when the run predates them keeping each track's names."""
    groups = out.get("track_groups") or {}
    if not groups or any("names" not in g for g in groups.values()):
        return None
    readings = {}
    for tid, g in groups.items():
        names = [n for n, c in (g.get("names") or {}).items() for _ in range(int(c))]
        readings[int(tid)] = names + [""] * max(int(g.get("crops") or 0) - len(names), 0)
    return readings


@login_required
@require_POST
def grouping_preview(request, pk, step_id):
    """What a grouping would do to a batch, without saving anything:
    tracks per group now and with it, and events counted before and after."""
    from . import aggregate, review
    from .regroup import group_step

    pipeline = get_object_or_404(Pipeline, pk=pk, user=request.user)
    step = _step(pipeline, step_id)
    try:
        body = json.loads(request.body or b"{}")
        grouping = taxon_groups._valid(body["grouping"])
        batch = body["batch"]
    except (KeyError, ValueError, TypeError, AttributeError) as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=400)
    cfg = taxon_groups.settings({**(step.get("config") or {}), "grouping": json.dumps(grouping)})
    leave_out = set(cfg["leave_out"]) if cfg["on"] else set()

    deadline = time.monotonic() + PREVIEW_SECONDS
    now, after = {}, {}
    events_now = events_after = clips = skipped = 0
    # Streamed a few at a time, never the whole batch at once (memory/76).
    finished = PipelineRun.objects.filter(batch_id=batch, user=request.user,
                                          status=PipelineRun.Status.COMPLETED)
    total = finished.count()
    for run in finished.iterator(chunk_size=10):
        _i, gstep = group_step(run)
        if gstep is None or time.monotonic() > deadline:
            skipped += 1
            continue
        out = (run.context or {}).get(gstep["id"]) or {}
        readings = _readings_of(out)
        if readings is None:
            try:
                readings = taxon_groups.crop_readings(out.get("result") or {})
            except Exception:  # noqa: BLE001 - one unreadable clip mustn't sink the preview
                readings = None
        if not readings:
            skipped += 1
            continue
        clips += 1
        new = taxon_groups.track_groups(readings, cfg, aggregate.run_job_id(run))
        for g in (out.get("track_groups") or {}).values():
            now[g["group"]] = now.get(g["group"], 0) + 1
        for g in new.values():
            after[g["group"]] = after.get(g["group"], 0) + 1
        counted = {t for t, g in new.items() if g["counted"]}
        for kind, o in aggregate.analyzer_outputs(run):
            if kind != "events":
                continue
            rows = o.get("rows") or []
            events_now += len(rows)
            keys = [review.event_key(r) for r in rows]
            keys += (o.get("filtered_out") or {}).get("event_keys") or []
            for k in keys:
                try:
                    tid = int(float(k.split("|")[2]))
                except (IndexError, ValueError):
                    continue
                if tid in counted or tid not in new:
                    events_after += 1
    order = cfg["groups"] + [taxon_groups.UNCLEAR]
    names = sorted(set(now) | set(after),
                   key=lambda g: order.index(g) if g in order else len(order))
    return JsonResponse({
        "ok": True, "clips": clips, "skipped": skipped, "of": total,
        "rows": [{"group": g, "now": now.get(g, 0), "after": after.get(g, 0),
                  "counted": g not in leave_out} for g in names],
        "events_now": events_now, "events_after": events_after,
    })
