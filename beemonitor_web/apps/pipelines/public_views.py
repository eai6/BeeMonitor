"""A batch's results for anyone holding its share link (memory/47).

No login. Every view starts from the token, and every clip, job or table it
serves is checked against that batch's runs, so the link reaches nothing else.
What is left out is deliberate: re-running, GPU time, error text, links into
the signed-in app, and site and location unless the owner shared them.
"""
from __future__ import annotations

from django.http import Http404, HttpResponse
from django.shortcuts import get_object_or_404, render
from django.views.decorators.http import require_GET

from . import aggregate, sharing


def _no_index(response):
    response["X-Robots-Tag"] = "noindex, nofollow"
    # The token is in the URL; keep it out of Referer headers to S3 and CDNs.
    response["Referrer-Policy"] = "no-referrer"
    return response


def _share_or_off(request, token):
    share = sharing.share_for_token(token)
    runs = sharing.shared_runs(share) if share else []
    if not runs:
        return None, None, _no_index(render(request, "pipelines/share_off.html", status=404))
    return share, runs, None


def _shared_or_404(token):
    share = sharing.share_for_token(token)
    if share is None:
        raise Http404("This link isn't shared.")
    runs = sharing.shared_runs(share)
    if not runs:
        raise Http404("This link isn't shared.")
    return share, runs


@require_GET
def public_batch(request, token):
    share, runs, off = _share_or_off(request, token)
    if off:
        return off

    from .registry import pipeline_input_kind
    from .views import _backfill_interactions_paths

    rows = aggregate.batch_rows(runs)
    outcome = aggregate.batch_summary(rows)
    devices = aggregate.batch_devices(rows)
    photos = pipeline_input_kind(runs[0].steps) == "photo"
    if photos:
        downloads = [{"kind": "photos", "label": "Photos",
                      "hint": "One row per insect: photo, time taken, box, class, "
                              "species and its confidence."}]
    else:
        sources, _ = aggregate.collect_sources(runs)
        _backfill_interactions_paths(sources)
        downloads = aggregate.available_downloads(sources, runs)

    for row in rows:
        outs = [o for k, o in aggregate.analyzer_outputs(row["run"]) if k == "interactions"]
        row["contact_sec"] = (round(sum(float(o.get("total_duration_sec") or 0) for o in outs), 1)
                              if outs else None)
    whens = [r["when"] for r in rows if r["when"]]
    sharing.record_view(share)
    return _no_index(render(request, "pipelines/batch_public.html", {
        "share": share,
        "token": token,
        "pipeline": runs[0].pipeline,
        "rows": rows,
        "outcome": outcome,
        "devices": devices,
        "contact": None if photos else sharing.contact_by_reference(runs),
        "downloads": downloads,
        "photos": photos,
        "first_when": min(whens) if whens else None,
        "last_when": max(whens) if whens else None,
        "running": outcome["running"],
    }))


@require_GET
def public_csv(request, token, kind):
    share, runs = _shared_or_404(token)
    try:
        filename, fieldnames, rows = sharing.batch_csv(runs, share.batch_id, kind)
    except KeyError:
        raise Http404("Unknown table.")
    except sharing.EmptyExport as e:
        return _no_index(HttpResponse(str(e), status=404, content_type="text/plain"))
    if not share.show_locations:
        fieldnames = sharing.without_locations(fieldnames)
    from .views import _csv_response
    return _no_index(_csv_response(filename, fieldnames, rows))


def _shared_video(token, video_pk):
    from apps.videos.models import Video

    share, runs = _shared_or_404(token)
    if not share.show_videos:
        raise Http404("Videos aren't shared on this link.")
    if video_pk not in {aggregate.run_video_id(r) for r in runs}:
        raise Http404("Not in this batch.")
    return get_object_or_404(Video.everything, pk=video_pk)


@require_GET
def public_video(request, token, video_pk):
    from apps.videos.views import stream_redirect

    return _no_index(stream_redirect(_shared_video(token, video_pk),
                                     sharing.PUBLIC_MEDIA_HOURS))


@require_GET
def public_thumbnail(request, token, video_pk):
    from apps.videos.views import thumbnail_redirect

    return _no_index(thumbnail_redirect(_shared_video(token, video_pk),
                                        sharing.PUBLIC_MEDIA_HOURS))


@require_GET
def public_overlay(request, token, job_pk):
    from apps.analysis.models import Job
    from apps.analysis.views import overlay_response

    share, runs = _shared_or_404(token)
    if not share.show_videos:
        raise Http404("Videos aren't shared on this link.")
    if job_pk not in {aggregate.run_job_id(r) for r in runs}:
        raise Http404("Not in this batch.")
    job = get_object_or_404(Job.objects.select_related("video"), pk=job_pk)
    return _no_index(overlay_response(request, job, cache_control="private, max-age=600"))
