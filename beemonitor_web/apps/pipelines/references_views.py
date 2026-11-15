"""A batch's References page (memory/81): its clips by place and day, the
layout each day has, and which finished runs were counted with an older one.
Its launcher edits a day, copies the day before's layout, puts a day back on
its site's, and recounts what's out of date, from one page."""

from __future__ import annotations

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import Http404, JsonResponse
from django.shortcuts import redirect, render
from django.urls import reverse
from django.utils.http import urlencode
from django.views.decorators.http import require_POST

from .models import PipelineRun


def _owned_days(request, batch_id):
    """The batch's rows and days, for its launcher only."""
    from . import listing, references, sharing

    if not sharing.owns_batch(request.user, batch_id):
        raise Http404("No such batch.")
    rows = listing.rows(PipelineRun.objects.filter(batch_id=batch_id))
    return rows, references.days(rows)


def _pending(batch_id) -> int:
    from .recount import CURRENT_RULES

    return PipelineRun.objects.filter(batch_id=batch_id, status=PipelineRun.Status.COMPLETED,
                                      counting_rules__lt=CURRENT_RULES).count()


@login_required
def batch_references(request, batch_id):
    from . import batches

    rows, days = _owned_days(request, batch_id)
    show = request.GET.get("show") or ""
    shown = [d for d in days if (not show or (show == "stale" and d["stale"])
                                 or (show == "own" and d["own"])
                                 or (show == "mixed" and d["mixed"]))]
    back = urlencode({"scope": "day", "back": request.get_full_path()})
    for i, day in enumerate(days):
        clip = day["videos"][0]
        day["key"] = clip.pk
        day["clip"] = clip
        day["edit_url"] = reverse("videos:layout", args=[clip.pk]) + "?" + back
        before = days[i - 1] if i and days[i - 1]["place"] == day["place"] else None
        day["previous"] = before["videos"][0].pk if before else None
        day["previous_date"] = before["date"] if before else None
        lay = day["layout"]
        day["shapes"] = ([{"box": n.get("box")} for n in lay["nests"] if n.get("box")]
                         if lay else [])
        day["roi_box"] = (lay["roi"] if lay else None)
    first = PipelineRun.objects.filter(batch_id=batch_id).select_related("pipeline").first()
    return render(request, "pipelines/batch_references.html", {
        "batch_id": batch_id,
        "batch_name": batches.title(batch_id, first.pipeline if first else None),
        "days": shown,
        "all_days": len(days),
        "show": show,
        "stale": sum(d["stale"] for d in days),
        "own_days": sum(1 for d in days if d["own"]),
        "mixed_days": sum(1 for d in days if d["mixed"]),
        "unknown": sum(d["unknown"] for d in days),
        "pending": _pending(batch_id),
        "back": request.get_full_path(),
    })


@login_required
def batch_references_progress(request, batch_id):
    from . import sharing

    if not sharing.owns_batch(request.user, batch_id):
        raise Http404("No such batch.")
    return JsonResponse({"pending": _pending(batch_id)})


def _day_clips(request, clip_pk):
    """The clips of ``clip_pk``'s day the user may edit (the layout editor's
    "this day")."""
    from apps.videos.models import Video
    from apps.videos.views import same_day_clips

    clips = Video.objects.all() if request.user.is_superuser else Video.deletable(request.user)
    video = clips.filter(pk=clip_pk).select_related("site", "device").first()
    if video is None:
        raise Http404("No such clip.")
    return video, same_day_clips(request.user, video)[1]


@login_required
@require_POST
def batch_references_action(request, batch_id):
    """``action``: recount (one day's out-of-date runs, or every one when no
    ``day``), copy (the day before's layout onto this day) or site (this
    day's clips back on their site's layout)."""
    from apps.devices.layouts import layout_for_video
    from apps.videos.models import Layout

    from . import recount

    rows, days = _owned_days(request, batch_id)
    action, day_key = request.POST.get("action"), request.POST.get("day")
    back = request.POST.get("back") or reverse("pipelines:batch_references", args=[batch_id])
    if not back.startswith("/pipelines/batch/"):
        back = reverse("pipelines:batch_references", args=[batch_id])
    day = next((d for d in days if str(d["videos"][0].pk) == str(day_key)), None)
    if action == "recount":
        runs = [pk for d in ([day] if day else days) if d for pk in d["stale_runs"]]
        n = recount.queue(runs)
        messages.success(request, f"Recounting {n} run{'s' if n != 1 else ''} against the layout "
                                  "their clips have now, from their saved tracks.")
        return redirect(back)
    if day is None:
        raise Http404("No such day.")
    _video, clips = _day_clips(request, day["videos"][0].pk)
    if action == "copy":
        source, _ = _day_clips(request, request.POST.get("from"))
        lay = layout_for_video(source)
        for v in clips:
            Layout.save_new(video=v, roi=lay["roi_override"], polygon=lay["roi_polygon"],
                            nests=lay["nest_layout"], user=request.user)
        what = f"Copied {lay['source_label'] or 'the day before'}'s layout to the {len(clips)} clips of this day."
    elif action == "site":
        # Their own copies go: the clips follow their site's layout again,
        # including when it's next changed.
        Layout.objects.filter(video__in=clips).delete()
        what = f"The {len(clips)} clips of this day use their site's layout again."
    else:
        raise Http404("No such action.")
    n = recount.queue([r.pk for r in recount.runs_for_clips(clips)])
    messages.success(request, what + (f" Recounting {n} run{'s' if n != 1 else ''}." if n else ""))
    return redirect(back)
