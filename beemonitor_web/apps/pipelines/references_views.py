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


def _checked(batch_id, days):
    """Days to look at whose tubes the launcher found right, as they are now
    (memory/82), become checked."""
    from .models import TubeDayCheck

    checks = {(c.place, c.day): c.checked_key for c in TubeDayCheck.objects.filter(batch_id=batch_id)}
    for day in days:
        if day["kind"] == "look" and checks.get((day["place"], day["date"])) == day["flag_key"]:
            day["kind"] = "checked"
    return days


# The References page's filters: ?show= -> which days.
_SHOW = {
    "stale": lambda d: d["stale"],
    "own": lambda d: d["own"],
    "mixed": lambda d: d["mixed"],
    "look": lambda d: d["kind"] == "look",
    "waiting": lambda d: d["waiting"],
    "hand": lambda d: d["kind"] == "hand",
}


@login_required
def batch_references(request, batch_id):
    from . import batches

    rows, days = _owned_days(request, batch_id)
    _checked(batch_id, days)
    show = request.GET.get("show") or ""
    shown = [d for d in days if show not in _SHOW or _SHOW[show](d)] if show else list(days)
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
        day["check_url"] = reverse("pipelines:batch_references_day", args=[batch_id, clip.pk])
    first = PipelineRun.objects.filter(batch_id=batch_id).select_related("pipeline").first()
    detected = any(d["detected"] or d["waiting"] or d["errors"] for d in days)
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
        # Detected tubes (memory/82).
        "detected_mode": detected,
        "look_days": sum(1 for d in days if d["kind"] == "look"),
        "waiting": sum(d["waiting"] for d in days),
        "detected_days": sum(1 for d in days if d["detected"]),
        "hand_days": sum(1 for d in days if d["kind"] == "hand"),
    })


@login_required
def batch_references_day(request, batch_id, clip_pk):
    """One day's detected tubes (memory/82): a clip's frame with the tubes as
    numbered, the site's layout they were numbered from, and the day's clips
    whose numbering is worth a look."""
    from apps.devices.layouts import shared_layout_for_video

    from . import batches

    rows, days = _owned_days(request, batch_id)
    _checked(batch_id, days)
    i, day = next(((i, d) for i, d in enumerate(days) if d["videos"][0].pk == int(clip_pk)),
                  (None, None))
    if day is None:
        raise Http404("No such day.")
    pks = {v.pk for v in day["videos"]}
    mine = [r for r in rows if r["video"] is not None and r["video"].pk in pks]
    detected = [r for r in mine if ((r.get("layout") or {}).get("tubes") or {}).get("state")]
    shown_pk = request.GET.get("clip")
    pick = next((r for r in detected if str(r["video"].pk) == str(shown_pk)), None)
    if pick is None and day.get("worst"):
        pick = next((r for r in detected if r["run"].pk == day["worst"]["run"]), None)
    pick = pick or (detected[0] if detected else None)
    tubes, nests, template = {}, [], []
    video = pick["video"] if pick else day["videos"][0]
    if pick is not None:
        run = PipelineRun.objects.only("context", "steps").get(pk=pick["run"].pk)
        for out in (run.context or {}).values():
            if isinstance(out, dict) and out.get("tubes"):
                tubes, nests = out["tubes"], out.get("nest_layout") or []
                break
        template = shared_layout_for_video(type(video).everything.select_related(
            "site", "device").get(pk=video.pk))["nest_layout"] or []
    marks = tubes.get("marks") or {}
    shapes = [{"id": n.get("id"), "box": n.get("box"), "name": n.get("name") or "",
               "mark": marks.get(str(n.get("id")), "matched")} for n in nests if n.get("box")]
    clips = []
    for r in detected:
        t = r["layout"]["tubes"]
        if t.get("flag") or t.get("state") in ("error", "pending"):
            clips.append({"video": r["video"], "why": _why(t), "url": "?clip=" + str(r["video"].pk),
                          "shown": pick is r})
    look = [d for d in days if d["kind"] == "look"]
    after = next((d for d in days[i + 1:] if d["kind"] == "look"), None)
    before = next((d for d in reversed(days[:i]) if d["kind"] == "look"), None)
    t = round(float(video.duration_seconds or 0) / 2, 1)
    back = reverse("pipelines:batch_references_day", args=[batch_id, clip_pk])
    first = PipelineRun.objects.filter(batch_id=batch_id).select_related("pipeline").first()
    return render(request, "pipelines/batch_references_day.html", {
        "batch_id": batch_id,
        "batch_name": batches.title(batch_id, first.pipeline if first else None),
        "day": day, "video": video, "tubes": tubes, "why": _why(tubes) if tubes else "",
        "shapes": shapes, "template": [{"box": n.get("box")} for n in template if n.get("box")],
        "aspect": (f"{video.width} / {video.height}" if getattr(video, "width", None)
                   and getattr(video, "height", None) else "16 / 9"),
        "frame_url": reverse("videos:frame", args=[video.pk]) + f"?t={t}", "t": t,
        "clips": clips, "agree": len(detected) - len(clips), "look_left": len(look),
        "next_url": (reverse("pipelines:batch_references_day", args=[batch_id, after["videos"][0].pk])
                     if after else ""),
        "previous_url": (reverse("pipelines:batch_references_day", args=[batch_id, before["videos"][0].pk])
                         if before else ""),
        "edit_url": (reverse("videos:layout", args=[video.pk]) + "?" + urlencode(
            {"scope": "day", "tubes_of": pick["run"].pk, "back": back})) if pick else "",
        "run_pk": pick["run"].pk if pick else None,
        "can_number": bool(getattr(video, "site_id", None)) and bool(nests),
        "back": back,
        "list_url": reverse("pipelines:batch_references", args=[batch_id]) + "?show=look",
    })


def _why(tubes) -> str:
    """A clip's detected numbering in words."""
    flag, state = tubes.get("flag") or "", tubes.get("state")
    if state == "pending":
        return "Waiting for its tubes to be found"
    if state == "error" or flag == "error":
        return tubes.get("error") or "Its tubes couldn't be found"
    if flag == "none_found":
        return "No tubes found"
    if flag == "no_template":
        return f"{tubes.get('new') or 0} tubes, numbered by position: the site has no layout to number them from"
    parts = [f"{tubes.get('matched') or 0} matched"]
    if tubes.get("kept"):
        parts.append(f"{tubes['kept']} not found, kept from the site's layout")
    if tubes.get("new"):
        parts.append(f"{tubes['new']} new")
    shift = tubes.get("shift") or [0, 0]
    if max(abs(shift[0]), abs(shift[1])) >= 0.25:
        parts.append(f"camera moved {max(abs(shift[0]), abs(shift[1])):.1f} tube widths")
    return " · ".join(parts)


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
    if action == "checked" and day is not None:
        # Its detected tubes look right (memory/82).
        from .models import TubeDayCheck

        TubeDayCheck.objects.update_or_create(
            batch_id=batch_id, place=day["place"], day=day["date"],
            defaults={"checked_key": day["flag_key"], "checked_by": request.user})
        messages.success(request, "Marked as looked at.")
        nxt = request.POST.get("next") or ""
        return redirect(nxt if nxt.startswith("/pipelines/batch/") and "//" not in nxt else back)
    if action == "numbering" and day is not None:
        return _make_numbering(request, batch_id, day, back)
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


def _make_numbering(request, batch_id, day, back):
    """The tubes one clip's detection found, as its site's layout: the
    numbering every clip there is matched to from now on (memory/82). Each
    day numbered from the old one is then out of date, and recounted."""
    from apps.videos.models import Layout, Video

    from . import recount

    run = PipelineRun.objects.filter(batch_id=batch_id, user=request.user,
                                     pk=request.POST.get("run")).only("context", "steps").first()
    out = next((o for o in ((run.context or {}).values() if run else [])
                if isinstance(o, dict) and (o.get("tubes") or {}).get("state") == "detected"), None)
    video = Video.everything.select_related("site").filter(pk__in=[v.pk for v in day["videos"]],
                                                          site__isnull=False).first()
    if out is None or video is None or not out.get("nest_layout"):
        messages.error(request, "No detected tubes to save for this site.")
        return redirect(back)
    site = video.site
    latest = Layout.current(site=site)
    nests = [{k: v for k, v in n.items() if k in ("id", "box", "name", "points")}
             for n in out["nest_layout"]]
    Layout.save_new(site=site, roi=latest.roi_override if latest else out.get("hotel_roi"),
                    polygon=latest.roi_polygon if latest else out.get("hotel_polygon"),
                    nests=nests, user=request.user)
    _rows, days = _owned_days(request, batch_id)
    stale = [pk for d in days if d["place"] == day["place"] for pk in d["stale_runs"]]
    n = recount.queue(stale)
    messages.success(request, f"Saved these {len(nests)} tubes as site {site.name}'s numbering. "
                              f"Renumbering and recounting {n} run{'s' if n != 1 else ''} there.")
    return redirect(back)
