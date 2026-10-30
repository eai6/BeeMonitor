"""Period review: every event of a device or site over hours or days, and the
trips in it (memory/62 §4, §4b). The logic is ``apps.pipelines.period``."""

from __future__ import annotations

import csv
import json
import uuid
from datetime import datetime, timedelta

from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import Http404, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, render
from django.urls import reverse
from django.views import View

from apps.pipelines import period
from apps.pipelines import review as event_review

from .models import Job


def _place(request):
    """``(device, site, start, end, notes)`` from the request, or Http404.

    ``batch=<uuid>`` fills in what's missing: the batch's one device (or one
    site) and its clips' time range."""
    from apps.devices.models import Device
    from apps.pipelines.aggregate import run_video_id
    from apps.pipelines.models import PipelineRun
    from apps.videos.models import Site, Video

    g = request.GET
    device = site = None
    start, end = period.parse_local(g.get("from")), period.parse_local(g.get("to"))
    if g.get("batch"):
        try:
            batch = uuid.UUID(g["batch"])
        except ValueError:
            raise Http404("No such batch.")
        vids = [v for r in PipelineRun.objects.filter(batch_id=batch) if (v := run_video_id(r))]
        videos = list(Video.objects.filter(pk__in=vids).select_related("device", "site"))
        if not videos:
            raise Http404("No such batch.")
        devices = {v.device_id for v in videos}
        sites = {v.site_id for v in videos}
        if len(devices) == 1 and None not in devices:
            device = videos[0].device
        elif len(sites) == 1 and None not in sites:
            site = videos[0].site
        starts = [s for s in (period.local_start(v) for v in videos) if s]
        if starts and not start:
            start = min(starts).replace(minute=0, second=0, microsecond=0)
        if starts and not end:
            last = max((period.local_start(v) + timedelta(seconds=float(v.duration_seconds or 0)))
                       for v in videos if period.local_start(v))
            end = last.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    if g.get("device"):
        device = get_object_or_404(Device, pk=g["device"])
    elif g.get("site_id"):
        site = get_object_or_404(Site, pk=g["site_id"])
    if device is None and site is None:
        raise Http404("Pick a device or a site.")
    if start is None:
        start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    if end is None or end <= start:
        end = start + timedelta(days=1)
    return device, site, start, end


def _gather(request):
    device, site, start, end = _place(request)
    # Opened from a batch: its clips show (and take votes on) the batch's runs.
    prefer = period.batch_runs(request.GET["batch"]) if request.GET.get("batch") else None
    data = period.gather(request.user, device=device, site=site, start=start, end=end,
                         prefer=prefer)
    if not data["clips"] and not data["skipped"]:
        # Nothing the user can see here: not a page about someone else's unit.
        from apps.devices.models import Device
        if device is not None and not Device.accessible(request.user).filter(pk=device.pk).exists():
            raise Http404("No such device.")
        if site is not None and site.user_id != request.user.pk:
            raise Http404("No such site.")
    data.update(device=device, site=site)
    return data


def _trip_votes(runs, user):
    """``{(run pk, exit key): {"mine": {...} or None, "accepted", "rejected"}}``."""
    out = {}
    for t in event_review.trip_reviews_for(runs):
        s = out.setdefault((t.own_run.pk, t.exit_key), {"mine": None, "accepted": 0, "rejected": 0})
        s[t.verdict] += 1
        if t.user_id == user.pk:
            s["mine"] = {"verdict": t.verdict, "entry_key": t.entry_key, "reason": t.reason}
    return out


class PeriodReviewView(LoginRequiredMixin, View):
    def get(self, request):
        from apps.devices.models import Device
        from apps.videos.models import Site

        from apps.pipelines import review as event_review

        data = _gather(request)
        clips_json, pairs = [], []
        for ci, c in enumerate(data["clips"]):
            evs = period.clip_events(c, request.user)
            pairs += [(ci, e) for e in evs]
            job, run_q = c["job"], f"?run={c['run'].pk}"
            clips_json.append({
                "job": job.pk, "run": str(c["run"].pk), "title": c["video"].title, "offset": c["offset"],
                "duration": c["duration"], "fps": c["fps"],
                "start": c["start"].isoformat(timespec="seconds"),
                "can_review": c["can_review"],
                "overlay_url": reverse("analysis:track_overlay", args=[job.pk]) + run_q,
                "stream_url": reverse("analysis:period_stream", args=[job.pk]),
                # The run is named so votes land on the run shown here.
                "review_url": reverse("analysis:event_review", args=[job.pk]) + run_q,
                "add_url": reverse("analysis:add_missed_event", args=[job.pk]) + run_q,
                # Events reviewers added as missed (memory/62 §3b), placed on
                # the period's axis like the detected ones.
                "added": [{"id": a["id"], "frame": a["frame"], "end_frame": a["end_frame"],
                           "action": a["action"], "reference": a["reference"],
                           "track_id": a["track_id"], "by": a["user"],
                           "t": round(c["offset"] + a["frame"] / float(c["fps"] or 25), 3)}
                          for a in event_review.added_rows(c["run"])],
                "crops_url": reverse("analysis:track_crops", args=[job.pk, 0]),
                "results_url": reverse("analysis:results", args=[job.pk]),
                "events": evs,
            })
        votes = _trip_votes([c["run"] for c in data["clips"]], request.user)
        trips = []
        for t in period.suggest_trips(pairs):
            (eci, ee), (nci, ne) = t["exit"], t["entry"]
            v = votes.get((data["clips"][eci]["run"].pk, ee["key"])) or {}
            trips.append({"exit": {"clip": eci, "key": ee["key"]},
                          "entry": {"clip": nci, "key": ne["key"]},
                          "reference": ee["reference"], "gap_sec": t["gap_sec"],
                          "same_marker": t["same_marker"], "mine": v.get("mine"),
                          "accepted": v.get("accepted", 0), "rejected": v.get("rejected", 0)})
        span = (data["end"] - data["start"]).total_seconds()
        payload = {"start": data["start"].isoformat(timespec="seconds"), "span": span,
                   "clips": clips_json, "trips": trips,
                   "trip_url": reverse("analysis:trip_review")}
        query = request.GET.copy()
        return render(request, "analysis/period_review.html", {
            "data": data, "payload": payload,
            "from_value": data["start"].strftime("%Y-%m-%dT%H:%M"),
            "to_value": data["end"].strftime("%Y-%m-%dT%H:%M"),
            "devices": Device.accessible(request.user).order_by("name"),
            "sites": Site.objects.filter(user=request.user).order_by("name"),
            "csv_url": reverse("analysis:period_trips_csv") + "?" + query.urlencode(),
            "any_review": any(c["can_review"] for c in data["clips"]),
        })


class PeriodStreamView(LoginRequiredMixin, View):
    """The raw clip for a job the user may open — invited batch members too,
    whom the Videos page's stream link doesn't admit."""

    def get(self, request, pk):
        from apps.videos.views import stream_redirect

        from .views import clip_access

        job = get_object_or_404(Job.objects.for_viewer(request.user).filter(
            clip_access(request.user)).select_related("video"), pk=pk)
        return stream_redirect(job.video)


class TripReviewView(LoginRequiredMixin, View):
    """Record, change or take back one reviewer's call on a suggested trip.

    JSON ``{exit_job, exit_key, entry_job|null, entry_key, verdict, reason,
    relinked}``. Only reviewers of the exit's run; the entry must be a clip
    they can see."""

    def post(self, request):
        from .views import clip_access

        try:
            body = json.loads(request.body or b"{}")
            exit_job = get_object_or_404(Job, pk=int(body["exit_job"]))
        except (KeyError, ValueError, TypeError):
            return JsonResponse({"ok": False, "error": "exit_job is required"}, status=400)
        exit_run = event_review.run_for_review(exit_job, body.get("exit_run"))
        if exit_run is None or not event_review.can_review(request.user, exit_run):
            raise Http404("You can't review this clip.")
        entry_run = None
        if body.get("entry_job"):
            entry_job = Job.objects.filter(clip_access(request.user), pk=body["entry_job"]).first()
            entry_run = (event_review.run_for_review(entry_job, body.get("entry_run"))
                         if entry_job else None)
            if entry_run is None:
                raise Http404("No such entry.")
        try:
            verdict = period.set_trip_vote(
                request.user, exit_run, str(body.get("exit_key", "")), entry_run,
                str(body.get("entry_key") or ""), body.get("verdict"), body.get("reason", ""),
                bool(body.get("relinked")))
        except ValueError as e:
            return JsonResponse({"ok": False, "error": str(e)}, status=400)
        return JsonResponse({"ok": True, "verdict": verdict})


def trips_csv_response(trip_reviews, filename):
    rows = period.trip_rows(trip_reviews)
    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = f'attachment; filename="{filename}"'
    writer = csv.DictWriter(response, fieldnames=period.TRIP_FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    return response


class PeriodTripsCsvView(LoginRequiredMixin, View):
    """The period's reviewed trips: only pairs a reviewer called."""

    def get(self, request):
        from apps.pipelines.models import TripReview

        data = _gather(request)
        runs = [c["run"] for c in data["clips"]]
        reviews = event_review.trip_reviews_for(runs)
        return trips_csv_response(TripReview.objects.filter(pk__in=[t.pk for t in reviews]),
                                  f"trips_{data['start']:%Y%m%d_%H%M}.csv")
