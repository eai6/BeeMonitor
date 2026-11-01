"""Training › Event classifiers (memory/70): event classifiers trained and
evaluated on review data."""

from __future__ import annotations

import threading

from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import HttpResponse
from django.shortcuts import redirect, render
from django.urls import reverse
from django.views import View

from .models import CustomModel


def _choices(request):
    from apps.pipelines import eventml

    available = eventml.batches_with_reviews(request.user)
    picked = request.GET.getlist("batch") or request.POST.getlist("batch")
    picked = [b["id"] for b in available if b["id"] in picked] or [b["id"] for b in available]
    left_out = request.GET.get("left_out", request.POST.get("left_out"))
    if left_out is None:
        left_out = eventml.default_left_out(request.user)
    return available, picked, left_out


class EventClassifiersView(LoginRequiredMixin, View):
    def get(self, request):
        from apps.annotations import leftout
        from apps.pipelines import eventml

        available, picked, left_out = _choices(request)
        summary = None
        if picked and request.GET.get("show"):
            _rows, summary = eventml.dataset(request.user, picked, left_out)
            summary.pop("missed_rows", None)
        models = list(CustomModel.objects.filter(user=request.user,
                                                 model_type="event_classifier"))
        builtin = next(((m.metrics or {}).get("builtin") for m in models
                        if (m.metrics or {}).get("builtin")), None)
        return render(request, "training/event_classifiers.html", {
            "available": available, "picked": picked, "left_out": left_out,
            "left_out_errors": leftout.parse(left_out)[1],
            "summary": summary, "models": models, "builtin": builtin,
            "training": any(m.status == CustomModel.Status.TRAINING for m in models),
        })

    def post(self, request):
        """Train the chosen models in the background."""
        from apps.pipelines import eventml

        _available, picked, left_out = _choices(request)
        algorithm = request.POST.get("algorithm") or "random_forest"
        if algorithm not in ("random_forest", "gradient_boosting"):
            algorithm = "random_forest"
        kinds = [k for k in request.POST.getlist("kind") if k in eventml.KINDS] or ["exit_enter"]
        for kind in kinds:
            label = "Exit/enter" if kind == "exit_enter" else "Stay"
            n = CustomModel.objects.filter(user=request.user, model_type="event_classifier",
                                           classes=list(eventml.KINDS[kind])).count() + 1
            m = CustomModel.objects.create(
                user=request.user, name=f"{label} model v{n}", model_type="event_classifier",
                base_model=algorithm, classes=list(eventml.KINDS[kind]),
                status=CustomModel.Status.TRAINING)
            threading.Thread(target=eventml.train_and_store,
                             args=(m.pk, picked, left_out, algorithm), daemon=True).start()
        q = "&".join([f"batch={b}" for b in picked])
        return redirect(f"{reverse('training:event_classifiers')}?{q}")


class EventDatasetCsvView(LoginRequiredMixin, View):
    """The training set as CSV (features + labels), for notebooks and the paper."""

    def get(self, request):
        from apps.pipelines import eventml

        _available, picked, left_out = _choices(request)
        rows, _summary = eventml.dataset(request.user, picked, left_out)
        resp = HttpResponse(eventml.csv_text(rows), content_type="text/csv")
        resp["Content-Disposition"] = 'attachment; filename="reviewed_events.csv"'
        return resp
