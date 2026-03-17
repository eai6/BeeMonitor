from django.contrib import messages
from django.contrib.auth.mixins import LoginRequiredMixin
from django.shortcuts import get_object_or_404
from django.urls import reverse, reverse_lazy
from django.views.generic import DetailView, FormView, ListView, TemplateView

from .forms import JobCreateForm
from .models import Job, JobResult


class JobListView(LoginRequiredMixin, ListView):
    template_name = "analysis/list.html"
    context_object_name = "jobs"
    paginate_by = 20

    def get_queryset(self):
        return Job.objects.filter(user=self.request.user).select_related("video")


class JobDetailView(LoginRequiredMixin, DetailView):
    template_name = "analysis/detail.html"
    context_object_name = "job"

    def get_queryset(self):
        return Job.objects.filter(user=self.request.user).select_related("video")


class JobCreateView(LoginRequiredMixin, FormView):
    template_name = "analysis/new.html"
    form_class = JobCreateForm

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs["user"] = self.request.user
        return kwargs

    def get_initial(self):
        initial = super().get_initial()
        video_id = self.request.GET.get("video")
        if video_id:
            initial["video"] = video_id
        return initial

    def form_valid(self, form):
        job = Job.objects.create(
            user=self.request.user,
            video=form.cleaned_data["video"],
            config={
                "detection_mode": form.cleaned_data["detection_mode"],
                "confidence_threshold": form.cleaned_data["confidence_threshold"],
            },
            status=Job.Status.QUEUED,
        )
        messages.success(self.request, f"Job #{job.pk} created and queued.")
        return super().form_valid(form)

    def get_success_url(self):
        return reverse("analysis:list")


class JobResultsView(LoginRequiredMixin, TemplateView):
    template_name = "analysis/results.html"

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        job = get_object_or_404(Job, pk=self.kwargs["pk"], user=self.request.user)
        ctx["job"] = job
        try:
            ctx["result"] = job.result
        except JobResult.DoesNotExist:
            ctx["result"] = None
        return ctx
