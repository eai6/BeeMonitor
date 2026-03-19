import json

from django.contrib import messages
from django.contrib.auth import login
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.views import LoginView as AuthLoginView
from django.contrib.auth.views import LogoutView as AuthLogoutView
from django.db.models import Sum, Count
from django.urls import reverse_lazy
from django.views.generic import CreateView, TemplateView

from apps.analysis.models import Job, GPU_TIERS
from .forms import UserRegistrationForm
from .models import UserProfile, TIER_LIMITS


class LoginView(AuthLoginView):
    template_name = "accounts/login.html"
    redirect_authenticated_user = True

    def get_success_url(self):
        return reverse_lazy("dashboard:dashboard")


class LogoutView(AuthLogoutView):
    next_page = reverse_lazy("accounts:login")


class RegisterView(CreateView):
    template_name = "accounts/register.html"
    form_class = UserRegistrationForm
    success_url = reverse_lazy("dashboard:dashboard")

    def form_valid(self, form):
        response = super().form_valid(form)
        login(self.request, self.object)
        messages.success(self.request, "Account created successfully.")
        return response


class UsageDashboardView(LoginRequiredMixin, TemplateView):
    template_name = "accounts/usage.html"

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        user = self.request.user

        # Ensure profile exists
        profile, _ = UserProfile.objects.get_or_create(user=user)
        ctx["profile"] = profile
        ctx["tier_limits"] = TIER_LIMITS

        # Job stats
        jobs = Job.objects.filter(user=user)
        completed = jobs.filter(status=Job.Status.COMPLETED)
        ctx["total_jobs"] = jobs.count()
        ctx["completed_jobs"] = completed.count()
        ctx["failed_jobs"] = jobs.filter(status=Job.Status.FAILED).count()
        ctx["processing_jobs"] = jobs.filter(status=Job.Status.PROCESSING).count()

        # Cost breakdown by GPU tier
        gpu_stats = (
            completed.values("gpu_tier")
            .annotate(
                count=Count("id"),
                total_seconds=Sum("execution_seconds"),
                total_cost=Sum("compute_cost_usd"),
            )
            .order_by("-count")
        )
        ctx["gpu_stats"] = list(gpu_stats)

        # Monthly cost data for chart
        from django.db.models.functions import TruncMonth
        monthly_costs = (
            completed.annotate(month=TruncMonth("completed_at"))
            .values("month")
            .annotate(cost=Sum("compute_cost_usd"), count=Count("id"))
            .order_by("month")
        )
        ctx["monthly_costs_json"] = json.dumps([
            {
                "month": m["month"].strftime("%Y-%m") if m["month"] else "unknown",
                "cost": float(m["cost"] or 0),
                "count": m["count"],
            }
            for m in monthly_costs
        ])

        return ctx
