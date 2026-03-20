import json

from django.contrib import messages
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.contrib.auth.views import LoginView as AuthLoginView
from django.contrib.auth.views import LogoutView as AuthLogoutView
from django.db import transaction
from django.db.models import Sum, Count
from django.shortcuts import redirect, get_object_or_404
from django.urls import reverse_lazy
from django.utils import timezone
from django.views.generic import CreateView, ListView, TemplateView, View

from apps.analysis.models import Job, GPU_TIERS
from .forms import UserRegistrationForm, CouponForm
from .models import Coupon, CouponRedemption, UserProfile, TIER_LIMITS


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

        # Coupon redemption history
        ctx["redemptions"] = CouponRedemption.objects.filter(user=user).select_related("coupon").order_by("-redeemed_at")

        return ctx


class RedeemCouponView(LoginRequiredMixin, View):
    def post(self, request):
        code = request.POST.get("coupon_code", "").strip()
        if not code:
            messages.error(request, "Please enter a promo code.")
            return redirect("accounts:usage")

        try:
            coupon = Coupon.objects.get(code__iexact=code)
        except Coupon.DoesNotExist:
            messages.error(request, "Invalid promo code.")
            return redirect("accounts:usage")

        if not coupon.is_valid:
            messages.error(request, "This promo code has expired or is no longer available.")
            return redirect("accounts:usage")

        # Check if user already redeemed this coupon
        if CouponRedemption.objects.filter(coupon=coupon, user=request.user).exists():
            messages.warning(request, "You have already redeemed this promo code.")
            return redirect("accounts:usage")

        profile, _ = UserProfile.objects.get_or_create(user=request.user)

        with transaction.atomic():
            coupon = Coupon.objects.select_for_update().get(pk=coupon.pk)
            if not coupon.is_valid:
                messages.error(request, "This promo code is no longer available.")
                return redirect("accounts:usage")

            previous_tier = profile.tier
            credits_added = 0

            if coupon.coupon_type == Coupon.CouponType.CREDITS:
                credits_added = coupon.credits_amount
                profile.monthly_credits += credits_added
                profile.save(update_fields=["monthly_credits"])
                messages.success(request, f"{credits_added} bonus credits added to your account!")

            elif coupon.coupon_type == Coupon.CouponType.TIER_UPGRADE:
                new_tier = coupon.upgrade_tier
                limits = TIER_LIMITS.get(new_tier, TIER_LIMITS["free"])
                profile.tier = new_tier
                profile.monthly_credits = limits["monthly_credits"]
                profile.max_concurrent_jobs = limits["max_concurrent_jobs"]
                profile.save(update_fields=["tier", "monthly_credits", "max_concurrent_jobs"])
                messages.success(
                    request,
                    f"Your account has been upgraded to {profile.get_tier_display()} for {coupon.upgrade_duration_days} days!",
                )

            CouponRedemption.objects.create(
                coupon=coupon,
                user=request.user,
                credits_added=credits_added,
                previous_tier=previous_tier,
                new_tier=profile.tier,
            )
            coupon.times_redeemed += 1
            coupon.save(update_fields=["times_redeemed"])

        return redirect("accounts:usage")


class StaffRequiredMixin(UserPassesTestMixin):
    def test_func(self):
        return self.request.user.is_staff


class CouponListView(LoginRequiredMixin, StaffRequiredMixin, ListView):
    model = Coupon
    template_name = "accounts/coupons.html"
    context_object_name = "coupons"
    ordering = ["-created_at"]

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        ctx["form"] = CouponForm()
        return ctx


class CouponCreateView(LoginRequiredMixin, StaffRequiredMixin, View):
    def post(self, request):
        form = CouponForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, f"Coupon '{form.instance.code}' created.")
        else:
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f"{field}: {error}")
        return redirect("accounts:coupons")


class CouponDeactivateView(LoginRequiredMixin, StaffRequiredMixin, View):
    def post(self, request, pk):
        coupon = get_object_or_404(Coupon, pk=pk)
        coupon.is_active = False
        coupon.save(update_fields=["is_active"])
        messages.success(request, f"Coupon '{coupon.code}' deactivated.")
        return redirect("accounts:coupons")
