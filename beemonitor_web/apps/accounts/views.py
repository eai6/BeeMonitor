from django.contrib import messages
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.contrib.auth.views import LoginView as AuthLoginView
from django.contrib.auth.views import LogoutView as AuthLogoutView
from django.db import transaction
from django.db.models import F, Sum
from django.shortcuts import redirect
from django.urls import reverse_lazy
from django.utils import timezone
from django.views.generic import CreateView, TemplateView, View

from apps.analysis.models import Job, GPU_TIERS
from .forms import UserRegistrationForm
from . import quota
from .models import AccountRequest, Coupon, CouponRedemption, EmailLog, UserProfile, TIER_LIMITS


class LoginView(AuthLoginView):
    template_name = "accounts/login.html"
    redirect_authenticated_user = True

    def get_success_url(self):
        return reverse_lazy("dashboard:dashboard")


class LogoutView(AuthLogoutView):
    """POST only (Django's default): a GET logout let any page sign you out
    with an <img> tag."""

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


def _usage_totals(user, month_start):
    """Credits and GPU-hours, this month and lifetime, from UsageEvent. Users see
    hours, never seconds."""
    def totals(qs):
        agg = qs.aggregate(cr=Sum("credits"), secs=Sum("gpu_seconds"))
        return agg["cr"] or 0, (agg["secs"] or 0) / 3600

    month = user.usage_events.filter(created_at__date__gte=month_start)
    month_cr, month_h = totals(month)
    life_cr, life_h = totals(user.usage_events.all())
    return {"used_this_month": month_cr, "hours_this_month": month_h,
            "month_by_kind": quota.by_kind(month),
            "lifetime_credits": life_cr, "lifetime_hours": life_h}


class ProfileView(LoginRequiredMixin, TemplateView):
    """Account → Profile: who you are, your email, your password.

    One page, three forms; the submit button's ``action`` says which one.
    """

    template_name = "accounts/profile.html"

    def get_context_data(self, **kwargs):
        from django.contrib.auth.forms import PasswordChangeForm

        from .forms import EmailChangeForm, ProfileForm

        ctx = super().get_context_data(**kwargs)
        user = self.request.user
        profile, _ = UserProfile.objects.get_or_create(user=user)
        ctx["profile"] = profile
        ctx["unlimited"] = quota.is_unlimited(user)
        ctx.setdefault("profile_form", ProfileForm(initial={
            "first_name": user.first_name, "last_name": user.last_name,
            "organization": profile.organization}))
        ctx.setdefault("email_form", EmailChangeForm(user))
        ctx.setdefault("password_form", PasswordChangeForm(user))
        return ctx

    def post(self, request):
        from django.contrib.auth import update_session_auth_hash
        from django.contrib.auth.forms import PasswordChangeForm

        from .emails import send
        from .forms import EmailChangeForm, ProfileForm

        user = request.user
        action = request.POST.get("action")
        if action == "profile":
            form = ProfileForm(request.POST)
            if form.is_valid():
                user.first_name = form.cleaned_data["first_name"]
                user.last_name = form.cleaned_data["last_name"]
                user.save(update_fields=["first_name", "last_name"])
                UserProfile.objects.update_or_create(
                    user=user, defaults={"organization": form.cleaned_data["organization"]})
                messages.success(request, "Profile saved.")
                return redirect("accounts:profile")
            return self.render_to_response(self.get_context_data(profile_form=form))

        if action == "email":
            form = EmailChangeForm(user, request.POST)
            if form.is_valid():
                old = user.email
                user.email = form.cleaned_data["new_email"]
                user.save(update_fields=["email"])
                if old:
                    send("email_changed", old, "Your BeeMonitor email was changed",
                         f"The email on your BeeMonitor account ({user.username}) was changed "
                         f"from {old} to {user.email}.\n\nIf you didn't do this, reset your "
                         "password and contact the BeeMonitor team.", user=user)
                messages.success(request, f"Your email is now {user.email}.")
                return redirect("accounts:profile")
            return self.render_to_response(self.get_context_data(email_form=form))

        if action == "password":
            form = PasswordChangeForm(user, request.POST)
            if form.is_valid():
                form.save()
                # Keeps this session; every other session's hash no longer
                # matches, so they are signed out.
                update_session_auth_hash(request, form.user)
                if user.email:
                    send("password_changed", user.email, "Your BeeMonitor password was changed",
                         f"The password on your BeeMonitor account ({user.username}) was just "
                         "changed. If you didn't do this, reset it from the sign-in page.",
                         user=user)
                messages.success(request, "Password changed. Other devices have been signed out.")
                return redirect("accounts:profile")
            return self.render_to_response(self.get_context_data(password_form=form))

        return redirect("accounts:profile")


class UsageDashboardView(LoginRequiredMixin, TemplateView):
    template_name = "accounts/usage.html"

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        user = self.request.user

        # Ensure profile exists
        profile, _ = UserProfile.objects.get_or_create(user=user)
        ctx["profile"] = profile
        ctx["tier_limits"] = TIER_LIMITS

        # GPU credits (apps/accounts/quota.py) — the balance that gates GPU work.
        ctx["credits"] = {
            "unlimited": quota.is_unlimited(user),
            "left": quota.available(user),
            "pending": quota.pending(user),
            **_usage_totals(user, timezone.localdate().replace(day=1)),
        }
        profile.refresh_from_db()
        ctx["requests"] = AccountRequest.objects.filter(user=user)[:10]
        ctx["open_kinds"] = set(AccountRequest.objects.filter(
            user=user, status=AccountRequest.Status.PENDING).values_list("kind", flat=True))
        ctx["starting_credits"] = quota.STARTING_CREDITS
        from apps.videos import retention
        ctx["deletions"] = [] if quota.is_unlimited(user) else retention.scheduled_for(user)
        ctx["request_form"] = kwargs.get("request_form")

        # Job stats
        jobs = Job.objects.filter(user=user)
        completed = jobs.filter(status=Job.Status.COMPLETED)
        ctx["total_jobs"] = jobs.count()
        ctx["completed_jobs"] = completed.count()
        ctx["failed_jobs"] = jobs.filter(status=Job.Status.FAILED).count()
        ctx["processing_jobs"] = jobs.filter(status=Job.Status.PROCESSING).count()

        # Credits per day for the last 30 days, by kind of work (UsageEvent).
        # Users see credits, not dollars.
        from . import charts
        from .models import UsageEvent
        days, series = charts.credits_by_period(user.usage_events.all(), periods=30, unit="day")
        ctx["chart"] = charts.stacked(
            days, [UsageEvent.Kind(k).label if k in UsageEvent.Kind.values else k for k in series],
            "Day", label_every=5)
        ctx["plans"] = [{"key": k, **v} for k, v in TIER_LIMITS.items()]

        # Coupon redemption history
        ctx["redemptions"] = CouponRedemption.objects.filter(user=user).select_related("coupon").order_by("-redeemed_at")

        return ctx


class RequestCreateView(LoginRequiredMixin, View):
    """Ask for more credits, Pro, or longer storage (memory/50 §4). Platform
    admins are emailed and decide on /platform/requests/."""

    def post(self, request):
        from datetime import timedelta

        from .emails import send
        from .forms import AccountRequestForm
        from .throttle import allow

        form = AccountRequestForm(request.user, request.POST)
        if not form.is_valid():
            for errors in form.errors.values():
                for e in errors:
                    messages.error(request, e)
            return redirect("accounts:usage")
        if not allow(f"request:{request.user.pk}", 5, timedelta(days=1)):
            messages.error(request, "That's a lot of requests for one day. Try again tomorrow.")
            return redirect("accounts:usage")
        req = form.save(commit=False)
        req.user = request.user
        req.save()
        from django.contrib.auth import get_user_model
        link = request.build_absolute_uri(f"/platform/requests/#request-{req.pk}")
        for admin in get_user_model().objects.filter(is_superuser=True, is_active=True).exclude(email=""):
            send("account_request", admin.email, f"BeeMonitor request: {req.summary} for {request.user.username}",
                 f"{request.user.username} ({req.contact_email}) asked for: {req.summary}.\n\n"
                 f"{req.note}\n\nReview it: {link}", user=admin)
        messages.success(request, "Request sent. We'll reply by email, and its status shows below.")
        return redirect("accounts:usage")


class RedeemCouponView(LoginRequiredMixin, View):
    def post(self, request):
        from datetime import timedelta

        from .throttle import allow

        code = request.POST.get("coupon_code", "").strip()
        if not code:
            messages.error(request, "Please enter a promo code.")
            return redirect("accounts:usage")
        # Codes are short: ten tries an hour per account stops guessing them.
        if not allow(f"coupon:{request.user.pk}", 10, timedelta(hours=1)):
            messages.error(request, "Too many promo code attempts. Try again in an hour.")
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
                # Granted credits never reset (quota.py); the old code added to
                # the monthly limit, which a reset wiped.
                credits_added = coupon.credits_amount
                UserProfile.objects.filter(pk=profile.pk).update(
                    granted_gpu_credits=F("granted_gpu_credits") + credits_added)
                messages.success(request, f"{credits_added} credits added to your account.")

            elif coupon.coupon_type == Coupon.CouponType.TIER_UPGRADE:
                new_tier = coupon.upgrade_tier
                limits = TIER_LIMITS.get(new_tier, TIER_LIMITS["standard"])
                profile.tier = new_tier
                profile.monthly_credits = limits["monthly_credits"]
                # The trial ends: quota.expire_tier drops it back to Standard.
                profile.tier_expires_at = timezone.now() + timezone.timedelta(days=coupon.upgrade_duration_days)
                profile.save(update_fields=["tier", "monthly_credits", "tier_expires_at"])
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


class SuperuserRequiredMixin(UserPassesTestMixin):
    def test_func(self):
        return self.request.user.is_superuser


class EmailStatusView(LoginRequiredMixin, SuperuserRequiredMixin, TemplateView):
    """Is email working? SES's own view of the account, plus what we tried to send."""

    template_name = "accounts/email_status.html"

    def get_context_data(self, **kwargs):
        from datetime import timedelta

        from .emails import ses_status

        ctx = super().get_context_data(**kwargs)
        since = timezone.now() - timedelta(days=7)
        recent = EmailLog.objects.filter(created_at__gte=since)
        ctx["ses"] = ses_status()
        ctx["sent_7d"] = recent.filter(status=EmailLog.Status.SENT).count()
        ctx["failed_7d"] = recent.filter(status=EmailLog.Status.FAILED).count()
        ctx["logs"] = EmailLog.objects.select_related("user")[:100]
        return ctx


class EmailTestView(LoginRequiredMixin, SuperuserRequiredMixin, View):
    """Send a test email to the signed-in superuser, through the real backend."""

    def post(self, request):
        from .emails import send

        to = request.POST.get("to", "").strip() or request.user.email
        if not to:
            messages.error(request, "Your account has no email address to send to.")
            return redirect("accounts:email_status")
        log = send("test", to, "BeeMonitor test email",
                   "This is a test email from BeeMonitor. If you can read it, email works.",
                   user=request.user)
        if log.status == EmailLog.Status.SENT:
            messages.success(request, f"Sent a test email to {to}.")
        else:
            messages.error(request, f"Sending to {to} failed: {log.error}")
        return redirect("accounts:email_status")


class CouponListView(LoginRequiredMixin, SuperuserRequiredMixin, View):
    """Promo codes moved to the platform dashboard (/platform/codes/)."""

    def get(self, request):
        return redirect("platform_admin:codes")
