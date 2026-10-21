"""/platform/: the super admin's dashboard (memory/50 §8).

Manage users and see what they are doing: the overview, the users table, one
user's page with the actions that change an account, promo codes, and the
activity and audit log. Superusers only; every change is written to the audit
log (AdminAction) with who made it.
"""

import csv
import secrets
from datetime import timedelta
from decimal import Decimal, InvalidOperation

from django.contrib import messages
from django.contrib.auth import get_user_model
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.core.paginator import Paginator
from django.db.models import Count, F, Q, Sum
from django.http import Http404, HttpResponse
from django.shortcuts import get_object_or_404, redirect
from django.utils import timezone
from django.utils.dateparse import parse_date
from django.views.generic import TemplateView, View

from apps.accounts import quota
from apps.accounts.forms import CouponForm, LoggedPasswordResetForm
from apps.accounts.models import (AdminAction, Coupon, CouponRedemption, LoginEvent, UsageEvent,
                                  UserProfile)
from apps.devices.models import Device

from . import stats

User = get_user_model()

SORTS = {
    "credits": "-month_credits", "lifetime": "-life_seconds", "storage": "-storage_bytes",
    "clips": "-clips", "units": "-units", "joined": "-date_joined", "login": "-last_login",
    "name": "username",
}
# Tiers an admin can give. Free and Enterprise are on their way out (memory/50 §10).
GRANTABLE_TIERS = [UserProfile.Tier.STANDARD, UserProfile.Tier.PRO]
PRO_DEFAULT_DAYS = 365
GPU_NAMES = {"ml.g4dn.xlarge": "T4", "ml.g6.xlarge": "L4", "ml.g5.xlarge": "A10G",
             "ml.g6e.xlarge": "L40S", "ml.p4d.24xlarge": "A100×8"}


class SuperuserMixin(LoginRequiredMixin, UserPassesTestMixin):
    def test_func(self):
        return self.request.user.is_superuser

    def handle_no_permission(self):
        if self.request.user.is_authenticated:
            raise Http404  # Don't advertise that the page exists.
        return super().handle_no_permission()


class PlatformPage(SuperuserMixin, TemplateView):
    section = ""

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        ctx["section"] = self.section
        return ctx


class OverviewView(PlatformPage):
    template_name = "platform_admin/overview.html"
    section = "overview"

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        ctx["o"] = stats.overview()
        weeks, series = stats.weekly_credits(UsageEvent.objects.all(), split="tier")
        ctx["chart"] = _chart(weeks, [UserProfile.Tier(s).label if s in UserProfile.Tier.values else "Other"
                                      for s in series])
        top = (stats.users_with_totals().filter(month_credits__gt=0).order_by("-month_credits")[:8])
        ctx["top"] = [{"user": u, "tier": stats.tier_label(u)} for u in top]
        ctx["recent"] = _recent_activity(12)
        return ctx


class UserListView(PlatformPage):
    template_name = "platform_admin/users.html"
    section = "users"

    def get(self, request, *args, **kwargs):
        if request.GET.get("format") == "csv":
            return self._csv(self._rows())
        return super().get(request, *args, **kwargs)

    def _rows(self):
        g = self.request.GET
        qs = stats.users_with_totals()
        q = g.get("q", "").strip()
        if q:
            qs = qs.filter(Q(username__icontains=q) | Q(email__icontains=q)
                           | Q(first_name__icontains=q) | Q(last_name__icontains=q)
                           | Q(profile__organization__icontains=q))
        tier = g.get("tier", "")
        if tier == "superuser":
            qs = qs.filter(is_superuser=True)
        elif tier in UserProfile.Tier.values:
            qs = qs.filter(profile__tier=tier, is_superuser=False)
        active = g.get("active", "")
        if active in ("7", "30"):
            cutoff = timezone.now() - timedelta(days=int(active))
            qs = qs.filter(Q(profile__last_seen_at__gte=cutoff) | Q(last_login__gte=cutoff))
        qs = qs.order_by(SORTS.get(g.get("sort", ""), SORTS["credits"]), "username")
        rows = [{"user": u, "tier": stats.tier_label(u), "status": stats.status(u),
                 "hours": u.life_seconds / 3600} for u in qs]
        status = g.get("status", "")
        if status:
            rows = [r for r in rows if r["status"][0].lower().replace(" ", "_") == status]
        return rows

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        rows = self._rows()
        page = Paginator(rows, 50).get_page(self.request.GET.get("page"))
        params = self.request.GET.copy()
        params.pop("page", None)
        ctx.update(page=page, total=len(rows), query=params.urlencode(), g=self.request.GET,
                   tiers=UserProfile.Tier.choices, sorts=SORTS)
        return ctx

    @staticmethod
    def _csv(rows):
        resp = HttpResponse(content_type="text/csv")
        resp["Content-Disposition"] = f'attachment; filename="beemonitor-users-{timezone.localdate()}.csv"'
        w = csv.writer(resp)
        w.writerow(["username", "email", "name", "organization", "tier", "credits_this_month",
                    "gpu_cost_usd_this_month", "lifetime_gpu_hours", "storage_gb", "units", "clips",
                    "joined", "last_login", "last_seen", "status"])
        for r in rows:
            u = r["user"]
            w.writerow([
                u.username, u.email, u.get_full_name(), getattr(getattr(u, "profile", None), "organization", ""),
                r["tier"], f"{u.month_credits:.2f}", f"{u.month_cost:.2f}", f"{u.life_seconds / 3600:.2f}",
                f"{u.storage_bytes / 1e9:.2f}", u.units, u.clips, u.date_joined.date(),
                u.last_login or "", u.last_seen or "", r["status"][0],
            ])
        return resp


class UserDetailView(PlatformPage):
    template_name = "platform_admin/user.html"
    section = "users"

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        u = get_object_or_404(stats.users_with_totals(), pk=kwargs["pk"])
        profile, _ = UserProfile.objects.get_or_create(user=u)
        quota.expire_tier(profile)
        events = UsageEvent.objects.filter(user=u)
        weeks, series = stats.weekly_credits(events, split="kind")
        life = events.aggregate(cost=Sum("cost_usd"), cr=Sum("credits"))
        ctx.update(
            u=u, profile=profile, tier=stats.tier_label(u), status=stats.status(u),
            unlimited=quota.is_unlimited(u),
            left=quota.available(u), allowance=quota.monthly_allowance(profile),
            pending=quota.pending(u), resets_on=quota.next_reset(),
            life_cost=life["cost"] or 0, life_credits=life["cr"] or 0, life_hours=u.life_seconds / 3600,
            chart=_chart(weeks, [UsageEvent.Kind(s).label if s in UsageEvent.Kind.values else s
                                 for s in series]),
            usage=[{"e": e, "gpu": GPU_NAMES.get(e.instance_type, e.instance_type),
                    "hours": e.gpu_seconds / 3600} for e in events.order_by("-created_at")[:25]],
            units=Device.objects.filter(owner=u).order_by("-last_seen_at"),
            logins=LoginEvent.objects.filter(user=u)[:15],
            history=AdminAction.objects.filter(target=u).select_related("actor")[:30],
            redemptions=CouponRedemption.objects.filter(user=u).select_related("coupon")
            .order_by("-redeemed_at"),
            grantable_tiers=[(t.value, t.label) for t in GRANTABLE_TIERS],
            pro_default_until=(timezone.localdate() + timedelta(days=PRO_DEFAULT_DAYS)).isoformat(),
        )
        return ctx


class UserActionView(SuperuserMixin, View):
    """One POST endpoint per change to an account. Each writes the audit log."""

    def post(self, request, pk, action):
        target = get_object_or_404(User, pk=pk)
        profile, _ = UserProfile.objects.get_or_create(user=target)
        reason = request.POST.get("reason", "").strip()[:300]
        handler = getattr(self, f"do_{action}", None)
        if handler is None:
            raise Http404
        handler(request, target, profile, reason)
        return redirect("platform_admin:user", pk=target.pk)

    def do_grant(self, request, target, profile, reason):
        try:
            amount = Decimal(request.POST.get("credits", "")).quantize(quota.FOUR)
        except (InvalidOperation, ValueError):
            amount = Decimal(0)
        if amount == 0 or abs(amount) > 100000:
            messages.error(request, "Enter a number of credits (a negative number takes credits away).")
            return
        UserProfile.objects.filter(pk=profile.pk).update(
            granted_gpu_credits=F("granted_gpu_credits") + amount)
        verb = "Granted" if amount > 0 else "Removed"
        AdminAction.record(request.user, "grant_credits",
                           f"{verb} {abs(amount):g} credits {'to' if amount > 0 else 'from'} {target.username}",
                           target=target, reason=reason, credits=str(amount))
        messages.success(request, f"{verb} {abs(amount):g} credits.")

    def do_tier(self, request, target, profile, reason):
        tier = request.POST.get("tier", "")
        if tier not in [t.value for t in GRANTABLE_TIERS]:
            messages.error(request, "Choose Standard or Pro.")
            return
        until = parse_date(request.POST.get("until", "") or "")
        if tier == UserProfile.Tier.STANDARD:
            until = None
        elif until is None or until <= timezone.localdate():
            messages.error(request, "Pro needs an end date in the future.")
            return
        old = profile.get_tier_display()
        profile.tier = tier
        profile.tier_expires_at = (timezone.make_aware(timezone.datetime.combine(until, timezone.datetime.min.time()))
                                   if until else None)
        profile.save(update_fields=["tier", "tier_expires_at"])
        summary = f"Changed {target.username} from {old} to {profile.get_tier_display()}"
        if until:
            summary += f" until {until:%b %-d, %Y}"
        AdminAction.record(request.user, "change_tier", summary, target=target, reason=reason,
                           previous=old, tier=tier, until=until.isoformat() if until else None)
        messages.success(request, summary + ".")

    def do_limits(self, request, target, profile, reason):
        raw = request.POST.get("monthly_credits", "").strip()
        if raw == "":
            value = None
        else:
            try:
                value = Decimal(raw).quantize(quota.FOUR)
            except (InvalidOperation, ValueError):
                messages.error(request, "Monthly credits must be a number, or empty for the tier's default.")
                return
            if value < 0:
                messages.error(request, "Monthly credits can't be negative.")
                return
        profile.monthly_gpu_credits_override = value
        profile.save(update_fields=["monthly_gpu_credits_override"])
        summary = (f"Set {target.username}'s free monthly credits to {value:g}" if value is not None
                   else f"Reset {target.username}'s free monthly credits to the tier's default")
        AdminAction.record(request.user, "set_limits", summary, target=target, reason=reason,
                           monthly_credits=str(value) if value is not None else None)
        messages.success(request, summary + ".")

    def do_suspend(self, request, target, profile, reason):
        if target == request.user:
            messages.error(request, "You can't suspend your own account.")
            return
        target.is_active = False
        target.save(update_fields=["is_active"])
        _end_sessions(target)
        AdminAction.record(request.user, "suspend", f"Suspended {target.username}", target=target, reason=reason)
        messages.success(request, f"{target.username} is suspended: signed out everywhere, and can't sign in "
                                  "or use API keys. Their units still upload.")

    def do_reactivate(self, request, target, profile, reason):
        target.is_active = True
        target.save(update_fields=["is_active"])
        AdminAction.record(request.user, "reactivate", f"Reactivated {target.username}", target=target,
                           reason=reason)
        messages.success(request, f"{target.username} can sign in again.")

    def do_reset(self, request, target, profile, reason):
        if not target.email:
            messages.error(request, f"{target.username} has no email address.")
            return
        form = LoggedPasswordResetForm({"email": target.email})
        form.is_valid()
        form.save(request=request, use_https=request.is_secure(),
                  subject_template_name="accounts/password_reset_subject.txt",
                  email_template_name="accounts/password_reset_email.html",
                  domain_override=request.get_host())
        AdminAction.record(request.user, "password_reset", f"Sent {target.username} a password reset link",
                           target=target, reason=reason)
        messages.success(request, f"Sent a reset link to {target.email}. Email status shows whether it went.")


class CodesView(PlatformPage):
    template_name = "platform_admin/codes.html"
    section = "codes"

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        ctx.setdefault("form", CouponForm(initial={"code": _new_code(), "max_redemptions": 50}))
        ctx["codes"] = Coupon.objects.order_by("-created_at")
        ctx["upgrade_tiers"] = [(t.value, t.label) for t in GRANTABLE_TIERS if t != UserProfile.Tier.STANDARD]
        return ctx

    def post(self, request):
        form = CouponForm(request.POST)
        if form.is_valid():
            coupon = form.save()
            gives = (f"+{coupon.credits_amount} credits" if coupon.coupon_type == Coupon.CouponType.CREDITS
                     else f"{coupon.get_upgrade_tier_display()} for {coupon.upgrade_duration_days} days")
            AdminAction.record(request.user, "create_code", f"Created promo code {coupon.code} ({gives})",
                               code=coupon.code)
            messages.success(request, f"Created {coupon.code}.")
            return redirect("platform_admin:codes")
        return self.render_to_response(self.get_context_data(form=form))


class CodeDeactivateView(SuperuserMixin, View):
    def post(self, request, pk):
        coupon = get_object_or_404(Coupon, pk=pk)
        coupon.is_active = False
        coupon.save(update_fields=["is_active"])
        AdminAction.record(request.user, "deactivate_code", f"Deactivated promo code {coupon.code}",
                           code=coupon.code)
        messages.success(request, f"Deactivated {coupon.code}.")
        return redirect("platform_admin:codes")


class ActivityView(PlatformPage):
    template_name = "platform_admin/activity.html"
    section = "activity"

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        week_ago = timezone.now() - timedelta(days=7)
        ctx["audit"] = AdminAction.objects.select_related("actor", "target")[:100]
        ctx["logins"] = LoginEvent.objects.select_related("user")[:100]
        failed = (LoginEvent.objects.filter(success=False, created_at__gte=week_ago)
                  .values("username").annotate(n=Count("id")).order_by("-n")[:20])
        ctx["failed"] = failed
        top = stats.users_with_totals().filter(month_credits__gt=0).order_by("-month_credits")[:10]
        ctx["top"] = [{"user": u, "tier": stats.tier_label(u)} for u in top]
        out = [u for u in User.objects.filter(is_active=True).select_related("profile")
               if not quota.is_unlimited(u) and quota.available(u) <= 0]
        ctx["out_of_credits"] = out
        return ctx


# ── helpers ──────────────────────────────────────────────────────────────────

def _new_code():
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no 0/O, 1/I
    return "BEE-" + "".join(secrets.choice(alphabet) for _ in range(8))


def _end_sessions(user):
    """Sign a user out everywhere: delete every session that belongs to them."""
    from django.contrib.sessions.models import Session

    for s in Session.objects.filter(expire_date__gte=timezone.now()).iterator():
        if str(s.get_decoded().get("_auth_user_id")) == str(user.pk):
            s.delete()


# Series colours: distinct in lightness as well as hue.
CHART_COLORS = ["#1d4ed8", "#fb923c", "#93c5fd", "#7c2d12", "#9ca3af", "#0f766e"]


def _chart(weeks, labels, height=160):
    """Stacked bars as pixel heights, so the template draws them with no JS."""
    peak = max([w["total"] for w in weeks] + [0]) or 1
    colors = [CHART_COLORS[i % len(CHART_COLORS)] for i in range(len(labels))]
    return {
        "series": [{"label": label, "color": c} for label, c in zip(labels, colors)],
        "peak": peak,
        "weeks": [{"week": w["week"], "total": w["total"],
                   "bars": [{"h": round(v / peak * height), "color": colors[i], "label": labels[i], "v": v}
                            for i, v in enumerate(w["values"]) if v]}
                  for w in weeks],
        "height": height,
    }


def _recent_activity(n):
    """Admin actions, failed sign-ins and new accounts, newest first."""
    items = [{"at": a.created_at, "text": a.summary, "kind": "admin"}
             for a in AdminAction.objects.select_related("actor")[:n]]
    items += [{"at": e.created_at, "text": f"Failed sign-in for {e.username or e.user}"
               + (f" from {e.ip}" if e.ip else ""), "kind": "login"}
              for e in LoginEvent.objects.filter(success=False)[:n]]
    items += [{"at": u.date_joined, "text": f"New user: {u.username}", "kind": "user"}
              for u in User.objects.order_by("-date_joined")[:n]]
    return sorted(items, key=lambda x: x["at"], reverse=True)[:n]
