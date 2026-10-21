"""/platform/: only superusers see it, and every change to an account is audited.

memory/50 §8. The pages render with real data; the actions change what the
credit gate (apps/accounts/quota.py) reads; suspension really locks the account.
"""

from datetime import timedelta
from decimal import Decimal

from django.contrib.auth import get_user_model
from django.core import mail
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from apps.accounts import quota
from apps.accounts.models import (AdminAction, APIKey, Coupon, LoginEvent, UsageEvent,
                                  UserProfile)

User = get_user_model()
PAGES = ["platform_admin:overview", "platform_admin:users", "platform_admin:codes",
         "platform_admin:activity"]


def usage(user, credits, kind="analysis", source="1"):
    """Charge ``credits`` of T4 time the way finished work is charged."""
    return quota.charge(user, kind, float(credits) * 3600, quota.VIDEO_INSTANCE, "job", source,
                        units=600)


class AccessTests(TestCase):
    def setUp(self):
        self.member = User.objects.create_user("member", password="pw-12345678")
        self.staff = User.objects.create_user("staffer", password="pw-12345678", is_staff=True)

    def test_signed_out_people_are_sent_to_sign_in(self):
        for name in PAGES:
            r = self.client.get(reverse(name))
            self.assertEqual(r.status_code, 302, name)
            self.assertIn("/accounts/login/", r["Location"])

    def test_members_and_staff_get_not_found_everywhere(self):
        target = User.objects.create_user("t")
        for user in (self.member, self.staff):
            self.client.force_login(user)
            for name in PAGES:
                self.assertEqual(self.client.get(reverse(name)).status_code, 404, (user, name))
            self.assertEqual(self.client.get(reverse("platform_admin:user", args=[target.pk])).status_code, 404)
            r = self.client.post(reverse("platform_admin:user_action", args=[target.pk, "grant"]),
                                 {"credits": "500"})
            self.assertEqual(r.status_code, 404)
        target.profile.refresh_from_db()
        self.assertEqual(target.profile.granted_gpu_credits, 0)

    def test_the_nav_link_is_only_for_superusers(self):
        self.client.force_login(self.member)
        self.assertNotContains(self.client.get(reverse("accounts:profile")), "/platform/")
        admin = User.objects.create_superuser("boss", "boss@example.com", "pw-12345678")
        self.client.force_login(admin)
        self.assertContains(self.client.get(reverse("accounts:profile")), "/platform/")


class PagesTests(TestCase):
    def setUp(self):
        self.admin = User.objects.create_superuser("boss", "boss@example.com", "pw-12345678")
        self.alice = User.objects.create_user("alice", email="alice@lab.edu", password="pw-12345678")
        self.bob = User.objects.create_user("bob", email="bob@lab.edu")
        usage(self.alice, "3.5", source="a1")
        usage(self.alice, "1.0", kind="sampling", source="a2")
        usage(self.bob, "12", source="b1")  # past the 10 free credits
        self.client.force_login(self.admin)

    def test_every_page_renders(self):
        for name in PAGES:
            self.assertEqual(self.client.get(reverse(name)).status_code, 200, name)
        r = self.client.get(reverse("platform_admin:user", args=[self.alice.pk]))
        self.assertContains(r, "alice@lab.edu")
        self.assertContains(r, "4.5")  # credits this month

    def test_overview_totals_come_from_usage_events(self):
        o = self.client.get(reverse("platform_admin:overview")).context["o"]
        self.assertEqual(o["users"], 3)
        self.assertEqual(o["credits_month"], Decimal("16.5"))

    def test_users_table_sorts_searches_and_flags_who_is_out_of_credits(self):
        rows = self.client.get(reverse("platform_admin:users")).context["page"].object_list
        self.assertEqual([r["user"].username for r in rows][:2], ["bob", "alice"])
        status = {r["user"].username: r["status"][0] for r in rows}
        self.assertEqual(status["bob"], "Out of credits")
        rows = self.client.get(reverse("platform_admin:users"), {"q": "alice"}).context["page"].object_list
        self.assertEqual([r["user"].username for r in rows], ["alice"])
        rows = self.client.get(reverse("platform_admin:users"),
                               {"status": "out_of_credits"}).context["page"].object_list
        self.assertEqual([r["user"].username for r in rows], ["bob"])

    def test_a_usage_join_does_not_multiply_totals(self):
        # Several clips and several charges: subqueries, not joins, so 4.5 stays 4.5.
        from apps.videos.models import Video
        for i in range(3):
            Video.objects.create(user=self.alice, title=f"c{i}.mp4",
                                 file_size_bytes=1000, storage_key=f"raw/{i}.mp4")
        rows = self.client.get(reverse("platform_admin:users"), {"q": "alice"}).context["page"].object_list
        u = rows[0]["user"]
        self.assertEqual(u.month_credits, Decimal("4.5"))
        self.assertEqual((u.clips, u.storage_bytes), (3, 3000))

    def test_csv_export(self):
        r = self.client.get(reverse("platform_admin:users"), {"format": "csv"})
        self.assertEqual(r["Content-Type"], "text/csv")
        body = r.content.decode()
        self.assertIn("alice,alice@lab.edu", body)
        self.assertEqual(len(body.strip().splitlines()), 4)


@override_settings(EMAIL_BACKEND="django.core.mail.backends.locmem.EmailBackend")
class ActionTests(TestCase):
    def setUp(self):
        self.admin = User.objects.create_superuser("boss", "boss@example.com", "pw-12345678")
        self.alice = User.objects.create_user("alice", email="alice@lab.edu", password="pw-12345678")
        self.client.force_login(self.admin)

    def act(self, action, **data):
        return self.client.post(reverse("platform_admin:user_action", args=[self.alice.pk, action]), data)

    def last_log(self):
        return AdminAction.objects.first()

    def test_grant_credits_adds_to_the_balance_and_is_audited(self):
        self.act("grant", credits="50", reason="Field season")
        self.assertEqual(quota.available(self.alice), Decimal(60))
        log = self.last_log()
        self.assertEqual((log.actor, log.target, log.action, log.reason),
                         (self.admin, self.alice, "grant_credits", "Field season"))
        self.act("grant", credits="-20")
        self.assertEqual(quota.available(self.alice), Decimal(40))

    def test_pro_with_an_end_date_then_back_to_standard_when_it_ends(self):
        until = timezone.localdate() + timedelta(days=365)
        self.act("tier", tier="pro", until=until.isoformat())
        self.assertTrue(quota.is_unlimited(self.alice))
        UserProfile.objects.filter(user=self.alice).update(tier_expires_at=timezone.now() - timedelta(minutes=1))
        self.assertFalse(quota.is_unlimited(self.alice))
        profile = UserProfile.objects.get(user=self.alice)
        self.assertEqual((profile.tier, profile.tier_expires_at), ("standard", None))
        self.assertEqual(self.last_log().action, "tier_expired")

    def test_pro_needs_a_future_end_date(self):
        self.act("tier", tier="pro", until="")
        self.assertEqual(UserProfile.objects.get(user=self.alice).tier, "free")
        self.act("tier", tier="enterprise", until="2030-01-01")
        self.assertEqual(UserProfile.objects.get(user=self.alice).tier, "free")

    def test_custom_monthly_credits_replace_the_tiers(self):
        self.act("limits", monthly_credits="100")
        self.assertEqual(quota.available(self.alice), Decimal(100))
        self.act("limits", monthly_credits="")
        self.assertEqual(quota.available(self.alice), Decimal(10))

    def test_suspend_signs_out_blocks_sign_in_and_api_keys(self):
        other = self.client_class()
        other.force_login(self.alice)
        _, raw = APIKey.create_key(self.alice, "k", "live")
        self.act("suspend", reason="abuse")
        self.alice.refresh_from_db()
        self.assertFalse(self.alice.is_active)
        self.assertEqual(other.get(reverse("accounts:profile")).status_code, 302)  # session ended
        self.assertFalse(self.client_class().login(username="alice", password="pw-12345678"))
        r = self.client_class().get("/api/v1/jobs/", HTTP_AUTHORIZATION=f"Bearer {raw}")
        self.assertIn(r.status_code, (401, 403))
        self.assertIn("suspended", r.json()["detail"])
        self.act("reactivate")
        self.alice.refresh_from_db()
        self.assertTrue(self.alice.is_active)

    def test_an_admin_cannot_suspend_themselves(self):
        self.client.post(reverse("platform_admin:user_action", args=[self.admin.pk, "suspend"]))
        self.admin.refresh_from_db()
        self.assertTrue(self.admin.is_active)

    def test_send_password_reset_emails_a_link(self):
        self.act("reset")
        self.assertEqual(len(mail.outbox), 1)
        self.assertIn("/accounts/reset/", mail.outbox[0].body)
        self.assertEqual(self.last_log().action, "password_reset")

    def test_unknown_action_is_not_found(self):
        self.assertEqual(self.act("delete").status_code, 404)


class CodesTests(TestCase):
    def setUp(self):
        self.admin = User.objects.create_superuser("boss", "boss@example.com", "pw-12345678")
        self.client.force_login(self.admin)

    def test_create_and_deactivate_are_audited(self):
        self.client.post(reverse("platform_admin:codes"), {
            "code": "FIELD-2026", "coupon_type": "credits", "credits_amount": "20",
            "upgrade_tier": "pro", "upgrade_duration_days": "30", "max_redemptions": "50",
            "is_active": "on", "note": "workshop"})
        c = Coupon.objects.get(code="FIELD-2026")
        self.assertEqual((c.credits_amount, c.upgrade_tier, c.note), (20, "", "workshop"))
        self.client.post(reverse("platform_admin:code_deactivate", args=[c.pk]))
        c.refresh_from_db()
        self.assertFalse(c.is_active)
        self.assertEqual(list(AdminAction.objects.values_list("action", flat=True)),
                         ["deactivate_code", "create_code"])

    def test_a_plan_trial_code_ends(self):
        Coupon.objects.create(code="PROTRIAL", coupon_type="tier_upgrade", upgrade_tier="pro",
                              upgrade_duration_days=30)
        alice = User.objects.create_user("alice")
        self.client.force_login(alice)
        self.client.post(reverse("accounts:redeem_coupon"), {"coupon_code": "protrial"})
        profile = UserProfile.objects.get(user=alice)
        self.assertEqual(profile.tier, "pro")
        self.assertAlmostEqual(profile.tier_expires_at, timezone.now() + timedelta(days=30),
                               delta=timedelta(minutes=1))

    def test_the_old_staff_coupon_page_is_gone(self):
        staff = User.objects.create_user("staffer", is_staff=True)
        self.client.force_login(staff)
        self.assertEqual(self.client.get("/accounts/coupons/").status_code, 403)


class TrackingTests(TestCase):
    def setUp(self):
        self.alice = User.objects.create_user("alice", password="pw-12345678")

    def test_sign_ins_and_failures_are_recorded(self):
        self.client.post(reverse("accounts:login"), {"username": "alice", "password": "wrong"},
                         HTTP_USER_AGENT="Firefox", REMOTE_ADDR="10.0.0.9")
        self.client.post(reverse("accounts:login"), {"username": "alice", "password": "pw-12345678"})
        self.client_class().post(reverse("accounts:login"), {"username": "nobody", "password": "x"})
        events = list(LoginEvent.objects.order_by("created_at", "id"))
        self.assertEqual([(e.user, e.success) for e in events],
                         [(self.alice, False), (self.alice, True), (None, False)])
        self.assertEqual((events[0].ip, events[0].user_agent), ("10.0.0.9", "Firefox"))
        self.assertEqual(events[2].username, "nobody")

    def test_api_sign_in_is_recorded(self):
        self.client.post("/api/v1/auth/login/", {"username": "alice", "password": "pw-12345678"})
        e = LoginEvent.objects.get()
        self.assertEqual((e.user, e.success, e.method), (self.alice, True, "api"))

    def test_old_sign_ins_are_pruned(self):
        LoginEvent.objects.create(user=self.alice, success=True,
                                  created_at=timezone.now() - timedelta(days=91))
        self.client.post(reverse("accounts:login"), {"username": "alice", "password": "pw-12345678"})
        self.assertEqual(LoginEvent.objects.count(), 1)

    def test_last_seen_is_written_at_most_every_five_minutes(self):
        self.client.force_login(self.alice)
        self.client.get(reverse("accounts:profile"))
        first = UserProfile.objects.get(user=self.alice).last_seen_at
        self.assertIsNotNone(first)
        self.client.get(reverse("accounts:profile"))
        self.assertEqual(UserProfile.objects.get(user=self.alice).last_seen_at, first)

    def test_last_seen_counts_api_key_calls(self):
        _, raw = APIKey.create_key(self.alice, "k", "live")
        self.client.get("/api/v1/jobs/", HTTP_AUTHORIZATION=f"Bearer {raw}")
        self.assertIsNotNone(UserProfile.objects.get(user=self.alice).last_seen_at)
