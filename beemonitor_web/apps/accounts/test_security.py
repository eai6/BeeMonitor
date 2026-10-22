"""Security fixes from the October 2026 audit (accounts side).

Sign-in lockout, the reset-email limit and its fixed domain, the promo-code
limit, POST-only logout, the credit hold that stops parallel launches from
spending one balance twice, script-safe JSON in templates, and email failures
that don't put the address in the application log.
"""

from decimal import Decimal
from unittest import mock

from django.contrib.auth import get_user_model
from django.core import mail
from django.template import Context, Template
from django.test import TestCase, override_settings
from django.urls import reverse

from apps.accounts import quota
from apps.accounts.models import Coupon, CreditHold, LoginEvent
from apps.videos.models import Video

User = get_user_model()
LOCMEM = "django.core.mail.backends.locmem.EmailBackend"


class LoginLockoutTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("alice", email="a@example.com", password="right-pass-123")

    def fail(self, n, **extra):
        for _ in range(n):
            self.client.post(reverse("accounts:login"), {"username": "alice", "password": "wrong"}, **extra)

    def test_ten_failures_lock_the_username_even_with_the_right_password(self):
        self.fail(10)
        self.assertFalse(self.client.login(username="alice", password="right-pass-123"))
        r = self.client.post(reverse("accounts:login"), {"username": "alice", "password": "right-pass-123"})
        self.assertNotIn("_auth_user_id", self.client.session)
        self.assertEqual(r.status_code, 200)

    def test_nine_failures_still_let_the_owner_in(self):
        self.fail(9)
        self.assertTrue(self.client.login(username="alice", password="right-pass-123"))

    def test_the_api_login_is_locked_too(self):
        self.fail(10)
        r = self.client.post("/api/v1/auth/login/", {"username": "alice", "password": "right-pass-123"})
        self.assertEqual(r.status_code, 401)

    def test_one_address_spraying_many_usernames_is_locked(self):
        for i in range(30):
            LoginEvent.objects.create(username=f"u{i}", success=False, ip="203.0.113.7")
        self.client.post(reverse("accounts:login"), {"username": "alice", "password": "right-pass-123"},
                         REMOTE_ADDR="203.0.113.7")
        self.assertNotIn("_auth_user_id", self.client.session)
        self.client.post(reverse("accounts:login"), {"username": "alice", "password": "right-pass-123"},
                         REMOTE_ADDR="198.51.100.2")
        self.assertIn("_auth_user_id", self.client.session)


@override_settings(EMAIL_BACKEND=LOCMEM, SITE_DOMAIN="beemonitor.example.org")
class PasswordResetTests(TestCase):
    def setUp(self):
        User.objects.create_user("alice", email="a@example.com", password="x")

    def test_the_link_uses_the_site_domain_not_the_host_header(self):
        self.client.post(reverse("accounts:password_reset"), {"email": "a@example.com"},
                         HTTP_HOST="evil.example.com")
        self.assertIn("https://beemonitor.example.org/accounts/reset/", mail.outbox[0].body.replace("http://", "https://"))
        self.assertNotIn("evil.example.com", mail.outbox[0].body)

    def test_at_most_five_reset_emails_a_day_per_address(self):
        for _ in range(8):
            self.client.post(reverse("accounts:password_reset"), {"email": "A@example.com"})
        self.assertEqual(len(mail.outbox), 5)


class CouponLimitTests(TestCase):
    def test_ten_promo_code_tries_an_hour(self):
        user = User.objects.create_user("bee")
        self.client.force_login(user)
        for i in range(10):
            self.client.post(reverse("accounts:redeem_coupon"), {"coupon_code": f"GUESS{i}"})
        Coupon.objects.create(code="REAL", coupon_type="credits", credits_amount=20)
        self.client.post(reverse("accounts:redeem_coupon"), {"coupon_code": "REAL"})
        self.assertEqual(user.profile.__class__.objects.get(user=user).granted_gpu_credits, 25)


class LogoutTests(TestCase):
    def test_logout_needs_a_post(self):
        user = User.objects.create_user("bee")
        self.client.force_login(user)
        self.assertEqual(self.client.get(reverse("accounts:logout")).status_code, 405)
        self.assertIn("_auth_user_id", self.client.session)
        self.client.post(reverse("accounts:logout"))
        self.assertNotIn("_auth_user_id", self.client.session)


class CreditHoldTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("bee")
        # Four 5-hour clips: ~5.35 credits each, 21.4 together, 25 available.
        self.clips = [Video.objects.create(user=self.user, title=f"c{i}", storage_key=f"k/{i}",
                                           file_size_bytes=1, duration_seconds=18000) for i in range(4)]

    def test_a_launch_holds_what_it_was_approved_for(self):
        quota.check(self.user, [quota.video_work(v) for v in self.clips])
        self.assertEqual(CreditHold.objects.filter(user=self.user).count(), 1)
        # A second launch at the same moment sees the first one's credits as spent.
        with self.assertRaises(quota.QuotaExceeded):
            quota.check(self.user, [quota.video_work(v) for v in self.clips])

    def test_holds_expire(self):
        quota.check(self.user, [quota.video_work(v) for v in self.clips])
        CreditHold.objects.update(expires_at="2000-01-01T00:00:00Z")
        quota.check(self.user, [quota.video_work(v) for v in self.clips])
        self.assertEqual(CreditHold.objects.count(), 1)

    def test_unlimited_accounts_hold_nothing(self):
        self.user.is_superuser = True
        self.user.save()
        quota.check(self.user, Decimal(500))
        self.assertFalse(CreditHold.objects.exists())


class ScriptJsonTests(TestCase):
    def test_a_name_cannot_close_the_script(self):
        import json
        out = Template("<script>var p = {{ data|js }};</script>").render(
            Context({"data": json.dumps([{"name": "</script><script>alert(1)</script>"}])}))
        self.assertEqual(out.count("</script>"), 1)
        self.assertIn("\\u003C/script\\u003E", out)


@override_settings(EMAIL_BACKEND=LOCMEM)
class EmailLogTests(TestCase):
    def test_a_failed_email_keeps_the_address_out_of_the_log(self):
        from apps.accounts import emails
        with mock.patch("django.core.mail.EmailMultiAlternatives.send",
                        side_effect=Exception("MessageRejected: secret@example.com is not verified")), \
                self.assertLogs("apps.accounts.emails", level="ERROR") as logs:
            log = emails.send("test", "secret@example.com", "Hi", "Body")
        self.assertNotIn("secret@example.com", "\n".join(logs.output))
        self.assertIn("secret@example.com", log.error)
