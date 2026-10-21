"""Every email attempt is logged with its outcome, and a failure never raises.

SES's sandbox rejected password-reset mail to real users in October 2026 while
the page still said "check your email" — nobody could see it. EmailLog is how
it shows now (memory/50 §11 step 0).
"""

from unittest import mock

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from apps.accounts import emails
from apps.accounts.models import EmailLog

User = get_user_model()
LOCMEM = "django.core.mail.backends.locmem.EmailBackend"


@override_settings(EMAIL_BACKEND=LOCMEM)
class SendTests(TestCase):
    def test_a_sent_email_is_logged(self):
        log = emails.send("test", "a@example.com", "Hi", "Body")
        self.assertEqual(log.status, EmailLog.Status.SENT)

    def test_a_rejected_email_is_logged_and_does_not_raise(self):
        with mock.patch("django.core.mail.EmailMultiAlternatives.send",
                        side_effect=Exception("MessageRejected: Email address is not verified")):
            log = emails.send("test", "a@example.com", "Hi", "Body")
        self.assertEqual(log.status, EmailLog.Status.FAILED)
        self.assertIn("not verified", log.error)

    def test_password_reset_failure_is_logged_but_the_page_looks_the_same(self):
        User.objects.create_user("alice", email="alice@example.com", password="x")
        with mock.patch("django.core.mail.EmailMultiAlternatives.send",
                        side_effect=Exception("MessageRejected")):
            r = self.client.post(reverse("accounts:password_reset"), {"email": "alice@example.com"})
        self.assertRedirects(r, reverse("accounts:password_reset_done"))
        log = EmailLog.objects.get()
        self.assertEqual((log.kind, log.status, log.to), ("password_reset", "failed", "alice@example.com"))


@override_settings(EMAIL_BACKEND=LOCMEM)
class EmailPageTests(TestCase):
    def test_only_superusers_see_the_email_page(self):
        u = User.objects.create_user("u", password="x")
        self.client.force_login(u)
        self.assertEqual(self.client.get(reverse("accounts:email_status")).status_code, 403)

    def test_the_test_button_sends_and_logs(self):
        admin = User.objects.create_superuser("root", email="root@example.com", password="x")
        self.client.force_login(admin)
        self.assertEqual(self.client.get(reverse("accounts:email_status")).status_code, 200)
        self.client.post(reverse("accounts:email_test"), {"to": "root@example.com"})
        self.assertEqual(EmailLog.objects.get().kind, "test")
