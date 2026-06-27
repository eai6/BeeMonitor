"""Password-reset flow tests.

Uses the in-memory (locmem) email backend so no SES/AWS is touched. Covers the
three things that matter: a known email triggers exactly one email with a working
link, an unknown email is indistinguishable (no email, no error — the
enumeration guard), and a valid link actually lets the user set a new password.
"""

from django.contrib.auth import get_user_model
from django.core import mail
from django.test import TestCase, override_settings
from django.urls import reverse

User = get_user_model()


@override_settings(EMAIL_BACKEND="django.core.mail.backends.locmem.EmailBackend")
class PasswordResetFlowTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username="alice", email="alice@example.com", password="old-passw0rd",
        )

    def test_known_email_sends_one_reset_message(self):
        resp = self.client.post(
            reverse("accounts:password_reset"), {"email": "alice@example.com"},
        )
        self.assertRedirects(resp, reverse("accounts:password_reset_done"))
        self.assertEqual(len(mail.outbox), 1)
        body = mail.outbox[0].body
        self.assertIn("/reset/", body)
        self.assertEqual(mail.outbox[0].subject, "Reset your BeeMonitor password")

    def test_unknown_email_does_not_reveal_existence(self):
        resp = self.client.post(
            reverse("accounts:password_reset"), {"email": "nobody@example.com"},
        )
        # Same destination as a real address...
        self.assertRedirects(resp, reverse("accounts:password_reset_done"))
        # ...but no mail is sent and nothing errors.
        self.assertEqual(len(mail.outbox), 0)

    def test_reset_link_lets_user_set_new_password(self):
        # Request the reset, then follow the link from the email.
        self.client.post(
            reverse("accounts:password_reset"), {"email": "alice@example.com"},
        )
        # Extract uidb64/token from the emailed URL.
        body = mail.outbox[0].body
        reset_path = next(
            line.split("://", 1)[1].split("/", 1)[1]
            for line in body.splitlines()
            if "/reset/" in line
        )
        reset_url = "/" + reset_path

        # First GET redirects to the "set-password" URL (token moved into session).
        resp = self.client.get(reset_url)
        self.assertEqual(resp.status_code, 302)
        set_password_url = resp.url

        resp = self.client.post(
            set_password_url,
            {"new_password1": "brand-new-pass99", "new_password2": "brand-new-pass99"},
        )
        self.assertRedirects(resp, reverse("accounts:password_reset_complete"))

        self.user.refresh_from_db()
        self.assertTrue(self.user.check_password("brand-new-pass99"))
        self.assertTrue(
            self.client.login(username="alice", password="brand-new-pass99")
        )
