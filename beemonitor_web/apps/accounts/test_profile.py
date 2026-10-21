"""Account → Profile: see your email, edit your name, change email and password."""

from django.contrib.auth import get_user_model
from django.core import mail
from django.test import Client, TestCase, override_settings
from django.urls import reverse

from apps.accounts.models import EmailLog, UserProfile

User = get_user_model()


@override_settings(EMAIL_BACKEND="django.core.mail.backends.locmem.EmailBackend")
class ProfileTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("bee", email="bee@example.com", password="old-passw0rd!")
        self.client.force_login(self.user)
        self.url = reverse("accounts:profile")

    def test_the_page_shows_username_and_email(self):
        r = self.client.get(self.url)
        self.assertContains(r, "bee@example.com")
        self.assertContains(r, "It can't be changed")

    def test_name_and_organization_are_saved(self):
        self.client.post(self.url, {"action": "profile", "first_name": "Ada",
                                    "last_name": "B", "organization": "Pollinator Lab"})
        self.user.refresh_from_db()
        self.assertEqual((self.user.first_name, self.user.last_name), ("Ada", "B"))
        self.assertEqual(UserProfile.objects.get(user=self.user).organization, "Pollinator Lab")

    def test_email_change_needs_the_password_and_tells_the_old_address(self):
        r = self.client.post(self.url, {"action": "email", "new_email": "new@example.com",
                                        "current_password": "wrong"})
        self.assertContains(r, "your current password.")
        self.user.refresh_from_db()
        self.assertEqual(self.user.email, "bee@example.com")

        self.client.post(self.url, {"action": "email", "new_email": "new@example.com",
                                    "current_password": "old-passw0rd!"})
        self.user.refresh_from_db()
        self.assertEqual(self.user.email, "new@example.com")
        self.assertEqual(mail.outbox[0].to, ["bee@example.com"])
        self.assertEqual(EmailLog.objects.get().kind, "email_changed")

    def test_an_email_used_by_another_account_is_refused(self):
        User.objects.create_user("other", email="Taken@example.com", password="x")
        r = self.client.post(self.url, {"action": "email", "new_email": "taken@example.com",
                                        "current_password": "old-passw0rd!"})
        self.assertContains(r, "Another account already uses this email")

    def test_password_change_keeps_this_session_and_ends_others(self):
        other = Client()
        other.force_login(self.user)
        self.client.post(self.url, {"action": "password", "old_password": "old-passw0rd!",
                                    "new_password1": "n3w-Passw0rd-long", "new_password2": "n3w-Passw0rd-long"})
        self.user.refresh_from_db()
        self.assertTrue(self.user.check_password("n3w-Passw0rd-long"))
        self.assertEqual(self.client.get(self.url).status_code, 200)        # still signed in here
        self.assertEqual(other.get(self.url).status_code, 302)              # signed out elsewhere
        self.assertEqual(mail.outbox[-1].subject, "Your BeeMonitor password was changed")

    def test_the_username_in_the_nav_opens_the_profile(self):
        r = self.client.get(reverse("accounts:usage"))
        self.assertContains(r, f'href="{self.url}"')
