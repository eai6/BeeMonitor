"""The API had a reset-password endpoint that set any account's password from
an email address alone: no token, no login. It is gone; resets go through the
emailed-token flow at /accounts/password_reset/.
"""
from django.contrib.auth import get_user_model
from django.test import TestCase

User = get_user_model()


class ResetPasswordEndpointGoneTests(TestCase):
    def test_reset_password_by_email_alone_is_not_possible(self):
        user = User.objects.create_user("victim", email="v@example.com", password="original-pass-1")
        r = self.client.post("/api/v1/auth/reset-password/",
                             {"email": "v@example.com", "new_password": "pwned12345"},
                             content_type="application/json")
        self.assertIn(r.status_code, (404, 405))
        user.refresh_from_db()
        self.assertTrue(user.check_password("original-pass-1"))
