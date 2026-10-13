"""Rebooting a unit from Advanced settings, and the panels hidden from it."""
from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from apps.devices.models import Device

User = get_user_model()


class RebootTests(TestCase):
    def setUp(self):
        self.owner = User.objects.create_user("rb", password="x")
        self.device, _ = Device.create_with_key(self.owner, "Amber Osmia")
        self.client.force_login(self.owner)

    def test_reboot_is_queued_for_the_next_check_in(self):
        r = self.client.post(reverse("devices:reboot", args=[self.device.pk]))

        self.assertEqual(r.status_code, 302)
        self.device.refresh_from_db()
        self.assertEqual(self.device.pending_command, "reboot")

    def test_someone_else_cannot_reboot_it(self):
        other = User.objects.create_user("nope", password="x")
        self.client.force_login(other)

        self.client.post(reverse("devices:reboot", args=[self.device.pk]))

        self.device.refresh_from_db()
        self.assertEqual(self.device.pending_command, "")

    def test_advanced_settings_offers_reboot_and_hides_cellular_access(self):
        html = self.client.get(reverse("devices:detail", args=[self.device.pk])).content.decode()

        self.assertIn(reverse("devices:reboot", args=[self.device.pk]), html)
        self.assertLess(html.index("<details"), html.index("Reboot device"))
        self.assertNotIn("Cellular access", html)
        self.assertNotIn("Open for debug", html)
