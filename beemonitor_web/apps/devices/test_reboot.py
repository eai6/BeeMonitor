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


class BoardTests(TestCase):
    """What the unit says it runs on, and light mode (memory/56)."""

    def setUp(self):
        self.owner = User.objects.create_user("bd", password="x")
        self.device, _ = Device.create_with_key(self.owner, "Amber Osmia")
        self.client.force_login(self.owner)

    def _page(self, metrics):
        self.device.heartbeats.create(metrics=metrics)
        return self.client.get(reverse("devices:detail", args=[self.device.pk])).content.decode()

    def test_a_light_unit_says_so(self):
        html = self._page({"board": "Raspberry Pi Zero 2 W", "mem_mb": 416, "profile": "lite"})
        self.assertIn("Raspberry Pi Zero 2 W", html)
        self.assertIn("light mode", html)

    def test_a_full_unit_shows_only_its_board(self):
        html = self._page({"board": "Raspberry Pi 4 Model B", "mem_mb": 3793, "profile": "full"})
        self.assertIn("Raspberry Pi 4 Model B", html)
        self.assertNotIn("light mode", html)

    def test_an_older_unit_that_reports_no_board_shows_nothing(self):
        self.assertNotIn("light mode", self._page({"cpu_temp_c": 50}))
