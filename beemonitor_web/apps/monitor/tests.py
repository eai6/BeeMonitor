"""Tests for the activity pages + frame-ingest wiring (no S3 needed)."""

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from apps.devices.models import Device

from .models import Activity, ActivityFrame


class ActivityPageTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        User = get_user_model()
        cls.user = User.objects.create_user(username="alice", password="pw12345!")
        cls.other = User.objects.create_user(username="bob", password="pw12345!")
        cls.device, _ = Device.create_with_key(owner=cls.user, name="hive-1")
        cls.activity = Activity.objects.create(
            device=cls.device, activity_uid="evt-1", started_at=timezone.now(),
            lat=40.8, lon=-77.8, peak_motion=1234,
        )
        # storage_key points nowhere real; the view presigns best-effort (None on fail).
        ActivityFrame.objects.create(activity=cls.activity, storage_key="x/y/z.jpg",
                                     kind="crop", motion_score=999)

    def test_list_requires_login(self):
        resp = self.client.get(reverse("monitor:activity_list"))
        self.assertEqual(resp.status_code, 302)  # redirect to login

    def test_list_and_detail_render(self):
        self.client.force_login(self.user)
        resp = self.client.get(reverse("monitor:activity_list"))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "hive-1")
        resp = self.client.get(reverse("monitor:activity_detail", args=[self.activity.pk]))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Analysis pending")  # no detections yet

    def test_detail_access_scoped_to_device(self):
        self.client.force_login(self.other)  # no access to alice's device
        resp = self.client.get(reverse("monitor:activity_detail", args=[self.activity.pk]))
        self.assertEqual(resp.status_code, 404)

    def test_device_filter(self):
        self.client.force_login(self.user)
        resp = self.client.get(reverse("monitor:activity_list"), {"device": self.device.pk})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(list(resp.context["activities"]), [self.activity])


class FrameIngestTests(TestCase):
    def setUp(self):
        User = get_user_model()
        self.user = get_user_model().objects.create_user(username="carol", password="pw12345!")
        self.device, self.raw_key = Device.create_with_key(owner=self.user, name="hive-2")

    def test_requires_device_auth(self):
        resp = self.client.post("/api/v1/devices/frames", {})
        self.assertEqual(resp.status_code, 401)

    def test_missing_activity_uid_is_400(self):
        resp = self.client.post(
            "/api/v1/devices/frames", {"meta": "{}"},
            HTTP_AUTHORIZATION=f"Bearer {self.raw_key}",
        )
        self.assertEqual(resp.status_code, 400)
