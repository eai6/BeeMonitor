"""ROI + reference-object layout history: every save is kept, a version is in
use from the heartbeat that delivers it, and analysis of a clip uses the layout
that was in use when it was recorded."""

import json
from datetime import timedelta

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from apps.videos.models import Video
from .layouts import layout_for_video
from .models import Device, DeviceLayoutVersion

User = get_user_model()

ROI_A = {"box": [0.1, 0.1, 0.5, 0.5]}
ROI_B = {"box": [0.2, 0.2, 0.9, 0.9]}
NESTS = [{"id": 1, "box": [0.2, 0.2, 0.3, 0.3]}]


class LayoutHistoryTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("lee", password="x")
        self.device, self.raw_key = Device.create_with_key(self.user, "hive")
        self.device.tz_offset_min = 0
        self.device.save()
        self.client.force_login(self.user)

    def _save(self, roi, nests):
        r = self.client.post(reverse("devices:roi_editor", args=[self.device.pk]),
                             data=json.dumps({"roi": roi, "nests": nests}),
                             content_type="application/json")
        self.assertEqual(r.status_code, 200)
        return r.json()

    def _beat(self):
        self.client.post(reverse("devices-heartbeat"), data={"metrics": "{}"},
                         HTTP_AUTHORIZATION=f"Bearer {self.raw_key}")

    def _clip(self, at):
        return Video.objects.create(user=self.user, device=self.device, title="c",
                                    storage_key="k.mp4", file_size_bytes=1,
                                    status=Video.Status.READY, recorded_at=at, metadata={})

    def _set_applied(self, number, at):
        DeviceLayoutVersion.objects.filter(device=self.device, number=number).update(applied_at=at)

    def test_each_change_is_a_new_version_and_repeats_are_not(self):
        self._save(ROI_A, NESTS)
        self._save(ROI_A, NESTS)  # unchanged
        d = self._save(ROI_B, NESTS)
        self.assertEqual(list(DeviceLayoutVersion.objects.filter(device=self.device)
                              .values_list("number", flat=True)), [2, 1])
        self.assertEqual([v["number"] for v in d["versions"]], [2, 1])
        self.assertEqual(d["versions"][0]["state"], "waiting")

    def test_a_version_is_in_use_from_the_beat_that_delivers_it(self):
        self._save(ROI_A, NESTS)
        v = DeviceLayoutVersion.objects.get(device=self.device)
        self.assertIsNone(v.applied_at)
        self._beat()
        v.refresh_from_db()
        self.assertIsNotNone(v.applied_at)
        applied = v.applied_at
        self._beat()  # a later beat doesn't move it
        v.refresh_from_db()
        self.assertEqual(v.applied_at, applied)

    def test_clips_use_the_layout_in_use_when_they_were_recorded(self):
        now = timezone.now()
        self._save(ROI_A, NESTS)
        self._set_applied(1, now - timedelta(days=10))
        self._save(ROI_B, [])
        self._set_applied(2, now - timedelta(days=2))

        before_any = layout_for_video(self._clip(now - timedelta(days=20)))
        old = layout_for_video(self._clip(now - timedelta(days=5)))
        new = layout_for_video(self._clip(now - timedelta(days=1)))
        self.assertIsNone(before_any["roi_override"])
        self.assertEqual(old["roi_override"], ROI_A["box"])
        self.assertEqual(old["nest_layout"][0]["id"], 1)
        self.assertEqual(new["roi_override"], ROI_B["box"])
        self.assertEqual(new["nest_layout"], [])

    def test_the_device_clock_offset_is_taken_into_account(self):
        now = timezone.now()
        self._save(ROI_A, NESTS)
        self._set_applied(1, now - timedelta(hours=3))
        self.device.tz_offset_min = -300  # Pi clock 5h behind UTC
        self.device.save()
        # Stored 4h ago on the Pi's wall clock = 1h ago in true UTC -> v1 in use.
        clip = self._clip(now - timedelta(hours=4))
        clip.device.refresh_from_db()
        self.assertEqual(layout_for_video(clip)["roi_override"], ROI_A["box"])

    def test_an_unsent_version_is_not_used_and_shows_as_never_used_once_replaced(self):
        now = timezone.now()
        self._save(ROI_A, NESTS)
        self._set_applied(1, now - timedelta(days=3))
        self._save(ROI_B, NESTS)       # never delivered...
        d = self._save({"box": [0.3, 0.3, 0.6, 0.6]}, NESTS)  # ...replaced first
        self._beat()
        clip = self._clip(now - timedelta(days=1))
        self.assertEqual(layout_for_video(clip)["roi_override"], ROI_A["box"])
        states = {v["number"]: v["state"] for v in self._save({"box": [0.3, 0.3, 0.6, 0.6]}, NESTS)["versions"]}
        self.assertEqual(states, {3: "in_use", 2: "never_used", 1: "used"})
        self.assertEqual(d["versions"][1]["state"], "never_used")

    def test_devices_without_history_use_their_current_layout(self):
        Device.objects.filter(pk=self.device.pk).update(roi_override=ROI_A["box"], nest_layout=NESTS)
        self.device.refresh_from_db()
        clip = self._clip(timezone.now())
        clip.device.refresh_from_db()
        self.assertEqual(layout_for_video(clip)["roi_override"], ROI_A["box"])

    def test_analysis_config_uses_the_clip_era_layout(self):
        from apps.analysis.views import _video_job_config
        now = timezone.now()
        self._save(ROI_A, NESTS)
        self._set_applied(1, now - timedelta(days=10))
        self._save(ROI_B, [])
        self._set_applied(2, now - timedelta(days=2))
        cfg = _video_job_config({}, self._clip(now - timedelta(days=5)), use_device_roi=True)
        self.assertEqual(cfg["hotel_roi"], ROI_A["box"])
        self.assertEqual(cfg["nest_layout"][0]["id"], 1)

    def test_editor_lists_the_history(self):
        self._save(ROI_A, NESTS)
        html = self.client.get(reverse("devices:roi_editor", args=[self.device.pk])).content.decode()
        self.assertIn("Layout history", html)
        self.assertIn('id="layout-versions"', html)
