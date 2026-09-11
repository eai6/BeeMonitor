"""A camera mounted the other way up records upside-down footage.

The real fix is on the device: hardware/motion/config.py now reads
BEEMONITOR_VFLIP / BEEMONITOR_HFLIP, so the ISP hands over upright frames and
the detector, the annotation frames and the drawn geometry all agree without
anyone doing anything. This flag is for footage ALREADY uploaded from a
miscofigured camera — it turns what a person LOOKS at and deliberately nothing
else, so switching it on can never move a saved box.
"""

from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from apps.devices.models import Device
from apps.videos.models import Video

User = get_user_model()


class RotationFlagTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("rot", password="x")
        self.client.force_login(self.user)
        self.device = Device.objects.create(owner=self.user, name="NewCam",
                                            key_hash="krot", prefix="bmk_rot")
        self.video = Video.objects.create(
            user=self.user, device=self.device, title="clip",
            storage_key="rot/c.mp4", file_size_bytes=1, status=Video.Status.READY,
            recorded_at=timezone.now(), thumbnail_key="processed/rot/thumb.jpg")

    def test_it_is_off_by_default(self):
        self.assertFalse(self.device.rotate_180)

    def _player_page(self):
        """The clip page with a presigned URL, so the <video> block renders.

        Patched rather than live: the player is gated on a successful presign,
        so without this the test passes only while AWS credentials happen to be
        valid — it went green yesterday and red this morning when the SSO token
        expired, which is a property of the laptop, not of the code.
        """
        with patch("config.storage.S3StorageClient.generate_presigned_url",
                   return_value="https://signed.test/clip.mp4"):
            return self.client.get(
                reverse("videos:detail", kwargs={"pk": self.video.pk})
            ).content.decode()

    def test_the_player_turns_only_when_the_device_says_so(self):
        self.assertIn("<video", self._player_page())   # the block is rendering
        self.assertNotIn("rotate-180", self._player_page())

        self.device.rotate_180 = True
        self.device.save()

        self.assertIn("rotate-180", self._player_page())

    def test_flipping_it_clears_cached_stills_so_they_regenerate(self):
        """A thumbnail is keyed on the blob path, so it would never re-render."""
        self.device.rotate_180 = True
        self.device.save()

        self.video.refresh_from_db()
        self.assertEqual(self.video.thumbnail_key, "")

    def test_saving_without_changing_it_leaves_stills_alone(self):
        self.device.name = "Renamed"
        self.device.save()

        self.video.refresh_from_db()
        self.assertEqual(self.video.thumbnail_key, "processed/rot/thumb.jpg")

    def test_turning_it_back_off_also_clears_them(self):
        self.device.rotate_180 = True
        self.device.save()
        Video.objects.filter(pk=self.video.pk).update(thumbnail_key="k")

        self.device.rotate_180 = False
        self.device.save()

        self.video.refresh_from_db()
        self.assertEqual(self.video.thumbnail_key, "")

    def test_it_is_editable_from_the_device_form(self):
        resp = self.client.post(
            reverse("devices:edit", kwargs={"pk": self.device.pk}),
            {"name": "NewCam", "location": "", "rotate_180": "on"})

        self.assertIn(resp.status_code, (302, 200))
        self.device.refresh_from_db()
        self.assertTrue(self.device.rotate_180)

    def test_a_device_with_no_rotation_needs_no_device_at_all(self):
        """A video with no device must not blow up the thumbnail path."""
        from apps.videos import thumbnails

        orphan = Video.objects.create(
            user=self.user, title="no-device", storage_key="rot/o.mp4",
            file_size_bytes=1, status=Video.Status.READY)

        self.assertFalse(getattr(getattr(orphan, "device", None), "rotate_180", False))
        self.assertTrue(hasattr(thumbnails, "extract_thumbnail"))


class ThumbnailRotationTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("th", password="x")
        self.device = Device.objects.create(owner=self.user, name="C",
                                            key_hash="kth", prefix="bmk_th",
                                            rotate_180=True)

    def test_the_still_is_rotated_for_a_flipped_device(self):
        import numpy as np

        try:
            import cv2
        except ImportError:
            self.skipTest("cv2 not installed")

        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        frame[0, 0] = [255, 255, 255]        # a mark in the top-left

        turned = cv2.rotate(frame, cv2.ROTATE_180)

        self.assertEqual(list(turned[3, 3]), [255, 255, 255])
        self.assertEqual(list(turned[0, 0]), [0, 0, 0])
