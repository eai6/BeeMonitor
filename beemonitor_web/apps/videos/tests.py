"""Device-share access tests for videos and the ecological data derived from them.

The contract: a user with a DeviceShare sees that device's videos and data.
- viewer  → read-only: sees videos + downloads data, cannot run/delete
- manager → read-write: additionally runs analysis and deletes
- owner   → full
- stranger→ nothing

Regression guard for the bug where shared users saw an empty Processing page.
"""

from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import RequestFactory, TestCase
from django.urls import reverse

from apps.devices.models import Device, DeviceShare
from apps.analysis.models import Job, JobResult
from apps.analysis.views import DownloadEventsCSVView
from .models import Video

User = get_user_model()


class DeviceShareVideoAccessTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.owner = User.objects.create_user("alice", password="x")
        cls.manager = User.objects.create_user("bob", password="x")
        cls.viewer = User.objects.create_user("carol", password="x")
        cls.stranger = User.objects.create_user("dave", password="x")

        cls.device = Device.objects.create(
            owner=cls.owner, name="hotel-1", key_hash="h1", prefix="bmk_1",
        )
        DeviceShare.objects.create(
            device=cls.device, user=cls.manager, role=DeviceShare.Role.MANAGER,
            created_by=cls.owner,
        )
        DeviceShare.objects.create(
            device=cls.device, user=cls.viewer, role=DeviceShare.Role.VIEWER,
            created_by=cls.owner,
        )

        # A video on the shared device, plus one private (no device) owned by alice.
        cls.shared_video = Video.objects.create(
            user=cls.owner, device=cls.device, title="shared-clip",
            storage_key="alice/shared-clip.mp4", file_size_bytes=1000,
            status=Video.Status.READY,
        )
        cls.private_video = Video.objects.create(
            user=cls.owner, device=None, title="private-clip",
            storage_key="alice/private-clip.mp4", file_size_bytes=1000,
            status=Video.Status.READY,
        )

    # ---- model helpers -------------------------------------------------
    def test_accessible_includes_shared_for_any_role(self):
        self.assertEqual(
            set(Video.accessible(self.owner).values_list("pk", flat=True)),
            {self.shared_video.pk, self.private_video.pk},
        )
        for u in (self.manager, self.viewer):
            self.assertEqual(
                set(Video.accessible(u).values_list("pk", flat=True)),
                {self.shared_video.pk}, f"{u} should see only the shared video",
            )
        self.assertEqual(list(Video.accessible(self.stranger)), [])

    def test_manageable_excludes_viewer(self):
        self.assertIn(self.shared_video, Video.manageable(self.owner))
        self.assertIn(self.shared_video, Video.manageable(self.manager))
        self.assertNotIn(self.shared_video, Video.manageable(self.viewer))
        self.assertEqual(list(Video.manageable(self.stranger)), [])

    def test_managed_by(self):
        self.assertTrue(self.shared_video.managed_by(self.owner))
        self.assertTrue(self.shared_video.managed_by(self.manager))
        self.assertFalse(self.shared_video.managed_by(self.viewer))
        self.assertFalse(self.shared_video.managed_by(self.stranger))

    # ---- Processing hub (the list page) --------------------------------
    def test_processing_hub_lists_shared_video_for_shared_users(self):
        url = reverse("analysis:processing")
        for u in (self.owner, self.manager, self.viewer):
            self.client.force_login(u)
            html = self.client.get(url).content.decode()
            self.assertIn("shared-clip", html, f"{u} should see the shared video")
        self.client.force_login(self.stranger)
        self.assertNotIn("shared-clip", self.client.get(url).content.decode())

    def test_processing_hub_hides_run_controls_from_viewer(self):
        url = reverse("analysis:processing")
        self.client.force_login(self.manager)
        self.assertIn("Run analysis", self.client.get(url).content.decode())
        self.client.force_login(self.viewer)
        self.assertNotIn("Run analysis", self.client.get(url).content.decode())

    # ---- detail view ---------------------------------------------------
    def test_detail_access(self):
        url = reverse("videos:detail", args=[self.shared_video.pk])
        self.client.force_login(self.viewer)
        resp = self.client.get(url)
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "view only")  # read-only badge, no delete
        self.client.force_login(self.manager)
        self.assertContains(self.client.get(url), "Delete Video")
        self.client.force_login(self.stranger)
        self.assertEqual(self.client.get(url).status_code, 404)

    def test_detail_shows_bee_detection_confidence(self):
        # metadata.bee (status + confidence) is shown on the video detail page.
        v = Video.objects.create(
            user=self.owner, device=self.device, title="bee-clip",
            storage_key="alice/bee-clip.mp4", file_size_bytes=1,
            status=Video.Status.READY,
            metadata={"bee_confirmed": True, "bee": {"status": "confirmed",
                      "confidence": 0.87, "taxon": "Apidae"}},
        )
        self.client.force_login(self.owner)
        resp = self.client.get(reverse("videos:detail", args=[v.pk]))
        self.assertContains(resp, "Bee detection")
        self.assertContains(resp, "87% conf")
        self.assertContains(resp, "Apidae")

    # ---- delete permission ---------------------------------------------
    def test_viewer_cannot_delete(self):
        self.client.force_login(self.viewer)
        resp = self.client.post(reverse("videos:delete", args=[self.shared_video.pk]))
        self.assertEqual(resp.status_code, 404)
        self.assertTrue(Video.objects.filter(pk=self.shared_video.pk).exists())

    @patch("config.storage.get_s3_client")
    def test_manager_can_delete(self, mock_s3):
        self.client.force_login(self.manager)
        resp = self.client.post(reverse("videos:delete", args=[self.shared_video.pk]))
        self.assertEqual(resp.status_code, 302)
        self.assertFalse(Video.objects.filter(pk=self.shared_video.pk).exists())
        mock_s3.return_value.delete_blob.assert_called()  # storage cleanup attempted, mocked

    # ---- ecological data scoping (no S3: test the filter directly) ------
    def test_ecological_results_follow_video_access(self):
        job = Job.objects.create(
            user=self.owner, video=self.shared_video,
            status=Job.Status.COMPLETED, modal_job_id="mid-1",
        )
        result = JobResult.objects.create(job=job, total_events=5)

        view = DownloadEventsCSVView()
        for u in (self.owner, self.manager, self.viewer):
            req = RequestFactory().get("/analytics/download-events/")
            req.user = u
            qs, _ = view._get_filtered_results(req)
            self.assertIn(result, qs, f"{u} should get the shared-device result")

        req = RequestFactory().get("/analytics/download-events/")
        req.user = self.stranger
        qs, _ = view._get_filtered_results(req)
        self.assertNotIn(result, qs)
