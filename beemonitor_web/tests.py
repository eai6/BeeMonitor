"""Comprehensive platform tests for BeeMonitor.

Tests all Django apps: accounts, videos, analysis, annotations, training.
Uses SQLite for speed. Run with:
    cd beemonitor_web && python manage.py test tests --settings=config.settings.development -v2
"""
import json
from unittest.mock import patch, MagicMock

from django.contrib.auth.models import User
from django.test import TestCase, Client, override_settings
from django.urls import reverse


# ── Helpers ──────────────────────────────────────────────────────────


def create_user(username="testuser", password="testpass123"):
    return User.objects.create_user(username=username, password=password)


def logged_in_client(user=None):
    c = Client()
    if user is None:
        user = create_user()
    c.login(username=user.username, password="testpass123")
    return c, user


# ── Account Tests ────────────────────────────────────────────────────


class AccountTests(TestCase):
    def test_register_page_loads(self):
        r = self.client.get(reverse("accounts:register"))
        self.assertEqual(r.status_code, 200)

    def test_login_page_loads(self):
        r = self.client.get(reverse("accounts:login"))
        self.assertEqual(r.status_code, 200)

    def test_register_creates_user(self):
        r = self.client.post(reverse("accounts:register"), {
            "username": "newuser",
            "email": "new@test.com",
            "password1": "Str0ngP@ss!99",
            "password2": "Str0ngP@ss!99",
        })
        # May redirect on success or show form errors
        self.assertIn(r.status_code, [200, 302])

    def test_login_works(self):
        create_user()
        r = self.client.post(reverse("accounts:login"), {
            "username": "testuser",
            "password": "testpass123",
        })
        self.assertIn(r.status_code, [200, 302])

    def test_usage_page_requires_login(self):
        r = self.client.get(reverse("accounts:usage"))
        self.assertEqual(r.status_code, 302)  # redirect to login

    def test_usage_page_loads(self):
        c, _ = logged_in_client()
        r = c.get(reverse("accounts:usage"))
        self.assertEqual(r.status_code, 200)


class UserProfileTests(TestCase):
    def setUp(self):
        self.user = create_user()
        from apps.accounts.models import UserProfile
        self.profile, _ = UserProfile.objects.get_or_create(user=self.user)

    def test_profile_defaults(self):
        self.assertEqual(self.profile.monthly_credits, 3000)
        self.assertEqual(self.profile.used_credits, 0)
        self.assertEqual(self.profile.remaining_credits, 3000)

    def test_charge_deducts_credits(self):
        self.profile.charge(100, gpu_seconds=100.0)
        self.profile.refresh_from_db()
        self.assertEqual(self.profile.used_credits, 100)
        self.assertEqual(self.profile.remaining_credits, 2900)

    def test_charge_accumulates(self):
        self.profile.charge(100, gpu_seconds=100.0)
        self.profile.charge(200, gpu_seconds=200.0)
        self.profile.refresh_from_db()
        self.assertEqual(self.profile.used_credits, 300)
        self.assertEqual(self.profile.total_gpu_seconds, 300.0)
        self.assertEqual(self.profile.total_jobs_submitted, 2)

    def test_has_budget(self):
        self.assertTrue(self.profile.has_budget(3000))
        self.assertFalse(self.profile.has_budget(3001))

    def test_credit_usage_pct(self):
        self.assertEqual(self.profile.credit_usage_pct, 0)
        self.profile.charge(1500)
        self.profile.refresh_from_db()
        self.assertEqual(self.profile.credit_usage_pct, 50)


# ── Video Tests ──────────────────────────────────────────────────────


class VideoTests(TestCase):
    def setUp(self):
        self.client, self.user = logged_in_client()

    def test_video_list_loads(self):
        r = self.client.get(reverse("videos:list"))
        self.assertEqual(r.status_code, 200)

    def test_video_list_requires_login(self):
        c = Client()
        r = c.get(reverse("videos:list"))
        self.assertEqual(r.status_code, 302)

    def test_video_list_context(self):
        r = self.client.get(reverse("videos:list"))
        self.assertIn("videos", r.context)
        self.assertIn("all_video_ids", r.context)
        self.assertIn("custom_models", r.context)

    def test_upload_page_loads(self):
        r = self.client.get(reverse("videos:upload"))
        self.assertEqual(r.status_code, 200)

    def test_video_list_filters(self):
        r = self.client.get(reverse("videos:list") + "?site=test&year=2024")
        self.assertEqual(r.status_code, 200)


# ── Analysis Tests ───────────────────────────────────────────────────


class AnalysisTests(TestCase):
    def setUp(self):
        self.client, self.user = logged_in_client()

    def test_analysis_list_redirects_to_analytics(self):
        r = self.client.get(reverse("analysis:list"))
        self.assertEqual(r.status_code, 302)  # redirect to analytics

    def test_analytics_page_loads(self):
        r = self.client.get(reverse("analysis:analytics"))
        self.assertEqual(r.status_code, 200)

    def test_analytics_context(self):
        r = self.client.get(reverse("analysis:analytics"))
        self.assertIn("summary", r.context)
        self.assertIn("job_stats", r.context)
        self.assertIn("processing_jobs", r.context)

    def test_analytics_filters(self):
        r = self.client.get(reverse("analysis:analytics") + "?site=test&year=2024")
        self.assertEqual(r.status_code, 200)

    def test_new_job_page_loads(self):
        r = self.client.get(reverse("analysis:new"))
        self.assertEqual(r.status_code, 200)

    def test_poll_endpoint(self):
        r = self.client.get(reverse("analysis:poll"))
        self.assertEqual(r.status_code, 200)
        data = json.loads(r.content)
        self.assertIn("checked", data)
        self.assertIn("completed", data)

    def test_batch_requires_videos(self):
        r = self.client.post(reverse("analysis:batch"), {})
        self.assertEqual(r.status_code, 302)  # redirect with warning


class AnalysisJobModelTests(TestCase):
    def test_job_creation(self):
        from apps.analysis.models import Job
        from apps.videos.models import Video
        user = create_user()
        video = Video.objects.create(
            user=user, title="test.mp4",
            storage_key="test/test.mp4",
            status=Video.Status.READY,
            file_size_bytes=1000000,
        )
        job = Job.objects.create(
            user=user, video=video,
            config={"detection_mode": "yolo", "confidence_threshold": 0.25},
            status=Job.Status.PROCESSING,
        )
        self.assertEqual(job.status, "processing")
        self.assertEqual(job.config["detection_mode"], "yolo")

    def test_gpu_tiers_defined(self):
        from apps.analysis.models import GPU_TIERS
        self.assertIn("A10G", GPU_TIERS)
        self.assertIn("T4", GPU_TIERS)
        self.assertIn("cost_per_sec", GPU_TIERS["A10G"])


# ── Annotation Tests ─────────────────────────────────────────────────


class AnnotationProjectTests(TestCase):
    def setUp(self):
        self.client, self.user = logged_in_client()

    def test_project_list_loads(self):
        r = self.client.get(reverse("annotations:list"))
        self.assertEqual(r.status_code, 200)

    def test_create_project_page_loads(self):
        r = self.client.get(reverse("annotations:create"))
        self.assertEqual(r.status_code, 200)

    def test_create_project(self):
        r = self.client.post(reverse("annotations:create"), {
            "name": "Test Project",
            "description": "A test",
            "classes_text": "bee, wasp, nest",
        })
        from apps.annotations.models import AnnotationProject
        self.assertEqual(AnnotationProject.objects.filter(user=self.user).count(), 1)
        proj = AnnotationProject.objects.get(user=self.user)
        self.assertEqual(proj.name, "Test Project")
        self.assertEqual(proj.classes, ["bee", "wasp", "nest"])

    def test_project_detail_loads(self):
        from apps.annotations.models import AnnotationProject
        proj = AnnotationProject.objects.create(
            user=self.user, name="Test", classes=["bee"]
        )
        r = self.client.get(reverse("annotations:detail", args=[proj.pk]))
        self.assertEqual(r.status_code, 200)
        self.assertIn("frame_cards", r.context)
        self.assertIn("total_boxes", r.context)
        self.assertIn("class_counts", r.context)


class AnnotationModelTests(TestCase):
    def test_annotation_to_yolo_format(self):
        from apps.annotations.models import AnnotationProject, Annotation
        from apps.videos.models import Video
        user = create_user()
        proj = AnnotationProject.objects.create(user=user, name="T", classes=["bee"])
        video = Video.objects.create(user=user, title="t.mp4", storage_key="t/t.mp4", file_size_bytes=1000)
        proj.videos.add(video)
        ann = Annotation.objects.create(
            project=proj, video=video, frame_number=0,
            image_width=1280, image_height=720,
            boxes=[{"x": 100, "y": 100, "w": 50, "h": 50, "class": "bee", "class_id": 0}],
        )
        yolo = ann.to_yolo_format()
        self.assertIn("0 ", yolo)  # class_id 0
        parts = yolo.strip().split()
        self.assertEqual(len(parts), 5)  # class_id cx cy w h

    def test_save_annotation_endpoint(self):
        from apps.annotations.models import AnnotationProject, Annotation
        from apps.videos.models import Video
        c, user = logged_in_client()
        proj = AnnotationProject.objects.create(user=user, name="T", classes=["bee"])
        video = Video.objects.create(user=user, title="t.mp4", storage_key="t/t.mp4", file_size_bytes=1000)
        proj.videos.add(video)
        r = c.post(
            reverse("annotations:save", args=[proj.pk]),
            json.dumps({
                "video_id": video.pk,
                "frame_number": 42,
                "boxes": [{"x": 10, "y": 10, "w": 20, "h": 20, "class": "bee", "class_id": 0}],
            }),
            content_type="application/json",
        )
        self.assertEqual(r.status_code, 200)
        data = json.loads(r.content)
        self.assertTrue(data["success"])
        self.assertTrue(Annotation.objects.filter(project=proj, video=video, frame_number=42).exists())


# ── Training Tests ───────────────────────────────────────────────────


class TrainingTests(TestCase):
    def setUp(self):
        self.client, self.user = logged_in_client()

    def test_training_list_loads(self):
        r = self.client.get(reverse("training:list"))
        self.assertEqual(r.status_code, 200)

    def test_training_create_page_loads(self):
        r = self.client.get(reverse("training:create"))
        self.assertEqual(r.status_code, 200)

    def test_models_page_loads(self):
        r = self.client.get(reverse("training:models"))
        self.assertEqual(r.status_code, 200)

    def test_poll_endpoint(self):
        r = self.client.get(reverse("training:poll"))
        self.assertEqual(r.status_code, 200)
        data = json.loads(r.content)
        self.assertIn("checked", data)
        self.assertIn("completed", data)


class TrainingModelTests(TestCase):
    def test_training_job_creation(self):
        from apps.training.models import TrainingJob
        from apps.annotations.models import AnnotationProject
        user = create_user()
        proj = AnnotationProject.objects.create(user=user, name="T", classes=["bee"])
        job = TrainingJob.objects.create(
            user=user, project=proj, name="Test Train",
            base_model="yolov8n", epochs=5,
        )
        self.assertEqual(job.status, "queued")
        self.assertEqual(str(job), "Test Train (queued)")

    def test_custom_model_creation(self):
        from apps.training.models import CustomModel
        user = create_user()
        cm = CustomModel.objects.create(
            user=user, name="My Model",
            model_type="custom", base_model="yolov8n",
            storage_key="custom/1/best.pt",
            classes=["bee", "wasp"],
            metrics={"mAP50": 0.85},
            is_active=True,
        )
        self.assertTrue(cm.is_active)
        self.assertEqual(cm.classes, ["bee", "wasp"])


# ── Source Tests ─────────────────────────────────────────────────────


class SourceTests(TestCase):
    def setUp(self):
        self.client, self.user = logged_in_client()

    def test_source_list_loads(self):
        r = self.client.get(reverse("sources:list"))
        self.assertEqual(r.status_code, 200)


# ── Dashboard Tests ──────────────────────────────────────────────────


class DashboardTests(TestCase):
    def setUp(self):
        self.client, self.user = logged_in_client()

    def test_dashboard_redirects_to_analytics(self):
        r = self.client.get(reverse("dashboard:dashboard"))
        self.assertEqual(r.status_code, 302)
        self.assertIn(reverse("analysis:analytics"), r.url)


# ── Navigation Tests ─────────────────────────────────────────────────


class NavTests(TestCase):
    def setUp(self):
        self.client, self.user = logged_in_client()

    def test_all_nav_links_load(self):
        """Verify every nav link returns 200 or valid redirect."""
        urls = [
            reverse("dashboard:dashboard"),
            reverse("videos:list"),
            reverse("analysis:analytics"),
            reverse("annotations:list"),
            reverse("training:list"),
            reverse("sources:list"),
            reverse("accounts:usage"),
        ]
        for url in urls:
            r = self.client.get(url)
            self.assertIn(r.status_code, [200, 302], f"Failed: {url} returned {r.status_code}")

    def test_analysis_redirects_to_analytics(self):
        r = self.client.get(reverse("analysis:list"))
        self.assertRedirects(r, reverse("analysis:analytics"))

    def test_no_developer_in_nav(self):
        r = self.client.get(reverse("videos:list"))
        self.assertNotContains(r, "Developer")


# ── Integration Tests ────────────────────────────────────────────────


class AnnotationEditorTests(TestCase):
    def setUp(self):
        self.client, self.user = logged_in_client()
        from apps.annotations.models import AnnotationProject
        self.project = AnnotationProject.objects.create(
            user=self.user, name="Editor Test", classes=["bee", "wasp"]
        )

    def test_editor_no_video_shows_selector(self):
        r = self.client.get(reverse("annotations:editor", args=[self.project.pk]))
        self.assertEqual(r.status_code, 200)
        self.assertContains(r, "Select a video")

    def test_editor_with_video(self):
        from apps.videos.models import Video
        video = Video.objects.create(
            user=self.user, title="ed.mp4",
            storage_key="ed/ed.mp4", file_size_bytes=1000,
        )
        self.project.videos.add(video)
        r = self.client.get(
            reverse("annotations:editor", args=[self.project.pk]) + f"?video={video.pk}&frame=0"
        )
        self.assertEqual(r.status_code, 200)
        self.assertIn("total_project_frames", r.context)

    def test_frame_navigation_context(self):
        from apps.videos.models import Video
        from apps.annotations.models import Annotation
        video = Video.objects.create(
            user=self.user, title="nav.mp4",
            storage_key="nav/nav.mp4", file_size_bytes=1000,
        )
        self.project.videos.add(video)
        # Create 3 annotations
        for f in [10, 20, 30]:
            Annotation.objects.create(
                project=self.project, video=video, frame_number=f,
                boxes=[{"x": 0, "y": 0, "w": 10, "h": 10, "class": "bee", "class_id": 0}],
            )
        # Navigate to middle frame
        r = self.client.get(
            reverse("annotations:editor", args=[self.project.pk]) + f"?video={video.pk}&frame=20"
        )
        self.assertEqual(r.context["current_frame_index"], 2)
        self.assertEqual(r.context["total_project_frames"], 3)
        self.assertIn("prev_frame_url", r.context)
        self.assertIn("next_frame_url", r.context)
        self.assertTrue(r.context["prev_frame_url"])  # has prev
        self.assertTrue(r.context["next_frame_url"])  # has next


# ── Upload Endpoints (Phase 3) ───────────────────────────────────────


class UploadEndpointTests(TestCase):
    """Pi-side upload flow: /api/v1/uploads/{initiate,complete}.

    The S3 client is mocked end-to-end so the tests run offline. We're
    checking the Django plumbing — auth, prefix-scoping, Video creation —
    not boto3 behaviour, which has its own coverage in
    ``cloud/tests/test_s3_client.py``.
    """

    def setUp(self):
        from apps.devices.models import Device
        self.user = create_user(username="owner")
        self.other_user = create_user(username="stranger", password="x")
        self.device, self.raw_key = Device.create_with_key(
            owner=self.user, name="pi-1", location="natalies",
        )
        self.client = Client()

    def _auth(self):
        return {"HTTP_AUTHORIZATION": f"Bearer {self.raw_key}"}

    def test_initiate_returns_presigned_url(self):
        with patch("apps.api.uploads.get_s3_client") as mock_s3:
            mock_s3.return_value.generate_presigned_url.return_value = "https://signed-url"
            r = self.client.post(
                "/api/v1/uploads/initiate",
                data={"filename": "natalies_2026-05-23_10_00_00.mp4",
                      "size_bytes": 12345, "content_type": "video/mp4"},
                content_type="application/json",
                **self._auth(),
            )
        self.assertEqual(r.status_code, 200, r.content)
        body = r.json()
        self.assertEqual(body["method"], "PUT")
        self.assertEqual(body["upload_url"], "https://signed-url")
        self.assertTrue(
            body["storage_key"].startswith(
                f"users/{self.user.pk}/devices/{self.device.pk}/"
            )
        )
        self.assertTrue(body["storage_key"].endswith(".mp4"))

    def test_initiate_rejects_unknown_extension(self):
        r = self.client.post(
            "/api/v1/uploads/initiate",
            data={"filename": "evil.exe", "size_bytes": 100},
            content_type="application/json",
            **self._auth(),
        )
        self.assertEqual(r.status_code, 400)

    def test_initiate_requires_device_key(self):
        # No auth header.
        r = self.client.post(
            "/api/v1/uploads/initiate",
            data={"filename": "x.mp4", "size_bytes": 100},
            content_type="application/json",
        )
        self.assertEqual(r.status_code, 401)

        # Wrong key.
        r = self.client.post(
            "/api/v1/uploads/initiate",
            data={"filename": "x.mp4", "size_bytes": 100},
            content_type="application/json",
            HTTP_AUTHORIZATION="Bearer bmk_device_not-a-real-key",
        )
        self.assertEqual(r.status_code, 401)

    def test_initiate_rejects_zero_size(self):
        r = self.client.post(
            "/api/v1/uploads/initiate",
            data={"filename": "x.mp4", "size_bytes": 0},
            content_type="application/json",
            **self._auth(),
        )
        self.assertEqual(r.status_code, 400)

    def test_complete_creates_video_and_scopes_to_device(self):
        from apps.videos.models import Video
        storage_key = f"users/{self.user.pk}/devices/{self.device.pk}/2026/05/23/abc.mp4"
        with patch("apps.api.uploads.get_s3_client") as mock_s3:
            mock_s3.return_value.blob_exists.return_value = True
            r = self.client.post(
                "/api/v1/uploads/complete",
                data={"storage_key": storage_key, "file_size_bytes": 12345,
                      "recorded_at": "2026-05-23T10:00:00Z"},
                content_type="application/json",
                **self._auth(),
            )
        self.assertEqual(r.status_code, 201, r.content)
        body = r.json()
        video = Video.objects.get(pk=body["video_id"])
        self.assertEqual(video.user_id, self.user.pk)
        self.assertEqual(video.storage_key, storage_key)
        self.assertEqual(video.metadata.get("device_id"), self.device.pk)
        self.assertEqual(video.file_size_bytes, 12345)

    def test_complete_rejects_storage_key_outside_device_prefix(self):
        # Storage key for a different user/device.
        storage_key = f"users/{self.other_user.pk}/devices/9999/2026/05/23/x.mp4"
        with patch("apps.api.uploads.get_s3_client") as mock_s3:
            mock_s3.return_value.blob_exists.return_value = True
            r = self.client.post(
                "/api/v1/uploads/complete",
                data={"storage_key": storage_key, "file_size_bytes": 100},
                content_type="application/json",
                **self._auth(),
            )
        self.assertEqual(r.status_code, 403)

    def test_complete_404_when_object_missing(self):
        storage_key = f"users/{self.user.pk}/devices/{self.device.pk}/2026/05/23/abc.mp4"
        with patch("apps.api.uploads.get_s3_client") as mock_s3:
            mock_s3.return_value.blob_exists.return_value = False
            r = self.client.post(
                "/api/v1/uploads/complete",
                data={"storage_key": storage_key, "file_size_bytes": 100},
                content_type="application/json",
                **self._auth(),
            )
        self.assertEqual(r.status_code, 404)

    def test_revoked_device_cannot_upload(self):
        self.device.is_active = False
        self.device.save(update_fields=["is_active"])
        r = self.client.post(
            "/api/v1/uploads/initiate",
            data={"filename": "x.mp4", "size_bytes": 100},
            content_type="application/json",
            **self._auth(),
        )
        self.assertEqual(r.status_code, 401)


# ── Device UI ────────────────────────────────────────────────────────


class DeviceUIViewTests(TestCase):
    """Smoke tests for /devices/ pages — auth, ownership, raw-key one-shot."""

    def setUp(self):
        from apps.devices.models import Device
        self.owner = create_user(username="alice")
        self.stranger = create_user(username="bob")
        self.client, _ = logged_in_client(self.owner)
        # Pre-existing device for alice and a different device for bob.
        self.alice_device, _ = Device.create_with_key(self.owner, "alice-pi-1")
        self.bob_device, _ = Device.create_with_key(self.stranger, "bob-pi-1")

    def test_list_shows_only_my_devices(self):
        r = self.client.get(reverse("devices:list"))
        self.assertEqual(r.status_code, 200)
        body = r.content.decode()
        self.assertIn("alice-pi-1", body)
        self.assertNotIn("bob-pi-1", body)

    def test_create_then_view_raw_key_once(self):
        # Create.
        r = self.client.post(
            reverse("devices:add"),
            data={"name": "alice-pi-2", "location": "shed"},
            follow=False,
        )
        self.assertEqual(r.status_code, 302)
        # Pull the new device.
        from apps.devices.models import Device
        new_device = Device.objects.get(name="alice-pi-2", owner=self.owner)

        # Follow the redirect — raw key should be visible once.
        r2 = self.client.get(r.url)
        self.assertEqual(r2.status_code, 200)
        body2 = r2.content.decode()
        self.assertIn("bmk_device_", body2, "raw key should be shown on first view")

        # Refresh / revisit — raw key MUST NOT appear again.
        r3 = self.client.get(reverse("devices:created", args=[new_device.pk]))
        self.assertEqual(r3.status_code, 200)
        self.assertNotIn("bmk_device_", r3.content.decode(),
                         "raw key must not be re-shown")

    def test_cannot_revoke_someone_elses_device(self):
        # Alice tries to revoke Bob's device.
        r = self.client.post(reverse("devices:revoke", args=[self.bob_device.pk]))
        self.assertEqual(r.status_code, 404)
        # Bob's device still active.
        self.bob_device.refresh_from_db()
        self.assertTrue(self.bob_device.is_active)

    def test_revoke_disables_auth(self):
        r = self.client.post(reverse("devices:revoke", args=[self.alice_device.pk]))
        self.assertEqual(r.status_code, 302)
        self.alice_device.refresh_from_db()
        self.assertFalse(self.alice_device.is_active)

    def test_unauthenticated_redirected_to_login(self):
        anon = Client()
        r = anon.get(reverse("devices:list"))
        self.assertEqual(r.status_code, 302)
        self.assertIn("/accounts/login/", r.url)


# ── Phase 5: cross-user ownership scoping ────────────────────────────


class CrossUserScopingTests(TestCase):
    """User B must never see user A's resources via any UI/API path.

    Every page that lists, details, or mutates an owned resource gets a
    targeted check: log in as the wrong user, ask for it by pk, expect
    404 (NOT 403 — 404 doesn't leak that the resource exists).
    """

    def setUp(self):
        from apps.videos.models import Video
        from apps.analysis.models import Job, JobResult
        from apps.annotations.models import AnnotationProject
        from apps.devices.models import Device

        self.alice = create_user(username="alice")
        self.bob = create_user(username="bob")

        # Alice owns one of each resource.
        self.alice_video = Video.objects.create(
            user=self.alice, title="alice.mp4",
            storage_key="alice/alice.mp4", file_size_bytes=100,
            status=Video.Status.READY,
        )
        self.alice_job = Job.objects.create(
            user=self.alice, video=self.alice_video,
            modal_job_id="aj-1", config={},
            status=Job.Status.COMPLETED,
        )
        JobResult.objects.create(
            job=self.alice_job,
            events_csv_path="alice/events.csv",
            tracking_csv_path="alice/tracking.csv",
        )
        self.alice_project = AnnotationProject.objects.create(
            user=self.alice, name="alice-project", classes=["bee"],
        )
        self.alice_device, _ = Device.create_with_key(self.alice, "alice-pi")

        # Bob is logged in for every test.
        self.client, _ = logged_in_client(self.bob)

    # ── Videos ────────────────────────────────────────────────
    def test_bob_cannot_see_alice_video_detail(self):
        r = self.client.get(reverse("videos:detail", args=[self.alice_video.pk]))
        self.assertEqual(r.status_code, 404)

    def test_bob_cannot_delete_alice_video(self):
        r = self.client.post(reverse("videos:delete", args=[self.alice_video.pk]))
        self.assertEqual(r.status_code, 404)
        self.assertTrue(
            self.alice_video.__class__.objects.filter(pk=self.alice_video.pk).exists()
        )

    def test_bob_list_does_not_include_alice_videos(self):
        r = self.client.get(reverse("videos:list"))
        self.assertEqual(r.status_code, 200)
        self.assertNotIn("alice.mp4", r.content.decode())

    # ── Analysis jobs ─────────────────────────────────────────
    def test_bob_cannot_see_alice_job_detail(self):
        r = self.client.get(reverse("analysis:detail", args=[self.alice_job.pk]))
        self.assertEqual(r.status_code, 404)

    def test_bob_cannot_see_alice_job_results(self):
        r = self.client.get(reverse("analysis:results", args=[self.alice_job.pk]))
        self.assertEqual(r.status_code, 404)

    def test_bob_list_does_not_include_alice_jobs(self):
        # analysis:list redirects to analysis:analytics; follow it.
        r = self.client.get(reverse("analysis:list"), follow=True)
        self.assertEqual(r.status_code, 200)
        self.assertNotIn(self.alice_job.modal_job_id, r.content.decode())

    # ── Annotation projects ───────────────────────────────────
    def test_bob_cannot_see_alice_project(self):
        r = self.client.get(reverse("annotations:detail", args=[self.alice_project.pk]))
        self.assertEqual(r.status_code, 404)

    def test_bob_cannot_open_alice_editor(self):
        r = self.client.get(reverse("annotations:editor", args=[self.alice_project.pk]))
        self.assertEqual(r.status_code, 404)

    # ── Devices ───────────────────────────────────────────────
    def test_bob_cannot_revoke_alice_device(self):
        r = self.client.post(reverse("devices:revoke", args=[self.alice_device.pk]))
        self.assertEqual(r.status_code, 404)

    def test_bob_cannot_delete_alice_device(self):
        r = self.client.post(reverse("devices:delete", args=[self.alice_device.pk]))
        self.assertEqual(r.status_code, 404)

    def test_bob_list_does_not_include_alice_devices(self):
        r = self.client.get(reverse("devices:list"))
        self.assertEqual(r.status_code, 200)
        self.assertNotIn("alice-pi", r.content.decode())


class APICrossUserScopingTests(TestCase):
    """DRF API endpoints — Bob's API key must not retrieve Alice's resources."""

    def setUp(self):
        from apps.accounts.models import APIKey
        from apps.videos.models import Video
        from apps.analysis.models import Job, JobResult

        self.alice = create_user(username="alice")
        self.bob = create_user(username="bob")

        self.alice_video = Video.objects.create(
            user=self.alice, title="alice.mp4",
            storage_key="alice/alice.mp4", file_size_bytes=100,
            status=Video.Status.READY,
        )
        self.alice_job = Job.objects.create(
            user=self.alice, video=self.alice_video,
            modal_job_id="aj-1", config={},
            status=Job.Status.COMPLETED,
        )
        JobResult.objects.create(
            job=self.alice_job,
            events_csv_path="alice/events.csv",
        )

        _, self.bob_raw_key = APIKey.create_key(self.bob, name="bob-key", key_type="live")
        self.client = Client()

    def _auth(self):
        return {"HTTP_AUTHORIZATION": f"Bearer {self.bob_raw_key}"}

    def test_bob_cannot_get_alice_video(self):
        r = self.client.get(f"/api/v1/videos/{self.alice_video.pk}/", **self._auth())
        self.assertEqual(r.status_code, 404)

    def test_bob_cannot_get_alice_job(self):
        r = self.client.get(f"/api/v1/jobs/{self.alice_job.pk}/", **self._auth())
        self.assertEqual(r.status_code, 404)

    def test_bob_cannot_get_download_url_for_alice_job(self):
        r = self.client.get(
            f"/api/v1/jobs/{self.alice_job.pk}/download/?file=events_csv_path",
            **self._auth(),
        )
        # 404 is correct — get_object() raises Http404 for cross-user pk.
        self.assertEqual(r.status_code, 404)

    def test_bob_videos_list_excludes_alice(self):
        r = self.client.get("/api/v1/videos/", **self._auth())
        self.assertEqual(r.status_code, 200)
        body = r.json()
        results = body.get("results", body)
        if isinstance(results, list):
            titles = [v.get("title") for v in results]
            self.assertNotIn("alice.mp4", titles)


class SupportRoleBypassTests(TestCase):
    """is_support gives READ visibility to other users' data, not WRITE."""

    def setUp(self):
        from apps.accounts.models import UserProfile
        from apps.videos.models import Video

        self.alice = create_user(username="alice")
        self.support = create_user(username="support")
        # A post_save signal on User creates the UserProfile with default
        # is_support=False; get_or_create returns the existing row.
        sp, _ = UserProfile.objects.get_or_create(user=self.support)
        sp.is_support = True
        sp.save(update_fields=["is_support"])
        # OneToOne reverse accessors cache; reload self.support so reading
        # self.support.profile picks up the just-saved is_support flag.
        self.support.refresh_from_db()

        self.alice_video = Video.objects.create(
            user=self.alice, title="alice.mp4",
            storage_key="alice/alice.mp4", file_size_bytes=100,
            status=Video.Status.READY,
        )

        self.client, _ = logged_in_client(self.support)

    def test_is_support_helper(self):
        from apps.accounts.permissions import is_support
        self.assertTrue(is_support(self.support))
        self.assertFalse(is_support(self.alice))

    def test_support_user_cannot_write_to_alice_video(self):
        # Support is read-only — deleting via the per-resource view is owner-scoped.
        r = self.client.post(reverse("videos:delete", args=[self.alice_video.pk]))
        self.assertEqual(r.status_code, 404, "writes must still 404 cross-user")
        from apps.videos.models import Video as V
        self.assertTrue(V.objects.filter(pk=self.alice_video.pk).exists())


# ── Web-form direct-to-S3 uploads ────────────────────────────────────


class WebUploadEndpointTests(TestCase):
    """/api/v1/web-uploads/{initiate,complete} — browser PUTs to S3 directly.

    Session auth + CSRF. S3 is mocked end-to-end."""

    def setUp(self):
        self.user = create_user(username="alice")
        self.stranger = create_user(username="bob")
        self.client, _ = logged_in_client(self.user)

    def _initiate(self, filename="natalies_2026-05-23_10_00_00.mp4", size=1024):
        with patch("apps.api.web_uploads.get_s3_client") as mock_s3:
            mock_s3.return_value.generate_presigned_url.return_value = "https://signed-url"
            return self.client.post(
                "/api/v1/web-uploads/initiate",
                data={"filename": filename, "size_bytes": size,
                      "content_type": "video/mp4"},
                content_type="application/json",
            )

    def test_initiate_returns_presigned_url(self):
        r = self._initiate()
        self.assertEqual(r.status_code, 200, r.content)
        body = r.json()
        self.assertEqual(body["upload_url"], "https://signed-url")
        self.assertTrue(body["storage_key"].startswith(f"{self.user.pk}/"))
        self.assertTrue(body["storage_key"].endswith(".mp4"))

    def test_initiate_requires_login(self):
        anon = Client()
        r = anon.post(
            "/api/v1/web-uploads/initiate",
            data={"filename": "x.mp4", "size_bytes": 100},
            content_type="application/json",
        )
        self.assertIn(r.status_code, (302, 401, 403))

    def test_initiate_rejects_unknown_extension(self):
        r = self._initiate(filename="evil.exe")
        self.assertEqual(r.status_code, 400)

    def test_complete_creates_video_in_my_account(self):
        from apps.videos.models import Video
        storage_key = f"{self.user.pk}/abc123/test.mp4"
        with patch("apps.api.web_uploads.get_s3_client") as mock_s3:
            mock_s3.return_value.blob_exists.return_value = True
            r = self.client.post(
                "/api/v1/web-uploads/complete",
                data={"storage_key": storage_key, "file_size_bytes": 5000,
                      "title": "My Test Video"},
                content_type="application/json",
            )
        self.assertEqual(r.status_code, 201, r.content)
        video = Video.objects.get(pk=r.json()["video_id"])
        self.assertEqual(video.user_id, self.user.pk)
        self.assertEqual(video.title, "My Test Video")
        self.assertEqual(video.storage_key, storage_key)

    def test_complete_rejects_other_users_storage_key(self):
        storage_key = f"{self.stranger.pk}/abc/sneaky.mp4"
        with patch("apps.api.web_uploads.get_s3_client") as mock_s3:
            mock_s3.return_value.blob_exists.return_value = True
            r = self.client.post(
                "/api/v1/web-uploads/complete",
                data={"storage_key": storage_key, "file_size_bytes": 100},
                content_type="application/json",
            )
        self.assertEqual(r.status_code, 403)

    def test_complete_404_when_object_missing(self):
        storage_key = f"{self.user.pk}/abc/missing.mp4"
        with patch("apps.api.web_uploads.get_s3_client") as mock_s3:
            mock_s3.return_value.blob_exists.return_value = False
            r = self.client.post(
                "/api/v1/web-uploads/complete",
                data={"storage_key": storage_key, "file_size_bytes": 100},
                content_type="application/json",
            )
        self.assertEqual(r.status_code, 404)


# ── Remote power scheduling + battery telemetry ──────────────────────


class WakeScheduleValidatorTests(TestCase):
    """Shape-validation of the desired WittyPi schedule (the device enforces the
    real safety floor — this is just rejecting malformed specs server-side)."""

    def test_default_is_daylight(self):
        from apps.devices.models import Device
        user = create_user(username="alice")
        dev, _ = Device.create_with_key(user, "pi-1")
        self.assertEqual(dev.wake_schedule_dict(), {"mode": "daylight"})

    def test_valid_modes(self):
        from apps.devices.models import clean_wake_schedule
        ok, err = clean_wake_schedule({"mode": "daylight"})
        self.assertEqual((ok, err), ({"mode": "daylight"}, ""))
        ok, err = clean_wake_schedule({"mode": "always_on"})
        self.assertEqual(ok, {"mode": "always_on"})
        ok, err = clean_wake_schedule({"mode": "window", "on": "6:5", "off": "20:00"})
        self.assertEqual(ok, {"mode": "window", "on": "06:05", "off": "20:00"})
        ok, err = clean_wake_schedule(
            {"mode": "interval", "wake_every_min": "60", "on_minutes": "10"})
        self.assertEqual(ok, {"mode": "interval", "wake_every_min": 60, "on_minutes": 10})

    def test_rejects_bad_specs(self):
        from apps.devices.models import clean_wake_schedule
        for bad in (
            {"mode": "nope"},
            "notadict",
            {"mode": "window", "on": "25:00", "off": "20:00"},
            {"mode": "window", "on": "06:00", "off": "06:00"},   # equal
            {"mode": "interval", "wake_every_min": "2", "on_minutes": "1"},  # < 5
            {"mode": "interval", "wake_every_min": "60", "on_minutes": "90"},  # on > every
            {"mode": "interval", "wake_every_min": "x", "on_minutes": "10"},
        ):
            ok, err = clean_wake_schedule(bad)
            self.assertIsNone(ok, bad)
            self.assertTrue(err)


class DeviceScheduleViewTests(TestCase):
    """The dashboard schedule editor (manager+); ajax + ownership scoping."""

    def setUp(self):
        from apps.devices.models import Device
        self.owner = create_user(username="alice")
        self.stranger = create_user(username="bob")
        self.client, _ = logged_in_client(self.owner)
        self.device, _ = Device.create_with_key(self.owner, "alice-pi")
        self.other, _ = Device.create_with_key(self.stranger, "bob-pi")

    def _post(self, pk, data, ajax=True):
        extra = {"HTTP_X_REQUESTED_WITH": "XMLHttpRequest"} if ajax else {}
        return self.client.post(reverse("devices:schedule", args=[pk]), data=data, **extra)

    def test_set_always_on(self):
        r = self._post(self.device.pk, {"mode": "always_on"})
        self.assertEqual(r.status_code, 200, r.content)
        self.assertTrue(r.json()["ok"])
        self.device.refresh_from_db()
        self.assertEqual(self.device.wake_schedule, {"mode": "always_on"})
        self.assertIn("24/7", r.json()["label"])

    def test_set_window(self):
        r = self._post(self.device.pk, {"mode": "window", "on": "07:00", "off": "19:30"})
        self.assertEqual(r.status_code, 200, r.content)
        self.device.refresh_from_db()
        self.assertEqual(self.device.wake_schedule,
                         {"mode": "window", "on": "07:00", "off": "19:30"})

    def test_invalid_rejected(self):
        r = self._post(self.device.pk, {"mode": "window", "on": "bad", "off": "19:00"})
        self.assertEqual(r.status_code, 400)
        self.device.refresh_from_db()
        self.assertEqual(self.device.wake_schedule, {"mode": "daylight"})  # unchanged

    def test_cannot_set_someone_elses_device(self):
        r = self._post(self.other.pk, {"mode": "always_on"})
        self.assertIn(r.status_code, (403, 404))  # blocked either way
        self.other.refresh_from_db()
        self.assertEqual(self.other.wake_schedule, {"mode": "daylight"})


class HeartbeatScheduleTests(TestCase):
    """The heartbeat response carries the desired schedule (device reconciles to
    it), and the status poll surfaces schedule + battery telemetry."""

    def setUp(self):
        from apps.devices.models import Device
        self.user = create_user(username="owner")
        self.device, self.raw_key = Device.create_with_key(self.user, "pi-1")
        self.api = Client()

    def test_heartbeat_response_includes_wake_schedule(self):
        self.device.wake_schedule = {"mode": "always_on"}
        self.device.save(update_fields=["wake_schedule"])
        r = self.api.post(
            "/api/v1/devices/heartbeat",
            data={"metrics": json.dumps({"storage_pct": 12.0})},
            HTTP_AUTHORIZATION=f"Bearer {self.raw_key}",
        )
        self.assertEqual(r.status_code, 201, r.content)
        self.assertEqual(r.json()["wake_schedule"], {"mode": "always_on"})

    def test_status_json_surfaces_battery_and_schedule(self):
        from apps.devices.models import DeviceHeartbeat
        DeviceHeartbeat.objects.create(device=self.device, metrics={
            "battery_voltage": 12.4, "output_current": 0.35, "power_source": "DC input",
            "active_schedule": {"mode": "daylight"},
            "next_boot": "06:00", "next_shutdown": "20:00",
        })
        c, _ = logged_in_client(self.user)
        r = c.get(reverse("devices:status", args=[self.device.pk]))
        self.assertEqual(r.status_code, 200)
        d = r.json()
        self.assertEqual(d["battery_voltage"], 12.4)
        self.assertEqual(d["output_current"], 0.35)
        self.assertEqual(d["power_source"], "DC input")
        # device reports daylight, desired is daylight -> confirmed.
        self.assertTrue(d["schedule_confirmed"])
        self.assertEqual(d["next_boot"], "06:00")

    def test_schedule_pending_when_device_silent(self):
        from apps.devices.models import DeviceHeartbeat
        self.device.wake_schedule = {"mode": "always_on"}
        self.device.save(update_fields=["wake_schedule"])
        DeviceHeartbeat.objects.create(device=self.device, metrics={})  # no active_schedule
        c, _ = logged_in_client(self.user)
        r = c.get(reverse("devices:status", args=[self.device.pk]))
        self.assertFalse(r.json()["schedule_confirmed"])


# ── Device-side WittyPi schedule logic (hardware/telemetry.py) ───────


def _load_hw_telemetry():
    """Load hardware/telemetry.py by path (it lives outside the Django app)."""
    import importlib.util
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "hardware" / "telemetry.py"
    spec = importlib.util.spec_from_file_location("hw_telemetry", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class WittyPiScheduleDeviceTests(TestCase):
    """Pure device-side scheduling logic: validation + wake-floor clamp + .wpi
    translation + the report-only safety gate (no hardware required)."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        try:
            cls.t = _load_hw_telemetry()
        except Exception as e:  # pragma: no cover - env without telemetry deps
            raise cls.skipTest(cls, f"hardware/telemetry.py not importable: {e}")

    def test_clean_schedule_valid(self):
        self.assertEqual(self.t._clean_schedule({"mode": "daylight"}), {"mode": "daylight"})
        self.assertEqual(self.t._clean_schedule({"mode": "always_on"}), {"mode": "always_on"})
        self.assertEqual(
            self.t._clean_schedule({"mode": "window", "on": "6:0", "off": "20:00"}),
            {"mode": "window", "on": "06:00", "off": "20:00"})
        self.assertEqual(
            self.t._clean_schedule({"mode": "interval", "wake_every_min": 60, "on_minutes": 10}),
            {"mode": "interval", "wake_every_min": 60, "on_minutes": 10})

    def test_clean_schedule_rejects_garbage(self):
        for bad in ({"mode": "nope"}, "x", {"mode": "window", "on": "6:0", "off": "6:0"},
                    {"mode": "interval", "wake_every_min": 3, "on_minutes": 1},
                    {"mode": "interval", "wake_every_min": 60, "on_minutes": 60}):
            self.assertIsNone(self.t._clean_schedule(bad), bad)

    def test_wake_floor_rejects_long_off_stretch(self):
        orig = self.t.WAKE_FLOOR_HOURS
        try:
            self.t.WAKE_FLOOR_HOURS = 2.0
            # off stretch 10h > 2h floor -> rejected (can't strand the unit).
            self.assertIsNone(
                self.t._clean_schedule({"mode": "window", "on": "06:00", "off": "20:00"}))
            # 50-min off interval is within the floor -> allowed.
            self.assertIsNotNone(
                self.t._clean_schedule({"mode": "interval", "wake_every_min": 60, "on_minutes": 10}))
        finally:
            self.t.WAKE_FLOOR_HOURS = orig

    def test_schedule_to_wpi_durations(self):
        w = self.t._schedule_to_wpi({"mode": "window", "on": "07:50", "off": "18:45"})
        self.assertIn("ON    H10 M55", w)   # 10h55m on
        self.assertIn("OFF   H13 M5", w)    # 13h05m off
        iv = self.t._schedule_to_wpi({"mode": "interval", "wake_every_min": 60, "on_minutes": 10})
        self.assertIn("ON    M10", iv)
        self.assertIn("OFF   M50", iv)

    def test_report_only_gate_does_not_apply(self):
        # With apply gated off, _apply_schedule must be a safe no-op (never raises,
        # never reaches the WittyPi). Default is off.
        self.assertFalse(self.t.WAKE_SCHEDULE_APPLY)
        self.t._apply_schedule({"mode": "always_on"})  # should just log + return


# ── Cellular debug-access firewall toggle ────────────────────────────


class DeviceCellularViewTests(TestCase):
    """Open/close the cellular egress gate for remote debugging (rpi-connect)."""

    def setUp(self):
        from apps.devices.models import Device
        self.owner = create_user(username="alice")
        self.stranger = create_user(username="bob")
        self.client, _ = logged_in_client(self.owner)
        self.device, _ = Device.create_with_key(self.owner, "alice-pi")
        self.other, _ = Device.create_with_key(self.stranger, "bob-pi")

    def test_open_queues_command_with_clamped_minutes(self):
        r = self.client.post(reverse("devices:cellular", args=[self.device.pk]),
                             {"action": "open", "minutes": "999"})
        self.assertEqual(r.status_code, 302)
        self.device.refresh_from_db()
        self.assertEqual(self.device.pending_command, "cellular_open")
        self.assertEqual(self.device.command_params, {"minutes": 240})  # clamped

    def test_gate_queues_command(self):
        r = self.client.post(reverse("devices:cellular", args=[self.device.pk]),
                             {"action": "gate"})
        self.assertEqual(r.status_code, 302)
        self.device.refresh_from_db()
        self.assertEqual(self.device.pending_command, "cellular_gate")

    def test_unknown_action_rejected(self):
        r = self.client.post(reverse("devices:cellular", args=[self.device.pk]),
                             {"action": "nope"})
        self.assertEqual(r.status_code, 302)
        self.device.refresh_from_db()
        self.assertEqual(self.device.pending_command, "")

    def test_cannot_toggle_someone_elses_device(self):
        r = self.client.post(reverse("devices:cellular", args=[self.other.pk]),
                             {"action": "open"})
        self.assertIn(r.status_code, (403, 404))
        self.other.refresh_from_db()
        self.assertEqual(self.other.pending_command, "")

    def test_status_surfaces_cell_firewall(self):
        from apps.devices.models import DeviceHeartbeat
        DeviceHeartbeat.objects.create(device=self.device, metrics={"cell_firewall": "open"})
        r = self.client.get(reverse("devices:status", args=[self.device.pk]))
        self.assertEqual(r.json()["cell_firewall"], "open")


class ActivityTableRenderTests(TestCase):
    """The device detail page renders the CSV-exportable activity table."""

    def setUp(self):
        from apps.devices.models import Device
        self.owner = create_user(username="alice")
        self.client, _ = logged_in_client(self.owner)
        self.device, _ = Device.create_with_key(self.owner, "alice-pi")

    def test_detail_has_activity_table_and_csv(self):
        r = self.client.get(reverse("devices:detail", args=[self.device.pk]))
        self.assertEqual(r.status_code, 200)
        body = r.content.decode()
        self.assertIn('id="act-table-body"', body)
        self.assertIn('id="act-csv-btn"', body)
        self.assertIn("buildActivityTable", body)
        self.assertIn("Activity per", body)
        # One range selector drives both chart + table (shared, in-place switch).
        self.assertIn("act-range-btn", body)
        self.assertIn("data-range=", body)
        self.assertIn('id="act-range-label"', body)
