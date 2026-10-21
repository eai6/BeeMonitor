"""Security fixes from the October 2026 audit (API side).

The credit gate on every pipeline launch, custom models only by what the
caller may use, a video's source only the caller's own, API passwords checked
like the web's, upload names without paths, and data-source credentials
encrypted at rest.
"""

from unittest import mock

from cryptography.fernet import Fernet
from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from rest_framework.test import APIClient

from apps.analysis.models import Job
from apps.pipelines import engine
from apps.pipelines.models import Pipeline, PipelineRun
from apps.sources.models import DataSource
from apps.training.models import CustomModel
from apps.videos.models import Video

User = get_user_model()
KEY = Fernet.generate_key().decode()


class Base(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("bee", password="x")
        self.api = APIClient()
        self.api.force_authenticate(self.user)
        # 100 hours of footage: ~107 credits, far over the 10 free.
        self.big = Video.objects.create(user=self.user, title="big", storage_key="k/big",
                                        file_size_bytes=1, status=Video.Status.READY,
                                        duration_seconds=100 * 3600)
        self.pipeline = Pipeline.objects.create(user=self.user, title="p", steps=[
            {"id": "v", "block_type": "input.video", "config": {}},
            {"id": "d", "block_type": "detect.bee", "config": {}, "inputs": {"video": "v"}},
        ])


class PipelineGateTests(Base):
    def test_the_api_pipeline_run_is_gated(self):
        r = self.api.post(f"/api/v1/pipelines/{self.pipeline.pk}/run/", {"video_ids": [self.big.pk]},
                          format="json")
        self.assertEqual(r.status_code, 402, r.content)
        self.assertFalse(PipelineRun.objects.exists())
        self.assertFalse(Job.objects.exists())

    def test_start_run_checks_credits_and_leaves_nothing_behind(self):
        from apps.accounts import quota
        run = PipelineRun.objects.create(pipeline=self.pipeline, user=self.user)
        with self.assertRaises(quota.QuotaExceeded):
            engine.start_run(run, steps=engine.steps_with_video(self.pipeline, self.big.pk))
        self.assertFalse(PipelineRun.objects.exists())
        self.assertFalse(Job.objects.exists())

    def test_rerun_is_gated(self):
        old = PipelineRun.objects.create(pipeline=self.pipeline, user=self.user,
                                         steps=engine.steps_with_video(self.pipeline, self.big.pk))
        self.client.force_login(self.user)
        r = self.client.post(f"/pipelines/{self.pipeline.pk}/run/{old.pk}/rerun/")
        self.assertEqual(r.status_code, 302)
        self.assertEqual(PipelineRun.objects.count(), 1)
        self.assertFalse(Job.objects.exists())


class CustomModelTests(Base):
    def setUp(self):
        super().setUp()
        self.small = Video.objects.create(user=self.user, title="s", storage_key="k/s", file_size_bytes=1,
                                          status=Video.Status.READY, duration_seconds=60)
        other = User.objects.create_user("other")
        self.private = CustomModel.objects.create(user=other, name="theirs", base_model="yolo",
                                                  storage_key="models/other/best.pt",
                                                  status=CustomModel.Status.READY)
        self.mine = CustomModel.objects.create(user=self.user, name="mine", base_model="yolo",
                                               storage_key="models/bee/best.pt",
                                               status=CustomModel.Status.READY)

    def create(self, config):
        return self.api.post("/api/v1/jobs/", {"video_id": self.small.pk, "config": config}, format="json")

    def test_someone_elses_private_model_is_refused_by_path_or_id(self):
        self.assertEqual(self.create({"custom_bee_model_path": "models/other/best.pt"}).status_code, 400)
        self.assertEqual(self.create({"custom_bee_model": self.private.pk}).status_code, 400)
        self.assertEqual(self.create({"custom_bee_model_path": "anything/at/all.pt"}).status_code, 400)
        self.assertFalse(Job.objects.exists())

    def test_own_model_by_id_is_stored_as_its_path(self):
        r = self.create({"custom_bee_model": self.mine.pk})
        self.assertEqual(r.status_code, 201, r.content)
        self.assertEqual(Job.objects.get().config["custom_bee_model_path"], "models/bee/best.pt")

    def test_a_published_model_is_usable(self):
        self.private.visibility = "public"
        self.private.save()
        self.assertEqual(self.create({"custom_bee_model": self.private.pk}).status_code, 201)

    def test_batch_submit_checks_models_too(self):
        r = self.api.post("/api/v1/jobs/batch_submit/",
                          {"video_ids": [self.small.pk], "config": {"custom_nest_model_path": "models/other/best.pt"}},
                          format="json")
        self.assertEqual(r.status_code, 400, r.content)


class VideoSourceTests(Base):
    def test_a_video_cannot_borrow_another_users_source(self):
        other = User.objects.create_user("other")
        theirs = DataSource.objects.create(user=other, name="s3", source_type="aws_s3", config_encrypted=b"x")
        r = self.api.patch(f"/api/v1/videos/{self.big.pk}/", {"source": theirs.pk}, format="json")
        self.assertEqual(r.status_code, 400)
        self.big.refresh_from_db()
        self.assertIsNone(self.big.source_id)


class RegisterTests(TestCase):
    def test_weak_passwords_are_refused(self):
        r = self.client.post("/api/v1/auth/register/",
                             {"username": "newbee", "email": "n@example.com", "password": "password"})
        self.assertEqual(r.status_code, 400)
        self.assertFalse(User.objects.filter(username="newbee").exists())


class UploadNameTests(Base):
    def test_path_segments_are_dropped_from_the_key(self):
        s3 = mock.Mock()
        s3.generate_presigned_url.return_value = "https://example/put"
        self.client.force_login(self.user)
        with mock.patch("apps.api.web_uploads.get_s3_client", return_value=s3):
            r = self.client.post("/api/v1/web-uploads/initiate",
                                 {"filename": "../../9/x/clip.mp4", "size_bytes": 10})
        self.assertEqual(r.status_code, 200, r.content)
        key = s3.generate_presigned_url.call_args[0][1]
        self.assertTrue(key.startswith(f"{self.user.pk}/"))
        self.assertTrue(key.endswith("/clip.mp4"))
        self.assertNotIn("..", key)


@override_settings(BEEMONITOR_CREDENTIAL_KEY=KEY)
class SourceCredentialTests(Base):
    def test_api_created_credentials_are_encrypted(self):
        from apps.sources import crypto
        r = self.api.post("/api/v1/sources/", {"name": "lab", "source_type": "aws_s3",
                                                "credentials": {"secret_access_key": "TOPSECRET"}},
                          format="json")
        self.assertIn(r.status_code, (200, 201), r.content)
        src = DataSource.objects.get(name="lab")
        self.assertNotIn(b"TOPSECRET", bytes(src.config_encrypted))
        self.assertEqual(crypto.decrypt(src)["secret_access_key"], "TOPSECRET")

    def test_old_plain_json_credentials_still_read(self):
        from apps.sources import crypto
        src = DataSource.objects.create(user=self.user, name="old", source_type="aws_s3",
                                        config_encrypted=b'{"bucket": "b"}')
        self.assertEqual(crypto.decrypt(src), {"bucket": "b"})
