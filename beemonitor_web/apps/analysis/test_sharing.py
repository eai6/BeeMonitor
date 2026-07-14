"""Device-share users (viewer/manager) can SEE analysis-run results on shared
devices' videos; write paths stay owner/manager. (memory/27 follow-up.)"""

from datetime import datetime, timezone as dt_timezone

from django.contrib.auth.models import User
from django.test import TestCase

from apps.analysis.models import Job, JobResult
from apps.devices.models import Device, DeviceShare
from apps.pipelines.models import Pipeline, PipelineRun
from apps.videos.models import Video

UTC = dt_timezone.utc


class RunResultSharingTest(TestCase):
    def setUp(self):
        self.owner = User.objects.create_user("owner", password="x")
        self.viewer = User.objects.create_user("viewer", password="x")
        self.manager = User.objects.create_user("manager", password="x")
        self.stranger = User.objects.create_user("stranger", password="x")

        self.device = Device.objects.create(owner=self.owner, name="unit-1")
        DeviceShare.objects.create(device=self.device, user=self.viewer,
                                   role="viewer", created_by=self.owner)
        DeviceShare.objects.create(device=self.device, user=self.manager,
                                   role="manager", created_by=self.owner)

        self.video = Video.objects.create(
            user=self.owner, device=self.device, title="sharedclip42", storage_key="k",
            file_size_bytes=0, site_name="S",
            recorded_at=datetime(2026, 7, 10, 12, 0, tzinfo=UTC))
        self.job = Job.objects.create(user=self.owner, video=self.video,
                                      status=Job.Status.COMPLETED)
        JobResult.objects.create(job=self.job, total_events=5,
                                 entry_count=2, exit_count=3)

    def _get(self, who, url):
        self.client.logout()
        self.client.login(username=who, password="x")
        return self.client.get(url)

    def test_shared_users_can_read_results_stranger_cannot(self):
        for path in (f"/analysis/{self.job.pk}/",
                     f"/analysis/{self.job.pk}/results/"):
            self.assertEqual(self._get("viewer", path).status_code, 200, path)
            self.assertEqual(self._get("manager", path).status_code, 200, path)
            self.assertEqual(self._get("stranger", path).status_code, 404, path)

    def test_processing_hub_shows_shared_devices_jobs(self):
        r = self._get("viewer", "/analysis/processing/")
        self.assertContains(r, "sharedclip42")
        r = self._get("stranger", "/analysis/processing/")
        self.assertNotContains(r, "sharedclip42")

    def test_annotated_render_is_manager_plus(self):
        # Viewer: 404 (write-ish, costs GPU). Manager: passes the gate — job has
        # no tracking data so it redirects with a warning instead of rendering.
        self.client.logout(); self.client.login(username="viewer", password="x")
        resp = self.client.post(f"/analysis/{self.job.pk}/annotate/")
        self.assertEqual(resp.status_code, 404)
        self.client.logout(); self.client.login(username="manager", password="x")
        resp = self.client.post(f"/analysis/{self.job.pk}/annotate/")
        self.assertEqual(resp.status_code, 302)  # gate passed; no-tracking redirect


class PipelineRunSharingTest(TestCase):
    def setUp(self):
        self.owner = User.objects.create_user("owner", password="x")
        self.viewer = User.objects.create_user("viewer", password="x")
        self.stranger = User.objects.create_user("stranger", password="x")
        self.device = Device.objects.create(owner=self.owner, name="unit-1")
        DeviceShare.objects.create(device=self.device, user=self.viewer,
                                   role="viewer", created_by=self.owner)
        self.video = Video.objects.create(
            user=self.owner, device=self.device, title="clip", storage_key="k",
            file_size_bytes=0,
            recorded_at=datetime(2026, 7, 10, 12, 0, tzinfo=UTC))
        self.pipe = Pipeline.objects.create(user=self.owner, title="p", steps=[])
        self.run = PipelineRun.objects.create(
            user=self.owner, pipeline=self.pipe, batch_id=None,
            steps=[{"id": "s1", "block_type": "input.video",
                    "config": {"video_id": self.video.pk}}])

    def _get(self, who, url):
        self.client.logout()
        self.client.login(username=who, password="x")
        return self.client.get(url)

    def test_run_csv_visible_to_shared_user_only(self):
        from django.urls import reverse
        url = reverse("pipelines:run_output_csv",
                      args=[self.pipe.pk, self.run.pk, "s1"])
        self.assertEqual(self._get("owner", url).status_code, 200)
        self.assertEqual(self._get("viewer", url).status_code, 200)
        self.assertEqual(self._get("stranger", url).status_code, 404)
