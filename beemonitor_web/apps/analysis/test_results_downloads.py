"""The results page offers the raw detections table, and only when it exists.

Detections are the one base table the batch page could hand back that the
per-clip page could not, so anyone wanting pre-tracking boxes for a single clip
had to export a batch of one. They are a genuinely different measurement from
tracking: tracking holds only detections the tracker associated into a
confirmed track, so the two counts legitimately differ.
"""

from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from apps.analysis.models import Job, JobResult
from apps.videos.models import Video

User = get_user_model()


class DetectionsDownloadTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("dl", password="x")
        self.client.force_login(self.user)
        self.video = Video.objects.create(
            user=self.user, title="clip", storage_key="dl/c.mp4",
            file_size_bytes=1, status=Video.Status.READY,
            recorded_at=timezone.now(), fps=25.0, width=1920, height=1080)
        self.job = Job.objects.create(user=self.user, video=self.video,
                                      status="completed", modal_job_id="dl-1")

    def _page(self):
        with patch("apps.analysis.views._generate_presigned_url",
                   side_effect=lambda p, **kw: f"https://signed.test/{p}" if p else ""):
            return self.client.get(
                reverse("analysis:results", kwargs={"pk": self.job.pk})
            ).content.decode()

    def test_a_result_with_detections_offers_the_download(self):
        JobResult.objects.create(job=self.job, detections_csv_path="dl/dl-1/detections.csv")

        html = self._page()

        self.assertIn("Detections CSV", html)
        self.assertIn("https://signed.test/dl/dl-1/detections.csv", html)

    def test_a_result_without_detections_offers_no_button(self):
        """A button that 404s reads as a broken download, not a skipped step.

        Presigning never checks existence, so the path must come from the
        result rather than being constructed from the job id.
        """
        JobResult.objects.create(job=self.job, tracking_csv_path="dl/dl-1/tracking.csv")

        html = self._page()

        self.assertNotIn("Detections CSV", html)
        self.assertIn("Tracking CSV", html)

    def test_detections_alone_still_renders_the_downloads_panel(self):
        """The panel's own gate has to know about the new table."""
        JobResult.objects.create(job=self.job, detections_csv_path="dl/dl-1/detections.csv")

        html = self._page()

        self.assertIn("Downloads", html)

    def test_a_stranger_cannot_see_the_clips_downloads(self):
        JobResult.objects.create(job=self.job, detections_csv_path="dl/dl-1/detections.csv")
        self.client.force_login(User.objects.create_user("nosy", password="x"))

        with patch("apps.analysis.views._generate_presigned_url", return_value="u"):
            resp = self.client.get(
                reverse("analysis:results", kwargs={"pk": self.job.pk}))

        self.assertEqual(resp.status_code, 404)
