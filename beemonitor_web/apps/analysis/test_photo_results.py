"""A photo job's results page shows the photo.

Job lookups read through ``Video.accessible``, which is clips only unless asked
for photos, so the run page's "Full job results" link 404'd on every photo.
"""

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from apps.analysis.models import Job, JobResult
from apps.pipelines.tests.test_photo_pipelines import PHOTO
from apps.videos.models import Video

User = get_user_model()


class PhotoResultsTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("pr", password="x")
        self.client.force_login(self.user)
        self.photo = Video.everything.create(user=self.user, title="p", storage_key="1/p.jpg",
                                             file_size_bytes=1, kind=Video.Kind.PHOTO)
        self.job = Job.objects.create(user=self.user, video=self.photo, status="completed",
                                      modal_job_id="pr-1", config={"task": "detect_photo"})
        JobResult.objects.create(job=self.job, summary_stats={"photo": PHOTO})

    def test_the_results_page_shows_the_photo_and_its_insects(self):
        resp = self.client.get(reverse("analysis:results", kwargs={"pk": self.job.pk}))

        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Bombus impatiens")
        self.assertNotContains(resp, "Generate annotated video")
        self.assertContains(resp, 'id="crop-viewer"', count=1)
        self.assertContains(resp, "data-crop ")

    def test_the_job_detail_page_opens(self):
        resp = self.client.get(reverse("analysis:detail", kwargs={"pk": self.job.pk}))

        self.assertEqual(resp.status_code, 200)

    def test_a_stranger_still_gets_a_404(self):
        self.client.force_login(User.objects.create_user("other", password="x"))

        resp = self.client.get(reverse("analysis:results", kwargs={"pk": self.job.pk}))

        self.assertEqual(resp.status_code, 404)
