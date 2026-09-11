"""The annotated-video proxy, and what it does when ffmpeg is not installed.

The worker writes its overlay with OpenCV's "mp4v" fourcc (MPEG-4 Part 2, see
config.video_codec), which Chrome and Safari will not play — so the re-encode is
load-bearing, not a nicety. The web image shipped without ffmpeg or ffprobe, so
every playback downloaded the whole video, failed both subprocess calls, logged
an ERROR, and redirected to a file the browser could not decode.
"""

from pathlib import Path
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from apps.analysis import video_proxy
from apps.analysis.models import Job, JobResult
from apps.videos.models import Video

User = get_user_model()


class ToolDetectionTests(TestCase):
    def setUp(self):
        video_proxy._tool_on_path.cache_clear()

    def tearDown(self):
        video_proxy._tool_on_path.cache_clear()

    def test_a_present_binary_is_reported_found(self):
        with patch("apps.analysis.video_proxy.shutil.which", return_value="/usr/bin/ffmpeg"):
            self.assertTrue(video_proxy._tool_on_path("ffmpeg"))

    def test_a_missing_binary_is_logged_as_a_deployment_defect(self):
        with patch("apps.analysis.video_proxy.shutil.which", return_value=None):
            with self.assertLogs("apps.analysis.video_proxy", level="ERROR") as logs:
                self.assertFalse(video_proxy._tool_on_path("ffmpeg"))

        self.assertIn("not installed in this image", logs.output[0])

    def test_the_answer_is_resolved_once_per_process(self):
        """A per-request which() call would re-log on every playback."""
        with patch("apps.analysis.video_proxy.shutil.which", return_value=None) as which:
            video_proxy._tool_on_path("ffmpeg")
            video_proxy._tool_on_path("ffmpeg")

        self.assertEqual(which.call_count, 1)


class ProxyFallbackTests(TestCase):
    def setUp(self):
        video_proxy._tool_on_path.cache_clear()
        self.user = User.objects.create_user("vp", password="x")
        self.client.force_login(self.user)
        self.video = Video.objects.create(
            user=self.user, title="clip", storage_key="vp/c.mp4",
            file_size_bytes=1, status=Video.Status.READY,
            recorded_at=timezone.now())
        self.job = Job.objects.create(user=self.user, video=self.video,
                                      status="completed", modal_job_id="vp-1")
        JobResult.objects.create(job=self.job,
                                 annotated_video_path="vp/vp-1/annotated_video.mp4")

    def tearDown(self):
        video_proxy._tool_on_path.cache_clear()

    def _get(self):
        return self.client.get(reverse("analysis:video_proxy",
                                       kwargs={"pk": self.job.pk}))

    def test_without_ffmpeg_it_redirects_without_downloading_the_video(self):
        """The download could not change the outcome, so it must not happen."""
        with patch("apps.analysis.video_proxy.shutil.which", return_value=None), \
             patch("apps.analysis.video_proxy._blob_exists", return_value=False), \
             patch("apps.analysis.video_proxy._download_blob") as download, \
             patch("apps.analysis.video_proxy._generate_presigned_url",
                   return_value="https://signed.test/original.mp4"):
            resp = self._get()

        self.assertEqual(resp.status_code, 302)
        self.assertEqual(resp["Location"], "https://signed.test/original.mp4")
        download.assert_not_called()

    def test_a_cached_h264_render_is_served_without_touching_ffmpeg(self):
        with patch("apps.analysis.video_proxy._blob_exists", return_value=True), \
             patch("apps.analysis.video_proxy._generate_presigned_url",
                   return_value="https://signed.test/cached_h264.mp4") as presign:
            resp = self._get()

        self.assertEqual(resp.status_code, 302)
        self.assertIn("cached_h264", resp["Location"])
        self.assertIn("_h264.mp4", presign.call_args[0][0])

    def test_with_ffmpeg_present_it_re_encodes_and_caches(self):
        def fake_download(container, blob, local):
            Path(local).write_bytes(b"not really a video")
            return True

        def fake_encode(src, dst):
            Path(dst).write_bytes(b"h264 bytes")
            return True

        with patch("apps.analysis.video_proxy.shutil.which", return_value="/usr/bin/ffmpeg"), \
             patch("apps.analysis.video_proxy._blob_exists", return_value=False), \
             patch("apps.analysis.video_proxy._download_blob", side_effect=fake_download), \
             patch("apps.analysis.video_proxy._is_h264", return_value=False), \
             patch("apps.analysis.video_proxy._reencode_to_h264",
                   side_effect=fake_encode) as encode, \
             patch("apps.analysis.video_proxy._upload_blob", return_value=True) as upload, \
             patch("apps.analysis.video_proxy._generate_presigned_url",
                   return_value="https://signed.test/new_h264.mp4"):
            resp = self._get()

        encode.assert_called_once()
        self.assertIn("_h264.mp4", upload.call_args[0][1])
        self.assertEqual(resp.status_code, 302)

    def test_a_stranger_cannot_stream_someone_elses_render(self):
        self.client.force_login(User.objects.create_user("nosy", password="x"))

        self.assertEqual(self._get().status_code, 404)


class EncoderErrorTests(TestCase):
    def test_a_missing_ffmpeg_binary_reads_as_a_deployment_defect(self):
        with patch("apps.analysis.video_proxy.subprocess.run",
                   side_effect=FileNotFoundError("ffmpeg")):
            with self.assertLogs("apps.analysis.video_proxy", level="ERROR") as logs:
                ok = video_proxy._reencode_to_h264("in.mp4", "out.mp4")

        self.assertFalse(ok)
        self.assertIn("deployment", logs.output[0])
