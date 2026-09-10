"""The review grid's still: which frame, and what happens when it can't be had.

The rule that matters is NOT frame 0. The recorder writes a 3-second pre-roll
in front of every motion clip (hardware/motion/config.py PRE_ROLL = 3.0), so
frame 0 is reliably an empty hotel — thumbnails of nothing, and a grid as
uniform as having none. We sample at the pre-roll mark, where motion starts.
"""

from unittest.mock import MagicMock, patch

from django.contrib.auth import get_user_model
from django.test import TestCase

from apps.videos import thumbnails
from apps.videos.models import Video

User = get_user_model()


class FakeCapture:
    """cv2.VideoCapture stand-in that records which frame was asked for."""

    CAP_PROP_FPS = 5
    CAP_PROP_FRAME_COUNT = 7
    CAP_PROP_POS_FRAMES = 1
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4

    def __init__(self, fps=25.0, total=300, opened=True, readable_from=0):
        self._fps, self._total, self._opened = fps, total, opened
        self._readable_from = readable_from
        self.seeks = []
        self.pos = 0
        self.released = False

    def isOpened(self):
        return self._opened

    def get(self, prop):
        return {self.CAP_PROP_FPS: self._fps,
                self.CAP_PROP_FRAME_COUNT: self._total,
                self.CAP_PROP_FRAME_WIDTH: 1920,
                self.CAP_PROP_FRAME_HEIGHT: 1080}.get(prop, 0)

    def set(self, prop, value):
        self.seeks.append(int(value))
        self.pos = int(value)

    def read(self):
        import numpy as np
        if self.pos < self._readable_from:
            return False, None
        return True, np.zeros((1080, 1920, 3), dtype=np.uint8)

    def release(self):
        self.released = True


def fake_cv2(capture):
    cv2 = MagicMock()
    cv2.CAP_PROP_FPS = FakeCapture.CAP_PROP_FPS
    cv2.CAP_PROP_FRAME_COUNT = FakeCapture.CAP_PROP_FRAME_COUNT
    cv2.CAP_PROP_POS_FRAMES = FakeCapture.CAP_PROP_POS_FRAMES
    cv2.CAP_PROP_FRAME_WIDTH = FakeCapture.CAP_PROP_FRAME_WIDTH
    cv2.CAP_PROP_FRAME_HEIGHT = FakeCapture.CAP_PROP_FRAME_HEIGHT
    cv2.VideoCapture.return_value = capture
    cv2.resize.side_effect = lambda frame, size, interpolation=None: frame
    cv2.imencode.return_value = (True, MagicMock(tobytes=lambda: b"jpeg"))
    return cv2


class FrameChoiceTests(TestCase):
    def test_samples_at_the_pre_roll_mark_not_frame_zero(self):
        cap = FakeCapture(fps=25.0, total=300)

        frame, _props = thumbnails._grab_frame(fake_cv2(cap), "clip.mp4")

        self.assertIsNotNone(frame)
        # 3.0 s x 25 fps — the moment motion was detected, not the empty
        # hotel three seconds earlier.
        self.assertEqual(cap.seeks[0], 75)

    def test_scales_the_target_with_the_clips_frame_rate(self):
        cap = FakeCapture(fps=50.0, total=600)

        thumbnails._grab_frame(fake_cv2(cap), "clip.mp4")

        self.assertEqual(cap.seeks[0], 150)

    def test_a_clip_shorter_than_the_pre_roll_falls_back_to_its_middle(self):
        cap = FakeCapture(fps=25.0, total=40)  # 1.6 s — 75 is past the end

        thumbnails._grab_frame(fake_cv2(cap), "clip.mp4")

        self.assertEqual(cap.seeks, [20])

    def test_an_unreadable_seek_falls_back_rather_than_giving_up(self):
        # Seeking works but no frame decodes until the very start.
        cap = FakeCapture(fps=25.0, total=300, readable_from=0)
        cap._readable_from = 1  # frame 0 readable only via the final fallback
        cv2 = fake_cv2(cap)

        frame, _props = thumbnails._grab_frame(cv2, "clip.mp4")

        self.assertIsNotNone(frame)

    def test_an_unopenable_file_yields_nothing(self):
        cap = FakeCapture(opened=False)

        self.assertEqual(thumbnails._grab_frame(fake_cv2(cap), "clip.mp4"), (None, {}))

    def test_the_capture_is_always_released(self):
        cap = FakeCapture()

        thumbnails._grab_frame(fake_cv2(cap), "clip.mp4")

        self.assertTrue(cap.released)


class ExtractionTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("alice", password="x")
        self.video = Video.objects.create(
            user=self.user, title="clip", storage_key="alice/clip.mp4",
            file_size_bytes=1, status=Video.Status.READY,
        )

    def test_stores_the_key_on_the_video(self):
        cap = FakeCapture()
        s3 = MagicMock()
        with patch.dict("sys.modules", {"cv2": fake_cv2(cap)}), \
             patch("config.storage.get_s3_client", return_value=s3):
            key = thumbnails.extract_thumbnail(self.video)

        self.assertEqual(key, "thumbs/alice_clip.mp4.jpg")
        self.video.refresh_from_db()
        self.assertEqual(self.video.thumbnail_key, key)
        s3.upload_stream.assert_called_once()

    def test_a_fully_known_clip_is_not_touched_again(self):
        self.video.thumbnail_key = "thumbs/already.jpg"
        self.video.fps, self.video.duration_seconds = 25.0, 12.0
        self.video.save(update_fields=["thumbnail_key", "fps", "duration_seconds"])
        s3 = MagicMock()

        with patch("config.storage.get_s3_client", return_value=s3):
            key = thumbnails.extract_thumbnail(self.video)

        self.assertEqual(key, "thumbs/already.jpg")
        s3.download_file.assert_not_called()

    def test_a_clip_with_a_still_but_no_measurements_is_still_probed(self):
        """Having a thumbnail used to mean never learning the clip's length.

        Every clip uploaded before the property probe existed has a still and
        no fps or duration, so the early return made "unknown length" permanent
        — which is why the batch page could not say how long anything was.
        """
        self.video.thumbnail_key = "thumbs/already.jpg"
        self.video.save(update_fields=["thumbnail_key"])
        s3 = MagicMock()

        with patch.dict("sys.modules", {"cv2": fake_cv2(FakeCapture())}), \
             patch("config.storage.get_s3_client", return_value=s3):
            thumbnails.extract_thumbnail(self.video)

        s3.download_file.assert_called_once()
        self.video.refresh_from_db()
        self.assertEqual(self.video.fps, 25.0)
        self.assertEqual(self.video.duration_seconds, 12.0)

    def test_probing_an_existing_still_does_not_re_upload_it(self):
        self.video.thumbnail_key = "thumbs/already.jpg"
        self.video.save(update_fields=["thumbnail_key"])
        s3 = MagicMock()

        with patch.dict("sys.modules", {"cv2": fake_cv2(FakeCapture())}), \
             patch("config.storage.get_s3_client", return_value=s3):
            key = thumbnails.extract_thumbnail(self.video)

        self.assertEqual(key, "thumbs/already.jpg")
        s3.upload_stream.assert_not_called()

    def test_a_clip_still_in_an_external_bucket_is_skipped(self):
        self.video.storage_key = "s3://someone-elses/clip.mp4"
        self.video.save(update_fields=["storage_key"])
        s3 = MagicMock()

        with patch("config.storage.get_s3_client", return_value=s3):
            self.assertEqual(thumbnails.extract_thumbnail(self.video), "")
        s3.download_file.assert_not_called()

    def test_a_failure_is_swallowed_so_an_upload_never_breaks(self):
        s3 = MagicMock()
        s3.download_file.side_effect = RuntimeError("S3 is having a day")

        with patch.dict("sys.modules", {"cv2": fake_cv2(FakeCapture())}), \
             patch("config.storage.get_s3_client", return_value=s3):
            self.assertEqual(thumbnails.extract_thumbnail(self.video), "")

        self.video.refresh_from_db()
        self.assertEqual(self.video.thumbnail_key, "")


class OnDemandTests(TestCase):
    """Clips uploaded before stills existed still get one, without a backfill."""

    def setUp(self):
        self.user = User.objects.create_user("bob", password="x")
        self.video = Video.objects.create(
            user=self.user, title="old clip", storage_key="bob/old.mp4",
            file_size_bytes=1, status=Video.Status.READY,
        )
        self.client.force_login(self.user)

    @staticmethod
    def _s3():
        s3 = MagicMock()
        s3.generate_presigned_url.return_value = "https://example.test/signed.jpg"
        return s3

    def test_the_endpoint_makes_a_missing_still_on_first_ask(self):
        cap = FakeCapture()
        with patch.dict("sys.modules", {"cv2": fake_cv2(cap)}), \
             patch("config.storage.get_s3_client", return_value=self._s3()):
            r = self.client.get(f"/videos/{self.video.pk}/thumb/")

        self.assertEqual(r.status_code, 302)
        self.video.refresh_from_db()
        self.assertTrue(self.video.thumbnail_key)

    def test_an_existing_still_is_served_without_decoding_again(self):
        self.video.thumbnail_key = "thumbs/have.jpg"
        self.video.save(update_fields=["thumbnail_key"])
        s3 = self._s3()

        with patch("config.storage.get_s3_client", return_value=s3):
            r = self.client.get(f"/videos/{self.video.pk}/thumb/")

        self.assertEqual(r.status_code, 302)
        s3.download_file.assert_not_called()

    def test_a_clip_with_no_extractable_frame_404s_rather_than_erroring(self):
        cap = FakeCapture(opened=False)
        with patch.dict("sys.modules", {"cv2": fake_cv2(cap)}), \
             patch("config.storage.get_s3_client", return_value=self._s3()):
            r = self.client.get(f"/videos/{self.video.pk}/thumb/")

        self.assertEqual(r.status_code, 404)

    def test_a_busy_server_defers_instead_of_queueing(self):
        """A screen of lazy images must not start 20 decodes at once; the card
        stays dark and the next scroll past it tries again."""
        with patch.object(thumbnails._slots, "acquire", return_value=False):
            self.assertEqual(thumbnails.extract_on_demand(self.video), "")

    def test_a_stranger_cannot_make_the_server_decode_someone_elses_clip(self):
        stranger = User.objects.create_user("mallory", password="x")
        self.client.force_login(stranger)
        s3 = self._s3()

        with patch("config.storage.get_s3_client", return_value=s3):
            r = self.client.get(f"/videos/{self.video.pk}/thumb/")

        self.assertEqual(r.status_code, 404)
        s3.download_file.assert_not_called()

    def test_the_stream_endpoint_is_scoped_the_same_way(self):
        stranger = User.objects.create_user("eve", password="x")
        self.client.force_login(stranger)

        self.assertEqual(self.client.get(f"/videos/{self.video.pk}/stream/").status_code, 404)
