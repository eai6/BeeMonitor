"""A batch's public share link (memory/47).

Anyone with the link can review the batch without signing in, and the link
reaches that batch and nothing else: not another batch's clips, not re-running,
not GPU time or error text, and not site or location unless the owner says so.
"""
from unittest import mock

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from apps.analysis.models import Job, JobResult
from apps.devices.models import Device
from apps.pipelines import sharing
from apps.pipelines.models import BatchShare, Pipeline, PipelineRun
from apps.videos.models import Video

User = get_user_model()

BATCH = "3f2a91c0-0000-4000-8000-00000000a11c"
OTHER_BATCH = "3f2a91c0-0000-4000-8000-00000000b22d"


class BatchShareTests(TestCase):
    def setUp(self):
        self.owner = User.objects.create_user("owner", password="x")
        self.device = Device.objects.create(
            owner=self.owner, name="Unit 3", key_hash="hs1", prefix="bmk_s1",
            location="North meadow plot 4")
        self.pipeline = Pipeline.objects.create(user=self.owner, title="Pollen assay")
        self.video, self.job = self._clip(BATCH, "a")
        self.other_video, self.other_job = self._clip(OTHER_BATCH, "b")
        failed_video = Video.objects.create(
            user=self.owner, device=self.device, title="f", storage_key="s/f.mp4",
            file_size_bytes=1, status=Video.Status.READY, recorded_at=timezone.now(),
            duration_seconds=600.0)  # a known length: the owner's page won't probe S3
        PipelineRun.objects.create(
            pipeline=self.pipeline, user=self.owner, batch_id=BATCH, status="failed",
            steps=[{"id": "v", "block_type": "input.video",
                    "config": {"video_id": str(failed_video.pk)}}],
            context={"v": {"artifact": "video", "video_id": failed_video.pk},
                     "t": {"error": "Traceback: /opt/program/secret_path.py exploded"}})

    def _clip(self, batch_id, name):
        video = Video.objects.create(
            user=self.owner, device=self.device, title=name, storage_key=f"s/{name}.mp4",
            file_size_bytes=1, status=Video.Status.READY, recorded_at=timezone.now(),
            duration_seconds=600.0)
        job = Job.objects.create(user=self.owner, video=video, status="completed", modal_job_id=f"m-{name}",
                                 execution_seconds=1234)
        JobResult.objects.create(job=job, tracking_csv_path=f"1/{name}/tracking_results.csv")
        PipelineRun.objects.create(
            pipeline=self.pipeline, user=self.owner, batch_id=batch_id, status="completed",
            steps=[{"id": "v", "block_type": "input.video",
                    "config": {"video_id": str(video.pk)}}],
            context={"v": {"artifact": "video", "video_id": video.pk},
                     "t": {"job_id": job.pk,
                           "result": {"tracking_csv_path": f"1/{name}/tracking_results.csv"}},
                     "i": {"artifact": "table", "table_kind": "interactions",
                           "interaction_count": 3, "total_duration_sec": 12.5,
                           "per_reference": [
                               {"id": "reference_12", "label": "Reference 12",
                                "interactions": 2, "partners": 2, "duration_sec": 10.0},
                               {"id": "reference_5", "label": "Reference 5",
                                "interactions": 1, "partners": 1, "duration_sec": 2.5}]}})
        return video, job

    def _turn_on(self, **options):
        self.client.force_login(self.owner)
        data = {"show_videos": "on", **options}
        self.client.post(reverse("pipelines:batch_share", args=[BATCH]), data)
        self.client.logout()
        return BatchShare.objects.get(batch_id=BATCH, revoked_at=None)

    def _public(self, share, name="public_batch", *args):
        return self.client.get(reverse(name, args=[share.token, *args]))

    # ── the owner ────────────────────────────────────────────────────────────

    def test_the_owner_turns_a_link_on_with_videos_and_without_locations(self):
        share = self._turn_on()

        self.assertTrue(share.show_videos)
        self.assertFalse(share.show_locations)
        self.client.force_login(self.owner)
        html = self.client.get(reverse("pipelines:batch_detail", args=[BATCH])).content.decode()
        self.assertIn(f"/s/{share.token}/", html)
        self.assertIn("Turn off link", html)

    def test_only_the_launcher_can_share(self):
        stranger = User.objects.create_user("stranger", password="x")
        self.client.force_login(stranger)

        r = self.client.post(reverse("pipelines:batch_share", args=[BATCH]), {"show_videos": "on"})

        self.assertEqual(r.status_code, 404)
        self.assertFalse(BatchShare.objects.exists())

    def test_turning_off_kills_the_link_and_turning_on_makes_a_new_one(self):
        old = self._turn_on()
        self.client.force_login(self.owner)
        self.client.post(reverse("pipelines:batch_share", args=[BATCH]), {"action": "off"})
        self.client.logout()

        r = self._public(old)
        self.assertEqual(r.status_code, 404)
        self.assertContains(r, "isn't shared any more", status_code=404)

        new = self._turn_on()
        self.assertNotEqual(new.token, old.token)
        self.assertEqual(self._public(old).status_code, 404)
        self.assertEqual(self._public(new).status_code, 200)

    # ── the public page ──────────────────────────────────────────────────────

    def test_anyone_with_the_link_sees_the_results_without_signing_in(self):
        share = self._turn_on()

        r = self._public(share)

        self.assertEqual(r.status_code, 200)
        self.assertEqual(r["X-Robots-Tag"], "noindex, nofollow")
        html = r.content.decode()
        self.assertIn("Pollen assay", html)
        self.assertIn("reference_12", html)
        self.assertIn("10.0 s", html)
        self.assertIn("80%", html)  # reference_12's share of 12.5 s of contact
        self.assertIn(f"/s/{share.token}/data/interactions.csv", html)

    def test_the_public_page_leaves_out_what_is_private(self):
        html = self._public(self._turn_on()).content.decode()

        self.assertNotIn("North meadow", html)       # location is off
        self.assertNotIn("secret_path", html)        # no error text
        self.assertIn("didn't finish", html)
        self.assertNotIn("GPU time", html)
        self.assertNotIn(reverse("pipelines:run_on_videos"), html)
        self.assertNotIn("/analysis/", html)         # nothing that needs a login
        self.assertNotIn("/videos/", html)

    def test_locations_show_when_the_owner_shares_them(self):
        html = self._public(self._turn_on(show_locations="on")).content.decode()

        self.assertIn("North meadow plot 4", html)

    def test_views_are_counted(self):
        share = self._turn_on()
        self._public(share)
        self._public(share)

        self.assertEqual(BatchShare.objects.get(pk=share.pk).view_count, 2)

    # ── media and tables under the link ─────────────────────────────────────

    def test_a_clip_in_the_batch_streams_with_a_short_lived_url(self):
        share = self._turn_on()
        s3 = mock.Mock()
        s3.generate_presigned_url.return_value = "https://s3.example/clip?sig"

        with mock.patch("config.storage.get_s3_client", return_value=s3):
            r = self._public(share, "public_video", self.video.pk)

        self.assertEqual(r.status_code, 302)
        self.assertEqual(r["Location"], "https://s3.example/clip?sig")
        self.assertEqual(s3.generate_presigned_url.call_args.args[2],
                         sharing.PUBLIC_MEDIA_HOURS)

    def test_the_link_cannot_reach_another_batchs_clips(self):
        share = self._turn_on()

        with mock.patch("config.storage.get_s3_client") as s3:
            self.assertEqual(self._public(share, "public_video", self.other_video.pk).status_code, 404)
            self.assertEqual(self._public(share, "public_thumbnail", self.other_video.pk).status_code, 404)
            self.assertEqual(self._public(share, "public_overlay", self.other_job.pk).status_code, 404)
        s3.assert_not_called()

    def test_with_videos_off_there_is_no_player_and_no_media(self):
        share = self._turn_on(show_videos="")
        share.show_videos = False
        share.save()

        html = self._public(share).content.decode()
        self.assertNotIn("data-original", html)
        self.assertNotIn("clip-viewer", html)
        self.assertEqual(self._public(share, "public_video", self.video.pk).status_code, 404)
        self.assertEqual(self._public(share, "public_overlay", self.job.pk).status_code, 404)

    def test_public_csvs_drop_site_and_location_unless_shared(self):
        share = self._turn_on()
        table = ("interactions_batch_3f2a91c0.csv",
                 ["video_title", "device_name", "site_name", "location", "duration_sec"],
                 [{"video_title": "a", "device_name": "Unit 3", "site_name": "Meadow",
                   "location": "North meadow plot 4", "duration_sec": 1.0}])

        with mock.patch.object(sharing, "batch_csv", return_value=table):
            body = self._public(share, "public_batch_csv", "interactions").content.decode()
        self.assertTrue(body.startswith("video_title,device_name,duration_sec"))
        self.assertNotIn("North meadow", body)

        share.show_locations = True
        share.save()
        with mock.patch.object(sharing, "batch_csv", return_value=table):
            body = self._public(share, "public_batch_csv", "interactions").content.decode()
        self.assertIn("North meadow plot 4", body)

    def test_an_unknown_token_is_the_link_off_page(self):
        r = self.client.get(reverse("public_batch", args=["not-a-real-token"]))

        self.assertEqual(r.status_code, 404)
        self.assertContains(r, "isn't shared any more", status_code=404)
