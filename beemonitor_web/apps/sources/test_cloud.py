"""Cloud storage (memory/54): connect an S3 bucket, import clips, keep them
there or copy them in, in the background, without ever using BeeMonitor's own
AWS role. Real S3 behaviour from moto: the user's bucket and ours.
"""

from unittest import mock

import boto3
from cryptography.fernet import Fernet
from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse
from moto import mock_aws

from apps.sources import copying, crypto, s3
from apps.sources.models import DataSource, SourceCopy
from apps.videos.models import Site, Video

User = get_user_model()
KEY = Fernet.generate_key().decode()
CLIP = b"\x00\x00\x00\x18ftypmp42" + b"x" * 2048
CREDS = {"bucket": "their-footage", "prefix": "2026/", "access_key_id": "AKIAUSER",
         "secret_access_key": "user-secret", "region": "us-east-1"}


@override_settings(BEEMONITOR_CREDENTIAL_KEY=KEY, AWS_S3_BUCKET_RAW_VIDEOS="our-raw",
                   AWS_S3_BUCKET_PROCESSED="our-processed", AWS_REGION="us-east-1")
class CloudTests(TestCase):
    def setUp(self):
        self.aws = mock_aws()
        self.aws.start()
        self.addCleanup(self.aws.stop)
        from config.storage import get_s3_client
        get_s3_client.cache_clear()
        self.addCleanup(get_s3_client.cache_clear)
        c = boto3.client("s3", region_name="us-east-1")
        for b in ("their-footage", "our-raw", "our-processed"):
            c.create_bucket(Bucket=b)
        c.put_object(Bucket="their-footage", Key="2026/june/nest_0412.mp4", Body=CLIP)
        c.put_object(Bucket="their-footage", Key="2026/june/notes.txt", Body=b"hi")
        c.put_object(Bucket="their-footage", Key="2026/top.mov", Body=CLIP)
        c.put_object(Bucket="their-footage", Key="private/elsewhere.mp4", Body=CLIP)
        self.s3 = c
        self.user = User.objects.create_user("bee", password="pw-12345678")
        self.client.force_login(self.user)
        # No ffmpeg in tests: measuring is checked separately.
        patcher = mock.patch("apps.sources.copying.measure")
        self.measure = patcher.start()
        self.addCleanup(patcher.stop)

    def source(self, **creds):
        return DataSource.objects.create(user=self.user, name="Lab", source_type="aws_s3",
                                         config_encrypted=crypto.encrypt({**CREDS, **creds}), is_connected=True)

    def run_copies(self):
        for pk in SourceCopy.objects.filter(status="queued").values_list("pk", flat=True):
            SourceCopy.objects.filter(pk=pk).update(status="running")
            copying.run(pk)

    # ── Connecting ──
    def test_connect_tests_the_bucket_and_never_shows_the_secret_again(self):
        data = {"name": "Lab", "s3_bucket": "their-footage", "s3_prefix": "2026", "s3_access_key_id": "AKIAUSER",
                "s3_secret_access_key": "user-secret", "s3_region": "us-east-1"}
        r = self.client.post(reverse("sources:add"), data)
        source = DataSource.objects.get()
        self.assertRedirects(r, reverse("sources:browse", args=[source.pk]), fetch_redirect_response=False)
        self.assertEqual(crypto.decrypt(source)["prefix"], "2026/")
        self.assertNotIn(b"user-secret", bytes(source.config_encrypted))
        r = self.client.post(reverse("sources:add"), {**data, "s3_bucket": "no-such-bucket"})
        self.assertContains(r, "no bucket by that name")
        self.assertNotContains(r, "user-secret")
        self.assertEqual(DataSource.objects.count(), 1)

    def test_a_source_without_keys_never_falls_back_to_our_role(self):
        with self.assertRaises(s3.SourceError):
            s3.client({"bucket": "their-footage"})
        from rest_framework.test import APIClient
        api = APIClient()
        api.force_authenticate(self.user)
        r = api.post("/api/v1/sources/", {"name": "x", "source_type": "aws_s3",
                                          "credentials": {"bucket": "their-footage"}}, format="json")
        self.assertEqual(r.status_code, 400)
        r = api.post("/api/v1/sources/", {"name": "g", "source_type": "gcs",
                                          "credentials": {"bucket": "b", "access_key_id": "a", "secret_access_key": "s"}},
                     format="json")
        self.assertEqual(r.status_code, 400)

    # ── Browsing ──
    def test_browse_shows_folders_and_videos_only(self):
        source = self.source()
        r = self.client.get(reverse("sources:browse", args=[source.pk]))
        self.assertEqual([f["name"] for f in r.context["files"]], ["top.mov"])
        self.assertEqual(r.context["folders"], ["june/"])
        r = self.client.get(reverse("sources:browse", args=[source.pk]), {"path": "june/"})
        self.assertEqual([f["name"] for f in r.context["files"]], ["nest_0412.mp4"])

    def test_another_users_source_is_not_found(self):
        source = self.source()
        self.client.force_login(User.objects.create_user("x"))
        self.assertEqual(self.client.get(reverse("sources:browse", args=[source.pk])).status_code, 404)

    # ── Importing ──
    def test_keep_in_my_bucket(self):
        source = self.source()
        site = Site.objects.create(user=self.user, name="North bank")
        self.client.post(reverse("sources:sync", args=[source.pk]),
                         {"path": "june/", "selected_files": ["2026/june/nest_0412.mp4"], "where": "bucket",
                          "site": site.pk})
        video = Video.objects.get()
        self.assertEqual(video.storage_key, "s3://their-footage/2026/june/nest_0412.mp4")
        self.assertEqual((video.site, video.file_size_bytes), (site, len(CLIP)))
        self.assertEqual(SourceCopy.objects.get().purpose, "measure")
        self.run_copies()
        video.refresh_from_db()
        self.assertTrue(video.storage_key.startswith("s3://"))          # nothing copied
        self.assertEqual(self.s3.list_objects_v2(Bucket="our-raw").get("KeyCount"), 0)
        self.measure.assert_called_once()

    def test_copy_into_beemonitor_streams_it_in_the_background(self):
        source = self.source()
        self.client.post(reverse("sources:sync", args=[source.pk]),
                         {"path": "june/", "selected_files": ["2026/june/nest_0412.mp4"], "where": "copy"})
        video = Video.objects.get()
        self.assertTrue(video.storage_key.startswith("s3://"))          # not in the request
        self.run_copies()
        video.refresh_from_db()
        self.assertEqual(video.storage_key, f"{self.user.pk}/{video.pk}/nest_0412.mp4")
        body = self.s3.get_object(Bucket="our-raw", Key=video.storage_key)
        self.assertEqual(body["Body"].read(), CLIP)
        self.assertEqual(body["ContentType"], "video/mp4")
        copy = SourceCopy.objects.get()
        self.assertEqual((copy.status, copy.bytes_done), ("done", len(CLIP)))
        self.assertEqual(video.metadata["remote_bucket"], "their-footage")   # still knows where it came from

    def test_importing_twice_skips_the_same_file(self):
        source = self.source()
        for _ in range(2):
            self.client.post(reverse("sources:sync", args=[source.pk]),
                             {"path": "", "selected_files": ["2026/top.mov"], "where": "bucket"})
        self.assertEqual(Video.objects.count(), 1)

    def test_the_site_comes_from_the_filename_and_is_made_if_new(self):
        source = self.source()
        self.s3.put_object(Bucket="their-footage", Key="2026/mendels_2024-04-16_15_21_24.mp4", Body=CLIP)
        self.s3.put_object(Bucket="their-footage", Key="2026/Oakridge_2024-04-16_15_30_00.mp4", Body=CLIP)
        self.s3.put_object(Bucket="their-footage", Key="2026/oakridge_2024-04-16_15_40_00.mp4", Body=CLIP)
        oak = Site.objects.create(user=self.user, name="oakridge")
        other = User.objects.create_user("other", password="pw-12345678")
        Site.objects.create(user=other, name="mendels")                  # someone else's
        r = self.client.post(reverse("sources:sync", args=[source.pk]),
                             {"selected_files": ["2026/mendels_2024-04-16_15_21_24.mp4",
                                                 "2026/Oakridge_2024-04-16_15_30_00.mp4",
                                                 "2026/oakridge_2024-04-16_15_40_00.mp4"],
                              "where": "bucket", "site": ""}, follow=True)
        mendels = Site.objects.get(user=self.user, name="mendels")
        by_title = {v.title: v for v in Video.objects.all()}
        self.assertEqual(by_title["mendels_2024-04-16_15_21_24"].site, mendels)
        self.assertEqual(by_title["Oakridge_2024-04-16_15_30_00"].site, oak)   # any capitals
        self.assertEqual(by_title["oakridge_2024-04-16_15_40_00"].site, oak)
        self.assertEqual(Site.objects.filter(user=self.user).count(), 2)
        self.assertEqual(by_title["mendels_2024-04-16_15_21_24"].recorded_at.hour, 15)
        self.assertContains(r, "New site from the filenames: mendels.")

    def test_no_site_and_a_chosen_site_override_the_filename(self):
        source = self.source()
        self.s3.put_object(Bucket="their-footage", Key="2026/mendels_2024-04-16_15_21_24.mp4", Body=CLIP)
        key = ["2026/mendels_2024-04-16_15_21_24.mp4"]
        self.client.post(reverse("sources:sync", args=[source.pk]),
                         {"selected_files": key, "where": "bucket", "site": "none"})
        self.assertIsNone(Video.objects.get().site)
        self.assertFalse(Site.objects.exists())
        Video.objects.all().delete()
        north = Site.objects.create(user=self.user, name="North bank")
        self.client.post(reverse("sources:sync", args=[source.pk]),
                         {"selected_files": key, "where": "bucket", "site": north.pk})
        self.assertEqual(Video.objects.get().site, north)

    def test_a_filename_without_a_site_gets_none(self):
        source = self.source()
        self.client.post(reverse("sources:sync", args=[source.pk]),
                         {"selected_files": ["2026/june/nest_0412.mp4"], "where": "bucket"})
        self.assertIsNone(Video.objects.get().site)
        self.assertFalse(Site.objects.exists())

    def test_odd_filenames_import_without_a_site(self):
        source = self.source()
        odd = ["2026/" + n for n in (
            "clip.mp4", "_2024-04-16_15_21_24.mp4", "   _2024-04-16_15_21_24.mp4",
            "x" * 250 + "_2024-04-16_15_21_24.mp4", "IMG 2024-04-16 15.21.24.mov",
            "mendels_2024-13-45_99_99_99.mp4")]
        for key in odd:
            self.s3.put_object(Bucket="their-footage", Key=key, Body=CLIP)
        r = self.client.post(reverse("sources:sync", args=[source.pk]),
                             {"selected_files": odd, "where": "bucket"}, follow=True)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(Video.objects.count(), len(odd))
        self.assertFalse(Site.objects.exists())       # an impossible date reads as no pattern

    def test_one_click_imports_everything_not_yet_imported(self):
        source = self.source()
        keys = [f"2026/bulk/mendels_2024-04-16_15_{m:02d}_00.mp4" for m in range(25)]
        for key in keys:
            self.s3.put_object(Bucket="their-footage", Key=key, Body=CLIP)
        url = reverse("sources:sync", args=[source.pk])
        self.client.post(url, {"path": "bulk/", "selected_files": keys[:10], "where": "bucket"})
        with mock.patch("apps.sources.views.IMPORT_BATCH", 4):       # several INSERT batches
            r = self.client.post(url, {"path": "bulk/", "import_all": "true", "where": "copy"}, follow=True)
        self.assertContains(r, "Imported 15 clips")
        self.assertContains(r, "10 were already imported.")
        self.assertEqual(Video.objects.count(), 25)
        self.assertEqual(Video.objects.filter(site__name="mendels").count(), 25)
        copies = SourceCopy.objects.filter(purpose="keep")
        self.assertEqual(copies.count(), 15)
        self.assertEqual(set(copies.values_list("bytes_total", flat=True)), {len(CLIP)})
        self.assertFalse(Video.objects.filter(source_copies__isnull=True).exists())

    def test_a_forged_key_outside_the_folder_is_ignored(self):
        source = self.source()
        self.client.post(reverse("sources:sync", args=[source.pk]),
                         {"selected_files": ["private/elsewhere.mp4"], "where": "copy"})
        self.assertFalse(Video.objects.exists())

    def test_import_a_whole_folder(self):
        source = self.source()
        self.client.post(reverse("sources:sync", args=[source.pk]), {"path": "", "import_all": "true", "where": "bucket"})
        self.assertEqual(sorted(Video.objects.values_list("title", flat=True)), ["nest_0412", "top"])

    def test_a_copy_that_fails_says_why_in_plain_words(self):
        source = self.source(secret_access_key="wrong")
        video = Video.objects.create(user=self.user, source=source, title="c", file_size_bytes=1,
                                     storage_key="s3://their-footage/2026/top.mov",
                                     metadata={"remote_bucket": "their-footage", "remote_key": "2026/top.mov"})
        copying.queue(video, SourceCopy.Purpose.KEEP)
        with mock.patch("apps.sources.s3.stream_copy", side_effect=s3.SourceError("Access denied to that bucket or folder.")):
            self.run_copies()
        copy = SourceCopy.objects.get()
        self.assertEqual(copy.status, "failed")
        self.assertIn("Access denied", copy.error)

    def test_one_copy_per_clip(self):
        source = self.source()
        video = Video.objects.create(user=self.user, source=source, title="c", file_size_bytes=1,
                                     storage_key="s3://their-footage/2026/top.mov")
        copying.queue(video, SourceCopy.Purpose.KEEP)
        copying.queue(video, SourceCopy.Purpose.KEEP)
        self.assertEqual(SourceCopy.objects.count(), 1)

    # ── Analysing a clip kept in the bucket ──
    def test_analysis_gets_a_temporary_copy_removed_a_day_later(self):
        from datetime import timedelta

        from django.utils import timezone

        from apps.analysis.views import _ingest_external_s3_to_storage
        source = self.source()
        video = Video.objects.create(user=self.user, source=source, title="c", file_size_bytes=len(CLIP),
                                     storage_key="s3://their-footage/2026/top.mov",
                                     metadata={"remote_bucket": "their-footage", "remote_key": "2026/top.mov"})
        key = _ingest_external_s3_to_storage(video, "job1")
        self.assertEqual(self.s3.get_object(Bucket="our-raw", Key=key)["Body"].read(), CLIP)
        video.refresh_from_db()
        self.assertTrue(video.storage_key.startswith("s3://"))          # the clip stays in their bucket
        SourceCopy.objects.update(finished_at=timezone.now() - timedelta(days=2))
        with mock.patch("threading.Thread"):
            copying.tick()
        self.assertEqual(self.s3.list_objects_v2(Bucket="our-raw").get("KeyCount"), 0)
        self.assertEqual(SourceCopy.objects.get().status, "removed")

    # ── The queue ──
    def _queued(self, n):
        source = self.source()
        for i in range(n):
            v = Video.objects.create(user=self.user, source=source, title=f"c{i}", file_size_bytes=1,
                                     storage_key=f"s3://their-footage/2026/c{i}.mp4",
                                     metadata={"remote_bucket": "their-footage", "remote_key": f"2026/c{i}.mp4"})
            copying.queue(v, SourceCopy.Purpose.KEEP)

    def test_six_copies_run_at_once(self):
        self._queued(9)
        with mock.patch("threading.Thread"):
            self.assertEqual(copying.tick()["copies_started"], 6)
            self.assertEqual(copying.tick()["copies_started"], 0)       # all slots taken
        self.assertEqual(SourceCopy.objects.filter(status="running").count(), 6)

    def test_a_finished_copy_starts_the_next_without_waiting_for_a_tick(self):
        self._queued(8)
        with mock.patch("threading.Thread"):
            copying.tick()
        first = SourceCopy.objects.filter(status="running").first()
        SourceCopy.objects.filter(pk=first.pk).update(status="done")
        with mock.patch("threading.Thread") as thread, mock.patch("apps.sources.copying.run"):
            copying._run_in_thread(first.pk)
        self.assertEqual(thread.call_count, 1)
        self.assertEqual(SourceCopy.objects.filter(status="running").count(), 6)

    def test_a_copy_gone_quiet_is_put_back_but_a_busy_one_is_not(self):
        from datetime import timedelta

        from django.utils import timezone
        self._queued(2)
        quiet, busy = SourceCopy.objects.all()
        old = timezone.now() - timedelta(minutes=11)
        SourceCopy.objects.filter(pk=quiet.pk).update(status="running", started_at=old, heartbeat_at=old)
        SourceCopy.objects.filter(pk=busy.pk).update(status="running", started_at=old,
                                                     heartbeat_at=timezone.now())
        with mock.patch("apps.sources.copying._fill_slots", return_value=0):
            copying.tick()
        self.assertEqual(SourceCopy.objects.get(pk=quiet.pk).status, "queued")
        self.assertEqual(SourceCopy.objects.get(pk=busy.pk).status, "running")

    def test_a_copy_for_one_analysis_is_never_requeued_or_counted(self):
        from datetime import timedelta

        from django.utils import timezone
        self._queued(1)
        temp = SourceCopy.objects.create(video=Video.objects.first(), purpose="temp", status="running",
                                         started_at=timezone.now() - timedelta(hours=2))
        with mock.patch("threading.Thread"):
            self.assertEqual(copying.tick()["copies_started"], 1)
        self.assertEqual(SourceCopy.objects.get(pk=temp.pk).status, "running")

    def test_progress_touches_the_heartbeat(self):
        source = self.source()
        self.client.post(reverse("sources:sync", args=[source.pk]),
                         {"selected_files": ["2026/june/nest_0412.mp4"], "where": "copy"})
        self.run_copies()
        copy = SourceCopy.objects.get()
        self.assertEqual(copy.status, "done")
        self.assertIsNotNone(copy.heartbeat_at)

    # ── The 3-month rule ──
    @override_settings(RAW_FOOTAGE_DELETION="on")
    def test_a_copied_clip_goes_back_to_their_bucket_instead_of_being_lost(self):
        from apps.videos import retention
        source = self.source()
        self.s3.put_object(Bucket="our-raw", Key="1/9/top.mov", Body=CLIP)
        video = Video.objects.create(user=self.user, source=source, title="c", file_size_bytes=len(CLIP),
                                     storage_key="1/9/top.mov", raw_delete_on="2000-01-01",
                                     metadata={"remote_bucket": "their-footage", "remote_key": "2026/top.mov"})
        with mock.patch("apps.videos.retention._notices_complete", return_value=True), \
                mock.patch("apps.videos.retention._send_due_notices", return_value=(0, 0)):
            retention.run()
        video.refresh_from_db()
        self.assertEqual(video.storage_key, "s3://their-footage/2026/top.mov")
        self.assertIsNone(video.raw_deleted_at)
        self.assertEqual(self.s3.list_objects_v2(Bucket="our-raw").get("KeyCount"), 0)

    # ── Pages ──
    def test_the_clip_page_offers_copy_into_beemonitor(self):
        source = self.source()
        video = Video.objects.create(user=self.user, source=source, title="c", file_size_bytes=1,
                                     storage_key="s3://their-footage/2026/top.mov",
                                     metadata={"remote_bucket": "their-footage", "remote_key": "2026/top.mov"})
        r = self.client.get(reverse("videos:detail", args=[video.pk]))
        self.assertContains(r, "Stored in your bucket")
        self.client.post(reverse("sources:copy_clip", args=[video.pk]))
        self.assertEqual(SourceCopy.objects.get().purpose, "keep")

    def test_linked_clips_dont_count_as_our_storage(self):
        from apps.platform_admin import stats
        source = self.source()
        Video.objects.create(user=self.user, source=source, title="c", file_size_bytes=5_000_000_000,
                             storage_key="s3://their-footage/2026/top.mov")
        Video.objects.create(user=self.user, title="u", file_size_bytes=100, storage_key="1/u.mp4")
        self.assertEqual(stats.overview()["storage_bytes"], 100)

    def test_the_list_page(self):
        self.source()
        r = self.client.get(reverse("sources:list"))
        self.assertContains(r, "Browse &amp; import")
        self.assertContains(self.client.get(reverse("analysis:processing")), reverse("sources:list"))


class ProbeTests(TestCase):
    def test_duration_fps_and_size_are_read_over_a_url(self):
        import json

        from apps.videos import frames
        user = User.objects.create_user("bee")
        video = Video.objects.create(user=user, title="c", storage_key="1/c.mp4", file_size_bytes=1)
        out = mock.Mock(returncode=0, stdout=json.dumps({
            "streams": [{"width": 1920, "height": 1080, "avg_frame_rate": "30000/1001"}],
            "format": {"duration": "2072.4"}}).encode())
        with mock.patch("config.storage.presigned_get", return_value="https://s3/x"), \
                mock.patch("subprocess.run", return_value=out):
            props = frames.probe(video)
        self.assertEqual(props, {"fps": 29.97, "width": 1920, "height": 1080, "duration_seconds": 2072.4})
