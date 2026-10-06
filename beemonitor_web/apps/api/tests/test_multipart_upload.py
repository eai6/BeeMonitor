"""Browser uploads in parts, recording time, sites, filters (memory/44)."""

import json
from datetime import datetime, timezone
from unittest import mock

from django.contrib.auth import get_user_model
from django.test import TestCase

from apps.api.multipart import part_size_for
from apps.videos.models import Site, Video

User = get_user_model()


class FakeS3:
    """The boto3 calls the multipart views make, recorded."""

    def __init__(self):
        self.calls = []
        self.exceptions = mock.Mock(NoSuchUpload=KeyError)

    def create_multipart_upload(self, **kw):
        self.calls.append(("create", kw))
        return {"UploadId": "U1"}

    def generate_presigned_url(self, method, ExpiresIn, Params):
        return f"https://s3/{Params['Key']}?part={Params['PartNumber']}"

    def list_parts(self, **kw):
        return {"Parts": [{"PartNumber": 1, "ETag": '"e1"', "Size": 16}], "IsTruncated": False}

    def complete_multipart_upload(self, **kw):
        self.calls.append(("complete", kw))

    def abort_multipart_upload(self, **kw):
        self.calls.append(("abort", kw))


class MultipartTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("up", password="x")
        self.client.force_login(self.user)
        self.s3 = FakeS3()
        wrapper = mock.Mock(_key=lambda k: k)
        patcher = mock.patch("apps.api.multipart._s3", return_value=(wrapper, self.s3, "raw"))
        patcher.start()
        self.addCleanup(patcher.stop)
        thumbs = mock.patch("apps.videos.thumbnails.queue_thumbnail")
        thumbs.start()
        self.addCleanup(thumbs.stop)

    def _post(self, path, body):
        return self.client.post(f"/api/v1/web-uploads/{path}", data=json.dumps(body),
                                content_type="application/json")

    def test_part_size_keeps_any_file_under_the_part_limit(self):
        for size in (1, 5 * 2**30, 100 * 2**30, 500 * 2**30):
            ps = part_size_for(size)
            self.assertGreaterEqual(ps, 16 * 2**20)
            self.assertLessEqual(-(-size // ps), 9000)

    def test_initiate_opens_an_upload_under_the_users_prefix(self):
        r = self._post("multipart/initiate", {"filename": "big.avi", "size_bytes": 7 * 2**30})
        self.assertEqual(r.status_code, 200, r.content)
        self.assertTrue(r.json()["storage_key"].startswith(f"{self.user.pk}/"))
        self.assertEqual(r.json()["upload_id"], "U1")

    def test_initiate_refuses_other_files(self):
        r = self._post("multipart/initiate", {"filename": "notes.txt", "size_bytes": 10})
        self.assertEqual(r.status_code, 400)

    def test_someone_elses_key_is_refused(self):
        r = self._post("multipart/sign", {"storage_key": "999/x/a.mp4", "upload_id": "U1",
                                          "part_numbers": [1]})
        self.assertEqual(r.status_code, 403)

    def test_sign_and_resume(self):
        key = f"{self.user.pk}/abc/a.mp4"
        r = self._post("multipart/sign", {"storage_key": key, "upload_id": "U1", "part_numbers": [2, 1]})
        self.assertEqual(sorted(r.json()["urls"]), ["1", "2"])
        r = self._post("multipart/parts", {"storage_key": key, "upload_id": "U1"})
        self.assertEqual(r.json()["parts"][0]["part_number"], 1)

    def _complete(self, name="GX010231.MP4", **extra):
        key = f"{self.user.pk}/abc/{name}"
        body = {"storage_key": key, "upload_id": "U1", "file_size_bytes": 123,
                "original_filename": name, "parts": [{"part_number": 1, "etag": '"e1"'}], **extra}
        r = self._post("multipart/complete", body)
        self.assertEqual(r.status_code, 201, r.content)
        return Video.objects.get(pk=r.json()["video_id"])

    def test_time_from_the_file_comes_first(self):
        v = self._complete("cam_2026-10-04_11_02_10.mp4", file_recorded_at="2026-10-04T09:12:40Z")
        self.assertEqual(v.metadata["recorded_at_source"], "file")
        self.assertEqual(v.recorded_at, datetime(2026, 10, 4, 9, 12, 40, tzinfo=timezone.utc))

    def test_an_unset_camera_clock_falls_back_to_the_name(self):
        v = self._complete("IMG_20261004_091240.mov", file_recorded_at="1904-01-01T00:00:00Z")
        self.assertEqual(v.metadata["recorded_at_source"], "filename")

    def test_then_the_users_start_time_then_upload_time(self):
        v = self._complete("trial4.mov", user_recorded_at="2026-10-04T09:30:00Z")
        self.assertEqual(v.metadata["recorded_at_source"], "user")
        v = self._complete("arena.avi")
        self.assertEqual(v.metadata["recorded_at_source"], "upload_time")
        self.assertTrue(v.metadata["needs_transcode"])

    def test_site_device_batch_all_optional(self):
        site = Site.objects.create(user=self.user, name="Mendel's Garden", lat=40.8, lon=-77.86)
        v = self._complete(site_id=site.id, batch="Pollen assay · trial 4")
        self.assertEqual((v.site, v.site_name), (site, "Mendel's Garden"))
        self.assertEqual(v.metadata["batch"], "Pollen assay · trial 4")
        self.assertEqual(v.metadata["uploaded_via"], "web")
        bare = self._complete("plain.mp4")
        self.assertIsNone(bare.site)

    def test_another_users_site_is_refused(self):
        other = User.objects.create_user("o", password="x")
        site = Site.objects.create(user=other, name="Theirs")
        key = f"{self.user.pk}/abc/a.mp4"
        r = self._post("multipart/complete", {"storage_key": key, "upload_id": "U1", "file_size_bytes": 1,
                                               "parts": [{"part_number": 1, "etag": "e"}], "site_id": site.id})
        self.assertEqual(r.status_code, 400)

    def test_duplicates_are_found_by_original_name_and_size(self):
        self._complete("GX010229.MP4")
        r = self._post("check", {"files": [{"name": "GX010229.MP4", "size": 123},
                                           {"name": "GX010229.MP4", "size": 999}]})
        self.assertEqual([d["size"] for d in r.json()["duplicates"]], [123])


class SiteAndFilterTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("st", password="x")
        self.client.force_login(self.user)

    def test_create_a_site_name_only_or_with_location(self):
        r = self.client.post("/videos/sites/", data=json.dumps({"name": "Lab arena"}),
                             content_type="application/json")
        self.assertEqual(r.status_code, 201)
        self.assertIsNone(r.json()["site"]["lat"])
        r = self.client.post("/videos/sites/", data=json.dumps({"name": "Field", "lat": "40.8",
                                                               "lon": "−77.86"}),
                             content_type="application/json")
        self.assertEqual(r.json()["site"]["lon"], -77.86)
        r = self.client.post("/videos/sites/", data=json.dumps({"name": "Bad", "lat": 120}),
                             content_type="application/json")
        self.assertEqual(r.status_code, 400)
        self.assertEqual(len(self.client.get("/videos/sites/").json()["sites"]), 2)

    def test_upload_filters(self):
        from apps.videos.workspace import apply_video_filters
        mk = lambda **m: Video.objects.create(user=self.user, title="c", storage_key="k",
                                              file_size_bytes=1, metadata=m)
        a = mk(uploaded_via="web", batch="trial 4", recorded_at_source="upload_time")
        b = mk(uploaded_via="web", batch="trial 5", recorded_at_source="file")
        c = mk(recorded_at_source="device")
        qs = Video.objects.filter(user=self.user)
        self.assertEqual(list(apply_video_filters(qs, {"batch": "trial 4"})), [a])
        self.assertEqual(set(apply_video_filters(qs, {"origin": "upload"})), {a, b})
        self.assertEqual(list(apply_video_filters(qs, {"origin": "unit"})), [c])
        self.assertEqual(list(apply_video_filters(qs, {"time": "unknown"})), [a])

    def test_bioclip_uses_the_uploads_site_when_there_is_no_unit(self):
        from apps.analysis.views import _candidate_taxa
        site = Site.objects.create(user=self.user, name="F", lat=40.8, lon=-77.86)
        v = Video.objects.create(user=self.user, title="c", storage_key="k", file_size_bytes=1, site=site)
        with mock.patch("apps.monitor.priors.region_taxa", return_value=["Osmia lignaria"]) as rt:
            self.assertEqual(_candidate_taxa(v), ["Osmia lignaria"])
        self.assertEqual(rt.call_args[0][:2], (40.8, -77.86))

    def test_upload_page_renders_and_old_batch_page_redirects(self):
        self.assertEqual(self.client.get("/videos/upload/").status_code, 200)
        self.assertRedirects(self.client.get("/videos/batch-upload/"), "/videos/upload/")


class TranscodeTests(TestCase):
    """Uploaded AVIs are sent for conversion and switched to the MP4."""

    def setUp(self):
        self.user = User.objects.create_user("tc", password="x")
        self.video = Video.objects.create(user=self.user, title="arena", file_size_bytes=1,
                                          storage_key=f"{self.user.pk}/abc/arena.avi",
                                          metadata={"needs_transcode": True})

    def test_dispatch_sends_once_then_collect_switches_to_the_mp4(self):
        from django.test import override_settings
        from apps.videos import transcode
        with override_settings(SAGEMAKER_ENDPOINT_NAME="ep"), \
                mock.patch("apps.analysis.views._put_inference_payload", return_value="s3://in/x") as put, \
                mock.patch("apps.analysis.views._invoke_endpoint_async", return_value=("o", "s3://f/x")):
            self.assertEqual(transcode.dispatch(), 1)
            self.assertEqual(transcode.dispatch(), 0)        # not sent twice
        self.assertEqual(put.call_args[0][1]["output_key"], f"{self.user.pk}/abc/arena.mp4")

        s3 = mock.Mock()
        s3.blob_exists.return_value = True
        with mock.patch("config.storage.get_s3_client", return_value=s3), \
                mock.patch("apps.videos.thumbnails.queue_thumbnail"):
            self.assertEqual(transcode.collect(), 1)
        self.video.refresh_from_db()
        self.assertEqual(self.video.storage_key, f"{self.user.pk}/abc/arena.mp4")
        self.assertEqual(self.video.metadata["original_key"], f"{self.user.pk}/abc/arena.avi")
        self.assertFalse(self.video.metadata["needs_transcode"])

    def test_a_failure_retries_once_then_gives_up(self):
        from apps.videos import transcode
        self.video.metadata = {"needs_transcode": True, "transcode_attempts": 1,
                               "transcode": {"state": "invoked", "output_key": "k.mp4",
                                             "failure_uri": "s3://f/x", "at": "2026-10-06T00:00:00+00:00"}}
        self.video.save()
        s3 = mock.Mock()
        s3.blob_exists.return_value = False
        with mock.patch("config.storage.get_s3_client", return_value=s3):
            transcode.collect()
        self.video.refresh_from_db()
        self.assertNotIn("transcode", self.video.metadata)   # queued again
        self.video.metadata = {**self.video.metadata, "transcode_attempts": 2,
                               "transcode": {"state": "invoked", "output_key": "k.mp4",
                                             "failure_uri": "s3://f/x", "at": "2026-10-06T00:00:00+00:00"}}
        self.video.save()
        with mock.patch("config.storage.get_s3_client", return_value=s3):
            transcode.collect()
        self.video.refresh_from_db()
        self.assertEqual(self.video.metadata["transcode"]["state"], "failed")

    def test_nothing_is_sent_without_an_endpoint(self):
        from django.test import override_settings
        from apps.videos import transcode
        with override_settings(SAGEMAKER_ENDPOINT_NAME=""):
            self.assertEqual(transcode.dispatch(), 0)


class PhotoUploadTests(MultipartTests):
    """Photos from any camera upload the same way and become kind=photo."""

    def test_a_jpeg_becomes_a_photo_with_its_exif_time(self):
        r = self._post("multipart/initiate", {"filename": "IMG_0042.JPG", "size_bytes": 5_000_000})
        self.assertEqual(r.status_code, 200)
        key = f"{self.user.pk}/abc/IMG_0042.JPG"
        r = self._post("multipart/complete", {
            "storage_key": key, "upload_id": "U1", "file_size_bytes": 5_000_000,
            "original_filename": "IMG_0042.JPG", "parts": [{"part_number": 1, "etag": "e"}],
            "file_recorded_at": "2026-10-04T09:12:40Z"})
        self.assertEqual(r.json()["kind"], "photo")
        photo = Video.everything.get(pk=r.json()["video_id"])
        self.assertEqual(photo.kind, Video.Kind.PHOTO)
        self.assertEqual(photo.metadata["recorded_at_source"], "file")
        self.assertFalse(Video.objects.filter(pk=photo.pk).exists())   # clips-only manager
