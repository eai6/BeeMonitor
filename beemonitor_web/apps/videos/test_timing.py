"""A clip's start and the real time of a moment in it (memory/60).

File names carry the site's local wall-clock, which used to be stored as if it
were UTC. With a site time zone it becomes the true instant; without one the
timestamps keep the wall-clock and carry no offset; an upload time is not a
recording time, so the timestamp is blank.
"""

import json
from datetime import datetime, timezone as dt_timezone
from unittest import mock

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from apps.devices.models import Device
from apps.videos import thumbnails, timing
from apps.videos.models import Site, Video

User = get_user_model()
NAME = "hotelA_2026-06-02_11_00_00.mp4"     # 11:00 wall-clock at the site


class Base(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("bee", password="pw-12345678")
        self.client.force_login(self.user)

    def clip(self, *, site=None, device=None, source="filename", filename=NAME, explicit=None):
        if source == "filename":
            when, src = Video.resolve_recorded_at(None, filename)
        elif source == "upload_time":
            when, src = Video.resolve_recorded_at(None, "clip.mp4")
        else:
            when, src = explicit, source
        return Video.objects.create(
            user=self.user, site=site, device=device, title=filename, storage_key=f"k/{filename}",
            file_size_bytes=1, recorded_at=when, metadata={"recorded_at_source": src})


class ClipStartTests(Base):
    def test_a_file_name_time_with_a_site_zone_is_the_true_instant(self):
        site = Site.objects.create(user=self.user, name="hotelA", tz_name="America/New_York")
        clip = self.clip(site=site)
        # 11:00 in New York in June is 15:00 UTC.
        self.assertEqual(clip.recorded_at, datetime(2026, 6, 2, 15, 0, tzinfo=dt_timezone.utc))
        self.assertEqual(timing.iso(timing.clip_start(clip)), "2026-06-02T11:00:00.000-04:00")
        self.assertEqual(clip.metadata["recorded_at_tz"], "America/New_York")
        self.assertEqual((clip.day, clip.hour), (2, 11))        # local, as before

    def test_a_file_name_time_without_a_site_zone_has_no_offset(self):
        site = Site.objects.create(user=self.user, name="hotelA")
        clip = self.clip(site=site)
        start = timing.clip_start(clip)
        self.assertIsNone(start.tzinfo)
        self.assertEqual(timing.iso(start), "2026-06-02T11:00:00.000")
        self.assertEqual(timing.timestamp_at(clip, 1.5), "2026-06-02T11:00:01.500")

    def test_a_device_time_is_shown_in_the_units_zone(self):
        sent = datetime(2026, 6, 2, 15, 0, tzinfo=dt_timezone.utc)
        unit = Device.objects.create(owner=self.user, name="Amber", key_hash="h", prefix="p",
                                     tz_name="Europe/Berlin")
        clip = self.clip(device=unit, source="device", explicit=sent)
        self.assertEqual(timing.iso(timing.clip_start(clip)), "2026-06-02T17:00:00.000+02:00")

    def test_a_device_without_a_zone_stays_utc(self):
        sent = datetime(2026, 6, 2, 15, 0, tzinfo=dt_timezone.utc)
        unit = Device.objects.create(owner=self.user, name="Amber", key_hash="h", prefix="p")
        clip = self.clip(device=unit, source="device", explicit=sent)
        self.assertEqual(timing.iso(timing.clip_start(clip)), "2026-06-02T15:00:00.000+00:00")

    def test_typed_and_file_metadata_times_are_instants(self):
        when = datetime(2026, 6, 2, 15, 0, tzinfo=dt_timezone.utc)
        site = Site.objects.create(user=self.user, name="s", tz_name="America/New_York")
        for source in ("user", "file"):
            clip = self.clip(site=site, source=source, explicit=when, filename=f"{source}.mp4")
            self.assertEqual(clip.recorded_at, when)      # not shifted: already an instant
            self.assertEqual(timing.iso(timing.clip_start(clip)), "2026-06-02T11:00:00.000-04:00")

    def test_an_upload_time_is_not_a_start(self):
        clip = self.clip(source="upload_time", filename="clip.mp4")
        self.assertIsNone(timing.clip_start(clip))
        self.assertEqual(timing.timestamp_at(clip, 3.0), "")

    def test_a_precomputed_start_is_accepted(self):
        start = datetime(2026, 6, 2, 11, 0)
        self.assertEqual(timing.timestamp_at(start, 0.0333), "2026-06-02T11:00:00.033")
        self.assertEqual(timing.timestamp_at(None, 1), "")
        self.assertEqual(timing.iso(None), "")


class RebaseTests(Base):
    def test_setting_a_zone_rebases_once_however_often_it_is_set(self):
        site = Site.objects.create(user=self.user, name="hotelA")
        clip = self.clip(site=site)
        typed = self.clip(site=site, source="user", filename="typed.mp4",
                          explicit=datetime(2026, 6, 2, 9, 0, tzinfo=dt_timezone.utc))
        self.assertEqual(clip.recorded_at, datetime(2026, 6, 2, 11, 0, tzinfo=dt_timezone.utc))

        site.set_time_zone("America/New_York")
        site.set_time_zone("America/New_York")
        clip.refresh_from_db()
        self.assertEqual(clip.recorded_at, datetime(2026, 6, 2, 15, 0, tzinfo=dt_timezone.utc))

        site.set_time_zone("Europe/Berlin")             # moved from the wall-clock, not from 15:00
        clip.refresh_from_db()
        self.assertEqual(clip.recorded_at, datetime(2026, 6, 2, 9, 0, tzinfo=dt_timezone.utc))
        self.assertEqual(clip.metadata["recorded_wall"], "2026-06-02T11:00:00")

        typed.refresh_from_db()                           # only file-name times move
        self.assertEqual(typed.recorded_at, datetime(2026, 6, 2, 9, 0, tzinfo=dt_timezone.utc))

    def test_an_unknown_zone_is_refused(self):
        site = Site.objects.create(user=self.user, name="hotelA")
        with self.assertRaises(ValueError):
            site.set_time_zone("Mars/Olympus")

    def test_a_clip_saved_again_does_not_shift_twice(self):
        site = Site.objects.create(user=self.user, name="hotelA", tz_name="America/New_York")
        clip = self.clip(site=site)
        clip.title = "renamed"
        clip.save()
        clip.save()
        clip.refresh_from_db()
        self.assertEqual(clip.recorded_at, datetime(2026, 6, 2, 15, 0, tzinfo=dt_timezone.utc))


class SiteTimeZoneViewTests(Base):
    def test_the_upload_page_sets_a_zone_and_rebases(self):
        site = Site.objects.create(user=self.user, name="hotelA")
        clip = self.clip(site=site)
        r = self.client.post(reverse("videos:site_timezone", args=[site.pk]),
                             json.dumps({"tz_name": "America/New_York"}), content_type="application/json")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["site"]["tz_name"], "America/New_York")
        self.assertEqual(r.json()["clips_moved"], 1)
        clip.refresh_from_db()
        self.assertEqual(clip.recorded_at.hour, 15)

    def test_a_bad_zone_is_an_error(self):
        site = Site.objects.create(user=self.user, name="hotelA")
        r = self.client.post(reverse("videos:site_timezone", args=[site.pk]),
                             json.dumps({"tz_name": "Nowhere"}), content_type="application/json")
        self.assertEqual(r.status_code, 400)

    def test_someone_elses_site_is_not_found(self):
        other = User.objects.create_user("other", password="pw-12345678")
        site = Site.objects.create(user=other, name="theirs")
        r = self.client.post(reverse("videos:site_timezone", args=[site.pk]), {"tz_name": "UTC"})
        self.assertEqual(r.status_code, 404)

    def test_a_new_site_from_the_upload_page_takes_its_zone(self):
        r = self.client.post(reverse("videos:sites"), json.dumps({"name": "hotelB", "tz_name": "Europe/Berlin"}),
                             content_type="application/json")
        self.assertEqual(r.status_code, 201)
        self.assertEqual(Site.objects.get(name="hotelB").tz_name, "Europe/Berlin")

    def test_the_sites_page_asks_for_a_missing_zone_and_saves_it(self):
        site = Site.objects.create(user=self.user, name="hotelA")
        html = self.client.get(reverse("videos:sites_page")).content.decode()
        self.assertIn("No time zone yet", html)
        self.assertIn('<option value="America/New_York">', html)
        self.client.post(reverse("videos:site_timezone", args=[site.pk]), {"tz_name": "America/New_York"})
        site.refresh_from_db()
        self.assertEqual(site.tz_name, "America/New_York")
        self.assertNotIn("No time zone yet", self.client.get(reverse("videos:sites_page")).content.decode())

    def test_the_sites_page_adds_a_site_with_a_zone(self):
        self.client.post(reverse("videos:sites_page"), {"name": "hotelC", "tz_name": "Asia/Tokyo"})
        self.assertEqual(Site.objects.get(name="hotelC").tz_name, "Asia/Tokyo")

    def test_the_clip_page_shows_the_local_start(self):
        site = Site.objects.create(user=self.user, name="hotelA", tz_name="America/New_York")
        clip = self.clip(site=site)
        unknown = self.clip(source="upload_time", filename="clip.mp4")
        s3 = mock.Mock(generate_presigned_url=mock.Mock(return_value="https://s3/x"))
        with mock.patch("config.storage.get_s3_client", return_value=s3):
            html = self.client.get(reverse("videos:detail", args=[clip.pk])).content.decode()
            self.assertIn("2026-06-02 11:00:00", html)
            html = self.client.get(reverse("videos:detail", args=[unknown.pk])).content.decode()
            self.assertIn("Unknown", html)


class FrameRateTests(TestCase):
    def probe(self, r, avg):
        out = json.dumps({"streams": [{"r_frame_rate": r, "avg_frame_rate": avg}]})
        with mock.patch("subprocess.run", return_value=mock.Mock(stdout=out)):
            return thumbnails.frame_rate_is_variable("x.mp4")

    def test_even_frames_are_constant(self):
        self.assertFalse(self.probe("30/1", "30/1"))
        self.assertFalse(self.probe("30000/1001", "2997/100"))

    def test_uneven_frames_are_variable(self):
        self.assertTrue(self.probe("30/1", "24123/1000"))

    def test_unknown_is_unknown(self):
        self.assertIsNone(self.probe("0/0", "0/0"))
        with mock.patch("subprocess.run", side_effect=OSError):
            self.assertIsNone(thumbnails.frame_rate_is_variable("x.mp4"))
