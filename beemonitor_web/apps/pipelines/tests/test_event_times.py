"""Events export in real time, from the clip's start (memory/60).

A unit's clips in the unit's zone; an uploaded clip named ``site_timestamp`` in
its site's zone; a clip whose start is unknown with a blank timestamp but its
seconds into the clip still filled, so the reader can add their own start.
"""

import csv
import io
from datetime import datetime, timezone as dt_timezone
from unittest import mock

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from apps.analysis.models import Job, JobResult
from apps.devices.models import Device
from apps.videos.models import Site, Video

User = get_user_model()

# Frame 50 at 25 fps: 2 s into the clip.
SAVED = [{"frame": 50, "track_id": 3, "action": "exit", "reference": "reference_1",
          "at_edge": False}]


class EventTimestampTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("times", password="x")
        self.client.force_login(self.user)

    def _clip(self, **kw):
        video = Video.objects.create(user=self.user, title=kw.pop("title", "clip.mp4"),
                                     storage_key="t/c.mp4", file_size_bytes=1,
                                     status=Video.Status.READY, fps=25.0, **kw)
        job = Job.objects.create(user=self.user, video=video, status="completed")
        JobResult.objects.create(job=job, summary_stats={"video_fps": 25.0})
        return job

    def _row(self, job):
        with mock.patch("apps.pipelines.executors.primitives_for_job", return_value=SAVED):
            body = self.client.get(reverse("analysis:results_csv",
                                           kwargs={"pk": job.pk, "kind": "events"})).content
        rows = list(csv.DictReader(io.StringIO(body.decode())))
        self.assertEqual(len(rows), 1)
        return rows[0]

    def test_a_units_clip_is_in_the_units_zone(self):
        unit = Device.objects.create(owner=self.user, name="Amber", key_hash="ht", prefix="pt",
                                     tz_name="Europe/Berlin")
        job = self._clip(device=unit,
                         recorded_at=datetime(2026, 6, 2, 15, 0, tzinfo=dt_timezone.utc),
                         metadata={"recorded_at_source": "device"})

        row = self._row(job)

        self.assertEqual(row["timestamp"], "2026-06-02T17:00:02.000+02:00")
        self.assertEqual(row["clip_sec"], "2.0")

    def test_an_uploaded_clip_is_in_its_sites_zone(self):
        site = Site.objects.create(user=self.user, name="hotelA", tz_name="America/New_York")
        name = "hotelA_2026-06-02_11_00_00.mp4"         # 11:00 at the site
        when, src = Video.resolve_recorded_at(None, name)
        job = self._clip(site=site, title=name, recorded_at=when,
                         metadata={"recorded_at_source": src})

        self.assertEqual(self._row(job)["timestamp"], "2026-06-02T11:00:02.000-04:00")

    def test_an_unknown_start_leaves_the_timestamp_blank(self):
        when, src = Video.resolve_recorded_at(None, "clip.mp4")     # upload time
        job = self._clip(recorded_at=when, metadata={"recorded_at_source": src})

        row = self._row(job)

        self.assertEqual(row["timestamp"], "")
        self.assertEqual(row["clip_sec"], "2.0")
