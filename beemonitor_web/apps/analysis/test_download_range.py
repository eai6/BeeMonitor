"""Tests for the from/to date-range filter on the CSV-download mixin."""

from datetime import datetime
from unittest import mock
from zoneinfo import ZoneInfo

from django.contrib.auth.models import User
from django.test import RequestFactory, TestCase, override_settings

from apps.analysis.models import Job, JobResult
from apps.analysis.views import _download_range_bounds, _FilteredJobsMixin
from apps.videos.models import Video

UTC = ZoneInfo("UTC")


class RangeBoundsTest(TestCase):
    def test_inclusive_end_of_day(self):
        start, end = _download_range_bounds("2026-07-01", "2026-07-05", UTC)
        self.assertEqual(start, datetime(2026, 7, 1, 0, 0, 0, 0, tzinfo=UTC))
        self.assertEqual(end, datetime(2026, 7, 5, 23, 59, 59, 999999, tzinfo=UTC))

    def test_partial_and_bad_input(self):
        s, e = _download_range_bounds("", "2026-07-05", UTC)
        self.assertIsNone(s)
        self.assertIsNotNone(e)
        s, e = _download_range_bounds("not-a-date", "", UTC)
        self.assertIsNone(s)
        self.assertIsNone(e)


@override_settings(TIME_ZONE="UTC", USE_TZ=True)
class DownloadRangeFilterTest(TestCase):
    def setUp(self):
        self.user = User.objects.create(username="dl")
        self.factory = RequestFactory()
        self.by_day = {}
        for i, day in enumerate((("2026-06-30", 12), ("2026-07-01", 12),
                                 ("2026-07-05", 23), ("2026-07-06", 0))):
            date_str, hour = day
            y, m, d = (int(x) for x in date_str.split("-"))
            v = Video.objects.create(
                user=self.user, title=date_str, storage_key=f"k{i}", file_size_bytes=1,
                recorded_at=datetime(y, m, d, hour, 30, tzinfo=UTC))
            job = Job.objects.create(user=self.user, video=v,
                                     status=Job.Status.COMPLETED, modal_job_id=f"j{i}")
            JobResult.objects.create(job=job)
            self.by_day[date_str] = v.pk

    def _run(self, **params):
        req = self.factory.get("/dl/", params)
        req.user = self.user
        return _FilteredJobsMixin()._get_filtered_results(req)

    def test_range_is_inclusive_of_both_end_days(self):
        qs, label = self._run(**{"from": "2026-07-01", "to": "2026-07-05"})
        titles = sorted(r.job.video.title for r in qs)
        # 07-01 12:30 and 07-05 23:30 in; 06-30 and 07-06 out.
        self.assertEqual(titles, ["2026-07-01", "2026-07-05"])
        self.assertIn("2026-07-01_to_2026-07-05", label)

    def test_open_ended_from(self):
        qs, _ = self._run(**{"from": "2026-07-05"})
        titles = sorted(r.job.video.title for r in qs)
        self.assertEqual(titles, ["2026-07-05", "2026-07-06"])

    def test_no_range_returns_all(self):
        qs, label = self._run()
        self.assertEqual(qs.count(), 4)
        self.assertEqual(label, "all")


class StreamCombinedCSVTest(TestCase):
    """The events download streams a combined CSV across jobs (parallel S3 fetch)."""

    def setUp(self):
        self.user = User.objects.create(username="s")
        self.client.force_login(self.user)
        for i in range(3):
            v = Video.objects.create(user=self.user, title=f"vid{i}", site_name="siteX",
                                     storage_key=f"k{i}", file_size_bytes=1,
                                     recorded_at=datetime(2026, 7, 3, 10, tzinfo=UTC))
            job = Job.objects.create(user=self.user, video=v,
                                     status=Job.Status.COMPLETED, modal_job_id=f"j{i}")
            JobResult.objects.create(job=job, events_csv_path=f"p/{i}/events.csv")

    def test_streams_header_and_prefixed_rows(self):
        class FakeS3:
            def download_to_stream(self, container, path, buf):
                # Each per-job CSV: one header + one data row tagged with its path.
                buf.write(f"frame,x\r\n{path},7\r\n".encode("utf-8"))

        from django.urls import reverse
        with mock.patch("apps.analysis.views.get_s3_client", return_value=FakeS3()):
            resp = self.client.get(reverse("analysis:download_events"))
            body = b"".join(resp.streaming_content).decode("utf-8")

        self.assertEqual(resp.status_code, 200)
        self.assertIn("attachment; filename=", resp["Content-Disposition"])
        lines = [ln for ln in body.splitlines() if ln]
        self.assertEqual(lines[0], "video_title,site_name,recorded_at,frame,x")  # header once
        self.assertEqual(len(lines), 4)                                          # header + 3 rows
        # Every job's row carries its video-title prefix + its own S3 path.
        for i in range(3):
            self.assertTrue(any(f"vid{i},siteX," in ln and f"p/{i}/events.csv,7" in ln
                                for ln in lines[1:]))
