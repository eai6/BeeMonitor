"""Tests for the persisted foraging-trips pipeline (apps.analysis.foraging).

The load-bearing invariant: trip BOUNDS are a pure read-time filter. Pairing
consumes the Exit on every Entry regardless of bounds, so trips stored paired
at the widest bounds, then filtered by duration, must equal trips paired
directly at the narrow bounds. Everything (batch page vs device chart sync,
adjustable bounds with no recompute) rests on that.
"""

from datetime import datetime, timedelta, timezone as dt_timezone

from django.contrib.auth.models import User
from django.test import TestCase

from apps.analysis import foraging
from apps.analysis.models import DailyForagingSummary, Job
from apps.devices.models import Device
from apps.pipelines import aggregate
from apps.videos.models import Video

UTC = dt_timezone.utc


def _events(rows):
    """[(nest, action, iso_time)] -> collect_events-shaped dicts."""
    return [{
        "action": action,
        "nest": str(nest),
        "time": t,
        "video": "v1",
        "video_pk": 1,
        "track_id": "",
    } for nest, action, t in rows]


class BoundsArePureFilterTest(TestCase):
    def test_wide_pairing_plus_duration_filter_equals_narrow_pairing(self):
        base = datetime(2026, 7, 10, 12, 0, tzinfo=UTC)
        rows = []
        # Nest 1: a 5s trip (below min), a 60s trip, a 3h trip (above max),
        # plus an unmatched exit. Nest 2: interleaved 20s and 9000s trips.
        rows += [("1", "Exit", base), ("1", "Entry", base + timedelta(seconds=5))]
        rows += [("1", "Exit", base + timedelta(minutes=5)),
                 ("1", "Entry", base + timedelta(minutes=6))]
        rows += [("1", "Exit", base + timedelta(hours=1)),
                 ("1", "Entry", base + timedelta(hours=4))]
        rows += [("1", "Exit", base + timedelta(hours=6))]
        rows += [("2", "Exit", base + timedelta(seconds=30)),
                 ("2", "Entry", base + timedelta(seconds=50)),
                 ("2", "Exit", base + timedelta(minutes=10)),
                 ("2", "Entry", base + timedelta(minutes=160))]

        narrow, _ = aggregate.aggregate_trips(
            [], aggregate.DEFAULT_MIN_SEC, aggregate.DEFAULT_MAX_SEC,
            events=_events(rows))
        wide, _ = aggregate.aggregate_trips(
            [], aggregate.FULL_MIN_SEC, aggregate.FULL_MAX_SEC,
            events=_events(rows))
        wide_filtered = [t for t in wide
                         if aggregate.DEFAULT_MIN_SEC <= t["duration_sec"]
                         <= aggregate.DEFAULT_MAX_SEC]

        self.assertEqual(
            [(t["nest"], t["exit_time"], t["duration_sec"]) for t in narrow],
            [(t["nest"], t["exit_time"], t["duration_sec"]) for t in wide_filtered],
        )
        # Sanity: the wide pairing really did keep the out-of-bounds pairs.
        self.assertGreater(len(wide), len(wide_filtered))

    def test_clamp_trip_bounds(self):
        self.assertEqual(aggregate.clamp_trip_bounds(None, None),
                         (aggregate.DEFAULT_MIN_SEC, aggregate.DEFAULT_MAX_SEC))
        self.assertEqual(aggregate.clamp_trip_bounds("30", "600"), (30.0, 600.0))
        self.assertEqual(aggregate.clamp_trip_bounds("-5", "999999"), (0.0, 86400.0))
        self.assertEqual(aggregate.clamp_trip_bounds("100", "50"), (100.0, 100.0))
        self.assertEqual(aggregate.clamp_trip_bounds("junk", "junk"),
                         (aggregate.DEFAULT_MIN_SEC, aggregate.DEFAULT_MAX_SEC))


class DeviceTripsReadTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("edward", password="x")
        self.device = Device.objects.create(owner=self.user, name="unit-1")

    def _summary(self, day, trips):
        return DailyForagingSummary.objects.create(
            user=self.user, site_name="SiteA", device=self.device,
            date=day, trips=trips)

    def test_window_and_bounds_filtering(self):
        base = datetime(2026, 7, 10, 23, 30, tzinfo=UTC)  # near a UTC day edge
        self._summary(base.date(), [
            [base.timestamp(), 60.0, "1", False],           # in window, in bounds
            [base.timestamp() + 60, 5.0, "1", False],       # below min bound
            [(base - timedelta(days=3)).timestamp(), 60.0, "2", True],  # out of window
        ])
        # Trip stored under the NEXT UTC date but inside the query window —
        # the ±1 day row pad must pick it up.
        nxt = base + timedelta(hours=1)  # 00:30 next day
        self._summary(nxt.date(), [[nxt.timestamp(), 120.0, "3", True]])

        got = foraging.device_trips(
            self.device, base - timedelta(hours=2), base + timedelta(hours=2))
        self.assertEqual(
            [t.timestamp() for t in got],
            [base.timestamp(), nxt.timestamp()],
        )
        # Bounds narrow at read time — no recompute involved.
        got = foraging.device_trips(
            self.device, base - timedelta(hours=2), base + timedelta(hours=2),
            min_sec=90, max_sec=7200)
        self.assertEqual([t.timestamp() for t in got], [nxt.timestamp()])

    def test_null_trips_rows_are_skipped(self):
        self._summary(datetime(2026, 7, 10, tzinfo=UTC).date(), None)
        got = foraging.device_trips(
            self.device,
            datetime(2026, 7, 9, tzinfo=UTC), datetime(2026, 7, 12, tzinfo=UTC))
        self.assertEqual(got, [])


class StaleFlagLifecycleTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("edward", password="x")
        self.device = Device.objects.create(owner=self.user, name="unit-1")

    def _job(self, device):
        video = Video.objects.create(
            user=self.user, device=device, title="clip", storage_key="k",
            file_size_bytes=0, site_name="SiteA",
            recorded_at=datetime(2026, 7, 10, 15, 0, tzinfo=UTC))
        return Job.objects.create(user=self.user, video=video)

    def test_mark_stale_creates_flagged_row(self):
        foraging.mark_stale_for_job(self._job(self.device))
        row = DailyForagingSummary.objects.get()
        self.assertTrue(row.stale)
        self.assertEqual(row.device_id, self.device.pk)
        self.assertEqual(str(row.date), "2026-07-10")

    def test_null_device_jobs_are_ignored(self):
        foraging.mark_stale_for_job(self._job(None))
        self.assertEqual(DailyForagingSummary.objects.count(), 0)

    def test_sweep_recomputes_and_clears(self):
        foraging.mark_stale_for_job(self._job(self.device))
        # No completed jobs with events CSVs exist -> recompute yields an empty
        # day, entirely offline (S3 reads are never attempted with no sources).
        done = foraging.sweep_stale()
        self.assertEqual(done, 1)
        row = DailyForagingSummary.objects.get()
        self.assertFalse(row.stale)
        self.assertEqual(row.trips, [])
        self.assertEqual(row.total_trips, 0)
