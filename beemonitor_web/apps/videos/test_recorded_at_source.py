"""A recording time and an upload time must not be indistinguishable.

`recorded_at` was filled by `explicit or filename or timezone.now()`, and the
third of those is not a recording time at all. A device that buffered a backlog
while offline and flushed it on reconnect stamped every clip in the backlog
with the flush time; those clips then landed on the wrong day, in the wrong
time-of-day bucket, in every time series — looking exactly like good data.

`resolve_recorded_at` still falls back, but now says so.
"""

from datetime import datetime, timezone as dt_timezone

from django.test import TestCase

from apps.videos.models import Video


class ResolveRecordedAtTests(TestCase):
    def test_a_device_supplied_timestamp_is_trusted_over_the_filename(self):
        sent = datetime(2026, 5, 1, 9, 30, tzinfo=dt_timezone.utc)

        when, source = Video.resolve_recorded_at(sent, "hotelA_2026-06-02_11_00_00.mp4")

        self.assertEqual(when, sent)
        self.assertEqual(source, "device")

    def test_the_filename_is_used_when_the_device_sent_nothing(self):
        when, source = Video.resolve_recorded_at(None, "hotelA_2026-06-02_11_00_00.mp4")

        self.assertEqual(source, "filename")
        self.assertEqual((when.year, when.month, when.day, when.hour), (2026, 6, 2, 11))

    def test_an_unparseable_name_falls_back_but_is_labelled_upload_time(self):
        when, source = Video.resolve_recorded_at(None, "clip.mp4")

        self.assertEqual(source, "upload_time")
        self.assertIsNotNone(when)

    def test_no_evidence_at_all_is_also_upload_time(self):
        self.assertEqual(Video.resolve_recorded_at(None, "")[1], "upload_time")

    def test_the_row_reports_whether_its_timestamp_was_measured(self):
        self.assertTrue(Video(metadata={"recorded_at_source": "device"})
                        .recorded_at_is_measured)
        self.assertTrue(Video(metadata={"recorded_at_source": "filename"})
                        .recorded_at_is_measured)
        self.assertFalse(Video(metadata={"recorded_at_source": "upload_time"})
                         .recorded_at_is_measured)

    def test_a_clip_ingested_before_this_existed_is_not_called_a_guess(self):
        # No provenance recorded — the old rows. Treat as measured rather than
        # retroactively casting doubt on the whole library.
        self.assertTrue(Video(metadata={}).recorded_at_is_measured)
        self.assertTrue(Video(metadata=None).recorded_at_is_measured)
