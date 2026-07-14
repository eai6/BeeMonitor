"""Tests for chunked tracking of long videos (spawn ranges + result merge)."""

import io
import os
from types import SimpleNamespace
from unittest import mock

from django.contrib.auth.models import User
from django.test import TestCase

from apps.analysis import views
from apps.analysis.models import Job
from apps.videos.models import Video


def _video(duration, fps=25.0):
    return SimpleNamespace(duration_seconds=duration, fps=fps)


class ChunkRangesTest(TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ, {"BEEMONITOR_CHUNK_TRACKING": "1"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_gated_off_by_default(self):
        with mock.patch.dict(os.environ, {"BEEMONITOR_CHUNK_TRACKING": ""}):
            self.assertIsNone(views._chunk_ranges(_video(9999), "yolo"))

    def test_short_and_unknown_duration_stay_single(self):
        self.assertIsNone(views._chunk_ranges(_video(600), "yolo"))     # < 1200s
        self.assertIsNone(views._chunk_ranges(_video(None), "sam3"))    # unknown
        self.assertIsNone(views._chunk_ranges(_video(0), "sam3"))

    def test_yolo_long_video_contiguous_chunks(self):
        ranges = views._chunk_ranges(_video(3600, fps=25), "yolo")  # 1h @25fps
        self.assertEqual(len(ranges), 3)
        self.assertEqual(ranges[0][0], 0)
        for (s1, e1), (s2, _e2) in zip(ranges, ranges[1:]):
            self.assertEqual(e1, s2)  # shared boundaries: no gap, no overlap
        self.assertIsNone(ranges[-1][1])  # last chunk runs to EOF

    def test_sam3_uses_tiny_chunks(self):
        ranges = views._chunk_ranges(_video(600, fps=25), "sam3")  # 10 min
        self.assertEqual(len(ranges), 10)  # 60s chunks
        # a 10-min video is a single invocation for YOLO
        self.assertIsNone(views._chunk_ranges(_video(600, fps=25), "yolo"))

    def test_track_id_remap(self):
        self.assertEqual(views._remap_chunk_track_id("7", 0), "7")
        self.assertEqual(views._remap_chunk_track_id("7", 2), "2000007")
        self.assertEqual(views._remap_chunk_track_id("x9", 1), "1_x9")
        # No collisions across chunks for realistic id ranges
        self.assertNotEqual(views._remap_chunk_track_id("1", 1),
                            views._remap_chunk_track_id("1000001", 0))


class _FakeS3:
    """Blob-path -> bytes store standing in for the processed bucket."""

    def __init__(self, blobs):
        self.blobs = dict(blobs)
        self.uploaded = {}

    def download_to_stream(self, container, path, buf):
        buf.write(self.blobs[path])

    def upload_stream(self, container, path, stream, content_type=""):
        self.uploaded[path] = stream.read()


EVENTS_CSV_0 = b"frame_number,track_id,nest,action\n100,1,3,Exit\n250,2,3,Entry\n"
EVENTS_CSV_1 = b"frame_number,track_id,nest,action\n1600,1,4,Exit\n"


class MergeChunkResultsTest(TestCase):
    def test_merge_concats_namespaces_and_sums(self):
        user = User.objects.create_user("ed", password="x")
        video = Video.objects.create(user=user, title="v", storage_key="k",
                                     file_size_bytes=0)
        job = Job.objects.create(user=user, video=video, modal_job_id="mid123")

        chunks = [
            {"i": 0, "result": {
                "events_csv_path": "u/c0/events.csv", "tracking_csv_path": "",
                "total_events": 2, "entry_count": 1, "exit_count": 1,
                "unique_tracks": 2, "nest_count": 3, "foraging_trip_count": 1,
                "interaction_count": 0, "execution_seconds": 100.5,
                "summary_stats": {"video_fps": 25.0}}},
            {"i": 1, "result": {
                "events_csv_path": "u/c1/events.csv", "tracking_csv_path": "",
                "total_events": 1, "entry_count": 0, "exit_count": 1,
                "unique_tracks": 1, "nest_count": 4, "foraging_trip_count": 0,
                "interaction_count": 0, "execution_seconds": 50.0,
                "summary_stats": {"video_fps": 25.0}}},
        ]
        fake = _FakeS3({"u/c0/events.csv": EVENTS_CSV_0,
                        "u/c1/events.csv": EVENTS_CSV_1})
        with mock.patch("config.storage.get_s3_client", return_value=fake):
            merged = views._merge_chunk_results(job, chunks)

        key = f"{user.pk}/mid123/events.csv"
        self.assertEqual(merged["events_csv_path"], key)
        body = fake.uploaded[key].decode()
        lines = body.strip().splitlines()
        self.assertEqual(len(lines), 4)  # header + 3 rows
        # Frame numbers untouched (already absolute); chunk-1 track ids namespaced.
        self.assertIn("100,1,3,Exit", lines[1])
        self.assertIn("1600,1000001,4,Exit", lines[3])

        self.assertEqual(merged["total_events"], 3)
        self.assertEqual(merged["exit_count"], 2)
        self.assertEqual(merged["unique_tracks"], 3)
        self.assertEqual(merged["nest_count"], 4)
        self.assertEqual(merged["execution_seconds"], 150.5)
        self.assertEqual(merged["summary_stats"]["chunked"], 2)
        self.assertEqual(merged["annotated_video_path"], "")
        self.assertEqual(merged["tracking_csv_path"], "")
