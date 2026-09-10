"""Analyzers must be able to read the paths a JobResult actually stores.

Every JobResult stores a bare key into the processed bucket
("42/abc123/tracking.csv"). `_read_csv` handled s3:// URLs and local files, so
that key fell through to `pd.read_csv("42/abc123/tracking.csv")` — a relative
filename that does not exist. Every local analyzer then took its "tracking CSV
not available" branch and reported the job summary instead of analysing
anything. It read as a data condition; it was plumbing.
"""

import csv
import io
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase

from apps.pipelines import ops

ROWS = [{"frame": 0, "track_id": 1, "cx": 0.5, "cy": 0.5}]


def _csv_bytes(rows=ROWS):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue().encode()


class ReadCsvPathTests(SimpleTestCase):
    def test_a_processed_bucket_key_is_fetched(self):
        s3 = MagicMock()

        def _download(container, key, buf):
            self.assertEqual(container, "processed")
            self.assertEqual(key, "42/abc123/tracking.csv")
            buf.write(_csv_bytes())

        s3.download_to_stream.side_effect = _download

        with patch("config.storage.get_s3_client", return_value=s3):
            df = ops._read_csv("42/abc123/tracking.csv")

        self.assertIsNotNone(df)
        self.assertEqual(len(df), 1)

    def test_a_local_file_still_wins_over_a_bucket_lookup(self):
        path = Path(tempfile.mkdtemp()) / "t.csv"
        path.write_bytes(_csv_bytes())
        s3 = MagicMock()

        with patch("config.storage.get_s3_client", return_value=s3):
            df = ops._read_csv(str(path))

        self.assertEqual(len(df), 1)
        s3.download_to_stream.assert_not_called()

    def test_an_empty_path_reads_nothing(self):
        self.assertIsNone(ops._read_csv(""))
        self.assertIsNone(ops._read_csv(None))

    def test_a_missing_object_degrades_to_none_rather_than_raising(self):
        s3 = MagicMock()
        s3.download_to_stream.side_effect = RuntimeError("404")

        with patch("config.storage.get_s3_client", return_value=s3):
            self.assertIsNone(ops._read_csv("nope/missing.csv"))

    def test_the_analyzer_loaders_go_through_it(self):
        s3 = MagicMock()
        s3.download_to_stream.side_effect = lambda c, k, b: b.write(_csv_bytes())

        with patch("config.storage.get_s3_client", return_value=s3):
            tracking = ops.load_tracking_df({"tracking_csv_path": "u/j/tracking.csv"})
            events = ops.load_events_df({"events_csv_path": "u/j/events.csv"})
            inter = ops.load_interactions_df({"interactions_csv_path": "u/j/i.csv"})

        for df in (tracking, events, inter):
            self.assertIsNotNone(df)


class PandasIsInstalledTests(SimpleTestCase):
    def test_the_image_ships_pandas(self):
        """Without it every analyzer silently reports the job summary."""
        import pandas  # noqa: F401

    def test_it_is_declared_in_the_web_requirements(self):
        from django.conf import settings

        base = (Path(settings.BASE_DIR) / "requirements" / "base.txt").read_text()

        self.assertIn("pandas", base)
