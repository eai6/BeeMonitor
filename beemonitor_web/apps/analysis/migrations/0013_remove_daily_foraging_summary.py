"""Drop the daily foraging-trips table (memory/60: trips are left to the user).

Its rows are written to one CSV in the processed bucket first —
``exports/daily_foraging_summary_<timestamp>.csv`` — and the table is dropped
only if that upload succeeded. An empty table (every test database, any fresh
install) has nothing to keep and is dropped straight away.
"""

import csv
import io

from django.db import migrations


FIELDS = ("id", "user_id", "site_name", "device_id", "date", "total_trips",
          "cross_video_trips", "avg_duration_sec", "median_duration_sec",
          "trips_per_nest", "trips_csv_path", "video_count", "computed_at", "trips")


def export_rows(apps, schema_editor):
    Summary = apps.get_model("analysis", "DailyForagingSummary")
    rows = list(Summary.objects.order_by("id").values(*FIELDS))
    if not rows:
        return
    from django.utils import timezone

    from config.storage import get_s3_client

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    key = f"exports/daily_foraging_summary_{timezone.now():%Y%m%dT%H%M%SZ}.csv"
    # Raises on failure, which aborts the migration and keeps the table.
    get_s3_client().upload_stream("processed", key, io.BytesIO(buf.getvalue().encode()),
                                  content_type="text/csv")


class Migration(migrations.Migration):

    dependencies = [
        ("analysis", "0012_job_gpu_seconds_job_stage_seconds_and_more"),
    ]

    operations = [
        migrations.RunPython(export_rows, migrations.RunPython.noop),
        migrations.DeleteModel(name="DailyForagingSummary"),
    ]
