"""Seed UsageEvent from finished analysis jobs, and move credit coupons to the new unit.

* Usage history: one event per completed Job, so estimates (quota.seconds_per_unit)
  use real runs from day one and lifetime usage per user is queryable. These
  events are history only: nobody's balance is charged for them, and the month
  starts at 0 for everyone (memory/50 §10).
* Coupons: ``credits_amount`` was GPU-seconds; a credit is now a GPU-hour on
  the video machine. Existing credit coupons are converted (at least 1), so an
  old 5,000-"credit" code doesn't hand out 5,000 GPU-hours.
"""

from decimal import Decimal

from django.db import migrations

TIER_INSTANCE = {"T4": "ml.g4dn.xlarge", "A10G": "ml.g5.xlarge", "L4": "ml.g6.xlarge",
                 "L40S": "ml.g6e.xlarge", "A100": "ml.p4d.24xlarge"}
CREDITS_PER_HOUR = {"ml.g4dn.xlarge": Decimal("1"), "ml.g6.xlarge": Decimal("1.5"),
                    "ml.g5.xlarge": Decimal("2"), "ml.g6e.xlarge": Decimal("2.5"),
                    "ml.p4d.24xlarge": Decimal("50")}


def backfill(apps, schema_editor):
    Job = apps.get_model("analysis", "Job")
    UsageEvent = apps.get_model("accounts", "UsageEvent")
    batch = []
    jobs = (Job.objects.filter(status="completed", execution_seconds__gt=0)
            .select_related("video").order_by("pk"))
    for job in jobs.iterator(chunk_size=2000):
        cfg = job.config or {}
        if cfg.get("task") == "detect_photo":
            kind, units = "photo", 1
        elif cfg.get("task") == "pre_annotate":
            kind, units = "auto_label", int(cfg.get("max_frames") or 0)
        else:
            kind, units = "analysis", float(getattr(job.video, "duration_seconds", 0) or 0)
        instance = TIER_INSTANCE.get(job.gpu_tier, "ml.g4dn.xlarge")
        secs = float(job.execution_seconds)
        batch.append(UsageEvent(
            user_id=job.user_id, kind=kind, gpu_seconds=secs, instance_type=instance,
            credits=(Decimal(str(secs)) / 3600 * CREDITS_PER_HOUR[instance]).quantize(Decimal("0.0001")),
            cost_usd=job.compute_cost_usd or 0, units=units,
            source_type="job", source_id=str(job.pk),
            created_at=job.completed_at or job.created_at,
        ))
        if len(batch) >= 2000:
            UsageEvent.objects.bulk_create(batch, ignore_conflicts=True)
            batch = []
    if batch:
        UsageEvent.objects.bulk_create(batch, ignore_conflicts=True)


def convert_coupons(apps, schema_editor):
    Coupon = apps.get_model("accounts", "Coupon")
    for c in Coupon.objects.filter(coupon_type="credits", credits_amount__gt=0):
        c.credits_amount = max(1, round(c.credits_amount / 3600))
        c.save(update_fields=["credits_amount"])


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0007_gpu_credits_usage_and_email_log"),
        ("analysis", "0012_job_gpu_seconds_job_stage_seconds_and_more"),
    ]

    operations = [
        migrations.RunPython(backfill, migrations.RunPython.noop),
        migrations.RunPython(convert_coupons, migrations.RunPython.noop),
    ]
