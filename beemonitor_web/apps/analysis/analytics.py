"""
Analytics functions for the Activity Visualization Dashboard.

Aggregate data from completed JobResults and the associated Video temporal
fields to produce activity charts and summaries.
"""

import logging
from collections import defaultdict

from django.db.models import Avg, Count, Sum

from .models import Job, JobResult

logger = logging.getLogger(__name__)


def get_activity_over_time(user, site_name=None, year=None, month=None):
    """
    Return daily event counts across all completed jobs with entry/exit breakdown.

    Returns a list of dicts: [{"date": "2024-06-01", "entries": 12, "exits": 8, "total": 20}, ...]
    """
    qs = JobResult.objects.filter(
        job__user=user,
        job__status=Job.Status.COMPLETED,
    ).select_related("job__video")

    if site_name:
        qs = qs.filter(job__video__site_name=site_name)
    if year:
        qs = qs.filter(job__video__year=year)
    if month:
        qs = qs.filter(job__video__month=month)

    # Group by date using the video's recorded_at or uploaded_at
    daily = defaultdict(lambda: {"entries": 0, "exits": 0, "total": 0})

    for result in qs:
        video = result.job.video
        date_val = video.recorded_at or video.uploaded_at
        if date_val:
            date_key = date_val.strftime("%Y-%m-%d")
        else:
            date_key = "unknown"
        daily[date_key]["entries"] += result.entry_count
        daily[date_key]["exits"] += result.exit_count
        daily[date_key]["total"] += result.total_events

    # Sort by date
    sorted_data = []
    for date_key in sorted(daily.keys()):
        sorted_data.append({
            "date": date_key,
            "entries": daily[date_key]["entries"],
            "exits": daily[date_key]["exits"],
            "total": daily[date_key]["total"],
        })
    return sorted_data


def get_period_averages(user, site_name=None, year=None, month=None):
    """
    Return average events per hour-of-day, per day-of-week, and per month.
    """
    qs = JobResult.objects.filter(
        job__user=user,
        job__status=Job.Status.COMPLETED,
    ).select_related("job__video")

    if site_name:
        qs = qs.filter(job__video__site_name=site_name)
    if year:
        qs = qs.filter(job__video__year=year)
    if month:
        qs = qs.filter(job__video__month=month)

    hourly_totals = defaultdict(list)
    daily_totals = defaultdict(list)
    monthly_totals = defaultdict(list)

    for result in qs:
        video = result.job.video
        dt = video.recorded_at or video.uploaded_at

        if dt:
            hourly_totals[dt.hour].append(result.total_events)
            daily_totals[dt.weekday()].append(result.total_events)
            monthly_totals[dt.month].append(result.total_events)

    def _avg(lst):
        return round(sum(lst) / len(lst), 1) if lst else 0

    return {
        "hourly": {h: _avg(hourly_totals[h]) for h in range(24)},
        "daily": {d: _avg(daily_totals[d]) for d in range(7)},
        "monthly": {m: _avg(monthly_totals[m]) for m in range(1, 13)},
    }


def get_cumulative_activity(user, site_name=None, year=None, month=None):
    """
    Return cumulative sum of events over time (sorted by date).
    """
    daily_data = get_activity_over_time(user, site_name=site_name, year=year, month=month)

    cumulative = 0
    result = []
    for entry in daily_data:
        cumulative += entry["total"]
        result.append({
            "date": entry["date"],
            "cumulative": cumulative,
        })
    return result


def get_nest_activity_heatmap(user, job_id=None):
    """
    Return per-nest event counts from summary_stats.

    Looks for nest-related data in JobResult.summary_stats. Expected format
    in summary_stats: {"nests": {"nest_1": {"entries": 5, "exits": 3}, ...}}

    Returns a list of dicts: [{"nest_id": "nest_1", "entries": 5, "exits": 3, "total": 8}, ...]
    """
    qs = JobResult.objects.filter(
        job__user=user,
        job__status=Job.Status.COMPLETED,
    )

    if job_id:
        qs = qs.filter(job_id=job_id)

    nest_totals = defaultdict(lambda: {"entries": 0, "exits": 0, "total": 0})

    for result in qs:
        stats = result.summary_stats or {}
        nests = stats.get("nests", {})
        if isinstance(nests, dict):
            for nest_id, counts in nests.items():
                if isinstance(counts, dict):
                    entries = counts.get("entries", 0)
                    exits = counts.get("exits", 0)
                    nest_totals[nest_id]["entries"] += entries
                    nest_totals[nest_id]["exits"] += exits
                    nest_totals[nest_id]["total"] += entries + exits

        # Also check for nest_count to create placeholder entries if no detailed data
        if not nests and result.nest_count > 0:
            for i in range(1, result.nest_count + 1):
                nest_id = f"nest_{i}"
                if nest_id not in nest_totals:
                    # Distribute events roughly across nests
                    entries = result.entry_count // result.nest_count
                    exits = result.exit_count // result.nest_count
                    nest_totals[nest_id]["entries"] += entries
                    nest_totals[nest_id]["exits"] += exits
                    nest_totals[nest_id]["total"] += entries + exits

    result_list = []
    for nest_id in sorted(nest_totals.keys()):
        result_list.append({
            "nest_id": nest_id,
            "entries": nest_totals[nest_id]["entries"],
            "exits": nest_totals[nest_id]["exits"],
            "total": nest_totals[nest_id]["total"],
        })
    return result_list


def get_summary_stats(user, site_name=None, year=None, month=None):
    """
    Return high-level summary statistics.

    Returns a dict:
    {
        "total_videos": int,
        "total_events": int,
        "total_entries": int,
        "total_exits": int,
        "avg_events_per_video": float,
        "total_unique_tracks": int,
        "completed_jobs": int,
    }
    """
    qs = JobResult.objects.filter(
        job__user=user,
        job__status=Job.Status.COMPLETED,
    ).select_related("job__video")

    if site_name:
        qs = qs.filter(job__video__site_name=site_name)
    if year:
        qs = qs.filter(job__video__year=year)
    if month:
        qs = qs.filter(job__video__month=month)

    agg = qs.aggregate(
        total_events=Sum("total_events"),
        total_entries=Sum("entry_count"),
        total_exits=Sum("exit_count"),
        total_tracks=Sum("unique_tracks"),
        completed_jobs=Count("id"),
        avg_events=Avg("total_events"),
    )

    # Count distinct videos with completed jobs
    video_ids = qs.values_list("job__video_id", flat=True).distinct()

    return {
        "total_videos": len(video_ids),
        "total_events": agg["total_events"] or 0,
        "total_entries": agg["total_entries"] or 0,
        "total_exits": agg["total_exits"] or 0,
        "avg_events_per_video": round(agg["avg_events"] or 0, 1),
        "total_unique_tracks": agg["total_tracks"] or 0,
        "completed_jobs": agg["completed_jobs"] or 0,
    }
