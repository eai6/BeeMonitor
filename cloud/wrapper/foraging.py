"""Foraging trip computation from BeeMonitor event data.

Pairs Exit→Entry events at the same nest to identify foraging trips,
with duration filtering to exclude noise.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def compute_foraging_trips(
    events_df: pd.DataFrame,
    fps: float = 30.0,
    min_sec: float = 10,
    max_sec: float = 7200,
) -> pd.DataFrame:
    """Compute foraging trips by pairing Exit→Entry events per nest.

    Args:
        events_df: Events DataFrame with columns: action, nest, frame_number, track_id
        fps: Video frames per second (for frame→seconds conversion)
        min_sec: Minimum trip duration in seconds (default 10s)
        max_sec: Maximum trip duration in seconds (default 7200s = 2 hours)

    Returns:
        DataFrame with columns: nest, exit_frame, entry_frame, exit_sec,
        entry_sec, duration_sec, exit_track_id, entry_track_id
    """
    if events_df is None or events_df.empty:
        return pd.DataFrame(columns=[
            "nest", "exit_frame", "entry_frame", "exit_sec",
            "entry_sec", "duration_sec", "exit_track_id", "entry_track_id",
        ])

    required = {"action", "nest", "frame_number"}
    if not required.issubset(events_df.columns):
        logger.warning("Events DataFrame missing required columns: %s", required - set(events_df.columns))
        return pd.DataFrame(columns=[
            "nest", "exit_frame", "entry_frame", "exit_sec",
            "entry_sec", "duration_sec", "exit_track_id", "entry_track_id",
        ])

    fps = max(fps, 1.0)  # guard against zero/negative

    trips = []
    for nest_id, nest_events in events_df.groupby("nest"):
        nest_sorted = nest_events.sort_values("frame_number").reset_index(drop=True)
        last_exit = None

        for _, row in nest_sorted.iterrows():
            if row["action"] == "Exit":
                last_exit = row
            elif row["action"] == "Entry" and last_exit is not None:
                exit_sec = last_exit["frame_number"] / fps
                entry_sec = row["frame_number"] / fps
                duration = entry_sec - exit_sec

                if min_sec <= duration <= max_sec:
                    trips.append({
                        "nest": nest_id,
                        "exit_frame": int(last_exit["frame_number"]),
                        "entry_frame": int(row["frame_number"]),
                        "exit_sec": round(exit_sec, 2),
                        "entry_sec": round(entry_sec, 2),
                        "duration_sec": round(duration, 2),
                        "exit_track_id": last_exit.get("track_id", ""),
                        "entry_track_id": row.get("track_id", ""),
                    })
                last_exit = None

    df = pd.DataFrame(trips)
    logger.info("Computed %d foraging trips from %d events", len(df), len(events_df))
    return df


def compute_trip_summary(trips_df: pd.DataFrame) -> dict:
    """Compute summary statistics from a foraging trips DataFrame.

    Returns:
        Dict with: total_trips, avg_duration_sec, median_duration_sec,
        min_duration_sec, max_duration_sec, trips_per_nest
    """
    if trips_df is None or trips_df.empty:
        return {
            "total_trips": 0,
            "avg_duration_sec": 0,
            "median_duration_sec": 0,
            "min_duration_sec": 0,
            "max_duration_sec": 0,
            "trips_per_nest": {},
        }

    durations = trips_df["duration_sec"]
    trips_per_nest = trips_df.groupby("nest").size().to_dict()
    # Convert nest keys to strings for JSON serialization
    trips_per_nest = {str(k): int(v) for k, v in trips_per_nest.items()}

    return {
        "total_trips": len(trips_df),
        "avg_duration_sec": round(durations.mean(), 1),
        "median_duration_sec": round(durations.median(), 1),
        "min_duration_sec": round(durations.min(), 1),
        "max_duration_sec": round(durations.max(), 1),
        "trips_per_nest": trips_per_nest,
    }


def compute_daily_foraging_trips(
    video_events: list,
    min_sec: float = 10,
    max_sec: float = 7200,
) -> pd.DataFrame:
    """Compute foraging trips across multiple videos for a site+day.

    Merges events from all videos using absolute timestamps, enabling
    detection of trips where a bee exits in one video and enters in another.

    Args:
        video_events: List of dicts, each with:
            - video_id: identifier for the source video
            - recorded_at: datetime when video recording started
            - fps: video frames per second
            - events_df: DataFrame with action, nest, frame_number, track_id
        min_sec: Minimum trip duration in seconds
        max_sec: Maximum trip duration in seconds

    Returns:
        DataFrame with columns: nest, exit_time, entry_time, duration_sec,
        exit_video_id, entry_video_id, exit_track_id, entry_track_id, is_cross_video
    """
    from datetime import timedelta

    all_events = []
    for ve in video_events:
        df = ve["events_df"]
        if df is None or df.empty:
            continue
        if "action" not in df.columns or "nest" not in df.columns or "frame_number" not in df.columns:
            continue

        recorded_at = ve["recorded_at"]
        fps = max(ve.get("fps", 30.0), 1.0)
        video_id = ve.get("video_id", "")

        for _, row in df.iterrows():
            sec_offset = row["frame_number"] / fps
            abs_time = recorded_at + timedelta(seconds=sec_offset)
            all_events.append({
                "action": row["action"],
                "nest": row["nest"],
                "absolute_time": abs_time,
                "video_id": video_id,
                "track_id": row.get("track_id", ""),
            })

    if not all_events:
        return pd.DataFrame(columns=[
            "nest", "exit_time", "entry_time", "duration_sec",
            "exit_video_id", "entry_video_id", "exit_track_id", "entry_track_id",
            "is_cross_video",
        ])

    merged = pd.DataFrame(all_events)

    trips = []
    for nest_id, nest_events in merged.groupby("nest"):
        nest_sorted = nest_events.sort_values("absolute_time").reset_index(drop=True)
        last_exit = None

        for _, row in nest_sorted.iterrows():
            if row["action"] == "Exit":
                last_exit = row
            elif row["action"] == "Entry" and last_exit is not None:
                duration = (row["absolute_time"] - last_exit["absolute_time"]).total_seconds()
                if min_sec <= duration <= max_sec:
                    trips.append({
                        "nest": nest_id,
                        "exit_time": last_exit["absolute_time"].isoformat(),
                        "entry_time": row["absolute_time"].isoformat(),
                        "duration_sec": round(duration, 2),
                        "exit_video_id": last_exit["video_id"],
                        "entry_video_id": row["video_id"],
                        "exit_track_id": last_exit["track_id"],
                        "entry_track_id": row["track_id"],
                        "is_cross_video": last_exit["video_id"] != row["video_id"],
                    })
                last_exit = None

    result = pd.DataFrame(trips)
    cross_count = result["is_cross_video"].sum() if not result.empty else 0
    logger.info("Daily trips: %d total, %d cross-video", len(result), cross_count)
    return result
