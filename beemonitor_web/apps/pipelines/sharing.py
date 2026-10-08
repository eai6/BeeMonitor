"""Public, read-only batch share links (memory/47).

A ``BatchShare`` token opens one batch to anyone holding the link. Everything
the public views read goes through here, scoped to that batch's runs launched
by the person who shared it, so the link cannot reach another batch's clips.
"""
from __future__ import annotations

import secrets

from django.utils import timezone

from .models import BatchShare, PipelineRun

# Columns left out of public CSVs unless the owner shares locations.
LOCATION_FIELDS = ("site_name", "location")

# Presigned media for a public viewer lasts an hour, not the app's 24.
PUBLIC_MEDIA_HOURS = 1


class EmptyExport(Exception):
    """A batch CSV with nothing in it; the message says why."""


def new_token() -> str:
    return secrets.token_urlsafe(16)


def live_share(batch_id):
    return BatchShare.objects.filter(batch_id=batch_id, revoked_at=None).first()


def share_for_token(token):
    """The live share for a token, or None (unknown or turned off)."""
    if not token or len(token) > 64:
        return None
    return BatchShare.objects.filter(token=token, revoked_at=None).first()


def owns_batch(user, batch_id) -> bool:
    """Only the launcher may share a batch: every run in it must be theirs."""
    runs = PipelineRun.objects.filter(batch_id=batch_id)
    return (getattr(user, "is_authenticated", False) and runs.exists()
            and not runs.exclude(user=user).exists())


def turn_on(user, batch_id, show_videos=True, show_locations=False):
    share = live_share(batch_id)
    if share is None:
        share = BatchShare.objects.create(
            batch_id=batch_id, token=new_token(), created_by=user,
            show_videos=show_videos, show_locations=show_locations)
    else:
        share.show_videos, share.show_locations = show_videos, show_locations
        share.save(update_fields=["show_videos", "show_locations"])
    return share


def turn_off(batch_id) -> int:
    return BatchShare.objects.filter(batch_id=batch_id, revoked_at=None).update(
        revoked_at=timezone.now())


def shared_runs(share):
    return list(PipelineRun.objects
                .filter(batch_id=share.batch_id, user=share.created_by)
                .select_related("pipeline").order_by("started_at", "id"))


def record_view(share):
    BatchShare.objects.filter(pk=share.pk).update(
        view_count=share.view_count + 1, last_viewed_at=timezone.now())


def contact_by_reference(runs):
    """Interactions and contact time per reference across the batch.

    Read from the interaction analyzers' own summaries, already saved with each
    run, so this costs no S3 reads.
    """
    from . import aggregate

    outputs = [out for run in runs if run.status == run.Status.COMPLETED
               for kind, out in aggregate.analyzer_outputs(run) if kind == "interactions"]
    if not outputs:
        return None
    totals = aggregate.aggregate_interactions(outputs)
    refs = totals["per_reference"]
    contact = sum(r["duration_sec"] for r in refs) or 0.0
    longest = max((r["duration_sec"] for r in refs), default=0.0) or 1.0
    for r in refs:
        r["share_pct"] = round(100 * r["duration_sec"] / contact) if contact else 0
        r["bar_pct"] = round(100 * r["duration_sec"] / longest)
    refs.sort(key=lambda r: (-r["duration_sec"], r["id"]))
    return {
        "interactions": sum(r["interactions"] for r in refs),
        "contact_sec": round(contact, 1),
        "per_reference": refs,
    }


def batch_csv(runs, batch_id, kind):
    """``(filename, fieldnames, rows)`` for one combined batch table.

    Raises ``EmptyExport`` with the reason when the batch has nothing for it,
    and ``KeyError`` for an unknown kind.
    """
    from . import aggregate

    stem = f"{kind}_batch_{str(batch_id)[:8]}.csv"
    if kind == "photos":
        rows = aggregate.photo_rows(runs)
        fields = ["photo_id", "taken_at", "run_id", "insect", "class", "confidence",
                  "species", "species_confidence", "x", "y", "w", "h"]
        return stem, fields, rows
    if kind == "tracks":
        from . import tracks
        sources, _ = aggregate.collect_sources(runs)
        rows = tracks.batch_track_rows(sources, aggregate.read_processed_csv,
                                       aggregate._provenance)
        if not rows:
            raise EmptyExport("No tracks in this batch — its pipeline has no tracking step.")
        return stem, aggregate.PROVENANCE_FIELDS + tracks.TRACK_FIELDS, rows
    path_key = {"events": "events_csv_path",
                "tracking": "tracking_csv_path",
                "interactions": "interactions_csv_path",
                "detections": "detections_csv_path"}[kind]

    # Events and interactions come from the analyzers, recomputed if need be.
    # They must NOT silently fall back to the worker's own file: that one
    # matches an insect to a reference by centroid distance under a flat 50 px,
    # so a bee inside a large flower never appears in it, and a download that
    # quietly hands back a different answer is worse than one that fails.
    if kind in aggregate.PRIMITIVE_KINDS:
        fieldnames, rows = aggregate.primitive_csv(runs, kind)
        if fieldnames:
            return stem, fieldnames, rows
        raise EmptyExport(
            f"No {kind} could be computed for this batch. That usually means no "
            "reference reached the analyzer — save the device's ROI and reference "
            "objects, or wire a Detect node for the reference class into it. The "
            "Tracking CSV is unaffected.")

    from .views import _backfill_interactions_paths
    sources, _ = aggregate.collect_sources(runs)
    if path_key == "interactions_csv_path":
        _backfill_interactions_paths(sources)
    fieldnames, rows = aggregate.combined_csv(sources, path_key)
    if fieldnames is None:
        raise EmptyExport(f"No {kind} data available for this batch — "
                          "none of its completed runs produced that CSV.")
    return stem, fieldnames, rows


def without_locations(fieldnames):
    return [f for f in fieldnames if f not in LOCATION_FIELDS]
