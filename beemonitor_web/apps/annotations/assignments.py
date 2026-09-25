"""Who is reviewing which frames, and how to hand them out.

Assignment is per frame. Sampling and SAM 3 label the frames in one GPU pass;
what people do is check those labels, and "give eai7 500 frames" is the unit a
manager thinks in (memory/39). It used to be per clip, back when a person
labelled every frame of a clip from scratch.

A frame with no ``assigned_to`` is in the pool. Reviewed frames are done and
never handed out again.
"""

from django.db.models import Count, Q
from django.utils import timezone

from .models import Annotation

# SQLite caps bound parameters; Postgres doesn't mind either way.
_CHUNK = 500


def pool(qs):
    """Frames still to review that nobody holds, out of any frame query."""
    return qs.filter(reviewed=False, assigned_to__isnull=True)


def workloads(project):
    """Per person: frames assigned, how many they have reviewed, what is left."""
    rows = (Annotation.objects.filter(project=project, assigned_to__isnull=False)
            .order_by().values("assigned_to_id", "assigned_to__username")
            .annotate(n_assigned=Count("id"),
                      n_reviewed=Count("id", filter=Q(reviewed=True))))
    out = []
    for r in sorted(rows, key=lambda r: r["assigned_to__username"].lower()):
        out.append({
            "user_id": r["assigned_to_id"],
            "username": r["assigned_to__username"],
            "assigned": r["n_assigned"],
            "reviewed": r["n_reviewed"],
            "left": r["n_assigned"] - r["n_reviewed"],
            "pct": round(100 * r["n_reviewed"] / r["n_assigned"]) if r["n_assigned"] else 0,
        })
    return out


def unassigned_count(project):
    return pool(Annotation.objects.filter(project=project)).count()


def pick(qs, count, order="spread"):
    """Ids of ``count`` frames from ``qs``.

    ``spread``: a few frames from many clips across the whole period. Round r
    takes each clip's r-th frame; when a round has more than is still wanted,
    evenly spaced ones are kept, so 500 frames out of 5,000 clips cover all of
    them rather than the first 500. ``clips``: whole clips, oldest first, so
    one person sees a clip through.
    """
    if count <= 0:
        return []
    rows = qs.order_by("video__recorded_at", "video_id", "frame_number") \
             .values_list("id", "video_id")
    if order == "clips":
        return [pk for pk, _v in rows[:count]]

    by_clip, clips = {}, []
    for pk, vid in rows.iterator(chunk_size=5000):
        if vid not in by_clip:
            by_clip[vid] = []
            clips.append(vid)
        by_clip[vid].append(pk)

    out, r = [], 0
    while len(out) < count:
        round_ = [by_clip[v][r] for v in clips if len(by_clip[v]) > r]
        if not round_:
            break
        want = count - len(out)
        if len(round_) > want:
            round_ = [round_[int(i * len(round_) / want)] for i in range(want)]
        out.extend(round_)
        r += 1
    return out


def assign(project, ids, user, by=None):
    """Give these frames to ``user``. Only frames still in the pool move, so
    two managers assigning at once cannot hand the same frame out twice.
    Returns how many were assigned."""
    now, done = timezone.now(), 0
    ids = list(ids)
    for i in range(0, len(ids), _CHUNK):
        done += pool(Annotation.objects.filter(project=project, pk__in=ids[i:i + _CHUNK])) \
            .update(assigned_to=user, assigned_by=by, assigned_at=now)
    return done


def release(project, user):
    """Return this person's unreviewed frames to the pool. Returns how many."""
    return (Annotation.objects.filter(project=project, assigned_to=user, reviewed=False)
            .update(assigned_to=None, assigned_by=None, assigned_at=None))
