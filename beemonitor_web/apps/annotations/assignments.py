"""Who is doing which clips, and how to hand work out.

Assignment is per clip. A clip is a coherent scene — one hotel, one stretch of
time — so two people labelling the same one keep re-deciding the same judgement
calls, and "who labelled this" stops having an answer.
"""

from django.db.models import Count, Q

from .models import ClipAssignment


def by_video(project):
    """``{video_id: ClipAssignment}`` for the whole project, in one query."""
    return {a.video_id: a for a in
            project.assignments.select_related("user").all()}


def workloads(project):
    """Per person: clips assigned, frames in them, frames they have labelled.

    The owner's view of who has what. Counted from the annotations rather than
    from the assignments, because "12 clips" says nothing about whether the
    work is nearly done or barely started.
    """
    from .models import Annotation

    rows = (ClipAssignment.objects.filter(project=project)
            .values("user_id", "user__username")
            .annotate(clips=Count("video_id", distinct=True))
            .order_by("user__username"))

    frames = (Annotation.objects.filter(project=project).order_by()
              .values("video__clip_assignments__user_id")
              .annotate(total=Count("id"),
                        labelled=Count("id", filter=~Q(boxes=[]))))
    by_user = {f["video__clip_assignments__user_id"]: f for f in frames}

    out = []
    for r in rows:
        f = by_user.get(r["user_id"], {})
        total = f.get("total", 0)
        labelled = f.get("labelled", 0)
        out.append({
            "user_id": r["user_id"],
            "username": r["user__username"],
            "clips": r["clips"],
            "frames": total,
            "labelled": labelled,
            "pct": round(100 * labelled / total) if total else 0,
        })
    return out


def unassigned(project):
    """Clips nobody has taken. Visible to everyone so nothing is stranded."""
    return project.videos.exclude(clip_assignments__project=project)


def assign(project, video_ids, user, by=None):
    """Give these clips to one person. Returns how many moved.

    Reassignment overwrites: a clip has at most one owner of the work, and
    silently refusing to move it would leave the page disagreeing with the
    database.
    """
    moved = 0
    for vid in video_ids:
        _obj, created = ClipAssignment.objects.update_or_create(
            project=project, video_id=vid,
            defaults={"user": user, "assigned_by": by},
        )
        moved += 1 if created else 1
    return moved


def distribute(project, video_ids, users, by=None):
    """Deal these clips round-robin. Returns ``{user_id: count}``.

    Hand-picking twenty clips is not something to make anyone do twice, and an
    even split is what people mean by "share this out" almost every time.
    """
    users = list(users)
    if not users:
        return {}
    tally = {}
    for i, vid in enumerate(sorted(video_ids)):
        user = users[i % len(users)]
        ClipAssignment.objects.update_or_create(
            project=project, video_id=vid,
            defaults={"user": user, "assigned_by": by})
        tally[user.id] = tally.get(user.id, 0) + 1
    return tally


def claim(project, video_id, user):
    """Take an unassigned clip for yourself.

    Refuses a clip someone else already holds — claiming is for the pool, not a
    way around an assignment. ``assigned_by`` stays null, which is what marks it
    as self-claimed rather than given.
    """
    existing = project.assignments.filter(video_id=video_id).first()
    if existing:
        return existing.user_id == user.id
    ClipAssignment.objects.create(project=project, video_id=video_id, user=user)
    return True


def release(project, video_id, user):
    """Give a clip back to the pool. Your own only, unless you manage."""
    a = project.assignments.filter(video_id=video_id).first()
    if a is None:
        return False
    if a.user_id != user.id and not project.allows(user, "manager"):
        return False
    a.delete()
    return True
