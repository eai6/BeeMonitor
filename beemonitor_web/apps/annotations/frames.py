"""The project's frames: filtered, counted and paged in the database.

One filter feeds three things that must agree: the review grid, the pool a
manager assigns from, and the editor's prev/next. The grid used to walk every
annotation in Python to apply the class filter and count boxes, and the editor
loaded every frame key to find its neighbours — fine at 60 frames, not at the
7,000 a GPU sampling run produces, or the 100,000 a season will.

Frames are ordered by (clip, frame number): both columns are non-null integers
on the unique index, so "next" and "previous" are a keyset lookup rather than
a position in a list.
"""

from django.core.paginator import Paginator
from django.db import connection
from django.db.models import Count, F, Func, IntegerField, Q, Sum

from .models import Annotation

PAGE_SIZE = 60

FILTER_KEYS = ("status", "who", "device", "cls", "from", "to")
STATUSES = ("review", "reviewed", "all")
ORDER = ("video_id", "frame_number")


def parse(params):
    """The frame filter from a GET/POST mapping. Unknown values are dropped."""
    f = {k: (params.get(k) or "").strip() for k in FILTER_KEYS}
    if f["status"] not in STATUSES:
        f["status"] = "review"
    if f["who"] not in ("", "me", "none") and not f["who"].isdigit():
        f["who"] = ""
    if not f["device"].isdigit():
        f["device"] = ""
    for k in ("from", "to"):
        if len(f[k]) != 10:
            f[k] = ""
    return f


def query(f):
    """The filter as a query string (no leading ``?``), for links that keep it.

    ``status`` is always in it when present: it is what tells the editor it
    was opened from the grid and should walk the grid's frames.
    """
    from urllib.parse import urlencode
    return urlencode({k: v for k, v in f.items() if v})


def base(project, f, user):
    """Every filter except the review status — the status toggle's counts
    come from this, so each option says how many it would show."""
    qs = Annotation.objects.filter(project=project)
    who = f.get("who")
    if who == "me":
        qs = qs.filter(assigned_to=user)
    elif who == "none":
        qs = qs.filter(assigned_to__isnull=True)
    elif who:
        qs = qs.filter(assigned_to_id=int(who))
    if f.get("device"):
        qs = qs.filter(video__device_id=int(f["device"]))
    if f.get("from"):
        qs = qs.filter(video__recorded_at__date__gte=f["from"])
    if f.get("to"):
        qs = qs.filter(video__recorded_at__date__lte=f["to"])
    if f.get("cls"):
        qs = with_class(qs, f["cls"])
    return qs


def with_status(qs, status):
    if status == "review":
        return qs.filter(reviewed=False)
    if status == "reviewed":
        return qs.filter(reviewed=True)
    return qs


def filtered(project, f, user):
    return with_status(base(project, f, user), f.get("status", "review"))


def with_class(qs, cls):
    """Frames with at least one box of this class."""
    if connection.vendor == "postgresql":
        return qs.filter(boxes__contains=[{"class": cls}])
    # SQLite (tests, local dev) has no JSON containment; the sets are small.
    ids = [pk for pk, boxes in qs.values_list("pk", "boxes")
           if any((b or {}).get("class") == cls for b in boxes or [])]
    return qs.filter(pk__in=ids)


def _box_count():
    fn = "jsonb_array_length" if connection.vendor == "postgresql" else "json_array_length"
    return Sum(Func(F("boxes"), function=fn, output_field=IntegerField()))


def metrics(project):
    """The page's three numbers, and the lines under them. Three queries."""
    from .models import FrameSamplingTask

    agg = Annotation.objects.filter(project=project).aggregate(
        n_frames=Count("id"),
        n_labelled=Count("id", filter=~Q(boxes=[])),
        n_reviewed=Count("id", filter=Q(reviewed=True)),
        n_assigned=Count("id", filter=Q(reviewed=False, assigned_to__isnull=False)),
        n_clips=Count("video", distinct=True),
        n_boxes=_box_count(),
    )
    # (Aliases can't share a name with a field — "reviewed" is one.)
    agg = {k[2:]: v for k, v in agg.items()}
    agg["clips_with_frames"] = agg.pop("clips")
    clips = project.videos.count()
    empty = (FrameSamplingTask.objects
             .filter(project=project, status=FrameSamplingTask.Status.COMPLETED)
             .exclude(video_id__in=Annotation.objects.filter(project=project)
                      .values("video_id"))
             .values("video_id").distinct().count())
    to_review = agg["frames"] - agg["reviewed"]
    return {
        "clips": clips,
        "clips_with_frames": agg["clips_with_frames"],
        "clips_empty": empty,
        "clips_unsampled": max(clips - agg["clips_with_frames"] - empty, 0),
        "frames": agg["frames"],
        "labelled": agg["labelled"],
        "unlabelled": agg["frames"] - agg["labelled"],
        "boxes": agg["boxes"] or 0,
        "reviewed": agg["reviewed"],
        "to_review": to_review,
        "assigned": agg["assigned"],
        "pct_reviewed": round(100 * agg["reviewed"] / agg["frames"]) if agg["frames"] else 0,
    }


def status_counts(qs):
    """``{review, reviewed, all}`` over ``base()`` — one query."""
    agg = qs.aggregate(n_all=Count("id"),
                       n_reviewed=Count("id", filter=Q(reviewed=True)))
    return {"all": agg["n_all"], "reviewed": agg["n_reviewed"],
            "review": agg["n_all"] - agg["n_reviewed"]}


def page(qs, number, size=PAGE_SIZE):
    """One page of frame cards, thumbnails presigned. Returns (page, cards)."""
    paginator = Paginator(
        qs.select_related("video", "video__device", "assigned_to").order_by(*ORDER),
        size)
    pg = paginator.get_page(number)
    cards = []
    for ann in pg.object_list:
        boxes = ann.boxes or []
        cards.append({
            "video_pk": ann.video_id,
            "video_title": ann.video.title,
            "device": ann.video.device.name if ann.video.device_id else "",
            "recorded_at": ann.video.recorded_at,
            "frame_number": ann.frame_number,
            "box_count": len(boxes),
            "classes": sorted({(b or {}).get("class", "unknown") for b in boxes}),
            "frame_image_path": ann.frame_image_path or "",
            "reviewed": ann.reviewed,
            "review_source": ann.review_source,
            "assignee": ann.assigned_to,
        })
    _presign(cards)
    return pg, cards


def _presign(cards):
    if not any(c["frame_image_path"] for c in cards):
        return
    try:
        from config.storage import get_s3_client
        s3 = get_s3_client()
        for c in cards:
            if c["frame_image_path"]:
                c["thumbnail_url"] = s3.generate_presigned_url(
                    "processed", c["frame_image_path"], expiry_hours=2)
    except Exception as e:  # thumbnails fall back to the frame_image view
        import logging
        logging.getLogger(__name__).warning("Failed to presign thumbnails: %s", e)


def neighbours(qs, video_id, frame):
    """(position, total, prev, next) of a frame within ``qs``.

    ``prev``/``next`` are ``(video_id, frame_number)`` or None. The frame need
    not be in ``qs``: after saving, a frame leaves the "to review" queue and
    next is still the one after it.
    """
    v, n = int(video_id), int(frame)
    before = Q(video_id__lt=v) | Q(video_id=v, frame_number__lt=n)
    after = Q(video_id__gt=v) | Q(video_id=v, frame_number__gt=n)
    prev = (qs.filter(before).order_by("-video_id", "-frame_number")
            .values_list(*ORDER).first())
    nxt = qs.filter(after).order_by(*ORDER).values_list(*ORDER).first()
    return qs.filter(before).count() + 1, qs.count(), prev, nxt
