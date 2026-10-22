"""Raw footage is deleted 3 months after upload, for Standard accounts (memory/50 §6).

The rules, all of which exist so that nobody is surprised:

* **Who.** Standard accounts only. Pro and platform admins keep everything; an
  account with a storage request waiting keeps everything until it's decided.
* **When.** A clip is due 89 days after upload (the day before S3 would move it
  to Glacier IR, which bills 90 days minimum), or the day after an approved
  "keep until" date. When a clip comes within 30 days of that, it is scheduled
  (``Video.raw_delete_on``) for that day but never less than 30 days out, so
  footage that is already old still gets the full notice.
* **Notices.** Emails 30, 15 and 7 days before. A clip is deleted only when all
  three were *sent* and each has had its full time: a notice that failed or
  went out late pushes the deletion back on its own.
* **What.** Only the raw file in the raw-videos bucket. The row, its results
  (detections, tracks, events, crops, annotations) and its thumbnail stay.

``RAW_FOOTAGE_DELETION`` decides what the daily pass does: ``dry_run`` (the
default) only reports, on the platform dashboard; ``on`` sends notices and
deletes; ``off`` does nothing.
"""

import logging
from datetime import timedelta

from django.conf import settings
from django.contrib.auth import get_user_model
from django.db.models import Count, Min, Q, Sum
from django.utils import timezone

from .models import Video

logger = logging.getLogger(__name__)

KEEP_DAYS = 89
NOTICE_DAYS = (30, 15, 7)
# Deleted per daily pass at most, so a mistake can't take everything at once.
MAX_DELETIONS_PER_RUN = 5000


def mode() -> str:
    m = getattr(settings, "RAW_FOOTAGE_DELETION", "dry_run")
    return m if m in ("off", "dry_run", "on") else "dry_run"


def _users_whose_footage_expires():
    """Standard, non-admin accounts with no storage request waiting."""
    from apps.accounts.models import AccountRequest, UserProfile

    waiting = AccountRequest.objects.filter(kind=AccountRequest.Kind.STORAGE,
                                            status=AccountRequest.Status.PENDING).values("user_id")
    return (get_user_model().objects.filter(is_superuser=False, profile__tier=UserProfile.Tier.STANDARD)
            .exclude(pk__in=waiting).select_related("profile"))


def _raw(qs):
    """Rows whose raw file is ours and still there."""
    return qs.filter(raw_deleted_at__isnull=True).exclude(storage_key="").exclude(
        storage_key__startswith="s3://")


def _coming_due(user, today):
    """This user's clips due within 30 days, by upload date and retention."""
    horizon = today + timedelta(days=NOTICE_DAYS[0])
    keep_until = user.profile.raw_retention_until
    if keep_until and keep_until >= horizon:
        return Video.everything.none()
    return _raw(Video.everything.filter(user=user)).filter(
        uploaded_at__date__lte=horizon - timedelta(days=KEEP_DAYS))


def _due_date(video, keep_until, today):
    natural = video.uploaded_at.date() + timedelta(days=KEEP_DAYS)
    if keep_until:
        natural = max(natural, keep_until + timedelta(days=1))
    return max(natural, today + timedelta(days=NOTICE_DAYS[0]))


def preview(today=None):
    """Per user, what is (or, in a dry run, would be) deleted next:
    [{user, clips, bytes, first_date}], biggest first. For the dashboard."""
    today = today or timezone.localdate()
    out = []
    for user in _users_whose_footage_expires():
        if mode() == "on":
            qs = _raw(Video.everything.filter(user=user, raw_delete_on__isnull=False))
            agg = qs.aggregate(n=Count("id"), b=Sum("file_size_bytes"), first=Min("raw_delete_on"))
            first = agg["first"]
        else:
            qs = _coming_due(user, today)
            agg = qs.aggregate(n=Count("id"), b=Sum("file_size_bytes"), up=Min("uploaded_at"))
            first = (max(agg["up"].date() + timedelta(days=KEEP_DAYS), today + timedelta(days=NOTICE_DAYS[0]))
                     if agg["up"] else None)
        if agg["n"]:
            out.append({"user": user, "clips": agg["n"], "bytes": agg["b"] or 0, "first_date": first})
    return sorted(out, key=lambda r: -r["bytes"])


def scheduled_for(user):
    """What the account page tells a user: [{date, clips, bytes}] by date."""
    if mode() != "on":
        return []
    rows = (_raw(Video.everything.filter(user=user, raw_delete_on__isnull=False))
            .values("raw_delete_on").annotate(n=Count("id"), b=Sum("file_size_bytes"))
            .order_by("raw_delete_on"))
    return [{"date": r["raw_delete_on"], "clips": r["n"], "bytes": r["b"] or 0} for r in rows]


def run(today=None) -> dict:
    """The daily pass. Returns counts for the log."""
    today = today or timezone.localdate()
    m = mode()
    if m != "on":
        return {"mode": m, "would_delete_users": len(preview(today)) if m == "dry_run" else 0}
    stats = {"mode": m, "scheduled": 0, "unscheduled": 0, "notices": 0, "failed_notices": 0,
             "deleted": 0, "deleted_bytes": 0}
    expiring = list(_users_whose_footage_expires())

    # Anyone no longer subject to deletion (now Pro, an admin, a request
    # waiting) loses their schedule; so do clips a newer "keep until" covers.
    keep_ids = {u.pk for u in expiring}
    stats["unscheduled"] += (Video.everything.filter(raw_delete_on__isnull=False, raw_deleted_at__isnull=True)
                             .exclude(user_id__in=keep_ids).update(raw_delete_on=None))
    for user in expiring:
        keep_until = user.profile.raw_retention_until
        if keep_until:
            stats["unscheduled"] += Video.everything.filter(
                user=user, raw_deleted_at__isnull=True, raw_delete_on__lte=keep_until).update(raw_delete_on=None)

    deleted_left = MAX_DELETIONS_PER_RUN
    for user in expiring:
        keep_until = user.profile.raw_retention_until
        for video in _coming_due(user, today).filter(raw_delete_on__isnull=True).only("id", "uploaded_at"):
            Video.everything.filter(pk=video.pk).update(raw_delete_on=_due_date(video, keep_until, today))
            stats["scheduled"] += 1
        for delete_on in (_raw(Video.everything.filter(user=user, raw_delete_on__isnull=False))
                          .values_list("raw_delete_on", flat=True).distinct().order_by("raw_delete_on")):
            sent, failed = _send_due_notices(user, delete_on, today)
            stats["notices"] += sent
            stats["failed_notices"] += failed
            if delete_on <= today and deleted_left > 0 and _notices_complete(user, delete_on, today):
                n, b = _delete(user, delete_on, deleted_left)
                deleted_left -= n
                stats["deleted"] += n
                stats["deleted_bytes"] += b
    logger.info("raw footage retention: %s", stats)
    return stats


def _group(user, delete_on):
    return _raw(Video.everything.filter(user=user, raw_delete_on=delete_on))


def _send_due_notices(user, delete_on, today):
    """Send each of the 30/15/7-day notices whose day has come and that hasn't
    gone out yet. A failed one is retried on the next pass."""
    from apps.accounts.emails import send
    from apps.accounts.models import EmailLog, RetentionNotice

    sent = failed = 0
    for days in NOTICE_DAYS:
        if today < delete_on - timedelta(days=days):
            continue
        if RetentionNotice.objects.filter(user=user, delete_on=delete_on, days_before=days,
                                          status=RetentionNotice.Status.SENT).exists():
            continue
        group = _group(user, delete_on)
        agg = group.aggregate(n=Count("id"), b=Sum("file_size_bytes"))
        if not agg["n"]:
            return sent, failed
        units = sorted({name for name in group.exclude(device=None).values_list("device__name", flat=True)})
        log = send("raw_footage_notice", user.email, _subject(delete_on, agg["n"]),
                   _body(user, delete_on, agg["n"], agg["b"] or 0, units), user=user) if user.email else None
        ok = log is not None and log.status == EmailLog.Status.SENT
        RetentionNotice.objects.create(user=user, delete_on=delete_on, days_before=days,
                                       status=RetentionNotice.Status.SENT if ok else RetentionNotice.Status.FAILED,
                                       clips=agg["n"], bytes=agg["b"] or 0, sent_on=today)
        if ok:
            sent += 1
        else:
            failed += 1
            break  # the later notices wait for this one
    return sent, failed


def _notices_complete(user, delete_on, today):
    """All three notices sent, each at least its own number of days ago."""
    from apps.accounts.models import RetentionNotice

    for days in NOTICE_DAYS:
        notice = RetentionNotice.objects.filter(user=user, delete_on=delete_on, days_before=days,
                                                status=RetentionNotice.Status.SENT).order_by("sent_on").first()
        if notice is None or notice.sent_on > today - timedelta(days=days):
            return False
    return True


def _delete(user, delete_on, limit):
    from apps.accounts.models import AdminAction
    from config.storage import get_s3_client

    s3 = get_s3_client()
    n = b = 0
    for video in _group(user, delete_on)[:limit]:
        try:
            s3.delete_blob("raw-videos", video.storage_key)
        except Exception:
            logger.exception("could not delete raw footage %s", video.storage_key)
            continue
        Video.everything.filter(pk=video.pk).update(raw_deleted_at=timezone.now())
        n += 1
        b += video.file_size_bytes or 0
    if n:
        AdminAction.record(None, "raw_deleted",
                           f"Deleted the raw footage of {n} clips ({b / 1e9:.1f} GB) for {user.username}",
                           target=user, delete_on=delete_on.isoformat(), clips=n, bytes=b)
    return n, b


def _subject(delete_on, clips):
    return f"BeeMonitor: raw footage of {clips} clip{'s' if clips != 1 else ''} will be deleted on {delete_on:%b} {delete_on.day}"


def _body(user, delete_on, clips, size, units):
    domain = getattr(settings, "SITE_DOMAIN", "") or "beemonitor.edwardamoah.com"
    from_units = f" from {', '.join(units)}" if units else ""
    return (
        f"Hello {user.username},\n\n"
        f"The raw footage of {clips} clip{'s' if clips != 1 else ''}{from_units} "
        f"({size / 1e9:.1f} GB) will be deleted on {delete_on:%B} {delete_on.day}, {delete_on.year}. "
        "On the Standard plan raw footage is kept for 3 months after upload.\n\n"
        "Everything derived from it stays: detections, tracks, events, crops and annotations. "
        "Deleted clips can't be analysed again.\n\n"
        f"To keep it longer, request longer storage: https://{domain}/accounts/usage/\n"
        "Nothing is deleted while a request is waiting.\n"
    )
