"""Publishing a dataset, and taking a copy of one.

Publishing makes a finished thing reusable; sharing invites named people to
work on an unfinished one. A public project is **readable and copyable, never
writable** — "improve upon it" means take a copy and diverge, because shared
write with strangers is a moderation problem and a dataset that changes
underneath its users cannot be cited.
"""

from django.db.models import Count, Q
from django.utils import timezone

from .models import Annotation, AnnotationProject

# Frames and boxes are the dataset. These describe the FIELD SITE, and for a bee
# hotel they amount to a movement log — so they are withheld unless the
# publisher opts in, once, knowingly.
FIELD_METADATA = ("device", "site_name", "recorded_at", "location")


def publish(project, *, include_metadata=False):
    project.visibility = AnnotationProject.Visibility.PUBLIC
    project.publish_metadata = bool(include_metadata)
    project.published_at = project.published_at or timezone.now()
    project.save(update_fields=["visibility", "publish_metadata", "published_at"])
    return project


def unpublish(project):
    """Stop offering it. Copies already taken are unaffected — they reference
    the same frame objects, which nothing deletes."""
    project.visibility = AnnotationProject.Visibility.PRIVATE
    project.save(update_fields=["visibility"])
    return project


def summary(project):
    """What a card needs to say about a dataset, without opening it."""
    rows = (Annotation.objects.filter(project=project).order_by()
            .aggregate(frames=Count("id", filter=~Q(boxes=[])),
                       total=Count("id")))
    boxes = sum(len(a.boxes or []) for a in
                Annotation.objects.filter(project=project).only("boxes"))
    videos = project.videos.all()
    return {
        "frames": rows["frames"] or 0,
        "boxes": boxes,
        "clips": videos.count(),
        # Device counts are shown even when the metadata is withheld: "four
        # hotels" says how varied the data is without saying where any of them
        # is. That is the useful half, and the safe half.
        "devices": videos.exclude(device=None).values("device_id").distinct().count(),
        "hours": sorted({v.hour for v in videos if v.hour is not None}),
        "classes": list(project.classes or []),
    }


def copy_for(project, user, name=None):
    """Duplicate a public project into ``user``'s account.

    Frames are REFERENCED, not duplicated: an Annotation carries a path into the
    processed bucket, and nothing ever deletes those objects — not project
    deletion, which only cascades rows, and not video deletion, which cleans raw
    video and job CSVs but never frame images. So a copy cannot break, and a few
    thousand JPEGs are not duplicated per copy.

    The clips themselves are NOT attached. A copy is a dataset — frames and
    boxes — and attaching the source videos would hand over footage the copier
    was never shared, which is the property the whole sharing model protects.
    """
    copy = AnnotationProject.objects.create(
        user=user,
        name=name or f"{project.name} (copy)",
        description=project.description,
        classes=list(project.classes or []),
        copied_from=project,
        copied_from_name=project.name,
    )

    Annotation.objects.bulk_create([
        Annotation(
            project=copy,
            video=a.video,
            frame_number=a.frame_number,
            boxes=a.boxes,
            frame_image_path=a.frame_image_path,
            image_width=a.image_width,
            image_height=a.image_height,
            # A copy has not been reviewed BY THE COPIER. Carrying the flag over
            # would claim someone signed off work they have never seen.
            reviewed=False,
            sampled_only=a.sampled_only,
        )
        for a in Annotation.objects.filter(project=project).select_related("video")
    ], batch_size=500)
    return copy
