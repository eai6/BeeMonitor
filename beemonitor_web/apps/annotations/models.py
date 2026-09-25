from django.conf import settings
from django.db import models


class AnnotationProject(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="annotation_projects",
    )
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    classes = models.JSONField(
        default=list,
        help_text='List of class labels, e.g. ["bee", "wasp", "nest"]',
    )
    videos = models.ManyToManyField(
        "videos.Video",
        blank=True,
        related_name="annotation_projects",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    # ── Publishing ───────────────────────────────────────────────────────
    class Visibility(models.TextChoices):
        PRIVATE = "private", "Private — only people you share it with"
        PUBLIC = "public", "Public — anyone can view and take a copy"

    visibility = models.CharField(
        max_length=10, choices=Visibility.choices, default=Visibility.PRIVATE,
    )
    published_at = models.DateTimeField(null=True, blank=True)
    # Field metadata is off by default. Frames and boxes are the dataset;
    # device, site and recording times are a field site's movement log, and
    # publishing them should be a decision made once, knowingly.
    publish_metadata = models.BooleanField(default=False)
    # Where this project came from, when it is a copy of a public one. Kept as
    # SET_NULL so the attribution survives the original being deleted — a copy
    # that outlives its source should still say where it came from.
    copied_from = models.ForeignKey(
        "self", on_delete=models.SET_NULL, null=True, blank=True,
        related_name="copies",
    )
    copied_from_name = models.CharField(max_length=200, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return self.name

    @property
    def is_public(self) -> bool:
        return self.visibility == self.Visibility.PUBLIC

    @staticmethod
    def public():
        """Everything published, newest first."""
        return (AnnotationProject.objects
                .filter(visibility=AnnotationProject.Visibility.PUBLIC)
                .select_related("user").order_by("-published_at", "-created_at"))

    @staticmethod
    def readable(user):
        """What this user may open: theirs, shared with them, or published.

        The read scope for pages that serve public work as well as private —
        ``accessible`` stays the private-only one so no existing caller widens
        by accident.
        """
        return AnnotationProject.objects.filter(
            models.Q(user=user) | models.Q(shares__user=user)
            | models.Q(visibility=AnnotationProject.Visibility.PUBLIC)
        ).distinct()

    # ── Access ───────────────────────────────────────────────────────────
    # Four scopes, each named after what it PERMITS, so a view picks the one
    # matching what it does rather than a role name it has to reason about.

    def role_for(self, user):
        """This user's role on the project, or None. Owner outranks any share."""
        if user is None or not getattr(user, "is_authenticated", False):
            return None
        if self.user_id == user.id:
            return "owner"
        share = self.shares.filter(user=user).first()
        return share.role if share else None

    def allows(self, user, level):
        """True when the user's role is at least ``level``."""
        role = self.role_for(user)
        return role is not None and _ROLE_RANK.get(role, 0) >= _ROLE_RANK[level]

    @staticmethod
    def accessible(user):
        """Projects the user may READ — owned, or shared at any role.

        Read paths only. Every write path must name the level it needs.
        """
        return AnnotationProject.objects.filter(
            models.Q(user=user) | models.Q(shares__user=user)).distinct()

    @staticmethod
    def reviewable(user):
        """Projects the user may check and fix boxes in (reviewer and above).

        Which frames, within the project, is ``may_edit_frame``'s question.
        """
        return AnnotationProject.objects.filter(
            models.Q(user=user)
            | models.Q(shares__user=user, shares__role__in=("reviewer", "manager"))
        ).distinct()

    @staticmethod
    def manageable(user):
        """Projects the user may spend GPU on and restructure (manager and above).

        Adding clips, sampling, auto-labelling, editing classes, assigning work.
        A labeller must not be able to spend someone else's GPU budget or change
        the class list halfway through a project — both quietly invalidate work
        already done.
        """
        return AnnotationProject.objects.filter(
            models.Q(user=user) | models.Q(shares__user=user, shares__role="manager")
        ).distinct()

    @staticmethod
    def owned(user):
        """Projects the user may delete, share or publish. Owner only."""
        return AnnotationProject.objects.filter(user=user)

    def may_edit_frame(self, user, frame):
        """Whether this user may save this frame (an ``Annotation``, or None
        for a frame nobody has sampled yet).

        A reviewer fixes the frames assigned to them, and anything nobody
        holds. A frame assigned to someone else is theirs: two people fixing
        the same boxes keep re-deciding the same judgement calls. Managers are
        not confined — they hand the work out and settle disputes.
        """
        if self.allows(user, "manager"):
            return True
        if not self.allows(user, "reviewer"):
            return False
        return frame is None or frame.assigned_to_id in (None, user.id)

    def save(self, *args, **kwargs):
        if not self.classes:
            self.classes = ["bee", "wasp", "nest"]
        super().save(*args, **kwargs)


# Linear ranks, the property that makes DeviceShare easy to reason about: a
# check is "at least this level", never a set membership. Adding a role that is
# not comparable to the others is how permission systems become guesswork.
# There is no annotator: SAM 3 labels, people review (migration 0013).
_ROLE_RANK = {"viewer": 1, "reviewer": 3, "manager": 4, "owner": 5}


class ProjectShare(models.Model):
    """Grants another account access to an annotation project.

    Shaped after ``devices.DeviceShare``, which solved the same problem for
    hardware. The owner keeps full control and is the only one who can manage
    shares or publish.

    What a share deliberately does NOT grant is access to the source footage.
    Annotating reads the sampled frames out of the processed bucket; the editor
    never touches ``raw-videos``. So inviting someone to label does not hand
    them a field site's recording times or locations, and that has to stay true
    — the obvious implementation, handing over ``Video.accessible``, throws it
    away.
    """

    class Role(models.TextChoices):
        VIEWER = "viewer", "Viewer — see frames and labels"
        REVIEWER = "reviewer", "Reviewer — check and fix the AI's boxes"
        MANAGER = "manager", "Manager — add clips, run sampling, assign frames"

    project = models.ForeignKey(
        "AnnotationProject", on_delete=models.CASCADE, related_name="shares",
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
        related_name="shared_projects",
    )
    role = models.CharField(max_length=16, choices=Role.choices, default=Role.REVIEWER)
    created_at = models.DateTimeField(auto_now_add=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True,
        related_name="+",
    )

    class Meta:
        unique_together = ("project", "user")
        ordering = ["user__username"]

    def __str__(self) -> str:
        return f"{self.project.name} -> {self.user} ({self.role})"


class ClipAssignment(models.Model):
    """Who was labelling one clip. RETIRED — kept read-only for one release.

    Work is now handed out per frame (``Annotation.assigned_to``): sampling
    and SAM 3 label the frames, and people review them, so "give eai7 500
    frames" is the unit a manager thinks in (memory/39). Migration 0012 copied
    each row onto its clip's unreviewed frames.

    Per clip rather than per frame: a clip is a coherent scene — one hotel, one
    stretch of time — so two people labelling the same one keep re-deciding the
    same judgement calls, and "who labelled this" stops having an answer.

    Unique on (project, video): a clip has at most one owner of the work. A
    clip with no row is in the unassigned pool, which anyone with a role may
    claim — claiming writes a row like any other, so no work happens off the
    books.
    """

    project = models.ForeignKey(
        "AnnotationProject", on_delete=models.CASCADE, related_name="assignments",
    )
    video = models.ForeignKey(
        "videos.Video", on_delete=models.CASCADE, related_name="clip_assignments",
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
        related_name="clip_assignments",
    )
    assigned_at = models.DateTimeField(auto_now_add=True)
    # Null when someone claimed it for themselves out of the pool.
    assigned_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True,
        related_name="+",
    )

    class Meta:
        unique_together = ("project", "video")
        ordering = ["video_id"]

    def __str__(self) -> str:
        return f"{self.video_id} -> {self.user}"

    @property
    def self_claimed(self) -> bool:
        return self.assigned_by_id is None or self.assigned_by_id == self.user_id


class Annotation(models.Model):
    project = models.ForeignKey(
        AnnotationProject,
        on_delete=models.CASCADE,
        related_name="annotations",
    )
    video = models.ForeignKey(
        "videos.Video",
        on_delete=models.CASCADE,
        related_name="annotations",
    )
    frame_number = models.IntegerField()
    image_width = models.IntegerField(default=1280)
    image_height = models.IntegerField(default=720)
    boxes = models.JSONField(
        default=list,
        help_text='List of {"x": float, "y": float, "w": float, "h": float, "class": str, "class_id": int}',
    )
    frame_image_path = models.CharField(
        max_length=500, blank=True, default="",
        help_text="S3 key (in processed bucket) of extracted frame JPEG",
    )

    class ReviewSource(models.TextChoices):
        NONE = "", "Not reviewed"
        HUMAN = "human", "Human"
        LLM = "llm", "LLM"

    reviewed = models.BooleanField(
        default=False, help_text="A human or the LLM has vetted these boxes.")
    sampled_only = models.BooleanField(
        default=False,
        help_text="This row is a sampled frame nobody has annotated yet — it "
                  "exists so the editor can navigate to the frame. Distinct from "
                  "a frame a human deliberately marked empty, which IS a valid "
                  "negative example; training excludes these but keeps those.",
    )
    review_source = models.CharField(
        max_length=10, choices=ReviewSource.choices, default="", blank=True)
    reviewed_at = models.DateTimeField(null=True, blank=True)
    reviewed_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True,
        related_name="+")

    # Who is reviewing this frame. Null = in the unassigned pool.
    assigned_to = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True,
        related_name="review_frames")
    assigned_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True,
        related_name="+")
    assigned_at = models.DateTimeField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = [("project", "video", "frame_number")]
        ordering = ["frame_number"]
        indexes = [
            # The review grid and the assign pool: to review, whose.
            # (Keyset order, project/video/frame, rides the unique index.)
            models.Index(fields=["project", "reviewed", "assigned_to"],
                         name="ann_review_queue"),
        ]

    def __str__(self):
        return f"Frame {self.frame_number} of {self.video.title}"

    def to_yolo_format(self) -> str:
        """Convert boxes to YOLO txt format: class_id cx cy w h (normalized)."""
        lines = []
        for box in self.boxes:
            cx = (box["x"] + box["w"] / 2) / self.image_width
            cy = (box["y"] + box["h"] / 2) / self.image_height
            w = box["w"] / self.image_width
            h = box["h"] / self.image_height
            lines.append(f'{box["class_id"]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}')
        return "\n".join(lines)


class PreAnnotationTask(models.Model):
    """A durable record of one pre-annotation run (one video).

    Replaces the old fire-and-forget daemon thread that polled S3 for 15 min:
    the SageMaker invocation is fired once, its output S3 URI is stored here,
    and the background reconciler finalizes the task (writes Annotation rows)
    when the result lands — so a deploy mid-run no longer loses the result and
    the UI can show progress / failure.
    """

    class Status(models.TextChoices):
        QUEUED = "queued", "Queued"          # created, not yet invoked
        PROCESSING = "processing", "Processing"  # invoked, awaiting result
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"
        CANCELLED = "cancelled", "Cancelled"  # user cancelled; any result is discarded

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
                             related_name="preannotation_tasks")
    project = models.ForeignKey(AnnotationProject, on_delete=models.CASCADE,
                                related_name="preannotation_tasks")
    video = models.ForeignKey("videos.Video", on_delete=models.CASCADE,
                              related_name="preannotation_tasks")
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.QUEUED)
    labeler = models.CharField(max_length=20, default="yolo")   # yolo | sam3 (custom rides yolo)
    custom_model_key = models.CharField(max_length=500, blank=True, default="")
    # Sampling / SAM3 knobs needed to build the payload at spawn time.
    params = models.JSONField(default=dict, blank=True)
    # SageMaker async result location (poll target) + failure location.
    output_uri = models.CharField(max_length=700, blank=True, default="")
    failure_uri = models.CharField(max_length=700, blank=True, default="")
    frames_written = models.IntegerField(default=0)
    error_message = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"PreAnnotate {self.video_id} [{self.status}]"


class FrameSamplingTask(models.Model):
    """Extract frames from one video so they can be annotated.

    Sampling used to be fused into the SAM 3 pre-annotation call: the only way to
    get frames was to pay for a g5 auto-labelling pass over the whole video. This
    task decouples the two — it decodes frames on the web container's CPU and
    writes an empty ``Annotation`` row per frame, so the editor can navigate them
    immediately and SAM 3 becomes an opt-in, per-frame action.

    Deliberately shaped like ``PreAnnotationTask`` (durable row, advanced by the
    background reconciler) so the two read the same way on the project page.
    """

    class Status(models.TextChoices):
        QUEUED = "queued", "Queued"
        PROCESSING = "processing", "Processing"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"
        CANCELLED = "cancelled", "Cancelled"

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
                             related_name="frame_sampling_tasks")
    project = models.ForeignKey(AnnotationProject, on_delete=models.CASCADE,
                                related_name="frame_sampling_tasks")
    video = models.ForeignKey("videos.Video", on_delete=models.CASCADE,
                              related_name="frame_sampling_tasks")
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.QUEUED)
    # {"method": "motion"|"interval", "sample_interval": int, "max_frames": int,
    #  motion only: "min_gap_s", "roi": "device"|"frame", "replace": bool}
    params = models.JSONField(default=dict, blank=True)
    frames_written = models.IntegerField(default=0)
    # Motion sampling's per-clip strip: {"profile": [0-100 per bucket],
    # "picked": [bucket indices], "frames": n}. Null for interval sampling.
    motion = models.JSONField(null=True, blank=True)
    # GPU sampling (SAMPLING_BACKEND=sagemaker): the batch this clip went out
    # in, and how many times it has been sent. See sampling_remote.py.
    batch = models.ForeignKey("SamplingBatch", null=True, blank=True,
                              on_delete=models.SET_NULL, related_name="tasks")
    attempts = models.PositiveSmallIntegerField(default=0)
    error_message = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [models.Index(fields=["status", "created_at"])]

    def __str__(self):
        return f"SampleFrames {self.video_id} [{self.status}]"


class SamplingBatch(models.Model):
    """A batch of clips sent to the GPU endpoint in one invocation (task
    ``sample_label``: sample and pre-label in one pass, memory/38).

    The dispatcher claims queued FrameSamplingTasks into a batch before
    invoking, and the collector claims the batch (INVOKED → COLLECTING) before
    writing results, so several web processes never send or record the same
    work twice. ``result_key`` is a fixed S3 location the GPU writes, so a
    result is never lost even if the async output location was.
    """

    class Status(models.TextChoices):
        CLAIMED = "claimed", "Claimed"          # tasks reserved, not yet invoked
        INVOKED = "invoked", "Invoked"          # on the GPU queue
        COLLECTING = "collecting", "Collecting" # one process is writing results
        COLLECTED = "collected", "Collected"
        FAILED = "failed", "Failed"

    batch_id = models.CharField(max_length=64, unique=True)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.CLAIMED)
    params = models.JSONField(default=dict, blank=True)
    result_key = models.CharField(max_length=300, blank=True, default="")
    output_uri = models.CharField(max_length=700, blank=True, default="")
    failure_uri = models.CharField(max_length=700, blank=True, default="")
    error = models.TextField(blank=True, default="")
    claimed_at = models.DateTimeField(auto_now_add=True)
    invoked_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["claimed_at"]
        indexes = [models.Index(fields=["status", "claimed_at"])]

    def __str__(self) -> str:
        return f"sampling batch {self.batch_id} ({self.status})"
