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

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return self.name

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
        return role is not None and _ROLE_RANK[role] >= _ROLE_RANK[level]

    @staticmethod
    def accessible(user):
        """Projects the user may READ — owned, or shared at any role.

        Read paths only. Every write path must name the level it needs.
        """
        return AnnotationProject.objects.filter(
            models.Q(user=user) | models.Q(shares__user=user)).distinct()

    @staticmethod
    def annotatable(user):
        """Projects the user may draw boxes in (annotator and above)."""
        return AnnotationProject.objects.filter(
            models.Q(user=user)
            | models.Q(shares__user=user,
                       shares__role__in=("annotator", "reviewer", "manager"))
        ).distinct()

    @staticmethod
    def reviewable(user):
        """Projects the user may sign off ANYONE's work in (reviewer and above).

        Scope is what separates this from ``annotatable``: an annotator is
        confined to their own assignments, a reviewer is not.
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

    def assigned_to(self, user):
        """The clips this user is responsible for labelling."""
        return self.videos.filter(clip_assignments__project=self,
                                  clip_assignments__user=user)

    def may_annotate_video(self, user, video_id):
        """Whether this user may draw on this clip.

        An annotator works only what is assigned to them — by someone else or by
        themselves out of the pool. Reviewers and above are not confined that
        way, because checking other people's work is the job.
        """
        if self.allows(user, "reviewer"):
            return True
        if not self.allows(user, "annotator"):
            return False
        return self.assignments.filter(video_id=video_id, user=user).exists()

    def save(self, *args, **kwargs):
        if not self.classes:
            self.classes = ["bee", "wasp", "nest"]
        super().save(*args, **kwargs)


# Linear ranks, the property that makes DeviceShare easy to reason about: a
# check is "at least this level", never a set membership. Adding a role that is
# not comparable to the others is how permission systems become guesswork.
_ROLE_RANK = {"viewer": 1, "annotator": 2, "reviewer": 3, "manager": 4, "owner": 5}


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
        ANNOTATOR = "annotator", "Annotator — label their assigned clips"
        REVIEWER = "reviewer", "Reviewer — label, and sign off anyone's work"
        MANAGER = "manager", "Manager — add clips, run sampling, assign work"

    project = models.ForeignKey(
        "AnnotationProject", on_delete=models.CASCADE, related_name="shares",
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
        related_name="shared_projects",
    )
    role = models.CharField(max_length=16, choices=Role.choices, default=Role.ANNOTATOR)
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
    """Who is labelling one clip.

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

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = [("project", "video", "frame_number")]
        ordering = ["frame_number"]

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
    # {"sample_interval": int, "max_frames": int}
    params = models.JSONField(default=dict, blank=True)
    frames_written = models.IntegerField(default=0)
    error_message = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"SampleFrames {self.video_id} [{self.status}]"
