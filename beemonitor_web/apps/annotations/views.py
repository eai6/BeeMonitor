import io
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor

from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404
from django.db import models
from django.urls import reverse, reverse_lazy
from django.views import View
from django.views.generic import CreateView, DetailView, ListView, TemplateView, UpdateView

import logging

from .forms import ProjectCreateForm
from .models import Annotation, AnnotationProject

logger = logging.getLogger(__name__)


def _preannotate_opts(request):
    """(sample_interval, max_frames, confidence) from POST, clamped, with
    settings-based defaults. Lets power users tune sampling per request."""
    from django.conf import settings

    def _clamp(name, default, lo, hi, cast=int):
        try:
            v = cast(request.POST.get(name) or default)
        except (TypeError, ValueError):
            v = default
        return max(lo, min(hi, v))

    return (
        _clamp("sample_interval", settings.PREANNOTATE_SAMPLE_INTERVAL, 1, 120),
        _clamp("max_frames", settings.PREANNOTATE_MAX_FRAMES, 10, 2000),
        _clamp("confidence", settings.PREANNOTATE_CONFIDENCE, 0.01, 0.9, cast=float),
    )


def _labeler_opts(request):
    """(labeler, custom_model_key). Pre-annotation is SAM 3 only — text-prompt
    labeling of each class is what seeds new domains, and it's the single GPU
    path we keep for the annotation editor. (YOLO / custom-model pre-annotation
    was removed to avoid a second endpoint's GPU time.)"""
    return "sam3", ""


def _sam3_opts(request):
    """(nms_iou, max_detections) from POST, clamped. SAM 3-specific quality knobs:
    nms_iou dedupes overlapping boxes across the per-class prompt passes (0 = off);
    max_detections caps boxes per class per frame."""
    def _num(name, default, lo, hi, cast):
        try:
            v = cast(request.POST.get(name) or default)
        except (TypeError, ValueError):
            v = default
        return max(lo, min(hi, v))

    return (
        _num("nms_iou", 0.5, 0.0, 1.0, float),
        _num("max_detections", 100, 1, 500, int),
    )


class ProjectListView(LoginRequiredMixin, ListView):
    model = AnnotationProject
    template_name = "annotations/list.html"
    context_object_name = "projects"
    paginate_by = 20

    def get_queryset(self):
        # Shared projects belong in the list, or an invitation goes nowhere.
        return AnnotationProject.accessible(self.request.user).select_related("user")

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        # A shared project has to say whose it is and what you may do in it —
        # otherwise the list mixes yours with other people's and reads as if
        # you own all of them.
        me = self.request.user
        roles = {s.project_id: s.role for s in
                 me.shared_projects.all()} if me.is_authenticated else {}
        for project in ctx["projects"]:
            project.my_role = "owner" if project.user_id == me.id else roles.get(project.pk)
            project.shared_by = None if project.my_role == "owner" else project.user
        return ctx


class ProjectCreateView(LoginRequiredMixin, CreateView):
    model = AnnotationProject
    form_class = ProjectCreateForm
    template_name = "annotations/create.html"

    def form_valid(self, form):
        form.instance.user = self.request.user
        return super().form_valid(form)

    def get_success_url(self):
        return reverse_lazy("annotations:detail", kwargs={"pk": self.object.pk})


class ProjectUpdateView(LoginRequiredMixin, UpdateView):
    """Edit a project's name, description, and classes. Classes are edited as
    structured rows (add / rename / remove); a rename propagates to every existing
    box (class_id preserved), a remove drops that class's boxes."""
    model = AnnotationProject
    form_class = ProjectCreateForm
    template_name = "annotations/settings.html"

    def get_queryset(self):
        # Editing the name, description and class list restructures the
        # project — a labeller changing classes mid-run invalidates finished work.
        return AnnotationProject.manageable(self.request.user)

    def get_initial(self):
        initial = super().get_initial()
        initial["classes_text"] = ", ".join(self.object.classes or [])
        return initial

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        # Per-class box counts so the UI can warn before removing a used class.
        counts = {}
        for boxes in self.object.annotations.values_list("boxes", flat=True):
            for b in (boxes or []):
                nm = b.get("class", "unknown")
                counts[nm] = counts.get(nm, 0) + 1
        ctx["class_rows"] = [
            {"index": i, "name": c, "count": counts.get(c, 0)}
            for i, c in enumerate(self.object.classes or [])
        ]
        return ctx

    def post(self, request, *args, **kwargs):
        from django.shortcuts import redirect
        from django.contrib import messages

        project = self.get_object()
        name = (request.POST.get("name") or "").strip()
        description = (request.POST.get("description") or "").strip()
        names = request.POST.getlist("class_name")
        origs = request.POST.getlist("class_orig")
        old_classes = project.classes or []

        # Rebuild the class list from the rows (order = display order) and a map
        # old_index -> new_index. A row's class_orig is its original index, or
        # "new". Blank/duplicate names are dropped.
        new_classes, remap, seen = [], {}, set()
        for nm, orig in zip(names, origs):
            nm = (nm or "").strip()
            if not nm or nm.lower() in seen:
                continue
            seen.add(nm.lower())
            new_idx = len(new_classes)
            new_classes.append(nm)
            if orig != "new":
                try:
                    remap[int(orig)] = new_idx
                except (TypeError, ValueError):
                    pass

        if not name:
            messages.error(request, "Project name is required.")
            return redirect("annotations:settings", pk=project.pk)
        if not new_classes:
            messages.error(request, "At least one class is required.")
            return redirect("annotations:settings", pk=project.pk)

        def _old_index(box):
            cid = box.get("class_id")
            if isinstance(cid, int) and 0 <= cid < len(old_classes):
                return cid
            nm = box.get("class")
            return old_classes.index(nm) if nm in old_classes else None

        migrated_frames = dropped_boxes = 0
        for ann in project.annotations.all():
            new_boxes, changed = [], False
            for box in (ann.boxes or []):
                oi = _old_index(box)
                if oi is None:
                    new_boxes.append(box)  # unresolvable — leave untouched
                    continue
                if oi in remap:
                    ni = remap[oi]
                    nb = {**box, "class_id": ni, "class": new_classes[ni]}
                    if nb != box:
                        changed = True
                    new_boxes.append(nb)
                else:  # class removed → drop its boxes
                    changed = True
                    dropped_boxes += 1
            if changed:
                ann.boxes = new_boxes
                ann.save(update_fields=["boxes"])
                migrated_frames += 1

        project.name = name
        project.description = description
        project.classes = new_classes
        project.save(update_fields=["name", "description", "classes"])

        msg = "Project settings saved."
        if migrated_frames:
            msg += f" Relabeled boxes on {migrated_frames} frame(s)."
        if dropped_boxes:
            msg += f" Dropped {dropped_boxes} box(es) from removed class(es)."
        messages.success(request, msg)
        return redirect("annotations:detail", pk=project.pk)

    def get_success_url(self):
        return reverse_lazy("annotations:detail", kwargs={"pk": self.object.pk})


class ProjectDeleteView(LoginRequiredMixin, View):
    """Delete a project (cascades its annotations + pre-annotation tasks)."""

    def post(self, request, pk):
        from django.shortcuts import redirect
        from django.contrib import messages

        # Deleting is the owner's alone.
        project = get_object_or_404(
            AnnotationProject.owned(request.user), pk=pk)
        name = project.name
        project.delete()
        messages.info(request, f"Deleted project '{name}' and its annotations.")
        return redirect("annotations:list")


class ProjectDetailView(LoginRequiredMixin, DetailView):
    """The project's clips, and — at ``/review/`` — its annotated frames.

    Both halves are rendered from one context builder because they always were:
    the frame grid lived at the bottom of the clip workspace, under the stage
    tiles, the failures, the filters, the assignment controls, the selection
    actions and a table of every clip. Choosing what to annotate and checking
    what came back are different sittings with different filters, so they are
    now different pages — but splitting the view as well would have meant
    maintaining two context builders, and the one that already existed for this
    (ReviewView) had drifted out of the URL conf without anyone noticing.
    """

    model = AnnotationProject
    template_name = "annotations/detail.html"
    context_object_name = "project"
    #: Set by the ``review`` URL. Same data, the other half of the template.
    review = False

    def get_template_names(self):
        return ["annotations/review.html"] if self.review else [self.template_name]

    def get_queryset(self):
        # Reading the project. Every write path below names its own level.
        return AnnotationProject.accessible(self.request.user)

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        project = self.object

        # Advance any in-flight pre-annotation tasks on page load (the
        # background reconciler also does this, so it works with no open tab).
        poll_preannotation_tasks(self.request.user)
        from .models import PreAnnotationTask
        active = list(
            PreAnnotationTask.objects.filter(
                project=project,
                status__in=[PreAnnotationTask.Status.QUEUED, PreAnnotationTask.Status.PROCESSING],
            ).select_related("video")
        )
        recent_failed = list(
            PreAnnotationTask.objects.filter(
                project=project, status=PreAnnotationTask.Status.FAILED,
            ).select_related("video")[:5]
        )
        ctx["preannot_active"] = active
        ctx["preannot_failed"] = recent_failed
        # One shared cause is worth stating once, above the list, instead of
        # repeating it on every line and leaving the reader to notice.
        ctx["preannot_all_timed_out"] = bool(recent_failed) and all(
            "timed out" in (t.error_message or "").lower() for t in recent_failed)

        from django.db.models import Count, Q as _Q

        from apps.analysis.views import _sanitize_site, _unsanitize_site
        from apps.devices.models import Device
        from apps.videos.models import Video

        videos = project.videos.all()
        ctx["project_videos"] = videos
        ctx["video_count"] = videos.count()

        # Per-video annotated-frame counts in ONE query. This used to be a COUNT
        # per video inside a loop — 45 queries for a 45-video project, growing
        # linearly with the project.
        counted = list(
            videos.annotate(
                annotation_count=Count("annotations",
                                       filter=_Q(annotations__project=project),
                                       distinct=True))
            .select_related("device")
        )

        # Filter the project's OWN videos. A flat list of every video stops being
        # usable past a few dozen. Same dimensions as the Processing hub, but
        # prefixed v_ so they can't collide with the add-videos modal's av_.
        vf = {k: self.request.GET.get("v_" + k, "").strip()
              for k in ("q", "device", "site", "year", "month", "day",
                        "hfrom", "hto", "state")}

        def _keep(v):
            if vf["q"] and vf["q"].lower() not in (v.title or "").lower():
                return False
            if vf["device"] and str(v.device_id or "") != vf["device"]:
                return False
            if vf["site"] and v.site_name != _unsanitize_site(vf["site"]):
                return False
            for field in ("year", "month", "day"):
                if vf[field] and str(getattr(v, field, "") or "") != vf[field]:
                    return False
            # Hour-of-day window: inclusive start, exclusive end. start > end
            # wraps past midnight (22 -> 4), matching the Processing hub.
            if vf["hfrom"] or vf["hto"]:
                hour = getattr(v, "hour", None)
                if hour is None:
                    return False
                lo = int(vf["hfrom"]) if vf["hfrom"] else 0
                hi = int(vf["hto"]) if vf["hto"] else 24
                inside = (lo <= hour < hi) if lo < hi else (hour >= lo or hour < hi)
                if lo != hi and not inside:
                    return False
            if vf["state"] == "annotated" and v.annotation_count == 0:
                return False
            if vf["state"] == "unannotated" and v.annotation_count > 0:
                return False
            return True

        try:
            filtered = [v for v in counted if _keep(v)]
        except (TypeError, ValueError):
            filtered = counted  # a malformed filter shows everything, not nothing
        filtered.sort(key=lambda v: (v.recorded_at is None, v.recorded_at, v.pk),
                      reverse=True)

        # Where each clip has actually got to. One aggregate, not a count per
        # clip — and the stage is what the page filters and acts on.
        from . import progress as progress_mod

        failed_ids = {t.video_id for t in ctx.get("preannot_failed") or []}
        states = progress_mod.per_video(self.object, failed_ids)
        stage = (self.request.GET.get("stage") or "").strip()
        if stage in progress_mod.STAGE_LABELS:
            if stage == "new":
                # These clips have no annotation rows to aggregate, so they are
                # the ones the aggregate never saw.
                filtered = [v for v in filtered
                            if not states.get(v.pk, {}).get("frames")
                            and v.pk not in failed_ids]
            else:
                wanted = progress_mod.filter_ids(states, len(counted), stage)
                filtered = [v for v in filtered if v.pk in wanted]

        # Who is doing which clip. One query for the whole project; the rows
        # carry it so the list can filter, colour and reassign without more.
        from . import assignments as assign_mod

        holders = assign_mod.by_video(self.object)
        assignee = (self.request.GET.get("assignee") or "").strip()
        if assignee == "none":
            filtered = [v for v in filtered if v.pk not in holders]
        elif assignee == "me":
            filtered = [v for v in filtered
                        if holders.get(v.pk)
                        and holders[v.pk].user_id == self.request.user.id]
        elif assignee.isdigit():
            filtered = [v for v in filtered
                        if holders.get(v.pk)
                        and holders[v.pk].user_id == int(assignee)]

        VIDEO_LIST_CAP = 500
        shown = progress_mod.decorate(filtered[:VIDEO_LIST_CAP], states, failed_ids)
        # Latest motion strip per clip (from its most recent motion sampling).
        from .models import FrameSamplingTask
        motion_by_video = {}
        for vid, motion in (FrameSamplingTask.objects
                            .filter(project=self.object, motion__isnull=False)
                            .order_by("video_id", "-created_at")
                            .values_list("video_id", "motion")):
            motion_by_video.setdefault(vid, motion)
        ctx["video_data"] = [{"video": v, "annotation_count": v.annotation_count,
                              "progress": v.progress,
                              "holder": holders.get(v.pk),
                              "holder_dot": person_colour(
                                  holders[v.pk].user_id) if v.pk in holders else "",
                              "mine": (v.pk in holders
                                       and holders[v.pk].user_id == self.request.user.id),
                              "motion": _motion_strip(motion_by_video.get(v.pk))}
                             for v in shown]

        # Everyone on the project, for the "assigned to" filter and the assign
        # control. Counts come from the same map, so the strip and the rows can
        # never disagree.
        from django.contrib.auth import get_user_model

        User = get_user_model()
        member_ids = [self.object.user_id] + list(
            self.object.shares.values_list("user_id", flat=True))
        held = {}
        for a in holders.values():
            held[a.user_id] = held.get(a.user_id, 0) + 1
        members = {u.pk: u for u in User.objects.filter(pk__in=member_ids)}
        ctx["members"] = [{
            "user": members[uid], "dot": person_colour(uid),
            "count": held.get(uid, 0),
            "is_you": uid == self.request.user.id,
        } for uid in member_ids if uid in members]
        ctx["assignee"] = assignee
        ctx["unassigned_count"] = sum(
            1 for v in counted if v.pk not in holders)
        ctx["my_role"] = self.object.role_for(self.request.user)
        ctx["can_assign"] = self.object.allows(self.request.user, "manager")
        ctx["can_annotate"] = self.object.allows(self.request.user, "annotator")
        ctx["my_clips"] = sum(1 for a in holders.values()
                              if a.user_id == self.request.user.id)
        ctx["stage"] = stage
        ctx["progress"] = progress_mod.summary(
            self.object, states, len(counted), failed_ids)
        ctx["video_filter"] = vf
        ctx["video_filter_on"] = any(vf.values())
        ctx["video_filtered_count"] = len(filtered)
        ctx["video_list_capped"] = len(filtered) > len(shown)
        ctx["video_annotated_count"] = sum(1 for v in counted if v.annotation_count > 0)
        ctx["hours"] = list(range(24))
        # Options come from the project's own videos, so a filter can never offer
        # a value that matches nothing here.
        ctx["video_filter_opts"] = {
            "devices": sorted({(v.device_id, v.device.name) for v in counted if v.device_id},
                              key=lambda d: d[1]),
            "sites": sorted({_sanitize_site(v.site_name) for v in counted if v.site_name}),
            "years": sorted({v.year for v in counted if v.year}),
            "months": sorted({v.month for v in counted if v.month}),
            "days": sorted({v.day for v in counted if v.day}),
        }

        # The clip picker moved to AddVideosWorkspaceView, which uses the
        # shared workspace filter. What stood here was a second, hand-rolled
        # copy — title/device/site/year/month/day/confirmed — that had already
        # drifted from apply_video_filters: no multi-hotel selection, no
        # time-of-day window, no date range, no "not yet analysed".

        # Build combined frame grid with filters
        filter_video = self.request.GET.get("video", "")
        filter_class = self.request.GET.get("class", "")
        filter_review = self.request.GET.get("review", "")  # ""|reviewed|unreviewed|human|llm
        ctx["filter_video"] = filter_video
        ctx["filter_class"] = filter_class
        ctx["filter_review"] = filter_review

        anns_qs = project.annotations.select_related("video").order_by("video__title", "frame_number")
        if filter_video:
            try:
                anns_qs = anns_qs.filter(video_id=int(filter_video))
            except (ValueError, TypeError):
                pass
        if filter_review == "reviewed":
            anns_qs = anns_qs.filter(reviewed=True)
        elif filter_review == "unreviewed":
            anns_qs = anns_qs.filter(reviewed=False)
        elif filter_review in ("human", "llm"):
            anns_qs = anns_qs.filter(review_source=filter_review)

        # Review progress across the whole project (not just the filtered page).
        proj_anns = project.annotations
        ctx["reviewed_count"] = proj_anns.filter(reviewed=True).count()
        ctx["reviewed_human"] = proj_anns.filter(review_source="human").count()
        ctx["reviewed_llm"] = proj_anns.filter(review_source="llm").count()

        # Project-wide totals for the top stat tiles — these must NOT change with
        # the video/class/review filters (a filter making them read 0 looked like
        # data loss). Class breakdown here also shows which labels actually exist
        # yet (e.g. is "nest tube" populated, or still pre-annotating?).
        proj_total_frames = 0
        proj_total_boxes = 0
        proj_class_counts = {}
        for boxes in proj_anns.values_list("boxes", flat=True):
            proj_total_frames += 1
            for b in (boxes or []):
                cls = b.get("class", "unknown")
                proj_class_counts[cls] = proj_class_counts.get(cls, 0) + 1
                proj_total_boxes += 1
        ctx["proj_total_frames"] = proj_total_frames
        ctx["proj_total_boxes"] = proj_total_boxes
        ctx["proj_class_counts"] = dict(sorted(proj_class_counts.items()))

        # Stats run over ALL matching annotations; the thumbnail grid is
        # paginated (presigning thousands of URLs per page-load is too slow).
        PAGE_SIZE = 500
        try:
            page = max(1, int(self.request.GET.get("page", 1)))
        except (TypeError, ValueError):
            page = 1
        win_start = (page - 1) * PAGE_SIZE
        win_end = win_start + PAGE_SIZE

        total_boxes = 0
        total_matching = 0  # index into the (class-)filtered set
        class_counts = {}
        frame_cards = []

        for ann in anns_qs:
            boxes = ann.boxes or []
            box_classes = sorted(set(b.get("class", "unknown") for b in boxes)) if boxes else []

            # Class filter
            if filter_class and filter_class not in box_classes:
                continue

            for b in boxes:
                cls = b.get("class", "unknown")
                class_counts[cls] = class_counts.get(cls, 0) + 1
            total_boxes += len(boxes)

            # Only build cards for the current page window.
            if win_start <= total_matching < win_end:
                frame_cards.append({
                    "video_pk": ann.video_id,
                    "video_title": ann.video.title,
                    "frame_number": ann.frame_number,
                    "box_count": len(boxes),
                    "classes": box_classes,
                    "frame_image_path": ann.frame_image_path or "",
                    "reviewed": ann.reviewed,
                    "review_source": ann.review_source,
                })
            total_matching += 1

        # Presigned URLs for frame thumbnails
        try:
            from config.storage import get_s3_client
            s3 = get_s3_client()
            for card in frame_cards:
                if card["frame_image_path"]:
                    card["thumbnail_url"] = s3.generate_presigned_url(
                        "processed", card["frame_image_path"], expiry_hours=2,
                    )
        except Exception as e:
            logger.warning("Failed to presign thumbnails: %s", e)

        ctx["total_annotations"] = total_matching
        ctx["total_boxes"] = total_boxes
        ctx["class_counts"] = class_counts
        ctx["frame_cards"] = frame_cards
        # Pagination for the grid.
        import math
        total_pages = max(1, math.ceil(total_matching / PAGE_SIZE)) if total_matching else 1
        page = min(page, total_pages)
        ctx["page"] = page
        ctx["total_pages"] = total_pages
        ctx["page_size"] = PAGE_SIZE
        ctx["page_start"] = win_start + 1 if frame_cards else 0
        ctx["page_end"] = win_start + len(frame_cards)
        ctx["has_prev_page"] = page > 1
        ctx["has_next_page"] = page < total_pages
        # Query string carrying the current filters (minus page) for page links.
        from urllib.parse import urlencode
        _pg = {k: v for k, v in (("video", filter_video), ("class", filter_class),
                                 ("review", filter_review)) if v}
        ctx["page_qs_prefix"] = ("?" + urlencode(_pg) + "&") if _pg else "?"

        # Defaults for the AI pre-annotate sampling controls.
        from django.conf import settings
        ctx["preannotate_defaults"] = {
            "sample_interval": settings.PREANNOTATE_SAMPLE_INTERVAL,
            "max_frames": settings.PREANNOTATE_MAX_FRAMES,
            "confidence": settings.PREANNOTATE_CONFIDENCE,
        }
        # Fine-tuned models selectable as the pre-annotation labeler.
        from apps.training.models import CustomModel
        ctx["custom_bee_models"] = CustomModel.objects.filter(
            user=self.request.user, is_active=True, status=CustomModel.Status.READY,
        ).exclude(storage_key="")

        return ctx


class RemoveVideoView(LoginRequiredMixin, View):
    """Remove one video (and its annotations) from an annotation project."""

    def post(self, request, pk):
        from django.shortcuts import redirect
        from django.contrib import messages

        # Removing a clip discards anyone's work on it.
        project = get_object_or_404(
            AnnotationProject.manageable(request.user), pk=pk)
        video_id = request.POST.get("video_id")
        if video_id:
            project.videos.remove(video_id)
            Annotation.objects.filter(project=project, video_id=video_id).delete()
            messages.info(request, "Video removed from the project.")
        return redirect("annotations:detail", pk=pk)


# Stable per-person colours, so the same face is the same colour on the people
# page, the clip list and every assignment chip.
PERSON_DOTS = ["#16a34a", "#b45309", "#0e7490", "#7c3aed", "#be123c", "#4d7c0f",
               "#0369a1", "#a16207"]


def person_colour(user_id):
    return PERSON_DOTS[(user_id or 0) % len(PERSON_DOTS)]


class PublicBrowseView(LoginRequiredMixin, TemplateView):
    """Datasets and models other people have published.

    Datasets are COPIED, models are USED — a dataset you build on has to be
    yours to change, a model is an artefact you point a pipeline at.
    """

    template_name = "annotations/browse.html"

    def get_context_data(self, **kwargs):
        from apps.training.models import CustomModel

        from . import publishing

        ctx = super().get_context_data(**kwargs)
        tab = self.request.GET.get("tab") or "datasets"

        datasets = []
        for project in AnnotationProject.public()[:60]:
            datasets.append({
                "project": project,
                "summary": publishing.summary(project),
                "mine": project.user_id == self.request.user.id,
            })

        models = (CustomModel.objects
                  .filter(visibility=CustomModel.Visibility.PUBLIC)
                  .select_related("user").order_by("-published_at", "-id")[:60])

        ctx.update({
            "tab": tab,
            "datasets": datasets,
            "models": models,
            "dataset_count": AnnotationProject.public().count(),
            "model_count": CustomModel.objects.filter(
                visibility=CustomModel.Visibility.PUBLIC).count(),
        })
        return ctx


class PublishProjectView(LoginRequiredMixin, View):
    """Publish or unpublish. Owner only — it is their data being offered."""

    def post(self, request, pk):
        from django.contrib import messages
        from django.shortcuts import redirect

        from . import publishing

        project = get_object_or_404(AnnotationProject.owned(request.user), pk=pk)
        if request.POST.get("visibility") == "public":
            publishing.publish(
                project, include_metadata=bool(request.POST.get("include_metadata")))
            messages.success(
                request,
                "Published. Anyone signed in can now view it and take a copy — "
                + ("recording times and site names are included."
                   if project.publish_metadata
                   else "recording times and site names are withheld."))
        else:
            publishing.unpublish(project)
            messages.info(
                request,
                "No longer listed. Copies people already took are unaffected.")
        return redirect("annotations:people", pk=pk)


class CopyProjectView(LoginRequiredMixin, View):
    """Take a copy of a published dataset into your own account."""

    def post(self, request, pk):
        from django.contrib import messages
        from django.shortcuts import redirect

        from . import publishing

        source = get_object_or_404(
            AnnotationProject, pk=pk,
            visibility=AnnotationProject.Visibility.PUBLIC)
        copy = publishing.copy_for(source, request.user,
                                   name=(request.POST.get("name") or "").strip() or None)
        messages.success(
            request,
            f"Copied “{source.name}” — {copy.annotations.count()} frame(s) are "
            "yours to change. The original is untouched.")
        return redirect("annotations:detail", pk=copy.pk)


class ProjectPeopleView(LoginRequiredMixin, TemplateView):
    """Who is on this project, what they may do, and how far along they are.

    Readable by everyone on the project — knowing who else is working on it is
    part of working on it — while every control is owner-only.
    """

    template_name = "annotations/people.html"

    def get_context_data(self, **kwargs):
        from . import assignments as assign_mod
        from .models import ProjectShare

        ctx = super().get_context_data(**kwargs)
        project = get_object_or_404(
            AnnotationProject.accessible(self.request.user), pk=kwargs["pk"])

        loads = {w["user_id"]: w for w in assign_mod.workloads(project)}
        people = [{
            "user": project.user, "role": "owner", "role_label": "Owner",
            "dot": person_colour(project.user_id),
            "is_you": project.user_id == self.request.user.id,
            "share": None, "load": loads.get(project.user_id),
        }]
        for share in project.shares.select_related("user").all():
            people.append({
                "user": share.user, "role": share.role,
                "dot": person_colour(share.user_id),
                "role_label": share.get_role_display(),
                "is_you": share.user_id == self.request.user.id,
                "share": share, "load": loads.get(share.user_id),
            })

        ctx.update({
            "project": project,
            "people": people,
            "roles": ProjectShare.Role.choices,
            "can_manage_people": project.user_id == self.request.user.id,
            "unassigned_count": assign_mod.unassigned(project).count(),
            "my_role": project.role_for(self.request.user),
        })
        return ctx


class ShareInviteView(LoginRequiredMixin, View):
    """Add a person to the project. Owner only."""

    def post(self, request, pk):
        from django.contrib import messages
        from django.contrib.auth import get_user_model
        from django.shortcuts import redirect

        from .models import ProjectShare

        project = get_object_or_404(AnnotationProject.owned(request.user), pk=pk)
        who = (request.POST.get("who") or "").strip()
        role = request.POST.get("role") or ProjectShare.Role.ANNOTATOR

        User = get_user_model()
        user = (User.objects.filter(username__iexact=who).first()
                or User.objects.filter(email__iexact=who).first())
        if user is None:
            messages.error(request, f"No account matches “{who}”.")
        elif user.id == project.user_id:
            messages.info(request, "You already own this project.")
        elif role not in dict(ProjectShare.Role.choices):
            messages.error(request, "Unknown role.")
        else:
            share, created = ProjectShare.objects.update_or_create(
                project=project, user=user,
                defaults={"role": role, "created_by": request.user})
            messages.success(
                request,
                f"{'Added' if created else 'Updated'} {user.username} as "
                f"{share.get_role_display().split('—')[0].strip().lower()}.")
        return redirect("annotations:people", pk=pk)


class ShareUpdateView(LoginRequiredMixin, View):
    """Change someone's role, or remove them. Owner only."""

    def post(self, request, pk):
        from django.contrib import messages
        from django.shortcuts import redirect

        from .models import ProjectShare

        project = get_object_or_404(AnnotationProject.owned(request.user), pk=pk)
        share = get_object_or_404(ProjectShare, project=project,
                                  pk=request.POST.get("share_id"))

        if request.POST.get("remove"):
            # Their assignments go back to the pool rather than vanishing with
            # them — the work is the project's, not theirs.
            freed = project.assignments.filter(user=share.user).delete()[0]
            name = share.user.username
            share.delete()
            messages.info(
                request,
                f"Removed {name}." + (f" {freed} clip(s) returned to the "
                                      "unassigned pool." if freed else ""))
        else:
            role = request.POST.get("role")
            if role in dict(ProjectShare.Role.choices):
                share.role = role
                share.save(update_fields=["role"])
                messages.success(request, f"{share.user.username} is now a "
                                          f"{role}.")
        return redirect("annotations:people", pk=pk)


class AssignClipsView(LoginRequiredMixin, View):
    """Hand clips out, or deal them round-robin. Manager and above."""

    def post(self, request, pk):
        from django.contrib import messages
        from django.contrib.auth import get_user_model
        from django.shortcuts import redirect

        from . import assignments as assign_mod

        project = get_object_or_404(AnnotationProject.manageable(request.user), pk=pk)
        video_ids = [int(v) for v in request.POST.getlist("video_ids") if str(v).isdigit()]
        if not video_ids:
            messages.warning(request, "No clips selected.")
            return redirect("annotations:detail", pk=pk)

        User = get_user_model()
        targets = User.objects.filter(pk__in=request.POST.getlist("assignee"))
        # Only people who are actually on the project, or the assignment names
        # someone who cannot open it.
        allowed = {project.user_id} | set(
            project.shares.values_list("user_id", flat=True))
        targets = [u for u in targets if u.id in allowed]

        if not targets:
            freed = project.assignments.filter(video_id__in=video_ids).delete()[0]
            messages.info(request, f"Returned {freed} clip(s) to the pool.")
        elif len(targets) == 1:
            moved = assign_mod.assign(project, video_ids, targets[0], by=request.user)
            messages.success(request,
                             f"Assigned {moved} clip(s) to {targets[0].username}.")
        else:
            tally = assign_mod.distribute(project, video_ids, targets, by=request.user)
            spread = ", ".join(f"{u.username} {tally.get(u.id, 0)}" for u in targets)
            messages.success(request, f"Split {len(video_ids)} clip(s): {spread}.")
        return redirect(request.POST.get("next") or f"/annotations/{pk}/")


class ClaimClipsView(LoginRequiredMixin, View):
    """Take clips from the unassigned pool, or give your own back."""

    def post(self, request, pk):
        from django.contrib import messages
        from django.shortcuts import redirect

        from . import assignments as assign_mod

        project = get_object_or_404(AnnotationProject.annotatable(request.user), pk=pk)
        video_ids = [int(v) for v in request.POST.getlist("video_ids") if str(v).isdigit()]
        releasing = bool(request.POST.get("release"))

        done = 0
        for vid in video_ids:
            if releasing:
                done += 1 if assign_mod.release(project, vid, request.user) else 0
            else:
                done += 1 if assign_mod.claim(project, vid, request.user) else 0

        if releasing:
            messages.info(request, f"Returned {done} clip(s) to the pool.")
        else:
            messages.success(request, f"Took {done} clip(s).")
            if done < len(video_ids):
                messages.warning(
                    request,
                    f"{len(video_ids) - done} were already taken by someone else.")
        return redirect(request.POST.get("next") or f"/annotations/{pk}/")


# One add is capped: every added clip is frame-sampled (up to max_frames each)
# on the web process, so a whole camera's history in one go would queue
# millions of frames. Bigger sets take several adds — ideally even spreads.
ADD_CAP = 1000
ADD_PAGE_SIZE = 200


def _frames_per_clip():
    from . import sampling
    return sampling.clamp_params({})["max_frames"]


def _seconds_per_sample():
    """Typical wall time of one clip's frame sampling, from recent finished
    tasks — for the "about N hours" on the confirm. None until there is data."""
    from .models import FrameSamplingTask
    rows = list(FrameSamplingTask.objects
                .filter(status=FrameSamplingTask.Status.COMPLETED,
                        started_at__isnull=False, completed_at__isnull=False)
                .order_by("-completed_at").values_list("started_at", "completed_at")[:50])
    secs = sorted((done - start).total_seconds() for start, done in rows if done > start)
    return round(secs[len(secs) // 2], 1) if secs else None


def _page_of_clips(qs, in_project, offset):
    """One page of the picker grid, newest first, marked if already added."""
    videos = list(qs.select_related("device")
                  .order_by("-recorded_at", "-uploaded_at", "-id")[offset:offset + ADD_PAGE_SIZE])
    for v in videos:
        v.in_project = v.pk in in_project
    return videos


def _motion_strip(motion):
    """The clip row's motion strip: bar heights (px) and which bars were picked."""
    if not motion or not motion.get("profile"):
        return None
    picked = set(motion.get("picked") or [])
    return {"bars": [{"h": max(1, round(v * 0.16)), "picked": i in picked}
                     for i, v in enumerate(motion["profile"])],
            "n_picked": len(picked)}


class AddVideosGridView(LoginRequiredMixin, View):
    """The next page of the picker grid ("Load older clips"), as HTML."""

    def get(self, request, pk):
        from django.template.loader import render_to_string
        from apps.devices.models import Device
        from apps.videos import workspace
        from apps.videos.models import Video

        project = get_object_or_404(AnnotationProject.manageable(request.user), pk=pk)
        params = request.GET
        try:
            offset = max(0, int(params.get("offset") or 0))
        except (TypeError, ValueError):
            offset = 0
        qs = workspace.apply_video_filters(Video.accessible(request.user), params)
        in_project = set(project.videos.values_list("pk", flat=True))
        membership = (params.get("member") or "").strip()
        if membership == "in":
            qs = qs.filter(pk__in=in_project)
        elif membership == "out":
            qs = qs.exclude(pk__in=in_project)
        videos = _page_of_clips(qs, in_project, offset)
        devices = Device.accessible(request.user).order_by("name")
        dots = workspace.dots_by_device(workspace.device_rows(devices, [], {}))
        html = render_to_string("videos/_review_grid.html", {
            "video_days": workspace.group_by_day(videos, dots),
            "can_select": True,
            "card_status_template": "annotations/_card_membership.html",
        }, request=request)
        next_offset = offset + len(videos)
        return JsonResponse({"html": html, "count": len(videos),
                             "next_offset": next_offset,
                             "has_more": len(videos) == ADD_PAGE_SIZE and next_offset < qs.count()})


class AddVideosWorkspaceView(LoginRequiredMixin, TemplateView):
    """Choosing clips to annotate, in the interface built for choosing clips.

    The picker used to be a checkbox list of filenames in a five-row scroll box
    capped at 500 — the decision that determines what the model learns, made by
    reading titles through a window. This is the Processing hub's rail and grid,
    shared rather than copied, plus the coverage map that says whether the
    sample is lopsided.
    """

    template_name = "annotations/add_videos.html"

    def get_context_data(self, **kwargs):
        from apps.devices.models import Device
        from apps.videos import workspace
        from apps.videos.models import Video

        from . import coverage as coverage_mod

        ctx = super().get_context_data(**kwargs)
        # Adding clips restructures the project and spends decode time.
        project = get_object_or_404(
            AnnotationProject.manageable(self.request.user), pk=kwargs["pk"])
        params = self.request.GET

        accessible = Video.accessible(self.request.user)
        qs = workspace.apply_video_filters(accessible, params)
        in_project = set(project.videos.values_list("pk", flat=True))

        devices = Device.accessible(self.request.user).order_by("name")
        per_device = dict(
            qs.exclude(device=None).order_by()
            .values_list("device_id").annotate(models.Count("id")))
        rows = workspace.device_rows(devices, workspace._values(params, "device"),
                                     per_device)
        dots = workspace.dots_by_device(rows)

        # Clips already in the project stay in the grid, marked. Hiding them is
        # what makes over-sampling one hotel invisible — the same reason a
        # reference with zero visits keeps its row.
        membership = (params.get("member") or "").strip()
        if membership == "in":
            qs = qs.filter(pk__in=in_project)
        elif membership == "out":
            qs = qs.exclude(pk__in=in_project)

        videos = _page_of_clips(qs, in_project, 0)

        ctx.update({
            "project": project,
            "f": workspace.current_filter(params),
            "opts": workspace.filter_options(accessible),
            "device_rows": rows,
            "video_days": workspace.group_by_day(videos, dots),
            "video_count": qs.count(),
            "page_size": ADD_PAGE_SIZE,
            "add_cap": ADD_CAP,
            "frames_per_clip": _frames_per_clip(),
            "seconds_per_sample": _seconds_per_sample(),
            "sample_workers": 2,
            "date_overview": workspace.date_overview(
                workspace.apply_video_filters(accessible, workspace.without_dates(params))),
            "grid_page_url": reverse("annotations:add_videos_grid", args=[project.pk]),
            "available_total": accessible.count(),
            "in_project_count": len(in_project),
            "member": membership,
            "member_choices": (("", "Show all"), ("out", "Not yet added"),
                               ("in", "Already added")),
            "coverage": coverage_mod.build(
                workspace.apply_video_filters(accessible, params),
                project.videos.all(), devices, dots),
            "card_status_template": "annotations/_card_membership.html",
            "filter_action": reverse("annotations:add_videos_page", args=[project.pk]),
        })
        return ctx


class AddVideosDraftView(LoginRequiredMixin, View):
    """Pre-select a balanced spread across the hotels and hours the project lacks."""

    def get(self, request, pk):
        from apps.videos import workspace
        from apps.videos.models import Video

        from . import coverage as coverage_mod

        # Part of adding clips.
        project = get_object_or_404(
            AnnotationProject.manageable(request.user), pk=pk)
        qs = workspace.apply_video_filters(Video.accessible(request.user), request.GET)
        if request.GET.get("even"):
            # "Select an even N across everything matching the filter".
            picks = coverage_mod.even_spread(qs, project.videos.all(), ADD_CAP)
        else:
            try:
                per_cell = max(1, min(int(request.GET.get("per_cell") or 2), 10))
            except (TypeError, ValueError):
                per_cell = 2
            picks = coverage_mod.draft(qs, project.videos.all(), per_cell=per_cell)
        return JsonResponse({"picks": picks,
                             "video_ids": [p["id"] for p in picks],
                             "count": len(picks)})


class AddVideosView(LoginRequiredMixin, View):
    """Add selected videos to an annotation project."""

    def post(self, request, pk):
        from django.shortcuts import redirect
        from django.contrib import messages

        # Adding clips restructures the project.
        project = get_object_or_404(
            AnnotationProject.manageable(request.user), pk=pk)
        from apps.videos.models import Video

        if request.POST.get("mode") == "filter":
            # "Add all N matching this filter": resolved here from the filter
            # the page was showing, so no id list has to cross the wire. Not
            # capped — the page confirms the size first.
            from apps.videos import workspace
            qs = workspace.apply_video_filters(Video.accessible(request.user), request.POST)
            video_ids = list(qs.exclude(pk__in=project.videos.values("pk"))
                             .values_list("pk", flat=True))
            return self._add(request, project, video_ids, skipped=0)

        # Picks that were never on screen (drafts) arrive as one comma list.
        video_ids = request.POST.getlist("video_ids") + [
            v for v in (request.POST.get("extra_ids") or "").split(",") if v.strip()]
        try:
            video_ids = sorted({int(v) for v in video_ids})
        except (TypeError, ValueError):
            messages.error(request, "Invalid clip selection.")
            return redirect("annotations:add_videos_page", pk=pk)

        if not video_ids:
            messages.warning(request, "No videos selected.")
            return redirect("annotations:detail", pk=pk)
        if len(video_ids) > ADD_CAP:
            messages.error(
                request, f"{len(video_ids)} clips selected — one add is limited to "
                f"{ADD_CAP:,}. Add them in batches, or use an even spread.")
            return redirect("annotations:add_videos_page", pk=pk)

        already = set(project.videos.filter(pk__in=video_ids).values_list("pk", flat=True))
        return self._add(request, project, [v for v in video_ids if v not in already],
                         skipped=len(already))

    def _add(self, request, project, video_ids, skipped):
        from django.shortcuts import redirect
        from django.contrib import messages
        from apps.videos.models import Video

        if not video_ids:
            messages.warning(request, "Nothing new to add — those clips are already in the project.")
            return redirect("annotations:detail", pk=project.pk)
        # Accessible, not owned: the picker lists clips from shared devices, so
        # restricting the add to owned ones silently drops half a selection.
        added_ids = list(Video.accessible(request.user).filter(pk__in=video_ids)
                         .values_list("pk", flat=True))
        through = project.videos.through
        through.objects.bulk_create(
            [through(annotationproject_id=project.pk, video_id=v) for v in added_ids],
            ignore_conflicts=True, batch_size=1000)
        added = len(added_ids)

        # Sample them straight away. Adding a clip and then remembering to
        # sample it were two steps that always ran together, and forgetting the
        # second left the clip looking added-but-empty with nothing saying why.
        # Re-sampling with different knobs stays available on the project page.
        from . import sampling
        from .models import FrameSamplingTask

        # Queued in one insert; the bounded decode pool works through them two
        # at a time (and the reconciler re-feeds any a deploy interrupts).
        # New clips are sampled by motion: their most active frames, not every
        # Nth one, so a labeller isn't handed a stack of empty frames.
        params = sampling.motion_params()
        tasks = FrameSamplingTask.objects.bulk_create([
            FrameSamplingTask(user=request.user, project=project, video_id=v, params=params)
            for v in added_ids], batch_size=1000)
        for task in tasks:
            sampling.spawn_sampling_async(task.pk)

        messages.success(
            request,
            f"Added {added} clip(s)"
            + (f" ({skipped} already in the project, skipped)" if skipped else "")
            + " and started sampling their most active frames — up to "
            f"{params['max_frames']} each, at least {params['min_gap_s']:g}s apart. "
            "Frames appear as they finish.")
        return redirect("annotations:detail", pk=project.pk)


def _editor_landing(project, user, video_id, frame):
    """Where the editor should open, or None to open what was asked for.

    With a clip: that frame if it was sampled, else the clip's first frame
    still needing labels (else its first frame). Without one: the first frame
    needing labels in the user's assigned clips, then in the whole project.
    "Needing labels" = no boxes and not reviewed. Order matches the editor's
    prev/next (clip title, then frame).
    """
    from .models import ClipAssignment

    frames = Annotation.objects.filter(project=project)
    todo = frames.filter(boxes=[], reviewed=False)
    order = ("video__title", "video_id", "frame_number")
    try:
        frame = int(frame) if frame not in (None, "") else None
    except (TypeError, ValueError):
        frame = None

    if video_id:
        if not project.videos.filter(pk=video_id).exists():
            return None
        if frame is not None and frames.filter(video_id=video_id, frame_number=frame).exists():
            return None
        hit = (todo.filter(video_id=video_id).order_by("frame_number").first()
               or frames.filter(video_id=video_id).order_by("frame_number").first())
    else:
        mine = ClipAssignment.objects.filter(project=project, user=user).values("video_id")
        hit = (todo.filter(video_id__in=mine).order_by(*order).first()
               or todo.order_by(*order).first()
               or frames.order_by(*order).first())
    if hit is None:
        return None
    return (reverse("annotations:editor", args=[project.pk])
            + f"?video={hit.video_id}&frame={hit.frame_number}")


class AnnotationEditorView(LoginRequiredMixin, TemplateView):
    template_name = "annotations/editor.html"

    def get(self, request, *args, **kwargs):
        """Return JSON for AJAX frame navigation, HTML for normal page load."""
        if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.GET.get("format") == "json":
            # Opening the editor is reading; SaveAnnotationView decides who
            # may actually draw, and on which clip.
            project = get_object_or_404(
                AnnotationProject.accessible(request.user), pk=self.kwargs["pk"])
            video_id = request.GET.get("video")
            frame_number = int(request.GET.get("frame", 0))
            boxes = []
            if video_id:
                try:
                    video = project.videos.get(pk=video_id)
                    ann = Annotation.objects.get(project=project, video=video, frame_number=frame_number)
                    boxes = ann.boxes
                except (Annotation.DoesNotExist, Exception):
                    boxes = []
            return JsonResponse({"boxes": boxes, "frame": frame_number})
        # Land on a real frame. "Annotate" opens the editor with no clip, and
        # clip links say frame=0 — which motion sampling rarely picks — so both
        # used to show an empty canvas with no prev/next.
        from django.shortcuts import redirect
        project = get_object_or_404(
            AnnotationProject.accessible(request.user), pk=self.kwargs["pk"])
        # jump=1: the editor's "go to frame N" — an unsampled frame on purpose.
        target = None if request.GET.get("jump") else _editor_landing(
            project, request.user, request.GET.get("video"), request.GET.get("frame"))
        if target:
            return redirect(target)
        return super().get(request, *args, **kwargs)

    def get_context_data(self, **kwargs):
        logger = logging.getLogger(__name__)

        ctx = super().get_context_data(**kwargs)
        try:
            project = get_object_or_404(
                AnnotationProject.accessible(self.request.user),
                pk=self.kwargs["pk"])
        except Exception as e:
            logger.error("Editor: project lookup failed: %s", e)
            raise

        video_id = self.request.GET.get("video")
        frame_number = int(self.request.GET.get("frame", 0))

        video = None
        boxes = []
        if video_id:
            try:
                video = project.videos.get(pk=video_id)
            except Exception as e:
                logger.error("Editor: video %s not in project %s: %s", video_id, project.pk, e)
                video = None
            try:
                annotation = Annotation.objects.get(
                    project=project, video=video, frame_number=frame_number
                )
                boxes = annotation.boxes
            except Annotation.DoesNotExist:
                boxes = []

        # Auto-ingest external-S3 video into our raw-videos bucket if needed
        ctx["transferring"] = False
        if video and video.storage_key.startswith("s3://"):
            try:
                import threading
                from apps.analysis.views import _ingest_external_s3_to_storage

                # Check if already transferring (avoid duplicate)
                if not getattr(video, '_transfer_started', False):
                    ctx["transferring"] = True
                    # Transfer in background — page will show spinner and auto-reload
                    thread = threading.Thread(
                        target=self._do_transfer,
                        args=(video.pk,),
                        daemon=True,
                    )
                    thread.start()
            except Exception as e:
                logger.error("Auto-transfer failed for video %s: %s", video.pk, e)

        ctx["project"] = project
        ctx["video"] = video
        ctx["frame_number"] = frame_number
        ctx["boxes"] = json.dumps(boxes)
        ctx["classes"] = json.dumps(project.classes)
        ctx["videos"] = project.videos.all()
        ctx["video_url"] = ""

        # Build ordered frame list for prev/next navigation across all project frames
        all_frames = list(
            Annotation.objects.filter(project=project)
            .order_by("video__title", "frame_number")
            .values_list("video_id", "frame_number")
        )
        current_key = (video.pk if video else None, frame_number)
        ctx["total_project_frames"] = len(all_frames)
        ctx["current_frame_index"] = 0
        ctx["prev_frame_url"] = ""
        ctx["next_frame_url"] = ""

        if all_frames and current_key in all_frames:
            idx = all_frames.index(current_key)
            ctx["current_frame_index"] = idx + 1
            base_url = f"/annotations/{project.pk}/edit/"
            if idx > 0:
                pv, pf = all_frames[idx - 1]
                ctx["prev_frame_url"] = f"{base_url}?video={pv}&frame={pf}"
            if idx < len(all_frames) - 1:
                nv, nf = all_frames[idx + 1]
                ctx["next_frame_url"] = f"{base_url}?video={nv}&frame={nf}"

        # Presigned URL for video playback
        if video and video.storage_key and not video.storage_key.startswith("s3://"):
            try:
                from config.storage import get_s3_client
                ctx["video_url"] = get_s3_client().generate_presigned_url(
                    "raw-videos", video.storage_key,
                )
            except Exception:
                pass

        return ctx

    @staticmethod
    def _do_transfer(video_pk):
        """Background thread: ingest external-S3 video into our raw-videos bucket."""
        import django
        django.setup()
        from django.db import connection
        try:
            from apps.videos.models import Video
            from apps.analysis.views import _ingest_external_s3_to_storage
            video = Video.objects.select_related("source").get(pk=video_pk)
            _ingest_external_s3_to_storage(video)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error("Background transfer failed for %s: %s", video_pk, e)
        finally:
            connection.close()


class TransferVideoView(LoginRequiredMixin, View):
    """Ingest an external-S3 video into our raw-videos bucket so frames are available."""

    def post(self, request, pk):
        from django.shortcuts import redirect
        from django.contrib import messages

        # Moving a clip between projects.
        project = get_object_or_404(
            AnnotationProject.manageable(request.user), pk=pk)
        video_id = request.POST.get("video_id")
        frame = request.POST.get("frame", 0)

        from apps.videos.models import Video
        # The clip has to be in this project. Filtering on ownership instead
        # would make every manager on a shared project unable to touch clips
        # the owner added — which is all of them.
        video = get_object_or_404(project.videos.all(), pk=video_id)

        if not video.storage_key.startswith("s3://"):
            messages.info(request, "Video is already in storage.")
            return redirect(f"/annotations/{pk}/edit/?video={video_id}&frame={frame}")

        try:
            from apps.analysis.views import _ingest_external_s3_to_storage
            _ingest_external_s3_to_storage(video)
            messages.success(request, "Video transferred. Frames now available.")
        except Exception as e:
            logger.error("Transfer failed for video %s: %s", video_id, e)
            messages.error(request, f"Transfer failed: {e}")

        return redirect(f"/annotations/{pk}/edit/?video={video_id}&frame={frame}")


class SaveAnnotationView(LoginRequiredMixin, View):
    def post(self, request, pk):
        # Drawing. The per-clip check below narrows it further: an annotator
        # works only what is assigned to them.
        project = get_object_or_404(
            AnnotationProject.annotatable(request.user), pk=pk)
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)

        video_id = data.get("video_id")
        frame_number = data.get("frame_number")
        boxes = data.get("boxes", [])

        if video_id is None or frame_number is None:
            return JsonResponse({"error": "video_id and frame_number are required"}, status=400)

        video = get_object_or_404(project.videos, pk=video_id)

        # The rule that actually confines an annotator: they draw on the clips
        # assigned to them, by someone else or by themselves out of the pool.
        # Reviewers and above are not confined that way, because checking other
        # people's work is the job.
        if not project.may_annotate_video(request.user, video.pk):
            return JsonResponse(
                {"error": "This clip is not assigned to you. Claim it from the "
                          "unassigned pool, or ask for it to be assigned."},
                status=403)

        # A human saving in the editor = a human review.
        from django.utils import timezone
        annotation, created = Annotation.objects.update_or_create(
            project=project,
            video=video,
            frame_number=frame_number,
            defaults={
                "boxes": boxes,
                "reviewed": True,
                "review_source": Annotation.ReviewSource.HUMAN,
                "reviewed_at": timezone.now(),
                # A human saved this frame, so it is no longer a bare sampled
                # placeholder — including when they saved it with no boxes, which
                # is a deliberate negative example.
                "sampled_only": False,
            },
        )

        return JsonResponse({
            "success": True,
            "created": created,
            "annotation_id": annotation.pk,
            "reviewed": True,
        })


# Bounded pool for the (short) pre-annotation SPAWN — transfer video if needed,
# then invoke_endpoint_async. Finalization is NOT here; the reconciler does it.
_PREANNOT_POOL = ThreadPoolExecutor(max_workers=3, thread_name_prefix="preannot-spawn")


def spawn_preannotation_async(task_pk: int) -> None:
    _PREANNOT_POOL.submit(_spawn_preannotation, task_pk)


def _spawn_preannotation(task_pk: int) -> None:
    """Fire the SageMaker async invocation for one PreAnnotationTask and store
    its output/failure S3 URIs. Durable: after this returns the reconciler can
    finalize the task even across a deploy. Runs in a bounded thread."""
    import django
    django.setup()
    from urllib.parse import urlparse
    import boto3
    from botocore.config import Config
    from django.conf import settings
    from django.db import connection
    from django.utils import timezone

    from .models import PreAnnotationTask

    try:
        task = PreAnnotationTask.objects.select_related("project", "video", "video__source").get(pk=task_pk)
    except PreAnnotationTask.DoesNotExist:
        connection.close()
        return

    # Cancel guard: a task cancelled while waiting in the pool must never reach
    # the GPU (an async invocation can't be recalled once sent). Also inert
    # against accidental double-spawn.
    if task.status != PreAnnotationTask.Status.QUEUED:
        logger.info("pre-annotate task %s skipped (status=%s)", task_pk, task.status)
        connection.close()
        return

    try:
        video = task.video
        blob_path = video.storage_key
        if blob_path.startswith("s3://"):
            from apps.analysis.views import _ingest_external_s3_to_storage
            blob_path = _ingest_external_s3_to_storage(video)

        labeler = task.labeler
        if labeler == "sam3":
            endpoint = settings.SAGEMAKER_SAM3_ENDPOINT_NAME
        else:
            endpoint = settings.SAGEMAKER_ENDPOINT_NAME
        if not endpoint:
            raise RuntimeError(f"{'SAM3' if labeler=='sam3' else 'YOLO'} endpoint not configured")

        in_bucket = settings.SAGEMAKER_INPUT_BUCKET
        region = settings.AWS_REGION
        p = task.params or {}
        payload = {
            "task": "pre_annotate",
            "job_id": f"preannot-{task_pk}",
            "user_id": str(task.user_id),
            "video_blob_path": blob_path,
            # Detector for the unified image: SAM 3 text-prompt (the only
            # pre-annotation labeler now). The endpoint runs it on the g5.
            "detector_kind": "sam3" if labeler == "sam3" else "yolo",
            # Detect only the requested labels (SAM 3 prompts / YOLO filter);
            # finalize re-maps class ids by name against the project's classes.
            "classes": p.get("target_labels") or task.project.classes,
            "sample_interval": p.get("sample_interval", 10),
            "max_frames": p.get("max_frames", 300),
            "confidence_threshold": p.get("confidence", 0.15),
            "selection": p.get("selection", "uniform"),
            "nms_iou": p.get("nms_iou", 0.5),
            "max_detections": p.get("max_detections", 100),
        }
        if p.get("frame_numbers"):  # per-frame pre-annotate (editor) targets exact frames
            payload["frame_numbers"] = p["frame_numbers"]
        if task.custom_model_key:
            payload["custom_bee_model_path"] = task.custom_model_key

        cfg = Config(connect_timeout=10, read_timeout=30, retries={"max_attempts": 2})
        s3 = boto3.client("s3", region_name=region, config=cfg)
        smrt = boto3.client("sagemaker-runtime", region_name=region, config=cfg)
        key = f"preannotate/{task.project_id}/{task_pk}-{labeler}.json"
        s3.put_object(Bucket=in_bucket, Key=key,
                      Body=json.dumps(payload).encode("utf-8"),
                      ContentType="application/json")
        # Last look before spending GPU money: the ingest + payload upload above
        # can be slow, so re-check that a cancel didn't land meanwhile.
        cur = (PreAnnotationTask.objects
               .filter(pk=task_pk).values_list("status", flat=True).first())
        if cur != PreAnnotationTask.Status.QUEUED:
            logger.info("pre-annotate task %s cancelled before invoke (status=%s)",
                        task_pk, cur)
            return

        resp = smrt.invoke_endpoint_async(
            EndpointName=endpoint,
            InputLocation=f"s3://{in_bucket}/{key}",
            ContentType="application/json",
            InferenceId=f"preannot-{task_pk}",
            # Default is 15 min; long videos need the platform max (1 h).
            InvocationTimeoutSeconds=3600,
        )
        out = urlparse(resp["OutputLocation"])
        fail_loc = resp.get("FailureLocation", "") or ""
        if not fail_loc:
            fail_loc = resp["OutputLocation"].replace(".out", ".failure")
        # Guarded: only a still-QUEUED row advances to PROCESSING. If a cancel
        # won the race after the invoke, the row stays CANCELLED — the
        # reconciler only polls QUEUED/PROCESSING, so the GPU result is
        # discarded (the documented cancel semantics).
        PreAnnotationTask.objects.filter(
            pk=task_pk, status=PreAnnotationTask.Status.QUEUED,
        ).update(
            status=PreAnnotationTask.Status.PROCESSING,
            output_uri=resp["OutputLocation"], failure_uri=fail_loc,
            started_at=timezone.now(),
        )
        logger.info("pre-annotate task %s invoked -> %s", task_pk, resp["OutputLocation"])
    except Exception as e:
        logger.exception("pre-annotate spawn failed for task %s", task_pk)
        # Guarded likewise: a cancelled task must not be re-marked FAILED.
        PreAnnotationTask.objects.filter(
            pk=task_pk, status=PreAnnotationTask.Status.QUEUED,
        ).update(
            status=PreAnnotationTask.Status.FAILED,
            error_message=str(e)[:1000], completed_at=timezone.now(),
        )
    finally:
        connection.close()


# Past this the async request has expired (SM async: 1h processing + 6h TTL).
_PREANNOT_TIMEOUT_HOURS = 8


def finalize_preannotation_task(task, s3) -> bool:
    """Check one PROCESSING task's output/failure S3 URI. On the result, write
    Annotation rows (protecting human-reviewed frames) and complete; on failure
    or timeout, mark failed. Returns True if the task reached a terminal state.
    Idempotent; safe to call repeatedly from the reconciler."""
    import json as _json
    from datetime import timedelta
    from urllib.parse import urlparse
    from django.utils import timezone
    from botocore.exceptions import ClientError

    from .models import Annotation, PreAnnotationTask

    if not task.output_uri:
        return False
    out = urlparse(task.output_uri)
    try:
        body = s3.get_object(Bucket=out.netloc, Key=out.path.lstrip("/"))["Body"].read()
        result = _json.loads(body)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code", "") not in ("NoSuchKey", "404", "NotFound"):
            logger.warning("pre-annotate poll error task %s: %s", task.pk, e)
            return False
        # No output yet — check for a platform failure, else timeout.
        if task.failure_uri:
            f = urlparse(task.failure_uri)
            try:
                fb = s3.get_object(Bucket=f.netloc, Key=f.path.lstrip("/"))["Body"].read()
                PreAnnotationTask.objects.filter(pk=task.pk).update(
                    status=PreAnnotationTask.Status.FAILED,
                    error_message=f"endpoint failure: {fb.decode('utf-8', 'replace')[:500]}",
                    completed_at=timezone.now())
                return True
            except ClientError:
                pass
        age = timezone.now() - (task.started_at or task.created_at)
        if age > timedelta(hours=_PREANNOT_TIMEOUT_HOURS):
            PreAnnotationTask.objects.filter(pk=task.pk).update(
                status=PreAnnotationTask.Status.FAILED,
                error_message="Timed out — no result. Re-run pre-annotation.",
                completed_at=timezone.now())
            return True
        return False

    frames = result.get("frames") or []
    width = result.get("video_width", 1280)
    height = result.get("video_height", 720)

    # Suggest-only (editor per-frame pre-annotate): don't touch the DB — the boxes
    # are read straight from this S3 output and shown for the human to Save.
    if (task.params or {}).get("suggest_only"):
        PreAnnotationTask.objects.filter(pk=task.pk).update(
            status=PreAnnotationTask.Status.COMPLETED,
            frames_written=len(frames), completed_at=timezone.now())
        return True

    proj_classes = task.project.classes or []
    class_id_by_name = {c: i for i, c in enumerate(proj_classes)}
    target_labels = (task.params or {}).get("target_labels") or proj_classes
    # Merge mode: this run targets only SOME classes (e.g. nest/hotel) — keep the
    # frame's existing boxes of OTHER classes (e.g. reviewed bee) and swap in only
    # the target-class detections. Full-class runs keep the old replace semantics.
    target_set = set(target_labels)
    merge_mode = 0 < len(target_set) < len(proj_classes)

    def _remap(box):
        """Ensure box class_id matches the project's ordering (the detector may
        have used a subset ordering); drop unknown classes."""
        name = box.get("class")
        if name not in class_id_by_name:
            return None
        return {**box, "class_id": class_id_by_name[name]}

    existing_by_frame = {}
    if merge_mode:
        existing_by_frame = {
            a.frame_number: a for a in Annotation.objects.filter(
                project_id=task.project_id, video_id=task.video_id)
        }
    else:
        human_frames = set(
            Annotation.objects.filter(
                project_id=task.project_id, video_id=task.video_id,
                review_source=Annotation.ReviewSource.HUMAN,
            ).values_list("frame_number", flat=True)
        )

    created = 0
    for fd in frames:
        fn = fd["frame_number"]
        detected = [b for b in (_remap(b) for b in fd.get("boxes", [])) if b]

        if merge_mode:
            existing = existing_by_frame.get(fn)
            kept = [b for b in (existing.boxes if existing else [])
                    if b.get("class") not in target_set]  # preserve other classes
            new = [b for b in detected if b.get("class") in target_set]
            boxes = kept + new
            if existing:
                # Preserve the frame's review status — we only ADD target boxes,
                # the user's existing (incl. human-reviewed) boxes are untouched.
                fields = {"boxes": boxes}
                if not existing.frame_image_path and fd.get("frame_image_path"):
                    fields["frame_image_path"] = fd["frame_image_path"]
                Annotation.objects.filter(pk=existing.pk).update(**fields)
            else:
                Annotation.objects.create(
                    project_id=task.project_id, video_id=task.video_id, frame_number=fn,
                    boxes=boxes, image_width=width, image_height=height,
                    frame_image_path=fd.get("frame_image_path", ""),
                    reviewed=False, review_source=Annotation.ReviewSource.NONE,
                )
            created += 1
        else:
            if fn in human_frames:
                continue
            Annotation.objects.update_or_create(
                project_id=task.project_id, video_id=task.video_id, frame_number=fn,
                defaults={
                    "boxes": detected,
                    "image_width": width, "image_height": height,
                    "frame_image_path": fd.get("frame_image_path", ""),
                    "reviewed": False,
                    "review_source": Annotation.ReviewSource.NONE,
                    "reviewed_at": None,
                },
            )
            created += 1

    PreAnnotationTask.objects.filter(pk=task.pk).update(
        status=PreAnnotationTask.Status.COMPLETED,
        frames_written=created, completed_at=timezone.now())

    exec_secs = result.get("execution_seconds", 0) or 0
    try:
        from apps.accounts.models import UserProfile
        profile, _ = UserProfile.objects.get_or_create(user_id=task.user_id)
        profile.charge(int(exec_secs) if exec_secs else 30, gpu_seconds=exec_secs or 30)
    except Exception as e:
        logger.error("pre-annotate credit charge failed task %s: %s", task.pk, e)
    logger.info("pre-annotate task %s complete: %d frames written", task.pk, created)
    return True


def poll_preannotation_tasks(user=None) -> int:
    """Finalize in-flight pre-annotation tasks. Reusable by the reconciler and a
    page poll. ``user=None`` polls all users. Idempotent; never raises."""
    import boto3
    from botocore.config import Config
    from django.conf import settings

    from .models import PreAnnotationTask
    try:
        qs = PreAnnotationTask.objects.filter(status=PreAnnotationTask.Status.PROCESSING)
        if user is not None:
            qs = qs.filter(user=user)
        tasks = list(qs.select_related("project", "video")[:100])
        finalized = 0
        if tasks:
            s3 = boto3.client("s3", region_name=getattr(settings, "AWS_REGION", "us-east-1"),
                              config=Config(connect_timeout=10, read_timeout=30, retries={"max_attempts": 2}))
            finalized = sum(1 for t in tasks if finalize_preannotation_task(t, s3))
        # Drain QUEUED tasks under a concurrency cap, after finalizing so slots
        # freed this pass refill immediately — same shape as
        # analysis.views._drain_queue. Without it, pre-annotating an N-video
        # project fires N async SageMaker invocations at once, with only the
        # 3-worker spawn pool (which serialises the *invoke*, not the GPU work)
        # as backpressure.
        drain_preannotation_queue()
        return finalized
    except Exception:
        logger.exception("poll_preannotation_tasks failed")
        return 0


def drain_preannotation_queue() -> int:
    """Promote QUEUED pre-annotation tasks while under the concurrency cap."""
    from django.conf import settings

    from .models import PreAnnotationTask

    try:
        cap = getattr(settings, "PREANNOTATION_MAX_CONCURRENT", 4)
        in_flight = PreAnnotationTask.objects.filter(
            status=PreAnnotationTask.Status.PROCESSING).count()
        slots = max(0, cap - in_flight)
        if not slots:
            return 0
        spawned = 0
        for task in PreAnnotationTask.objects.filter(
            status=PreAnnotationTask.Status.QUEUED,
        ).order_by("created_at")[:slots]:
            # _spawn_preannotation re-checks the status before invoking, so a
            # task cancelled between here and the pool never reaches the GPU.
            spawn_preannotation_async(task.pk)
            spawned += 1
        return spawned
    except Exception:
        logger.exception("drain_preannotation_queue failed")
        return 0


class PreAnnotateView(LoginRequiredMixin, View):
    """Create a durable pre-annotation task for one video and fire it."""

    def post(self, request, pk):
        from django.shortcuts import redirect
        from django.contrib import messages

        # Spends GPU.
        project = get_object_or_404(
            AnnotationProject.manageable(request.user), pk=pk)
        from apps.videos.models import Video
        video = get_object_or_404(project.videos.all(),
                                  pk=request.POST.get("video_id"))

        task = _create_preannotation_task(request, project, video)
        spawn_preannotation_async(task.pk)
        engine = "SAM 3" if task.labeler == "sam3" else "AI"
        messages.info(request, f"Pre-annotating '{video.title}' with {engine}. "
                               "This runs in the background — refresh to see progress.")
        return redirect("annotations:detail", pk=pk)


def _create_preannotation_task(request, project, video):
    """Build a PreAnnotationTask row from the request's pre-annotate options."""
    from .models import PreAnnotationTask

    sample_interval, max_frames, confidence = _preannotate_opts(request)
    labeler, custom_model_key = _labeler_opts(request)
    # Frame selection is always "every Nth" (uniform). The DINOv2 "diverse"
    # selection was removed — it added embedding GPU work we don't need.
    selection = "uniform"
    nms_iou, max_detections = _sam3_opts(request)
    # Which labels to (re)detect. A strict subset means "add/refresh only these
    # classes" — finalize MERGES them into existing annotations, so other-class
    # boxes (e.g. your reviewed bee boxes) are preserved. Empty/all = every class.
    proj_classes = project.classes or []
    posted = set(request.POST.getlist("target_labels"))
    target_labels = [c for c in proj_classes if c in posted] if posted else list(proj_classes)
    return PreAnnotationTask.objects.create(
        user=request.user, project=project, video=video,
        labeler=labeler, custom_model_key=custom_model_key,
        params={
            "sample_interval": sample_interval, "max_frames": max_frames,
            "confidence": confidence, "selection": selection,
            "nms_iou": nms_iou, "max_detections": max_detections,
            "target_labels": target_labels,
        },
    )


class PreAnnotateAllView(LoginRequiredMixin, View):
    """Run AI pre-annotation on all videos in a project, or on the subset
    checked in the video list (video_ids)."""

    def post(self, request, pk):
        from django.shortcuts import redirect
        from django.contrib import messages

        # Spends GPU, on every sampled frame.
        project = get_object_or_404(
            AnnotationProject.manageable(request.user), pk=pk)
        videos = project.videos.all()
        video_ids = request.POST.getlist("video_ids")
        if video_ids:
            videos = videos.filter(pk__in=video_ids)

        if not videos.exists():
            messages.warning(request, "No videos in this project.")
            return redirect("annotations:detail", pk=pk)

        # One durable QUEUED task per video. The drain below promotes only as
        # many as the concurrency cap allows and the reconciler picks up the
        # rest, so a 200-video project doesn't fire 200 GPU invocations at once.
        count = 0
        labeler = "yolo"
        for video in videos:
            if not video.storage_key:
                continue
            task = _create_preannotation_task(request, project, video)
            labeler = task.labeler
            count += 1
        started = drain_preannotation_queue()

        engine = "SAM 3" if labeler == "sam3" else "AI"
        queued = max(0, count - started)
        msg = (f"{engine} pre-annotation started for {started} video(s). Runs in "
               "the background — refresh to see progress.")
        if queued:
            msg += f" {queued} more queued; they start as slots free up."
        messages.info(request, msg)
        return redirect("annotations:detail", pk=pk)


class CancelPreAnnotationView(LoginRequiredMixin, View):
    """Cancel in-flight pre-annotation. POST task_id to cancel one, or nothing to
    cancel all active tasks for the project. Marks tasks CANCELLED so the
    reconciler (which only polls QUEUED/PROCESSING) stops finalizing them.
    Tasks not yet sent to the GPU are truly skipped (the spawn worker re-checks
    status before invoking); invocations already sent can't be recalled — they
    finish in the background and their result is discarded, same as cancelling
    a run on the Processing page."""

    def post(self, request, pk):
        from django.shortcuts import redirect
        from django.contrib import messages
        from django.utils import timezone

        from .models import PreAnnotationTask

        # Cancelling other people's GPU work.
        project = get_object_or_404(
            AnnotationProject.manageable(request.user), pk=pk)
        active = PreAnnotationTask.objects.filter(
            project=project, user=request.user,
            status__in=[PreAnnotationTask.Status.QUEUED, PreAnnotationTask.Status.PROCESSING],
        )
        task_id = request.POST.get("task_id")
        if task_id:
            active = active.filter(pk=task_id)
        n = active.update(status=PreAnnotationTask.Status.CANCELLED,
                          error_message="Cancelled by user.", completed_at=timezone.now())
        messages.info(request, f"Cancelled {n} pre-annotation task(s)." if n
                      else "No active pre-annotation to cancel.")
        return redirect("annotations:detail", pk=pk)


class FrameImageView(LoginRequiredMixin, View):
    """Return a JPG image of a specific video frame.

    First tries to serve a pre-saved frame from S3 (uploaded during
    pre-annotation). Falls back to extracting from the video if needed.
    """

    def get(self, request, pk):
        video_id = request.GET.get("video")
        frame_number = int(request.GET.get("frame", 0))
        draw_boxes = request.GET.get("boxes", "false") == "true"

        # This is the hinge. Check ownership and shares silently do not work —
        # a collaborator sees an empty editor. Check nothing and every project's
        # frames leak. The rule is: the project must be readable by this user,
        # and the clip must be in it.
        project = get_object_or_404(AnnotationProject.accessible(request.user), pk=pk)
        video = get_object_or_404(project.videos.all(), pk=video_id)

        try:
            ann = Annotation.objects.get(project=project, video=video,
                                         frame_number=frame_number)
        except Annotation.DoesNotExist:
            ann = None

        # Prefer the pre-saved JPEG, but FALL BACK to extracting from the video
        # when that object is missing/broken — otherwise a stale frame_image_path
        # renders permanently blank even though the video is available.
        if ann and ann.frame_image_path:
            resp = self._serve_from_storage(ann.frame_image_path, ann if draw_boxes else None)
            if resp is not None:
                return resp
            logger.warning("FrameImageView: saved frame %s missing — extracting from video",
                           ann.frame_image_path)

        return self._extract_from_video(video, frame_number)

    def _serve_from_storage(self, blob_path, ann_for_boxes=None):
        """Serve a pre-saved JPEG frame from the processed S3 bucket.
        Returns None (not a 404) on failure so the caller can fall back."""
        try:
            import io
            from config.storage import get_s3_client

            buf = io.BytesIO()
            get_s3_client().download_to_stream("processed", blob_path, buf)
            data = buf.getvalue()
            if not data:
                return None

            if ann_for_boxes and ann_for_boxes.boxes:
                import cv2
                import numpy as np
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 255, 255), (255, 0, 255)]
                for box in ann_for_boxes.boxes:
                    x, y, w, h = int(box["x"]), int(box["y"]), int(box["w"]), int(box["h"])
                    color = colors[box.get("class_id", 0) % len(colors)]
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    label = box.get("class", "")
                    cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                _, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                data = encoded.tobytes()

            response = HttpResponse(data, content_type="image/jpeg")
            response["Cache-Control"] = "public, max-age=3600"
            return response
        except Exception as e:
            logger.warning("FrameImageView S3 miss for %s: %s", blob_path, e)
            return None  # let the caller fall back to video extraction

    def _extract_from_video(self, video, frame_number):
        """Fallback: download video and extract frame (slow)."""
        blob_path = video.storage_key
        if not blob_path or blob_path.startswith("s3://"):
            return HttpResponse(status=404)

        try:
            import cv2
            import tempfile
            import os
            from config.storage import get_s3_client

            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tmp.close()
            get_s3_client().download_file("raw-videos", blob_path, tmp.name)

            cap = cv2.VideoCapture(tmp.name)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            cap.release()
            os.unlink(tmp.name)

            if not ret:
                return HttpResponse(status=404)

            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return HttpResponse(buf.tobytes(), content_type="image/jpeg")
        except Exception as e:
            logger.error("FrameImageView extract error: %s", e, exc_info=True)
            return HttpResponse(status=500)


class ExportProjectView(LoginRequiredMixin, View):
    """Export YOLO dataset with images extracted from videos."""

    def get(self, request, pk):
        # A viewer may export: the dataset is what sharing is for.
        project = get_object_or_404(
            AnnotationProject.accessible(request.user), pk=pk)
        # Same rule as the training payload: un-annotated sampled frames are
        # navigation placeholders, not labelled data.
        annotations = (project.annotations.exclude(sampled_only=True)
                       .select_related("video").order_by("video", "frame_number"))

        if not annotations.exists():
            from django.contrib import messages
            messages.warning(request, "No annotations to export.")
            return HttpResponse(status=302, headers={"Location": f"/annotations/{pk}/"})

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            # Write data.yaml
            class_lines = "\n".join(
                f"  {i}: {cls}" for i, cls in enumerate(project.classes)
            )
            data_yaml = (
                f"path: .\n"
                f"train: images/train\n"
                f"val: images/val\n"
                f"nc: {len(project.classes)}\n"
                f"names:\n{class_lines}\n"
            )
            zf.writestr("data.yaml", data_yaml)

            from config.storage import get_s3_client
            s3 = get_s3_client()
            video_cache = {}  # video_pk -> (cv2.VideoCapture, tmp_path)

            # Deterministic split matching the training container's (sorted tail
            # → val). val_percent from ?val_percent= (default 20, = the training
            # job default) so "what you export" == "what training splits".
            try:
                val_percent = int(request.GET.get("val_percent", 20))
            except (TypeError, ValueError):
                val_percent = 20
            val_percent = max(0, min(90, val_percent))
            ann_list = list(annotations)
            split_idx = len(ann_list) - max(1, len(ann_list) * val_percent // 100) if ann_list else 0

            for i, ann in enumerate(ann_list):
                split = "val" if i >= split_idx else "train"
                base_name = f"{ann.video.title}_f{ann.frame_number:06d}"
                zf.writestr(f"labels/{split}/{base_name}.txt", ann.to_yolo_format())

                # Prefer the pre-saved frame JPEG (processed bucket) — the
                # tracker/pre-annotator already extracted it. Re-extract from
                # video only as a fallback.
                img_bytes = None
                if ann.frame_image_path:
                    try:
                        # `io` is imported at module level. Re-importing it here
                        # made the name local to this whole method, so the
                        # io.BytesIO() forty lines above raised UnboundLocalError
                        # and export failed for everyone, every time.
                        b = io.BytesIO()
                        s3.download_to_stream("processed", ann.frame_image_path, b)
                        img_bytes = b.getvalue()
                    except Exception as e:
                        logger.warning("export: saved frame %s missing (%s) — re-extracting",
                                       ann.frame_image_path, e)
                if img_bytes is None:
                    try:
                        if ann.video.pk not in video_cache:
                            blob_path = ann.video.storage_key
                            if blob_path and not blob_path.startswith("s3://"):
                                import tempfile, cv2
                                tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                                tmp.close()
                                s3.download_file("raw-videos", blob_path, tmp.name)
                                video_cache[ann.video.pk] = (cv2.VideoCapture(tmp.name), tmp.name)
                        if ann.video.pk in video_cache:
                            import cv2
                            cap, _ = video_cache[ann.video.pk]
                            cap.set(cv2.CAP_PROP_POS_FRAMES, ann.frame_number)
                            ret, frame = cap.read()
                            if ret:
                                _, img_buf = cv2.imencode(".jpg", frame)
                                img_bytes = img_buf.tobytes()
                    except Exception as e:
                        logger.error("Failed to extract frame %d from video %s: %s",
                                     ann.frame_number, ann.video.pk, e)
                if img_bytes is not None:
                    zf.writestr(f"images/{split}/{base_name}.jpg", img_bytes)

            # Cleanup video captures
            for cap, tmp_path in video_cache.values():
                cap.release()
                import os
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        buf.seek(0)
        response = HttpResponse(buf.read(), content_type="application/zip")
        response["Content-Disposition"] = (
            f'attachment; filename="{project.name}_yolo_dataset.zip"'
        )
        return response


class PreAnnotateFrameView(LoginRequiredMixin, View):
    """Per-frame pre-annotate: run the SAM 3 detector on just the current frame
    (async on the GPU endpoint) and return a task id to poll. suggest_only — the
    result is shown as suggestions; nothing is written until the human Saves."""

    def post(self, request, pk):
        from django.conf import settings
        from .models import PreAnnotationTask

        # One frame in the editor: an annotation aid, bounded, and how
        # labelling is actually done.
        project = get_object_or_404(
            AnnotationProject.annotatable(request.user), pk=pk)
        if not getattr(settings, "SAGEMAKER_SAM3_ENDPOINT_NAME", ""):
            return JsonResponse({"error": "SAM 3 endpoint isn't configured on this server."},
                                status=400)
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)
        video = get_object_or_404(project.videos, pk=data.get("video_id"))
        frame_number = int(data.get("frame_number", 0))

        # Which project classes to detect (default all), plus any free-text custom
        # labels. A new custom label is added to the project's classes so its boxes
        # persist and it's available everywhere (Active Class, training/export).
        target = [c for c in (project.classes or []) if c in (data.get("target_labels") or [])]
        if not target and not data.get("custom_labels"):
            target = list(project.classes or [])
        custom = [str(c).strip() for c in (data.get("custom_labels") or []) if str(c).strip()]
        added = [c for c in custom if c not in (project.classes or [])]
        if added:
            project.classes = (project.classes or []) + added
            project.save(update_fields=["classes"])
        prompt_classes = target + custom

        task = PreAnnotationTask.objects.create(
            user=request.user, project=project, video=video, labeler="sam3",
            params={
                "frame_numbers": [frame_number], "max_frames": 1,
                "confidence": settings.PREANNOTATE_CONFIDENCE,
                "selection": "uniform", "nms_iou": 0.5, "max_detections": 100,
                "target_labels": prompt_classes or list(project.classes or []),
                "suggest_only": True,
            },
        )
        spawn_preannotation_async(task.pk)
        return JsonResponse({"success": True, "task_id": task.pk})


class PreAnnotateFrameStatusView(LoginRequiredMixin, View):
    """Poll a per-frame pre-annotate task. Reads the async S3 output directly and
    returns the frame's boxes when ready (no DB write — that happens on Save)."""

    def get(self, request, pk):
        import boto3
        from urllib.parse import urlparse
        from django.conf import settings
        from django.utils import timezone
        from botocore.exceptions import ClientError

        from .models import PreAnnotationTask

        # Polling the above.
        project = get_object_or_404(
            AnnotationProject.annotatable(request.user), pk=pk)
        task = get_object_or_404(PreAnnotationTask, pk=request.GET.get("task"),
                                 project=project, user=request.user)
        if task.status == PreAnnotationTask.Status.FAILED:
            return JsonResponse({"status": "failed", "error": task.error_message})

        frame_number = ((task.params or {}).get("frame_numbers") or [0])[0]
        s3 = boto3.client("s3", region_name=settings.AWS_REGION)

        if task.output_uri:
            out = urlparse(task.output_uri)
            try:
                body = s3.get_object(Bucket=out.netloc, Key=out.path.lstrip("/"))["Body"].read()
                result = json.loads(body)
                PreAnnotationTask.objects.filter(pk=task.pk).update(
                    status=PreAnnotationTask.Status.COMPLETED, completed_at=timezone.now())
                return JsonResponse({"status": "completed",
                                     "boxes": self._frame_boxes(result, frame_number, project),
                                     "classes": project.classes or []})
            except ClientError as e:
                if e.response.get("Error", {}).get("Code", "") not in ("NoSuchKey", "404", "NotFound"):
                    return JsonResponse({"status": "processing"})
                if task.failure_uri:
                    f = urlparse(task.failure_uri)
                    try:
                        s3.get_object(Bucket=f.netloc, Key=f.path.lstrip("/"))
                        PreAnnotationTask.objects.filter(pk=task.pk).update(
                            status=PreAnnotationTask.Status.FAILED,
                            error_message="SAM 3 endpoint failure", completed_at=timezone.now())
                        return JsonResponse({"status": "failed", "error": "SAM 3 endpoint failure"})
                    except ClientError:
                        pass
        return JsonResponse({"status": "processing"})

    @staticmethod
    def _frame_boxes(result, frame_number, project):
        """Boxes for the target frame, with class ids re-mapped to project order."""
        classes = project.classes or []
        cid = {c: i for i, c in enumerate(classes)}
        for fd in (result.get("frames") or []):
            if fd.get("frame_number") == frame_number:
                out = []
                for b in fd.get("boxes", []):
                    name = b.get("class")
                    if name in cid:
                        out.append({**b, "class_id": cid[name]})
                return out
        return []




# ── Frame sampling (decoupled from SAM 3) ─────────────────────────────────────

class SampleFramesView(LoginRequiredMixin, View):
    """Sample frames from the project's videos so they can be annotated.

    The cheap half of what used to be one fused SAM 3 call: this only decodes and
    uploads frames (web CPU, no GPU). Auto-labelling stays a separate, opt-in
    action — per project or, in the editor, per frame.
    """

    def post(self, request, pk):
        from django.contrib import messages
        from django.shortcuts import redirect

        from . import sampling
        from .models import FrameSamplingTask

        # Decodes every clip; a labeller must not start it.
        project = get_object_or_404(
            AnnotationProject.manageable(request.user), pk=pk)
        videos = project.videos.all()
        video_ids = request.POST.getlist("video_ids") or (
            [request.POST["video_id"]] if request.POST.get("video_id") else []
        )
        if video_ids:
            videos = videos.filter(pk__in=video_ids)
        videos = list(videos)
        if not videos:
            messages.warning(request, "No videos selected to sample.")
            return redirect("annotations:detail", pk=pk)

        # The sampling panel's fields carry an "s_" prefix: the same form also
        # feeds Auto-label, which reads its own sample_interval/max_frames.
        # Unprefixed keys still work for other callers (the editor, the API).
        def _field(key):
            v = request.POST.get("s_" + key)
            return v if v is not None else request.POST.get(key)
        params = sampling.clamp_params({
            key: _field(key)
            for key in ("method", "sample_interval", "max_frames", "min_gap_s", "roi", "replace")
            if _field(key) is not None
        })
        for video in videos:
            task = FrameSamplingTask.objects.create(
                user=request.user, project=project, video=video, params=params,
            )
            sampling.spawn_sampling_async(task.pk)

        how = (f"the {params['max_frames']} most active frame(s), at least "
               f"{params['min_gap_s']:g}s apart,"
               if params["method"] == "motion" else
               f"up to {params['max_frames']} frame(s), every {params['sample_interval']} frames,")
        messages.info(
            request,
            f"Sampling {how} from {len(videos)} video(s). "
            "Refresh to see them appear — no GPU is used.",
        )
        return redirect("annotations:detail", pk=pk)


class CancelSamplingView(LoginRequiredMixin, View):
    """Cancel this project's queued/running frame-sampling tasks."""

    def post(self, request, pk):
        from django.contrib import messages
        from django.shortcuts import redirect
        from django.utils import timezone

        from .models import FrameSamplingTask

        # Cancelling other people's work.
        project = get_object_or_404(
            AnnotationProject.manageable(request.user), pk=pk)
        cancelled = FrameSamplingTask.objects.filter(
            project=project,
            status__in=[FrameSamplingTask.Status.QUEUED,
                        FrameSamplingTask.Status.PROCESSING],
        ).update(status=FrameSamplingTask.Status.CANCELLED,
                 completed_at=timezone.now())
        messages.info(request, f"Cancelled {cancelled} sampling task(s).")
        return redirect("annotations:detail", pk=pk)
