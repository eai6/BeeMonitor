"""Cloud storage: connect an S3 bucket, browse it, import clips (memory/54).

Imported clips either stay in the user's bucket (copied only while a run
reads them) or are copied into their BeeMonitor storage, in the background
(apps/sources/copying.py). Everything that touches the bucket goes through
apps/sources/s3.py, which never uses BeeMonitor's own AWS role.
"""

import logging

from django.contrib import messages
from django.contrib.auth.mixins import LoginRequiredMixin
from django.db.models import Count, Q
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse_lazy
from django.utils import timezone
from django.views import View
from django.views.generic import FormView, ListView

from apps.videos.models import Site, Video

from . import copying, s3
from .crypto import decrypt as _decrypt_credentials  # noqa: F401  (older imports use this name)
from .crypto import decrypt, encrypt, fernet as _get_fernet
from .forms import SourceCreateForm
from .models import DataSource, SourceCopy

logger = logging.getLogger(__name__)

# Videos per import at most, so one click can't queue a whole archive.
MAX_IMPORT = 2000


class SourceListView(LoginRequiredMixin, ListView):
    template_name = "sources/list.html"
    context_object_name = "sources"

    def get_queryset(self):
        return DataSource.objects.filter(user=self.request.user).annotate(
            clips=Count("videos", distinct=True),
            copying=Count("copies", filter=Q(copies__purpose=SourceCopy.Purpose.KEEP,
                                              copies__status__in=["queued", "running"]), distinct=True),
        )

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        ctx["recent_copies"] = (SourceCopy.objects.filter(video__user=self.request.user,
                                                          purpose=SourceCopy.Purpose.KEEP)
                                .select_related("video").order_by("-created_at")[:15])
        return ctx


class SourceCreateView(LoginRequiredMixin, FormView):
    template_name = "sources/add.html"
    form_class = SourceCreateForm
    success_url = reverse_lazy("sources:list")

    def form_valid(self, form):
        credentials = form.get_credentials()
        try:
            _get_fernet()  # fails early when the key is missing
        except ValueError:
            messages.error(self.request, "Cloud storage isn't available right now.")
            logger.error("BEEMONITOR_CREDENTIAL_KEY is not set")
            return self.form_invalid(form)
        # Tested before it's saved: a source that can't read its bucket is no use.
        try:
            message = s3.test(credentials)
        except s3.SourceError as e:
            form.add_error(None, str(e))
            return self.form_invalid(form)
        source = DataSource.objects.create(
            user=self.request.user, name=form.cleaned_data["name"],
            source_type=DataSource.SourceType.AWS_S3, config_encrypted=encrypt(credentials),
            is_connected=True)
        logger.info("AUDIT: user %s connected data source %s (s3, %s)",
                    self.request.user.pk, source.pk, credentials.get("bucket"))
        messages.success(self.request, f"{message} Browse it to import clips.")
        return redirect("sources:browse", pk=source.pk)


class SourceTestView(LoginRequiredMixin, View):
    def post(self, request, pk):
        source = get_object_or_404(DataSource, pk=pk, user=request.user)
        try:
            messages.success(request, s3.test(decrypt(source)))
            ok = True
        except s3.SourceError as e:
            messages.error(request, str(e))
            ok = False
        DataSource.objects.filter(pk=pk).update(is_connected=ok)
        return redirect("sources:list")


class SourceDeleteView(LoginRequiredMixin, View):
    def post(self, request, pk):
        source = get_object_or_404(DataSource, pk=pk, user=request.user)
        name = source.name
        source.delete()
        logger.info("AUDIT: user %s deleted data source %s", request.user.pk, pk)
        messages.success(request, f"Removed {name}. Clips copied into BeeMonitor stay; clips kept in that "
                                  "bucket can no longer be analysed.")
        return redirect("sources:list")


def _imported(user, bucket):
    """{remote key: Video} already imported from this bucket."""
    out = {}
    for v in Video.objects.filter(user=user, metadata__remote_bucket=bucket).only("pk", "storage_key", "metadata"):
        out[(v.metadata or {}).get("remote_key")] = v
    return out


class SourceBrowseView(LoginRequiredMixin, View):
    """One folder of a bucket at a time; pick clips and import them."""

    def get(self, request, pk):
        source = get_object_or_404(DataSource, pk=pk, user=request.user)
        path = request.GET.get("path", "")
        if ".." in path.split("/"):
            path = ""
        ctx = {"source": source, "path": path, "error": "", "folders": [], "files": [],
               "next_token": "", "sites": Site.objects.filter(user=request.user)}
        crumbs, acc = [], ""
        for part in [p for p in path.split("/") if p]:
            acc += part + "/"
            crumbs.append({"name": part, "path": acc})
        ctx["crumbs"] = crumbs
        try:
            creds = decrypt(source)
            ctx["bucket"], ctx["prefix"] = creds.get("bucket", ""), creds.get("prefix", "")
            page = s3.browse(creds, path, request.GET.get("token", ""))
            done = _imported(request.user, creds["bucket"])
            copies = {c.video_id: c for c in SourceCopy.objects.filter(
                video__in=[v for v in done.values()], purpose=SourceCopy.Purpose.KEEP).order_by("created_at")}
            for f in page["files"]:
                video = done.get(f["key"])
                f["video"] = video
                f["copy"] = copies.get(video.pk) if video else None
                f["copied"] = bool(video and not video.storage_key.startswith("s3://"))
            ctx.update(page)
        except s3.SourceError as e:
            ctx["error"] = str(e)
        return render(request, "sources/browse.html", ctx)


class SourceSyncView(LoginRequiredMixin, View):
    """Import the chosen clips (or every video under a folder)."""

    def post(self, request, pk):
        source = get_object_or_404(DataSource, pk=pk, user=request.user)
        path = request.POST.get("path", "")
        keep_here = request.POST.get("where", "bucket") == "copy"
        site = None
        if request.POST.get("site"):
            site = Site.objects.filter(user=request.user, pk=request.POST["site"]).first()
        back = redirect(f"{reverse_lazy('sources:browse', args=[pk])}?path={path}")
        try:
            creds = decrypt(source)
            if request.POST.get("import_all") == "true":
                files = s3.list_videos(creds, path, limit=MAX_IMPORT)
            else:
                keys = request.POST.getlist("selected_files")[:MAX_IMPORT]
                base = creds.get("prefix") or ""
                # Only keys under this source's folder: a forged key can't
                # reach elsewhere in the bucket.
                files = [{"key": k, "name": k.rsplit("/", 1)[-1], "size": s3.head(creds, k)}
                         for k in keys if k.startswith(base) and k.lower().endswith(s3.VIDEO_EXTENSIONS)]
        except s3.SourceError as e:
            messages.error(request, str(e))
            return back
        if not files:
            messages.warning(request, "No videos selected.")
            return back

        bucket = creds["bucket"]
        done = _imported(request.user, bucket)
        imported = skipped = 0
        for f in files:
            if f["key"] in done:
                skipped += 1
                continue
            site_name, recorded_at = Video.parse_timestamp_from_filename(f["name"])
            video = Video.objects.create(
                user=request.user, source=source, site=site,
                site_name=site.name if site else (site_name or ""),
                title=f["name"].rsplit(".", 1)[0], storage_key=s3.remote_uri(bucket, f["key"]),
                file_size_bytes=f["size"] or 0, status=Video.Status.READY, recorded_at=recorded_at,
                metadata={"source_id": source.pk, "source_type": source.source_type,
                          "remote_key": f["key"], "remote_bucket": bucket})
            copying.queue(video, SourceCopy.Purpose.KEEP if keep_here else SourceCopy.Purpose.MEASURE)
            imported += 1
        DataSource.objects.filter(pk=pk).update(last_synced_at=timezone.now())
        logger.info("AUDIT: user %s imported %d clips from source %s (%s)", request.user.pk, imported, pk,
                    "copy" if keep_here else "keep in bucket")
        msg = f"Imported {imported} clip{'s' if imported != 1 else ''}"
        msg += " — copying into your BeeMonitor storage in the background." if keep_here else \
            " — they stay in your bucket and are read from it when analysed."
        if skipped:
            msg += f" {skipped} were already imported."
        messages.success(request, msg)
        return back


class CopyIntoBeeMonitorView(LoginRequiredMixin, View):
    """Switch a clip kept in the user's bucket to a full copy in BeeMonitor."""

    def post(self, request, video_pk):
        video = get_object_or_404(Video.objects.filter(user=request.user), pk=video_pk)
        if not video.storage_key.startswith("s3://") or not video.source_id:
            messages.info(request, "This clip is already in your BeeMonitor storage.")
        else:
            copying.queue(video, SourceCopy.Purpose.KEEP)
            messages.success(request, "Copying into your BeeMonitor storage in the background.")
        return redirect("videos:detail", pk=video.pk)
