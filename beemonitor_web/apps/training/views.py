import logging
import uuid

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.mixins import LoginRequiredMixin
from django.shortcuts import redirect
from django.urls import reverse_lazy
from django.views import View
from django.views.generic import CreateView, DetailView, FormView, ListView

from .forms import TrainingCreateForm, ModelUploadForm
from .models import CustomModel, TrainingJob

logger = logging.getLogger(__name__)


class TrainingListView(LoginRequiredMixin, ListView):
    model = TrainingJob
    template_name = "training/list.html"
    context_object_name = "jobs"
    paginate_by = 20

    def get_queryset(self):
        return TrainingJob.objects.filter(user=self.request.user).select_related("project")


class TrainingCreateView(LoginRequiredMixin, CreateView):
    model = TrainingJob
    form_class = TrainingCreateForm
    template_name = "training/new.html"

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs["user"] = self.request.user
        return kwargs

    def form_valid(self, form):
        form.instance.user = self.request.user
        return super().form_valid(form)

    def get_success_url(self):
        return reverse_lazy("training:detail", kwargs={"pk": self.object.pk})


class TrainingDetailView(LoginRequiredMixin, DetailView):
    model = TrainingJob
    template_name = "training/detail.html"
    context_object_name = "job"

    def get_queryset(self):
        return TrainingJob.objects.filter(user=self.request.user).select_related("project")

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        try:
            ctx["custom_model"] = self.object.custom_model
        except CustomModel.DoesNotExist:
            ctx["custom_model"] = None
        return ctx


class CustomModelListView(LoginRequiredMixin, ListView):
    model = CustomModel
    template_name = "training/models.html"
    context_object_name = "models_list"
    paginate_by = 20

    def get_queryset(self):
        return CustomModel.objects.filter(user=self.request.user).select_related("training_job")


class CustomModelDetailView(LoginRequiredMixin, DetailView):
    model = CustomModel
    template_name = "training/model_detail.html"
    context_object_name = "model"

    def get_queryset(self):
        return CustomModel.objects.filter(user=self.request.user).select_related("training_job")


class UploadModelView(LoginRequiredMixin, FormView):
    """Upload a custom .pt model file."""
    template_name = "training/upload_model.html"
    form_class = ModelUploadForm

    def form_valid(self, form):
        model_file = self.request.FILES["model_file"]
        name = form.cleaned_data["name"]
        model_type = form.cleaned_data["model_type"]
        classes_text = form.cleaned_data.get("classes", "")
        classes = [c.strip() for c in classes_text.split(",") if c.strip()] if classes_text else []

        # Upload to Azure Blob Storage
        upload_id = uuid.uuid4().hex[:12]
        blob_path = f"custom/{self.request.user.pk}/{upload_id}/{model_file.name}"

        try:
            from azure.storage.blob import BlobServiceClient
            conn_str = settings.AZURE_STORAGE_CONNECTION_STRING
            if conn_str:
                service = BlobServiceClient.from_connection_string(conn_str)
                blob = service.get_blob_client("models", blob_path)
                blob.upload_blob(model_file, overwrite=True)
        except Exception as e:
            messages.error(self.request, f"Upload failed: {e}")
            return self.form_invalid(form)

        CustomModel.objects.create(
            user=self.request.user,
            name=name,
            model_type=model_type,
            base_model="uploaded",
            azure_model_path=blob_path,
            classes=classes,
            metrics={"source": "uploaded", "file_size": model_file.size},
            is_active=True,
        )

        messages.success(self.request, f"Model '{name}' uploaded successfully.")
        return redirect("training:models")
