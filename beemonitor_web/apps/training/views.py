from django.contrib.auth.mixins import LoginRequiredMixin
from django.urls import reverse_lazy
from django.views.generic import CreateView, DetailView, ListView

from .forms import TrainingCreateForm
from .models import CustomModel, TrainingJob


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
