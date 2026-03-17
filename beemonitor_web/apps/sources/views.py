from django.contrib import messages
from django.contrib.auth.mixins import LoginRequiredMixin
from django.urls import reverse_lazy
from django.views.generic import CreateView, ListView

from .forms import SourceCreateForm
from .models import DataSource


class SourceListView(LoginRequiredMixin, ListView):
    template_name = "sources/list.html"
    context_object_name = "sources"

    def get_queryset(self):
        return DataSource.objects.filter(user=self.request.user)


class SourceCreateView(LoginRequiredMixin, CreateView):
    template_name = "sources/add.html"
    form_class = SourceCreateForm
    success_url = reverse_lazy("sources:list")

    def form_valid(self, form):
        form.instance.user = self.request.user
        form.instance.config_encrypted = b""  # placeholder
        messages.success(self.request, "Data source added.")
        return super().form_valid(form)
