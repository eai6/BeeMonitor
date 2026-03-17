from django.contrib import messages
from django.contrib.auth.mixins import LoginRequiredMixin
from django.shortcuts import redirect
from django.views.generic import TemplateView

from apps.accounts.models import APIKey
from apps.developer.models import UsageLog


class DeveloperPortalView(LoginRequiredMixin, TemplateView):
    template_name = "developer/index.html"

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        user = self.request.user
        ctx["api_keys"] = APIKey.objects.filter(user=user).order_by("-created_at")
        ctx["recent_usage"] = UsageLog.objects.filter(user=user)[:20]
        ctx["total_requests"] = UsageLog.objects.filter(user=user).count()
        ctx["active_keys"] = APIKey.objects.filter(user=user, is_active=True).count()
        ctx["key_type_choices"] = APIKey.KeyType.choices
        # Pass the newly created raw key if present in session
        ctx["new_raw_key"] = self.request.session.pop("new_raw_key", None)
        return ctx

    def post(self, request, *args, **kwargs):
        action = request.POST.get("action")
        if action == "create_key":
            name = request.POST.get("key_name", "").strip()
            key_type = request.POST.get("key_type", "test")
            if name:
                instance, raw_key = APIKey.create_key(
                    user=request.user, name=name, key_type=key_type
                )
                request.session["new_raw_key"] = raw_key
                messages.success(request, f"API key '{name}' created. Copy it now -- it will not be shown again.")
            else:
                messages.error(request, "Key name is required.")
        elif action == "revoke_key":
            key_id = request.POST.get("key_id")
            APIKey.objects.filter(pk=key_id, user=request.user).update(is_active=False)
            messages.success(request, "API key revoked.")
        return redirect("developer:index")
