from django.contrib import messages
from django.contrib.auth import login
from django.contrib.auth.views import LoginView as AuthLoginView
from django.contrib.auth.views import LogoutView as AuthLogoutView
from django.urls import reverse_lazy
from django.views.generic import CreateView

from .forms import UserRegistrationForm


class LoginView(AuthLoginView):
    template_name = "accounts/login.html"
    redirect_authenticated_user = True

    def get_success_url(self):
        return reverse_lazy("dashboard:dashboard")


class LogoutView(AuthLogoutView):
    next_page = reverse_lazy("accounts:login")


class RegisterView(CreateView):
    template_name = "accounts/register.html"
    form_class = UserRegistrationForm
    success_url = reverse_lazy("dashboard:dashboard")

    def form_valid(self, form):
        response = super().form_valid(form)
        login(self.request, self.object)
        messages.success(self.request, "Account created successfully.")
        return response
