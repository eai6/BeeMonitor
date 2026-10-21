from django import forms
from django.contrib.auth.forms import PasswordResetForm, UserCreationForm
from django.contrib.auth.models import User
from django.template import loader

from .models import Coupon


class UserRegistrationForm(UserCreationForm):
    email = forms.EmailField(required=True)

    class Meta:
        model = User
        fields = ("username", "email", "password1", "password2")

    def clean_email(self):
        # Django's User.email isn't unique; enforce it (case-insensitively) here
        # so we don't create duplicates that later break password reset.
        email = self.cleaned_data["email"]
        if User.objects.filter(email__iexact=email).exists():
            raise forms.ValidationError(
                "An account with this email already exists. Try logging in or "
                "resetting your password.")
        return email


class LoggedPasswordResetForm(PasswordResetForm):
    """Django's reset form, sending through emails.send so every attempt is
    logged with SES's answer. The page still looks the same either way."""

    def send_mail(self, subject_template_name, email_template_name, context,
                  from_email, to_email, html_email_template_name=None):
        from .emails import send

        subject = "".join(loader.render_to_string(subject_template_name, context).splitlines())
        body = loader.render_to_string(email_template_name, context)
        html = (loader.render_to_string(html_email_template_name, context)
                if html_email_template_name else None)
        send("password_reset", to_email, subject, body, html=html, user=context.get("user"))


class CouponForm(forms.ModelForm):
    class Meta:
        model = Coupon
        fields = [
            "code", "coupon_type", "credits_amount",
            "upgrade_tier", "upgrade_duration_days",
            "max_redemptions", "is_active", "expires_at",
        ]
        widgets = {
            "expires_at": forms.DateTimeInput(attrs={"type": "datetime-local"}),
        }
