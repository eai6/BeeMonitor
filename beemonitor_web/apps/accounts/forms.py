from django import forms
from django.contrib.auth.forms import PasswordResetForm, UserCreationForm
from django.contrib.auth.models import User
from django.template import loader

from .models import AccountRequest, Coupon


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


class ProfileForm(forms.Form):
    """Name and organization. The username is not editable: it is how people
    sign in and how they are told apart in logs and on the admin dashboard."""

    first_name = forms.CharField(max_length=150, required=False)
    last_name = forms.CharField(max_length=150, required=False)
    organization = forms.CharField(max_length=200, required=False)


class EmailChangeForm(forms.Form):
    """Change the account email. Needs the current password, and the address
    must not belong to another account (password reset finds users by email).

    No confirmation email to the new address yet: while SES is in its sandbox
    it wouldn't arrive. The old address is told instead (memory/50)."""

    new_email = forms.EmailField(label="New email")
    current_password = forms.CharField(widget=forms.PasswordInput, label="Current password")

    def __init__(self, user, *args, **kwargs):
        self.user = user
        super().__init__(*args, **kwargs)

    def clean_current_password(self):
        pw = self.cleaned_data["current_password"]
        if not self.user.check_password(pw):
            raise forms.ValidationError("That isn't your current password.")
        return pw

    def clean_new_email(self):
        email = self.cleaned_data["new_email"].strip()
        if email.lower() == (self.user.email or "").lower():
            raise forms.ValidationError("That is already your email.")
        if User.objects.filter(email__iexact=email).exclude(pk=self.user.pk).exists():
            raise forms.ValidationError("Another account already uses this email.")
        return email


class LoggedPasswordResetForm(PasswordResetForm):
    """Django's reset form, sending through emails.send so every attempt is
    logged with SES's answer. The page still looks the same either way."""

    def save(self, *args, domain_override=None, **kwargs):
        # The link's domain is the site's own, never the request's Host header:
        # a forged Host would otherwise mail the victim a link to another site
        # carrying their reset token.
        from django.conf import settings
        return super().save(*args, domain_override=settings.SITE_DOMAIN or domain_override, **kwargs)

    def send_mail(self, subject_template_name, email_template_name, context,
                  from_email, to_email, html_email_template_name=None):
        from datetime import timedelta

        from .emails import send
        from .throttle import allow

        # Five a day per address: the form is public, so without this anyone
        # could fill a stranger's inbox (and spend the SES quota).
        if not allow(f"reset:{to_email.lower()}", 5, timedelta(days=1)):
            return
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
            "max_redemptions", "is_active", "expires_at", "note",
        ]
        widgets = {
            "expires_at": forms.DateTimeInput(attrs={"type": "datetime-local"}),
        }

    def clean(self):
        data = super().clean()
        if data.get("coupon_type") == Coupon.CouponType.CREDITS:
            data["upgrade_tier"] = ""
            if not data.get("credits_amount"):
                self.add_error("credits_amount", "A credits code needs a number of credits.")
        elif data.get("coupon_type") == Coupon.CouponType.TIER_UPGRADE:
            data["credits_amount"] = 0
            if not data.get("upgrade_tier"):
                self.add_error("upgrade_tier", "Choose the plan this code gives.")
        return data


class AccountRequestForm(forms.ModelForm):
    """One request for credits, Pro or longer storage. One open request of each
    kind at a time; Pro accounts don't ask for Pro."""

    class Meta:
        model = AccountRequest
        fields = ["kind", "credits", "until", "organization", "contact_email", "note"]

    def __init__(self, user, *args, **kwargs):
        self.user = user
        super().__init__(*args, **kwargs)

    def clean(self):
        from django.utils import timezone

        data = super().clean()
        kind = data.get("kind")
        if kind and AccountRequest.objects.filter(user=self.user, kind=kind,
                                                  status=AccountRequest.Status.PENDING).exists():
            raise forms.ValidationError("You already have one of these waiting for approval.")
        if kind == AccountRequest.Kind.CREDITS:
            credits = data.get("credits")
            if credits is None or credits <= 0 or credits > 100000:
                self.add_error("credits", "Ask for between 1 and 100,000 credits.")
            data["until"] = None
        elif kind == AccountRequest.Kind.STORAGE:
            until = data.get("until")
            if until is None or until <= timezone.localdate():
                self.add_error("until", "Choose a date in the future to keep your footage until.")
            data["credits"] = None
        elif kind == AccountRequest.Kind.PRO:
            if getattr(getattr(self.user, "profile", None), "tier", "") == "pro":
                raise forms.ValidationError("You're already on Pro.")
            data["credits"] = data["until"] = None
        return data
