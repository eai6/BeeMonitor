from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

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
