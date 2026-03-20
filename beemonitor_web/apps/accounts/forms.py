from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

from .models import Coupon


class UserRegistrationForm(UserCreationForm):
    email = forms.EmailField(required=True)

    class Meta:
        model = User
        fields = ("username", "email", "password1", "password2")


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
