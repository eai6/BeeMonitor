from django import forms


class DeviceCreateForm(forms.Form):
    name = forms.CharField(
        max_length=100,
        help_text="A nickname for this Pi, e.g. 'field-site-1' or 'pi-natalies-hive-2'.",
    )
    location = forms.CharField(
        max_length=200,
        required=False,
        help_text="Optional. Free-text. Where the Pi is physically deployed.",
    )
