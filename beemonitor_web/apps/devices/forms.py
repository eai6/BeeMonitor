from django import forms

from .models import Device


class DeviceCreateForm(forms.Form):
    name = forms.CharField(
        max_length=100,
        help_text="A nickname for this Pi, e.g. 'field-site-1' or 'pi-natalies-hive-2'.",
    )
    location = forms.CharField(
        max_length=200,
        required=False,
        help_text="Optional label, e.g. 'north hedgerow'.",
    )
    lat = forms.FloatField(
        required=False,
        min_value=-90,
        max_value=90,
        help_text="Optional. Decimal degrees, e.g. 40.7934.",
    )
    lon = forms.FloatField(
        required=False,
        min_value=-180,
        max_value=180,
        help_text="Optional. Decimal degrees, e.g. -77.8600.",
    )

    def clean(self):
        cleaned = super().clean()
        lat, lon = cleaned.get("lat"), cleaned.get("lon")
        # Coordinates only make sense as a pair.
        if (lat is None) != (lon is None):
            raise forms.ValidationError("Set both latitude and longitude, or neither.")
        return cleaned


class DeviceEditForm(forms.ModelForm):
    """Edit a device's label + deployment coordinates (name/location/lat/lon)."""

    class Meta:
        model = Device
        fields = ["name", "location", "lat", "lon"]
        help_texts = {
            "name": "A nickname for this Pi.",
            "location": "Optional label, e.g. 'north hedgerow'.",
            "lat": "Optional. Decimal degrees, e.g. 40.7934.",
            "lon": "Optional. Decimal degrees, e.g. -77.8600.",
        }

    def clean(self):
        cleaned = super().clean()
        lat, lon = cleaned.get("lat"), cleaned.get("lon")
        if (lat is None) != (lon is None):
            raise forms.ValidationError("Set both latitude and longitude, or neither.")
        return cleaned
