from django import forms

from apps.videos.models import Video

DETECTION_MODE_CHOICES = [
    ("detect_and_track", "Detect & Track"),
    ("detect_only", "Detect Only"),
    ("count_only", "Count Only"),
]


class JobCreateForm(forms.Form):
    video = forms.ModelChoiceField(queryset=Video.objects.none())
    detection_mode = forms.ChoiceField(choices=DETECTION_MODE_CHOICES)
    confidence_threshold = forms.FloatField(
        min_value=0.1,
        max_value=1.0,
        initial=0.5,
        widget=forms.NumberInput(attrs={"step": "0.05"}),
    )

    def __init__(self, *args, user=None, **kwargs):
        super().__init__(*args, **kwargs)
        if user:
            self.fields["video"].queryset = Video.objects.filter(
                user=user, status=Video.Status.READY
            )
