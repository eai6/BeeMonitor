from django import forms

from .models import Video


class VideoUploadForm(forms.ModelForm):
    video_file = forms.FileField(
        help_text="Upload a video file for analysis.",
    )

    class Meta:
        model = Video
        fields = ("title",)
