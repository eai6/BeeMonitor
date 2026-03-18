from django import forms

from .models import Video


class VideoUploadForm(forms.ModelForm):
    video_file = forms.FileField(
        help_text="Upload a video file for analysis.",
    )

    class Meta:
        model = Video
        fields = ("title",)


class MultipleFileInput(forms.ClearableFileInput):
    allow_multiple_selected = True


class MultipleFileField(forms.FileField):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("widget", MultipleFileInput(attrs={"accept": "video/*"}))
        super().__init__(*args, **kwargs)

    def clean(self, data, initial=None):
        if not data and initial:
            return initial
        if not isinstance(data, (list, tuple)):
            data = [data]
        return [super(MultipleFileField, self).clean(d) for d in data]


class VideoBatchUploadForm(forms.Form):
    video_files = MultipleFileField(
        help_text="Select multiple video files to upload.",
    )
    site_name = forms.CharField(
        max_length=200,
        required=False,
        help_text="Optional: override site name for all uploaded videos. If blank, site name is parsed from filenames.",
        widget=forms.TextInput(attrs={"placeholder": "e.g. hive_station_alpha"}),
    )
