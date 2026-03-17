from django import forms

from .models import DataSource


class SourceCreateForm(forms.ModelForm):
    class Meta:
        model = DataSource
        fields = ("name", "source_type")
