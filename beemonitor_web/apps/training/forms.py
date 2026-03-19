from django import forms

from apps.annotations.models import AnnotationProject

from .models import TrainingJob


class TrainingCreateForm(forms.ModelForm):
    project = forms.ModelChoiceField(
        queryset=AnnotationProject.objects.none(),
        widget=forms.Select(attrs={
            "class": "w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-amber-500 focus:border-amber-500",
        }),
    )

    class Meta:
        model = TrainingJob
        fields = ["project", "name", "base_model", "epochs", "image_size", "batch_size", "gpu_tier"]
        widgets = {
            "name": forms.TextInput(attrs={
                "class": "w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-amber-500 focus:border-amber-500",
                "placeholder": "My Training Job",
            }),
            "base_model": forms.Select(attrs={
                "class": "w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-amber-500 focus:border-amber-500",
            }),
            "epochs": forms.NumberInput(attrs={
                "class": "w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-amber-500 focus:border-amber-500",
                "min": 1,
                "max": 500,
            }),
            "image_size": forms.NumberInput(attrs={
                "class": "w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-amber-500 focus:border-amber-500",
                "min": 320,
                "max": 1280,
                "step": 32,
            }),
            "batch_size": forms.NumberInput(attrs={
                "class": "w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-amber-500 focus:border-amber-500",
                "min": 1,
                "max": 128,
            }),
            "gpu_tier": forms.Select(attrs={
                "class": "w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-amber-500 focus:border-amber-500",
            }),
        }

    def __init__(self, *args, user=None, **kwargs):
        super().__init__(*args, **kwargs)
        if user:
            self.fields["project"].queryset = AnnotationProject.objects.filter(user=user)
