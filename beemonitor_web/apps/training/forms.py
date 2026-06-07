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

    # Training is per-instance-hour (no scale-to-zero), so restrict to the
    # affordable single-GPU instances — drop L40S/A100 (ml.p4d.24xlarge is
    # ~$37/hr). Labels name the SageMaker instance each maps to (see
    # training/views.py _INSTANCE_BY_TIER).
    TRAINING_GPU_CHOICES = [
        ("T4", "T4 — ml.g4dn.xlarge (cheapest, slower)"),
        ("L4", "L4 — ml.g6.xlarge (balanced)"),
        ("A10G", "A10G — ml.g5.xlarge (fast, recommended)"),
    ]

    def __init__(self, *args, user=None, **kwargs):
        super().__init__(*args, **kwargs)
        if user:
            self.fields["project"].queryset = AnnotationProject.objects.filter(user=user)
        self.fields["gpu_tier"].choices = self.TRAINING_GPU_CHOICES
        if not self.initial.get("gpu_tier"):
            self.initial["gpu_tier"] = "A10G"


class ModelUploadForm(forms.Form):
    name = forms.CharField(max_length=200, widget=forms.TextInput(attrs={
        "class": "w-full border border-gray-300 rounded px-3 py-2 text-sm",
        "placeholder": "My Custom Nest Model",
    }))
    model_type = forms.ChoiceField(choices=[
        ("nest_detection", "Nest Detection"),
        ("bee_tracking", "Bee/Species Tracking"),
        ("custom", "Other / Custom"),
    ], widget=forms.Select(attrs={
        "class": "w-full border border-gray-300 rounded px-3 py-2 text-sm",
    }))
    model_file = forms.FileField(
        help_text="Upload a .pt YOLO model file",
        widget=forms.ClearableFileInput(attrs={"accept": ".pt,.pth,.onnx"}),
    )
    classes = forms.CharField(
        required=False,
        help_text="Comma-separated class names (e.g., bee, wasp, hover_fly)",
        widget=forms.TextInput(attrs={
            "class": "w-full border border-gray-300 rounded px-3 py-2 text-sm",
            "placeholder": "bee, wasp, hover_fly",
        }),
    )
