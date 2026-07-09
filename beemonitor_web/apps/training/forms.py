from django import forms
from django.conf import settings

from apps.annotations.models import AnnotationProject

from .models import CustomModel, TrainingJob

_SELECT_CSS = "w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-amber-500 focus:border-amber-500"

# S3 key (models bucket) of the production bee-tracking model the inference
# pipeline uses (cloud/wrapper/model_manager.py: MODEL_VERSION/bee_tracking.pt).
DEFAULT_BEE_MODEL_KEY = getattr(
    settings, "BEEMONITOR_DEFAULT_BEE_MODEL_KEY", "v1/bee_tracking.pt")
DEFAULT_BEE_MODEL_LABEL = "BeeMonitor current bee-tracking model"


class TrainingCreateForm(forms.ModelForm):
    project = forms.ModelChoiceField(
        queryset=AnnotationProject.objects.none(),
        widget=forms.Select(attrs={"class": _SELECT_CSS}),
    )
    # One select for both fine-tune sources ("default", "cm:<pk>") and
    # from-scratch architectures (yolov8n, ...); clean() maps fine-tune
    # picks onto init_weights_key + a plain base_model value.
    base_model = forms.ChoiceField(
        label="Base Model",
        widget=forms.Select(attrs={"class": _SELECT_CSS}),
    )

    class Meta:
        model = TrainingJob
        fields = ["project", "name", "base_model", "frame_filter",
                  "frames_with_class",
                  "epochs", "image_size", "batch_size", "val_percent", "gpu_tier"]
        widgets = {
            "name": forms.TextInput(attrs={
                "class": "w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-amber-500 focus:border-amber-500",
                "placeholder": "My Training Job",
            }),
            "frames_with_class": forms.CheckboxInput(attrs={
                "class": "rounded border-gray-300 text-amber-600 focus:ring-amber-500",
            }),
            "frame_filter": forms.Select(attrs={
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
            "val_percent": forms.NumberInput(attrs={
                "class": "w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-amber-500 focus:border-amber-500",
                "min": 5,
                "max": 50,
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

        # Fine-tune sources (the production bee model + the user's ready
        # custom/trained models) and from-scratch architectures share the
        # Base Model select, as optgroups.
        finetune_choices = [("default", f"{DEFAULT_BEE_MODEL_LABEL} (fine-tune)")]
        self._custom_models = {}
        if user:
            for cm in CustomModel.objects.filter(
                user=user, status=CustomModel.Status.READY, is_active=True,
            ).exclude(storage_key=""):
                self._custom_models[cm.pk] = cm
                map50 = (cm.metrics or {}).get("mAP50")
                suffix = f" — mAP50 {map50}" if map50 else ""
                finetune_choices.append((f"cm:{cm.pk}", f"{cm.name}{suffix} (fine-tune)"))
        self.fields["base_model"].choices = [
            ("Fine-tune an existing model", finetune_choices),
            ("Train from scratch (COCO-pretrained)", list(TrainingJob.BaseModel.choices)),
        ]
        if not self.initial.get("base_model"):
            self.initial["base_model"] = "default"

    def clean_val_percent(self):
        v = self.cleaned_data.get("val_percent")
        if v is None:
            return 20
        if not 5 <= v <= 50:
            raise forms.ValidationError("Validation split must be between 5% and 50%.")
        return v

    def clean(self):
        cleaned = super().clean()

        # Class subset — checkboxes named "class_subset" (dynamic per project,
        # so validated here rather than as a ChoiceField). Empty stored subset
        # means "all classes"; unchecking everything is an error.
        project = cleaned.get("project")
        if project is not None:
            proj_classes = project.classes or []
            selected = set(self.data.getlist("class_subset"))
            subset = [c for c in proj_classes if c in selected]  # keep project order
            if not subset:
                raise forms.ValidationError("Select at least one class to train on.")
            # All classes selected → store empty (= train on everything).
            self.instance.class_subset = [] if len(subset) == len(proj_classes) else subset

        choice = cleaned.get("base_model") or ""
        valid_archs = {v for v, _ in TrainingJob.BaseModel.choices}
        if choice == "default":
            self.instance.init_weights_key = DEFAULT_BEE_MODEL_KEY
            self.instance.init_from_label = DEFAULT_BEE_MODEL_LABEL
            cleaned["base_model"] = TrainingJob.BaseModel.YOLOV8N
        elif choice.startswith("cm:"):
            cm = self._custom_models.get(int(choice[3:]))
            if cm is None:
                raise forms.ValidationError("Selected fine-tune model is unavailable.")
            self.instance.init_weights_key = cm.storage_key
            self.instance.init_from_label = cm.name
            cleaned["base_model"] = (
                cm.base_model if cm.base_model in valid_archs else TrainingJob.BaseModel.YOLOV8N)
        return cleaned


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
