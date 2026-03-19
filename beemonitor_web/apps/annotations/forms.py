from django import forms

from .models import AnnotationProject


class ProjectCreateForm(forms.ModelForm):
    classes_text = forms.CharField(
        label="Classes",
        help_text="Comma-separated list of class labels, e.g. bee, wasp, nest",
        initial="bee, wasp, nest",
        widget=forms.TextInput(attrs={
            "class": "w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-amber-500 focus:border-amber-500",
            "placeholder": "bee, wasp, nest",
        }),
    )

    class Meta:
        model = AnnotationProject
        fields = ["name", "description"]
        widgets = {
            "name": forms.TextInput(attrs={
                "class": "w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-amber-500 focus:border-amber-500",
                "placeholder": "My Annotation Project",
            }),
            "description": forms.Textarea(attrs={
                "class": "w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-amber-500 focus:border-amber-500",
                "rows": 3,
                "placeholder": "Optional description...",
            }),
        }

    def clean_classes_text(self):
        raw = self.cleaned_data["classes_text"]
        classes = [c.strip() for c in raw.split(",") if c.strip()]
        if not classes:
            raise forms.ValidationError("At least one class label is required.")
        return classes

    def save(self, commit=True):
        instance = super().save(commit=False)
        instance.classes = self.cleaned_data["classes_text"]
        if commit:
            instance.save()
        return instance
