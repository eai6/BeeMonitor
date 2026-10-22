from django import forms



class SourceCreateForm(forms.Form):
    """Connect an S3 bucket (memory/54: S3 only for now). The secret is never
    written back into the page."""

    name = forms.CharField(max_length=200)
    s3_bucket = forms.CharField(max_length=200, label="Bucket")
    s3_prefix = forms.CharField(max_length=500, required=False, label="Folder (optional)",
                                help_text="Only this folder is visible, e.g. videos/field-site-1/")
    s3_access_key_id = forms.CharField(max_length=200, label="Access key ID")
    s3_secret_access_key = forms.CharField(max_length=200, label="Secret access key",
                                           widget=forms.PasswordInput(render_value=False))
    s3_region = forms.CharField(max_length=50, required=False, label="Region", initial="us-east-1")

    def clean_s3_bucket(self):
        bucket = self.cleaned_data["s3_bucket"].strip().removeprefix("s3://").strip("/")
        if not bucket or "/" in bucket:
            raise forms.ValidationError("Just the bucket name, e.g. pollinator-lab-footage. Put a folder below.")
        return bucket

    def clean_s3_prefix(self):
        prefix = self.cleaned_data.get("s3_prefix", "").strip().lstrip("/")
        return prefix + "/" if prefix and not prefix.endswith("/") else prefix

    def get_credentials(self) -> dict:
        return {
            "bucket": self.cleaned_data["s3_bucket"],
            "prefix": self.cleaned_data.get("s3_prefix", ""),
            "access_key_id": self.cleaned_data["s3_access_key_id"].strip(),
            "secret_access_key": self.cleaned_data["s3_secret_access_key"].strip(),
            "region": (self.cleaned_data.get("s3_region") or "us-east-1").strip(),
        }
