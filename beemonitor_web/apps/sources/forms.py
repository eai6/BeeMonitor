from django import forms

from .models import DataSource


class SourceCreateForm(forms.Form):
    name = forms.CharField(max_length=200)
    source_type = forms.ChoiceField(choices=DataSource.SourceType.choices)

    # AWS S3
    s3_bucket = forms.CharField(max_length=200, required=False, label="S3 Bucket Name")
    s3_prefix = forms.CharField(max_length=500, required=False, label="Key Prefix (optional)",
                                help_text="e.g. videos/field-site-1/")
    s3_access_key_id = forms.CharField(max_length=200, required=False, label="Access Key ID")
    s3_secret_access_key = forms.CharField(max_length=200, required=False, label="Secret Access Key",
                                           widget=forms.PasswordInput(render_value=True))
    s3_region = forms.CharField(max_length=50, required=False, label="Region", initial="us-east-1")

    # Azure Blob
    azure_container = forms.CharField(max_length=200, required=False, label="Container Name")
    azure_connection_string = forms.CharField(max_length=1000, required=False, label="Connection String",
                                              widget=forms.Textarea(attrs={"rows": 2}))

    # GCS
    gcs_bucket = forms.CharField(max_length=200, required=False, label="GCS Bucket Name")
    gcs_service_account_json = forms.CharField(required=False, label="Service Account JSON",
                                               widget=forms.Textarea(attrs={"rows": 4}),
                                               help_text="Paste the full JSON key file content")

    # Google Drive
    gdrive_folder_id = forms.CharField(max_length=200, required=False, label="Folder ID",
                                       help_text="The ID from the Google Drive folder URL")
    gdrive_oauth_token = forms.CharField(max_length=2000, required=False, label="OAuth Token",
                                         widget=forms.Textarea(attrs={"rows": 2}))

    def clean(self):
        cleaned = super().clean()
        st = cleaned.get("source_type")

        if st == "aws_s3":
            if not cleaned.get("s3_bucket"):
                self.add_error("s3_bucket", "Bucket name is required for S3.")
            if not cleaned.get("s3_access_key_id"):
                self.add_error("s3_access_key_id", "Access Key ID is required.")
            if not cleaned.get("s3_secret_access_key"):
                self.add_error("s3_secret_access_key", "Secret Access Key is required.")
        elif st == "azure_blob":
            if not cleaned.get("azure_container"):
                self.add_error("azure_container", "Container name is required.")
            if not cleaned.get("azure_connection_string"):
                self.add_error("azure_connection_string", "Connection string is required.")
        elif st == "gcs":
            if not cleaned.get("gcs_bucket"):
                self.add_error("gcs_bucket", "Bucket name is required.")
        elif st == "google_drive":
            if not cleaned.get("gdrive_folder_id"):
                self.add_error("gdrive_folder_id", "Folder ID is required.")
            if not cleaned.get("gdrive_oauth_token"):
                self.add_error("gdrive_oauth_token", "OAuth token is required.")

        return cleaned

    def get_credentials(self) -> dict:
        """Return a dict of credentials for the selected source type."""
        st = self.cleaned_data["source_type"]
        if st == "aws_s3":
            return {
                "bucket": self.cleaned_data["s3_bucket"],
                "prefix": self.cleaned_data.get("s3_prefix", ""),
                "access_key_id": self.cleaned_data["s3_access_key_id"],
                "secret_access_key": self.cleaned_data["s3_secret_access_key"],
                "region": self.cleaned_data.get("s3_region", "us-east-1"),
            }
        elif st == "azure_blob":
            return {
                "container": self.cleaned_data["azure_container"],
                "connection_string": self.cleaned_data["azure_connection_string"],
            }
        elif st == "gcs":
            return {
                "bucket": self.cleaned_data["gcs_bucket"],
                "service_account_json": self.cleaned_data.get("gcs_service_account_json", ""),
            }
        elif st == "google_drive":
            return {
                "folder_id": self.cleaned_data["gdrive_folder_id"],
                "oauth_token": self.cleaned_data["gdrive_oauth_token"],
            }
        return {}
