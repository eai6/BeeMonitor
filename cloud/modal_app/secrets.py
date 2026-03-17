"""Modal Secret references for BeeMonitor Cloud.

Secrets must be created in the Modal dashboard or via CLI:

    modal secret create azure-storage \\
        AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=..."

    modal secret create aws-creds \\
        AWS_ACCESS_KEY_ID="..." \\
        AWS_SECRET_ACCESS_KEY="..."

    modal secret create gcp-creds \\
        GOOGLE_APPLICATION_CREDENTIALS_JSON="..."
"""

import modal

azure_secret = modal.Secret.from_name("azure-storage")
aws_secret = modal.Secret.from_name("aws-creds")
gcp_secret = modal.Secret.from_name("gcp-creds")
