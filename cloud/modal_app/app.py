"""Main Modal application definition for BeeMonitor Cloud.

Usage:
    modal serve cloud/modal_app/app.py    # Dev with live-reload
    modal deploy cloud/modal_app/app.py   # Production deployment
"""

import modal

from cloud.modal_app.image import beemonitor_image
from cloud.modal_app.volumes import model_volume, MODEL_VOLUME_MOUNT
from cloud.modal_app.secrets import azure_secret

app = modal.App("beemonitor-cloud", image=beemonitor_image)


# ── Health check endpoint ─────────────────────────────────────────────

@app.function()
@modal.fastapi_endpoint(method="GET")
def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "beemonitor-cloud"}
