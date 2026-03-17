"""Modal Volume definitions for persistent model storage."""

import modal

# Persistent volume for ML model weights.
# Models are downloaded from Azure once and cached here across invocations.
model_volume = modal.Volume.from_name("beemonitor-models", create_if_missing=True)

MODEL_VOLUME_MOUNT = "/vol/models"
