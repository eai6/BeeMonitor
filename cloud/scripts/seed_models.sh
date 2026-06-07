#!/bin/bash
# Seed the base BeeMonitor model weights into the S3 models bucket.
#
# The SageMaker inference + training containers load these at runtime via
# ModelManager.ensure_models(), which expects them at:
#   s3://<models-bucket>/<version>/nest_detection.pt
#   s3://<models-bucket>/<version>/bee_tracking.pt
#   s3://<models-bucket>/<version>/event_classifier_model.pkl
#
# Without this, every analysis / pre-annotation invocation 404s on HeadObject.
# Run once per environment, and whenever the weights in models/ change:
#
#   AWS_PROFILE=ecomorph \
#   MODELS_BUCKET=beemonitor-dev-models-495331821764 \
#     bash cloud/scripts/seed_models.sh
set -euo pipefail

BUCKET="${MODELS_BUCKET:?set MODELS_BUCKET (e.g. beemonitor-dev-models-<acct>)}"
VERSION="${MODEL_VERSION:-v1}"      # must match cloud/wrapper/model_manager.py MODEL_VERSION
REGION="${AWS_REGION:-us-east-1}"
DIR="$(cd "$(dirname "$0")/../../models" && pwd)"

for f in nest_detection.pt bee_tracking.pt event_classifier_model.pkl; do
    [ -f "$DIR/$f" ] || { echo "missing $DIR/$f" >&2; exit 1; }
    echo "uploading $f -> s3://$BUCKET/$VERSION/$f"
    aws s3 cp "$DIR/$f" "s3://$BUCKET/$VERSION/$f" --region "$REGION"
done
echo "Seeded $VERSION base models to s3://$BUCKET/$VERSION/"
