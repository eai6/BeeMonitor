"""Flask serving shim for the BioCLIP endpoint — the SageMaker BYOC contract.

GET /ping -> 200 once the classifier is loaded; POST /invocations -> classify.
Mirrors ../serve.py but for the CPU BioCLIP handler. The model loads once at
import (gunicorn --workers 1), so the first request after a serverless cold
start is served by an already-warm classifier.
"""

import logging

from flask import Flask, Response, request

import inference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("beemonitor.bioclip.serve")

JSON = "application/json"
app = Flask(__name__)

_model = None
_load_error = None
try:
    logger.info("serve: loading BioCLIP classifier ...")
    _model = inference.model_fn()
    logger.info("serve: classifier ready")
except Exception as exc:  # noqa: BLE001
    _load_error = exc
    logger.exception("serve: classifier failed to load")


@app.route("/ping", methods=["GET"])
def ping():
    healthy = _model is not None
    return Response(
        response='{"status": "healthy"}\n' if healthy
        else f'{{"status": "unhealthy", "error": "{_load_error}"}}\n',
        status=200 if healthy else 503,
        mimetype=JSON,
    )


@app.route("/invocations", methods=["POST"])
def invocations():
    if _model is None:
        return Response('{"status": "error", "error": "model not loaded"}\n',
                        status=503, mimetype=JSON)
    content_type = request.content_type or JSON
    accept = request.headers.get("Accept", JSON)
    if accept == "*/*":
        accept = JSON
    try:
        image = inference.input_fn(request.get_data(), content_type)
        prediction = inference.predict_fn(image, _model)
        body, out_type = inference.output_fn(prediction, accept)
    except Exception as exc:  # noqa: BLE001
        logger.exception("serve: invocation failed")
        return Response(f'{{"status": "error", "error": "{exc}"}}\n',
                        status=400, mimetype=JSON)
    return Response(response=body, status=200, mimetype=out_type)
