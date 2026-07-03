"""Flask serving shim for the SAM 3 auto-labeler endpoint — the SageMaker BYOC
contract. Mirrors ../bioclip/serve.py.

GET /ping -> 200 once the model is loaded; POST /invocations -> segment. The model
loads once at import (gunicorn --workers 1), so the first request after a
scale-from-zero cold start is served by an already-warm model.
"""

import logging

from flask import Flask, Response, request

import inference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("beemonitor.sam3.serve")

JSON = "application/json"
app = Flask(__name__)

_model = None
_load_error = None
try:
    logger.info("serve: loading SAM 3 ...")
    _model = inference.model_fn()
    logger.info("serve: SAM 3 ready")
except Exception as exc:  # noqa: BLE001
    _load_error = exc
    logger.exception("serve: SAM 3 failed to load")


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
        parsed = inference.input_fn(request.get_data(), content_type)
        prediction = inference.predict_fn(parsed, _model)
        body, out_type = inference.output_fn(prediction, accept)
    except Exception as exc:  # noqa: BLE001
        logger.exception("serve: invocation failed")
        return Response(f'{{"status": "error", "error": "{exc}"}}\n',
                        status=400, mimetype=JSON)
    return Response(response=body, status=200, mimetype=out_type)
