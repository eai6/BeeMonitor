"""
SageMaker BYOC serving wrapper for the BeeMonitor GPU endpoint.

A SageMaker container must answer two HTTP routes on port 8080:

    GET  /ping         -> 200 once the container is healthy
    POST /invocations  -> run inference

This module is a thin Flask shim over the SageMaker inference contract in
``inference.py``. Mirrors EcoMorph's pattern. Engine is loaded once at module
import — i.e. in the gunicorn worker process, not the master (CUDA contexts
cannot be inherited across a fork, which is why gunicorn runs --workers 1
--preload off).
"""

import logging

from flask import Flask, Response, request

import inference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("beemonitor.serve")

JSON = "application/json"

app = Flask(__name__)

_pipeline = None
_load_error = None
try:
    logger.info("serve: building CloudPipeline ...")
    _pipeline = inference.model_fn()
    logger.info("serve: pipeline ready")
except Exception as exc:  # noqa: BLE001
    _load_error = exc
    logger.exception("serve: pipeline failed to build")


@app.route("/ping", methods=["GET"])
def ping():
    healthy = _pipeline is not None
    return Response(
        response='{"status": "healthy"}\n' if healthy
        else f'{{"status": "unhealthy", "error": "{_load_error}"}}\n',
        status=200 if healthy else 503,
        mimetype=JSON,
    )


@app.route("/invocations", methods=["POST"])
def invocations():
    if _pipeline is None:
        return Response(
            response='{"status": "error", "error": "pipeline not loaded"}\n',
            status=503, mimetype=JSON,
        )

    content_type = request.content_type or JSON
    accept = request.headers.get("Accept", JSON)
    if accept == "*/*":
        accept = JSON

    try:
        payload = inference.input_fn(request.get_data(), content_type)
        prediction = inference.predict_fn(payload, _pipeline)
        body, out_type = inference.output_fn(prediction, accept)
    except Exception as exc:  # noqa: BLE001
        logger.exception("serve: invocation failed")
        return Response(
            response='{"status": "error", "error": %r}\n' % str(exc),
            status=500, mimetype=JSON,
        )

    return Response(response=body, status=200, mimetype=out_type)


if __name__ == "__main__":
    # Local debug only — in the container gunicorn serves this app.
    app.run(host="0.0.0.0", port=8080)
