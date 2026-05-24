"""
SageMaker inference handler for the BeeMonitor GPU endpoint.

Implements the SageMaker Python inference contract:

    model_fn(model_dir)             -> load CloudPipeline once per container
    input_fn(request_body, ctype)   -> parse {video_storage_key, job_id, ...}
    predict_fn(payload, pipeline)   -> run CloudPipeline.process
    output_fn(prediction, accept)   -> serialize PipelineResult to JSON

Request body (application/json):
    {
        "job_id":   "<unique-id-for-the-analysis-job>",
        "user_id":  "<owner-id>",
        "video_blob_path": "users/7/devices/3/2026/05/.../uuid.mp4",
        "detection_mode": "yolo",            # optional
        "confidence_threshold": 0.25,        # optional
        "visualize": true,                   # optional
        "custom_nest_model_path": "...",     # optional
        "custom_bee_model_path": "...",      # optional
    }

Response body (application/json) is the dict form of ``PipelineResult``,
plus ``status``, ``execution_seconds``, and ``device``.
"""

import json
import logging
import os
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("beemonitor.handler")

JSON_CONTENT_TYPE = "application/json"


def model_fn(model_dir=None):
    """Build the CloudPipeline once per container.

    Loading is lazy: ``CloudPipeline.__init__`` doesn't load any model
    weights — those come down from the S3 ``models`` bucket on the first
    ``process()`` call via ``ModelManager.ensure_models()``. That means
    cold-start time is dominated by the network pull of the YOLO weights
    (~50 MB) on the very first invocation, not by container boot.
    """
    logger.info("model_fn: building CloudPipeline (model_dir=%s)", model_dir)
    # Imports here (not at module top) so the SageMaker contract module is
    # importable on the CPU dev box for tests where torch+cuda aren't present.
    from cloud.wrapper.pipeline import CloudPipeline
    pipeline = CloudPipeline()
    logger.info("model_fn: pipeline ready")
    return pipeline


def input_fn(request_body, content_type):
    if content_type != JSON_CONTENT_TYPE:
        raise ValueError(
            f"unsupported content_type {content_type!r}; expected {JSON_CONTENT_TYPE}"
        )
    if isinstance(request_body, (bytes, bytearray)):
        request_body = request_body.decode("utf-8")
    payload = json.loads(request_body)

    required = ("job_id", "user_id", "video_blob_path")
    missing = [k for k in required if not payload.get(k)]
    if missing:
        raise ValueError(f"missing required keys: {', '.join(missing)}")
    return payload


def predict_fn(payload, pipeline):
    """Run BeeMonitor analysis on one video. Returns a serializable dict."""
    job_id = payload["job_id"]
    user_id = str(payload["user_id"])
    video_blob_path = payload["video_blob_path"]

    started = time.time()
    logger.info("predict_fn: job=%s video=%s", job_id, video_blob_path)

    try:
        result = pipeline.process(
            job_id=job_id,
            user_id=user_id,
            video_blob_path=video_blob_path,
            detection_mode=payload.get("detection_mode", "yolo"),
            confidence_threshold=float(payload.get("confidence_threshold", 0.25)),
            ml_threshold=float(payload.get("ml_threshold", 0.6)),
            visualize=bool(payload.get("visualize", True)),
            two_mode_tracking=bool(payload.get("two_mode_tracking", True)),
            custom_nest_model_path=payload.get("custom_nest_model_path", "") or "",
            custom_bee_model_path=payload.get("custom_bee_model_path", "") or "",
        )
    except Exception as exc:
        logger.exception("predict_fn: pipeline failed for job %s", job_id)
        return {
            "status": "failed",
            "job_id": job_id,
            "user_id": user_id,
            "error_message": str(exc),
            "execution_seconds": round(time.time() - started, 2),
        }

    out = result.to_dict()
    out["status"] = "completed"
    out["execution_seconds"] = round(time.time() - started, 2)
    out["device"] = _detect_device()
    return out


def output_fn(prediction, accept):
    accept = accept or JSON_CONTENT_TYPE
    if accept == "*/*":
        accept = JSON_CONTENT_TYPE
    if accept != JSON_CONTENT_TYPE:
        raise ValueError(f"unsupported accept {accept!r}; expected {JSON_CONTENT_TYPE}")
    return json.dumps(prediction), JSON_CONTENT_TYPE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_device() -> str:
    """Report which device the pipeline ran on (for response telemetry)."""
    try:
        import torch
        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.get_device_name(0)}"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"
