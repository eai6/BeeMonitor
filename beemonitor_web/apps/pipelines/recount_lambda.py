"""The recount worker on Lambda (memory/81): ``python -m apps.pipelines.recount_lambda``.

It runs in the web's own image, so it recounts with the same code as the web.
The image has no Lambda runtime client, and needs none: this is Lambda's
runtime API loop (fetch the next event, answer it), a few lines of urllib.

Each SQS message carries ``{"runs": [pk, ...]}``. A run is claimed before it's
recounted (recount.recount_claimed), so a run the web's loop or another
worker took is skipped. A message whose runs couldn't be read at all is
reported failed and comes back, up to the queue's dead-letter limit.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import urllib.request

logger = logging.getLogger("apps.pipelines.recount_lambda")

_API = "http://{}/2018-06-01/runtime".format(os.environ.get("AWS_LAMBDA_RUNTIME_API", ""))


def load_secrets():
    """The Django secrets, from Secrets Manager: ``BEEMONITOR_SECRETS`` maps an
    environment variable to a secret's ARN (App Runner fills them itself)."""
    names = json.loads(os.environ.get("BEEMONITOR_SECRETS") or "{}")
    if not names:
        return
    import boto3

    sm = boto3.client("secretsmanager", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    for var, arn in names.items():
        if not os.environ.get(var):
            os.environ[var] = sm.get_secret_value(SecretId=arn)["SecretString"]


def handle(event) -> dict:
    """One SQS batch: recount its runs. Returns the messages to try again."""
    from django.db import close_old_connections

    from . import recount

    close_old_connections()
    failed, done = [], 0
    for record in event.get("Records") or []:
        try:
            pks = json.loads(record["body"])["runs"]
        except (KeyError, TypeError, ValueError):
            logger.error("recount worker: unreadable message %s", record.get("messageId"))
            continue
        try:
            for pk in pks:
                done += recount.recount_claimed(pk)
        except Exception:
            logger.exception("recount worker: message %s failed", record.get("messageId"))
            failed.append({"itemIdentifier": record["messageId"]})
    logger.info("recount worker: recounted %s runs", done)
    return {"batchItemFailures": failed}


def _call(path, data=None):
    req = urllib.request.Request(_API + path, data=data,
                                 headers={"Content-Type": "application/json"} if data else {})
    return urllib.request.urlopen(req, timeout=None)


def main():
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    try:
        load_secrets()
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
        os.environ["BEEMONITOR_RECONCILE_DISABLED"] = "1"     # no web loops in here
        import django

        django.setup()
    except Exception as exc:
        logger.exception("recount worker: could not start")
        _call("/init/error", json.dumps({"errorMessage": str(exc),
                                         "errorType": type(exc).__name__}).encode())
        raise
    while True:
        with _call("/invocation/next") as res:
            request_id = res.headers["Lambda-Runtime-Aws-Request-Id"]
            event = json.loads(res.read() or b"{}")
        try:
            body = json.dumps(handle(event)).encode()
            _call(f"/invocation/{request_id}/response", body).close()
        except Exception as exc:
            logger.exception("recount worker: invocation failed")
            _call(f"/invocation/{request_id}/error", json.dumps(
                {"errorMessage": str(exc), "errorType": type(exc).__name__}).encode()).close()


if __name__ == "__main__":
    main()
