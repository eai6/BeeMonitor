"""Recounts sent to an SQS queue, for Lambda workers (memory/81).

The web's recount loop is one thread in each of two workers on one vCPU,
beside the page requests: a 6,306-clip batch's recount took most of a day
and slowed every page while it ran. With ``RECOUNT_QUEUE_URL`` set, queued
runs go to SQS in messages of ``PER_MESSAGE``, and a Lambda
(recount_lambda.py) recounts them, several at a time. The web loop only
takes what the queue's workers have left for ``LEFT_SECONDS``.
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

# Runs a message: a few minutes of a worker's time. A run whose tubes are
# detected first takes ~30-60 s (memory/82), and a message must finish inside
# the worker's 15 minutes.
PER_MESSAGE = 5
# A run asked for this long ago and still not recounted is the web's again.
LEFT_SECONDS = 30 * 60


def enabled() -> bool:
    from django.conf import settings

    return bool(getattr(settings, "RECOUNT_QUEUE_URL", ""))


def send(pks) -> bool:
    """Send these runs to the queue. False when there's no queue or it can't
    be reached: the web's loop recounts them then."""
    from django.conf import settings

    url = getattr(settings, "RECOUNT_QUEUE_URL", "")
    pks = [str(pk) for pk in pks]
    if not url or not pks:
        return False
    try:
        import boto3

        sqs = boto3.client("sqs", region_name=getattr(settings, "AWS_REGION", "us-east-1"))
        bodies = [pks[i:i + PER_MESSAGE] for i in range(0, len(pks), PER_MESSAGE)]
        for i in range(0, len(bodies), 10):          # SQS takes 10 a call
            out = sqs.send_message_batch(QueueUrl=url, Entries=[
                {"Id": str(n), "MessageBody": json.dumps({"runs": body})}
                for n, body in enumerate(bodies[i:i + 10])])
            if out.get("Failed"):
                raise RuntimeError(f"{len(out['Failed'])} messages not sent: {out['Failed'][0]}")
    except Exception:
        logger.exception("recount queue: could not send %s runs; the web recounts them", len(pks))
        return False
    return True
