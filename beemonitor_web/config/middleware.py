"""Small operational middlewares."""

import logging
import time

logger = logging.getLogger("beemonitor.slow")

# Anything slower than this gets a log line — App Runner 504s at a hard 120s,
# so slow requests must be visible in the logs long before users see timeouts.
SLOW_REQUEST_SECONDS = 5.0


class SlowRequestLogMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        start = time.monotonic()
        response = self.get_response(request)
        elapsed = time.monotonic() - start
        if elapsed >= SLOW_REQUEST_SECONDS:
            logger.warning("slow request: %.1fs %s %s (status %s)",
                           elapsed, request.method, request.get_full_path(),
                           getattr(response, "status_code", "?"))
        return response
