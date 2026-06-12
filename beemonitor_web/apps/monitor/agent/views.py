"""Streaming chat endpoint for the monitoring agent (mirrors the setup assistant).

POST JSON: {device_id?, conversation_id?, message}
Returns a text/event-stream of JSON events (see client.stream_reply). User
messages are secret-redacted before they reach the API or the DB.
"""

import json
import logging

from django.conf import settings
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import JsonResponse, StreamingHttpResponse
from django.utils import timezone
from django.views import View

from apps.devices.models import Device
from apps.setup.assistant.safety import redact_secrets  # reuse the redactor

from . import client
from .tools import _online
from ..models import AgentConversation, AgentMessage

logger = logging.getLogger(__name__)


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


class AgentSendView(LoginRequiredMixin, View):
    def post(self, request):
        if not client.is_enabled():
            return JsonResponse(
                {"error": "The monitoring agent isn't configured on this server."},
                status=503)
        try:
            body = json.loads(request.body or "{}")
        except ValueError:
            return JsonResponse({"error": "Invalid JSON."}, status=400)

        message = (body.get("message") or "").strip()
        if not message:
            return JsonResponse({"error": "Empty message."}, status=400)

        # Per-user daily safety valve (shared limit with the setup assistant).
        since = timezone.now() - timezone.timedelta(days=1)
        recent = AgentMessage.objects.filter(
            conversation__user=request.user, role="user", created_at__gte=since).count()
        if recent >= settings.ASSISTANT_DAILY_MESSAGE_LIMIT:
            return JsonResponse(
                {"error": "Daily message limit reached. Try again later."}, status=429)

        device = None
        device_id = body.get("device_id")
        if device_id:
            device = Device.accessible(request.user).filter(pk=device_id).first()

        conv = None
        conv_id = body.get("conversation_id")
        if conv_id:
            conv = AgentConversation.objects.filter(pk=conv_id, user=request.user).first()
        if conv is None:
            conv = AgentConversation.objects.create(user=request.user, device=device)

        clean = redact_secrets(message)  # redact BEFORE persisting or sending
        AgentMessage.objects.create(conversation=conv, role="user", content=clean)

        history = [{"role": m.role, "content": m.content}
                   for m in conv.messages.all()]
        device_state = None
        if device:
            device_state = {"device_id": device.pk, "name": device.name,
                            "location": device.location, "lat": device.lat,
                            "lon": device.lon, **_online(device)}

        def event_stream():
            yield _sse({"type": "meta", "conversation_id": conv.pk})
            collected, meta = [], {}
            for ev in client.stream_reply(history, request.user, device_state=device_state):
                if ev["type"] == "text":
                    collected.append(ev["text"])
                elif ev["type"] == "done":
                    meta = ev.get("meta", {})
                yield _sse(ev)
            text = redact_secrets("".join(collected))
            if text:
                AgentMessage.objects.create(
                    conversation=conv, role="assistant", content=text, meta=meta)

        resp = StreamingHttpResponse(event_stream(), content_type="text/event-stream")
        resp["Cache-Control"] = "no-cache"
        resp["X-Accel-Buffering"] = "no"
        return resp
