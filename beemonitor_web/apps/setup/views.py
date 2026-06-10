"""Guided walkthrough views.

The walkthrough is device-scoped: pick (or create) a device, choose the unit
type (WiFi vs cellular — this branches which steps apply), then work through the
phases. Progress persists in SetupSession/SetupStepState so it resumes, and live
checks gate the experience on real device state.
"""

import logging

from django.conf import settings
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect
from django.utils import timezone
from django.utils.text import slugify
from django.views import View
from django.views.generic import TemplateView

from apps.devices.models import Device

from . import content
from .checks import run_check
from .models import SetupSession, SetupStepState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _device_values(request, device) -> dict:
    """Per-device placeholder substitutions for command blocks."""
    key = None
    if device is not None:
        # Peeked (not popped) so the key stays usable across the whole setup.
        key = (request.session.get(f"setup_key:{device.pk}")
               or request.session.get(f"device_key:{device.pk}"))
    return {
        "device_key": key or content.DEFAULTS["device_key"],
        "api_base": settings.BEEMONITOR_DEVICE_API_BASE,
        "record_dir": content.DEFAULTS["record_dir"],
        "hostname": (slugify(device.name) or "beemonitor") if device else "beemonitor",
        "_has_key": bool(key),
    }


def _session_for(user, device) -> SetupSession:
    sess, _ = SetupSession.objects.get_or_create(user=user, device=device)
    return sess


def _state_map(session) -> dict:
    return {s.step_id: s for s in session.step_states.all()}


def _ordered_steps(unit_type: str):
    # unset → show the WiFi subset as the default preview until they choose.
    ut = unit_type if unit_type in ("wifi", "cellular") else "wifi"
    return content.steps_for(ut)


def _next_pending(session, unit_type) -> str:
    states = _state_map(session)
    for st in _ordered_steps(unit_type):
        s = states.get(st["id"])
        if not s or s.status in ("pending", "active", "failed"):
            return st["id"]
    return ""


def _build_view_phases(session, unit_type, values):
    """Group steps by phase with per-step status + rendered commands."""
    states = _state_map(session) if session else {}
    steps = _ordered_steps(unit_type)
    by_phase = {}
    total = done = 0
    for st in steps:
        state = states.get(st["id"])
        status = state.status if state else "pending"
        total += 1
        if status in ("passed", "skipped"):
            done += 1
        by_phase.setdefault(st["phase"], []).append({
            **st,
            "rendered_command": content.render_command(st["command"], values),
            "status": status,
            "state_detail": state.detail if state else "",
        })
    phases = []
    for p in content.PHASES:
        if p["id"] in by_phase:
            phases.append({**p, "steps": by_phase[p["id"]]})
    pct = int(round(done / total * 100)) if total else 0
    return phases, pct, done, total


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

class SetupIndexView(LoginRequiredMixin, TemplateView):
    """Pick a device to set up (or create one)."""

    template_name = "setup/index.html"

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        devices = []
        for d in Device.accessible(self.request.user):
            sess = SetupSession.objects.filter(
                user=self.request.user, device=d).first()
            devices.append({
                "device": d,
                "unit_type": sess.unit_type if sess else "unset",
                "in_progress": bool(sess and sess.current_step),
            })
        ctx["devices"] = devices
        return ctx


class WalkthroughView(LoginRequiredMixin, TemplateView):
    """The guided stepper for one device."""

    template_name = "setup/walkthrough.html"

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        device = get_object_or_404(
            Device.accessible(self.request.user), pk=kwargs["pk"])
        session = _session_for(self.request.user, device)
        if not session.current_step:
            first = _ordered_steps(session.unit_type)[0]["id"]
            session.current_step = first
            session.save(update_fields=["current_step"])

        values = _device_values(self.request, device)
        phases, pct, done, total = _build_view_phases(
            session, session.unit_type, values)

        ctx.update({
            "device": device,
            "session": session,
            "unit_chosen": session.unit_type in ("wifi", "cellular"),
            "unit_type": session.unit_type,
            "phases": phases,
            "progress_pct": pct,
            "steps_done": done,
            "steps_total": total,
            "current_step": session.current_step,
            "has_key": values["_has_key"],
            "assistant_enabled": bool(settings.ANTHROPIC_API_KEY),
        })
        return ctx


class SetUnitTypeView(LoginRequiredMixin, View):
    """POST: choose WiFi vs cellular (the branch)."""

    def post(self, request, pk):
        device = get_object_or_404(Device.accessible(request.user), pk=pk)
        session = _session_for(request.user, device)
        unit = request.POST.get("unit_type")
        if unit in ("wifi", "cellular"):
            session.unit_type = unit
            session.current_step = _ordered_steps(unit)[0]["id"]
            session.save(update_fields=["unit_type", "current_step"])
        return redirect("setup:walkthrough", pk=pk)


class StepActionView(LoginRequiredMixin, View):
    """POST: complete / skip / goto a step (self-attested transitions)."""

    def post(self, request, pk):
        device = get_object_or_404(Device.accessible(request.user), pk=pk)
        session = _session_for(request.user, device)
        action = request.POST.get("action")
        step_id = request.POST.get("step_id", "")
        if content.step_by_id(step_id) is None:
            return JsonResponse({"error": "unknown step"}, status=400)

        if action == "goto":
            session.current_step = step_id
            session.save(update_fields=["current_step"])
            return JsonResponse({"ok": True, "current_step": step_id})

        status = {"complete": "passed", "skip": "skipped"}.get(action)
        if not status:
            return JsonResponse({"error": "unknown action"}, status=400)
        state, _ = SetupStepState.objects.get_or_create(
            session=session, step_id=step_id)
        state.status = status
        state.last_checked_at = timezone.now()
        state.save()
        session.current_step = _next_pending(session, session.unit_type)
        session.save(update_fields=["current_step"])
        return JsonResponse({"ok": True, "status": status,
                            "current_step": session.current_step})


class VerifyView(LoginRequiredMixin, View):
    """GET JSON: run a live check against the real device and record the result.

    Polled by the stepper; flips a step to passed/failed automatically when the
    device's actual state changes (e.g. first heartbeat → 'device online ✓').
    """

    def get(self, request, pk, check_id):
        device = get_object_or_404(Device.accessible(request.user), pk=pk)
        result = run_check(check_id, device)

        # Record against any step whose verify == this check.
        session = _session_for(request.user, device)
        for st in _ordered_steps(session.unit_type):
            if st.get("verify") == check_id:
                state, _ = SetupStepState.objects.get_or_create(
                    session=session, step_id=st["id"])
                if result["status"] == "pass":
                    state.status = "passed"
                elif result["status"] == "fail" and state.status != "passed":
                    state.status = "failed"
                state.detail = result["detail"][:300]
                state.last_checked_at = timezone.now()
                state.save()
        return JsonResponse(result)
