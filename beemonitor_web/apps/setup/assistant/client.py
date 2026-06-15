"""Claude client for the setup tutor/debugger.

A workflow-with-augmented-LLM, not an autonomous agent: Django owns the step
state and conversation; each turn we assemble a request with a cached static
prefix (role + rules + the full setup guide) plus a small volatile block (the
user's current step + live device snapshot), then run a short read-only tool
loop and stream the text back.

Grounding: the whole guide is stuffed into a cached system block (cache reads are
~0.1x), so the model answers from our docs. Tools are read-only. The assistant
proposes — it never executes — and is told to flag any state-changing command for
explicit human confirmation.
"""

import json
import logging

from django.conf import settings

from .tools import TOOL_DEFS, run_tool
from .. import content

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 4


def _content_params(blocks):
    """Rebuild assistant content blocks as clean input params.

    The SDK's model_dump() carries response-only fields (e.g. parsed_output)
    that the Messages API rejects on input, so we keep only what's allowed.
    """
    out = []
    for b in blocks:
        t = getattr(b, "type", None)
        if t == "text":
            out.append({"type": "text", "text": b.text})
        elif t == "tool_use":
            out.append({"type": "tool_use", "id": b.id,
                        "name": b.name, "input": b.input})
    return out

_SYSTEM_RULES = """You are the BeeMonitor Setup Assistant — a patient tutor and \
debugging copilot for people building a Raspberry Pi bee-monitoring field device \
(camera + motion-gated recorder, cellular/WiFi telemetry, systemd services).

How to help:
- GROUND every factual claim in the setup guide provided below. If the guide \
doesn't cover something, say so plainly — it is better to say "I'm not sure" \
than to guess. Do not invent commands, file paths, or service names.
- TEACH, don't dump. When the user is learning a step, guide with short \
explanations and one action at a time. Ask a clarifying question when their \
situation is ambiguous.
- BUT when the user is clearly blocked on a concrete failure, switch to direct, \
prescriptive mode: give the exact command and explain why in one line.
- Use your read-only tools to look at the user's REAL device (status, recent \
heartbeats) before theorizing. Investigate, then advise.

Safety — non-negotiable:
- You can only read. You never run anything on the device.
- If you recommend a command that changes state (deletes/formats/reboots/stops \
a service/edits system files), present it as a copy-paste block, state plainly \
what it will do and that it is irreversible if so, and tell the user to confirm \
before running it. Never present a destructive command as routine.
- Never ask for or repeat secrets (device keys, passwords, tokens). If the user \
pastes one, ignore its value.

Keep answers concise and skimmable. Reference the current step the user is on."""


def _system_blocks(current_step: str, device_state: dict | None):
    guide = content.as_markdown()
    static = (_SYSTEM_RULES +
              "\n\n=== BEEMONITOR SETUP GUIDE (your knowledge base) ===\n" + guide)
    blocks = [{
        "type": "text",
        "text": static,
        # Cache the big static prefix; cache reads cost ~10% of input.
        "cache_control": {"type": "ephemeral"},
    }]
    # Volatile context AFTER the cache breakpoint so it never busts the cache.
    ctx_lines = []
    if current_step:
        st = content.step_by_id(current_step)
        if st:
            ctx_lines.append(f"User's current step: \"{st['title']}\" "
                             f"(phase: {st['phase']}).")
    else:
        # Support mode (no setup step): the device is already deployed from the
        # golden image. Act as an ongoing support/debug copilot — diagnose live
        # issues with your tools and help fix them, and explain the dashboard
        # controls (telemetry rate, power schedule, motion tuning, WiFi/cellular,
        # software update, bee confirmation, crop-upload cap). The setup guide
        # below is background on how the system works, not a script to walk.
        ctx_lines.append(
            "Mode: SUPPORT for an already-deployed (golden-image) unit — not a "
            "fresh build. Investigate the live device first, then help the user "
            "diagnose, fix, or understand it (including the dashboard controls).")
    if device_state:
        ctx_lines.append("Live device snapshot: " + json.dumps(device_state))
    if ctx_lines:
        blocks.append({"type": "text", "text": "\n".join(ctx_lines)})
    return blocks


def is_enabled() -> bool:
    return bool(settings.ANTHROPIC_API_KEY)


def stream_reply(messages, user, current_step="", device_state=None):
    """Generator of event dicts:

        {"type": "text", "text": "..."}     incremental answer tokens
        {"type": "tool", "name": "..."}     a read-only tool was invoked
        {"type": "done", "text": "...", "meta": {...}}
        {"type": "error", "text": "..."}
    """
    if not is_enabled():
        yield {"type": "error", "text": "The AI assistant isn't configured "
               "(no ANTHROPIC_API_KEY set on the server)."}
        return

    try:
        import anthropic
    except Exception:  # pragma: no cover
        yield {"type": "error", "text": "The anthropic package isn't installed "
               "on the server."}
        return

    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    system = _system_blocks(current_step, device_state)
    convo = list(messages)  # [{role, content}]
    full_text = []
    tools_used = []

    try:
        for _ in range(MAX_TOOL_ROUNDS + 1):
            with client.messages.stream(
                model=settings.ASSISTANT_MODEL,
                max_tokens=settings.ASSISTANT_MAX_TOKENS,
                system=system,
                tools=TOOL_DEFS,
                messages=convo,
            ) as stream:
                for text in stream.text_stream:
                    full_text.append(text)
                    yield {"type": "text", "text": text}
                final = stream.get_final_message()

            if final.stop_reason != "tool_use":
                break

            # Run the requested read-only tools, feed results back, loop.
            convo.append({"role": "assistant",
                          "content": _content_params(final.content)})
            results = []
            for block in final.content:
                if getattr(block, "type", None) == "tool_use":
                    tools_used.append(block.name)
                    yield {"type": "tool", "name": block.name}
                    out = run_tool(block.name, block.input or {}, user)
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(out),
                    })
            convo.append({"role": "user", "content": results})
        yield {"type": "done", "text": "".join(full_text),
               "meta": {"tools": tools_used}}
    except Exception as e:  # pragma: no cover - API/network hiccup
        logger.exception("assistant stream failed")
        yield {"type": "error", "text": f"Assistant error: {e}"}
