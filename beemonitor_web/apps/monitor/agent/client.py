"""Claude client for the taxonomic monitoring agent (Phase 3).

A read-only, streaming tool-loop — a near-copy of ``apps/setup/assistant/client``
with taxonomy tools instead of setup tools. The agent reads the structured
Observations + telemetry/GPS through read-only tools and produces natural-language
insight; it never changes state. No-op (graceful notice) when ANTHROPIC_API_KEY
is unset. See ``memory/15_monitoring_agent_design.md`` §12.
"""

import json
import logging

from django.conf import settings

from .tools import TOOL_DEFS, run_tool

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 5

_SYSTEM = """You are the BeeMonitor Monitoring Agent — a field-ecology analyst for \
a network of camera hives that record insect activity. Each motion event ("activity") \
has a few cropped frames that BioCLIP identifies into a taxon, aggregated into \
observations; devices report GPS and telemetry.

How to help:
- GROUND every claim in the data. Use your read-only tools to look at the real \
observations, activities, and device context before answering — investigate, then \
report. Never invent a species, count, or date the data doesn't support.
- Speak at the right taxonomic level. Field crops are imperfect: if confidence is \
low or only genus/family is supported, say so rather than asserting a species. \
Report confidence when it matters.
- Use the location prior (region_species) to sanity-check identifications — flag a \
species that isn't plausible for the site, and note it may be a misidentification \
or a notable record.
- Summarize like an ecologist: which taxa, how often, when, and any change over \
time or anomaly worth a human's attention.

Boundaries:
- You can only READ. You never change device or data state.
- Never ask for or repeat secrets (device keys, passwords). If the user pastes \
one, ignore its value.

Keep answers concise and skimmable; lead with the finding."""


def _content_params(blocks):
    out = []
    for b in blocks:
        t = getattr(b, "type", None)
        if t == "text":
            out.append({"type": "text", "text": b.text})
        elif t == "tool_use":
            out.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
    return out


def _system_blocks(device_state: "dict | None"):
    blocks = [{
        "type": "text",
        "text": _SYSTEM,
        "cache_control": {"type": "ephemeral"},  # cache the static prefix (~0.1x reads)
    }]
    if device_state:  # volatile context AFTER the cache breakpoint
        blocks.append({"type": "text",
                       "text": "Current device context: " + json.dumps(device_state)})
    return blocks


def is_enabled() -> bool:
    return bool(settings.ANTHROPIC_API_KEY)


def stream_reply(messages, user, device_state=None):
    """Generator of event dicts: text / tool / done / error (see setup assistant)."""
    if not is_enabled():
        yield {"type": "error", "text": "The monitoring agent isn't configured "
               "(no ANTHROPIC_API_KEY set on the server)."}
        return
    try:
        import anthropic
    except Exception:  # pragma: no cover
        yield {"type": "error", "text": "The anthropic package isn't installed."}
        return

    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    system = _system_blocks(device_state)
    convo = list(messages)
    full_text, tools_used = [], []

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

            convo.append({"role": "assistant", "content": _content_params(final.content)})
            results = []
            for block in final.content:
                if getattr(block, "type", None) == "tool_use":
                    tools_used.append(block.name)
                    yield {"type": "tool", "name": block.name}
                    out = run_tool(block.name, block.input or {}, user)
                    results.append({"type": "tool_result", "tool_use_id": block.id,
                                    "content": json.dumps(out)})
            convo.append({"role": "user", "content": results})
        yield {"type": "done", "text": "".join(full_text), "meta": {"tools": tools_used}}
    except Exception as e:  # pragma: no cover - API/network hiccup
        logger.exception("monitor agent stream failed")
        yield {"type": "error", "text": f"Agent error: {e}"}
