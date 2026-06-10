# BeeMonitor — Guided Hardware Setup + AI Tutor/Debugger Design

**Status:** Proposal for review (not yet implemented)
**Author:** Drafted with Claude Code, 2026-06-10
**Goal:** Put the hardware setup instructions on the dashboard as a guided,
interactive walkthrough, and layer an AI assistant on top that tutors people
through the build and debugs problems (and code) *with* them.

---

## 1. Motivation

Today the only setup path is `hardware/README.md` — ~1,350 lines of dense,
correct, but intimidating instructions a non-expert (a teacher, a collaborating
researcher) has to read end-to-end on GitHub. We want:

1. **A guided walkthrough on the dashboard** — step-by-step, progress-tracked,
   resumable, with the user's *real* device key/endpoint pre-filled into every
   command, and live "is it working yet?" checks driven by actual device
   telemetry.
2. **An AI tutor/debugger** — a Claude-powered assistant embedded next to the
   walkthrough that knows where the user is, can read the device's real
   telemetry/logs, teaches rather than dumps answers, and helps debug failures
   and code interactively — with hard safety rails around destructive actions.

This builds directly on existing infrastructure: device creation already mints a
`bmk_device_` key and shows it once (`DeviceCreatedView`), telemetry already
streams heartbeats with service-health + WiFi state, and "online" is already
derived from `last_seen_at`. The walkthrough closes the loop between "I created a
device in the web UI" and "the physical unit is alive in the field."

---

## 2. How it fits the existing codebase

(From a survey of `beemonitor_web/`.)

- **Stack:** Django + Tailwind (CDN, green/"amber"-keyed EcoMorph palette) +
  Alpine.js + HTMX, WhiteNoise static, PWA service worker. `markdown-it-py` is
  already a dependency.
- **Apps:** `accounts, analysis, annotations, api, dashboard, developer, devices,
  docs, pwa, sources, training, videos`. New work lands in a **new `apps/setup/`**
  app (keeps the walkthrough state model + views self-contained) plus a small
  **`apps/assistant/`** for the AI tutor (or fold both into `setup/` initially).
- **Nav:** `templates/components/navbar.html` — brand = Analytics home,
  "Fine-tuning" dropdown, Devices link, Account dropdown. Add a **"Set up a
  device"** entry (and surface it prominently from the empty-state of the device
  list and from `devices/created.html`).
- **Device key reuse:** `DeviceCreatedView` already pops a one-shot `raw_key`
  from the session. The walkthrough reuses the *same* mechanism to bake the key
  + `BEEMONITOR_API_BASE` into copy-paste blocks. After the one-shot window the
  guide shows a placeholder + a "re-issue key" link (we never persist plaintext).
- **Telemetry we can poll for live validation** (already reported each beat):
  `recorder_active`, `uploader_active`, `cellular_active`, `wifi_state`,
  `wifi_ssid`, `storage_pct`, `code_commit`, `last_activity_at`, plus
  `last_seen_at`/online. This is the backbone of "auto-detect the device is
  online" without any new device-side code.
- **No LLM usage today.** No `anthropic` SDK anywhere; secrets come from env vars
  (and AWS Secrets Manager in prod). The assistant adds `anthropic` to
  `requirements.txt` and an `ANTHROPIC_API_KEY` setting. <do it all i will provide the key>

---

## 3. Design principles (from research)

Hardware/IoT onboarding research (NN/G wizards & smart-device onboarding, W3C
mobile a11y, Particle/Home Assistant/Azure DPS prior art) converges on five
high-leverage decisions, all adopted here:

1. **Interactive validation gating** — confirm each phase actually worked
   (service active, heartbeat received, frame captured) before advancing, shown
   as *live device state*, not a self-reported checkbox. This is the #1
   differentiator between a good and a frustrating setup.
2. **Pre-fill commands with the user's real values** + one-click copy.
   Transcription of keys/IDs is the top silent-failure cause.
3. **Server-side progress persistence + resume.** Field setup gets interrupted
   (power, signal, walking away). Treat it as a stateful process, not a page.
4. **Single source of truth** — generate both the web wizard *and*
   `hardware/README.md` from one structured content definition so they never
   drift.
5. **Mobile-first + accessible** — the real environment is a phone outdoors:
   vertical stepper, one action per screen, ≥44px targets, ARIA live regions for
   the polling status.

Supporting patterns: branch early on **WiFi-only vs cellular** unit (conditional
disclosure, not two wizards); show **expected output** ("what you should see",
incl. physical signals like LEDs) beside every command; **inline** error
callouts at the step, not in an appendix; make every block **idempotent /
re-runnable**; let the user **skip / "I'll fix later"** so they're never
hard-locked in the field; descriptive step labels, 3–6 top-level phases,
all-steps-visible-up-front.

---

## 4. Part A — The guided walkthrough

### 4.1 Structured content model (single source of truth)

Define the guide as **structured data** (Python module or YAML loaded at import,
versioned in-repo under `apps/setup/content/`). One record per step:

```
Phase  -> ordered group (3–6 total): e.g. Flash, Software, Configure,
          Services, Power (WittyPi), Connectivity
Step   -> { id, phase, title,
            concept (why/what, short),
            command_template (with {{device_key}}, {{api_base}}, {{hostname}}),
            expected_output (text + optional "physical signal" note),
            verify (id of a validation check, or null),
            common_errors [ {symptom, fix} ],
            time_estimate, difficulty, optional (bool),
            applies_to: wifi | cellular | both }
```

A small generator renders `hardware/README.md` from this same data (docs-as-code),
so the README becomes a *build artifact*, not a hand-maintained duplicate. Initial
content is a faithful port of the current Steps 0–10 (see README structure):
Step 0 flash · 1 code · 2 deps · 3 dirs+calibration seed · 4 camera focus ·
5 credentials/env · 6 services · 7 enable+boot · 8 WittyPi (field) ·
9 Pi Connect (optional) · 10 cellular (cellular-only). Steps 8/10 are gated by
the WiFi-vs-cellular branch.

### 4.2 Walkthrough state (resume)

New model `SetupSession` (in `apps/setup`):

```
SetupSession(user, device FK nullable, unit_type[wifi|cellular],
             current_step, created_at, updated_at)
SetupStepState(session, step_id, status[pending|active|passed|failed|skipped],
               last_checked_at, detail)
```

On return, drop the user at the first incomplete/non-validated step. State is
per-user+device so switching browser/phone resumes cleanly.

### 4.3 UI

- Extends `templates/base.html`; **vertical stepper** (Alpine `x-data`) with
  states completed/current/upcoming/**error**; all phases visible up front.
- Each step: concept blurb · pre-filled command block with **copy button**
  (reuse `devices/created.html` `copyKey()` pattern) · "what you should see" ·
  inline error callouts (Tailwind admonition) · a **Verify** button.
- **Continue is disabled until the step's verify check passes**, with a
  "Skip / I'll fix later" escape.
- Branch step early: "Which unit are you building? WiFi/bench vs Cellular field"
  → filters `applies_to`.
- Mobile: one action per screen, big touch targets, status in an ARIA live
  region.

### 4.4 Live validation endpoints (the differentiator)

JSON endpoints under `/setup/verify/<check_id>/?device=<pk>` that read **real**
state and return structured pass/fail. All are server-side; the browser polls
via Alpine every ~2–3s and renders plain-language status:

| Check | How (uses existing telemetry) |
|-------|-------------------------------|
| `device_online` | `last_seen_at` within `DEVICE_ONLINE_GRACE_SECONDS` (first heartbeat arrived) |
| `recorder_running` | latest heartbeat `recorder_active == true` |
| `uploader_running` | `uploader_active == true` |
| `cellular_up` | `cellular_active == true` (cellular units) |
| `wifi_connected` | `wifi_state == connected` + `wifi_ssid` present |
| `camera_ok` | a recent on-demand image arrived (reuse request-image flow) → show thumbnail |
| `storage_ok` | `storage_pct` present + sane |

Because these read heartbeat data we *already* collect, **no Pi-side code
changes are needed** for v1 validation. "Your device just came online ✓" flips
automatically when the first beat lands.

---

## 5. Part B — The AI tutor/debugger

### 5.1 Architecture: a workflow + augmented LLM (not an autonomous agent)

Django (ASGI; we're on App Runner) owns the step state machine and conversation
persistence. Each turn the backend assembles a Claude Messages API request and
streams the reply over SSE to a chat panel docked beside the walkthrough.

**Grounding (anti-hallucination), for our moderate doc set — recommended:**
- **Stuff the full setup guide + troubleshooting into context and cache it.**
  Prompt caching makes cache *reads* ~0.1× input price, so re-sending the guide
  every turn costs ~10% of nominal. Order is strictly tools → system → messages;
  put `cache_control:{"type":"ephemeral"}` on the last *static* block.
- **Enable Citations** (`document` blocks, `citations.enabled`) using
  **custom-content** documents (one block per step/troubleshooting bullet) so a
  citation maps cleanly to "Step 4." `cited_text` is free on output. Note:
  **Citations are incompatible with Structured Outputs** — don't use both.
- **RAG/embeddings: deferred.** Only worth it if the corpus grows large or
  changes constantly. Single guide → context-stuffing + citations wins.
- System-prompt hardening: explicitly allow **"I don't know"**, restrict answers
  to the provided guide, quote-first for long sections.

**Context layers per turn:**
- *Cached static prefix:* role/persona, dual tutor↔debugger mode rules, safety
  policy, the full guide (cached citation document blocks), read-only tool defs.
- *Small volatile block (after the cache breakpoint):* the user's **current
  walkthrough step** + a tiny device-health snapshot (online / last error).
  Keeping volatile data out of the cached prefix is critical — leaking
  timestamps/telemetry into it kills the cache hit.
- *Conversation history:* API is stateless; we persist and resend it each turn,
  mostly as cache reads.

### 5.2 Pedagogy (teach, don't dump)

- Scaffold with hints + decomposition; Socratic "why/how" before correcting;
  formative, specific feedback on what the user actually typed; one sub-step at a
  time to manage cognitive load; preserve learner agency ("friendly but
  demanding").
- **Enforce step-at-a-time in app logic, not the prompt alone** — only ask Claude
  to help with the *current* step (research shows a single prompt can't reliably
  enforce scaffolding).
- **Dual mode:** Socratic/guided for learning steps, but **direct &
  prescriptive when the user is blocked on a concrete failure** (give the exact
  command + why). The system prompt encodes the switch.

### 5.3 Debugging *with* the user (tool use)

Claude pulls real evidence via **read-only client tools** backed by data we
already have:

- `get_device_status(device_id)` → online, last heartbeat, service states, code commit
- `get_recent_heartbeats(device_id, n)` → recent telemetry incl. errors
- `get_service_status(device_id, service)` → recorder/uploader/cellular flags
- `lookup_troubleshooting(symptom)` → returns matching guide chunks as citation docs

Tool-use loop: `stop_reason:"tool_use"` → backend executes → returns
`tool_result` → repeat until text. Keep the toolset small and unambiguous;
`strict:true` schemas. For things we can't read (a command on the Pi), Claude
asks the user to run a **read-only** diagnostic and paste output; pasted
output/logs are treated as **untrusted data** (prompt-injection surface), wrapped
in delimited blocks.

### 5.4 Safety rails (non-negotiable, the dominant risk)

- **Read-only by default.** Every tool Claude calls autonomously is read-only.
- **No destructive command auto-runs, ever.** Any write/delete/reformat/reboot/
  `rm -rf`/`dd`/`mkfs`/partition op is surfaced to the user as text with a plain-
  English consequence explanation and requires **explicit, typed UI
  confirmation**. A cheap **Haiku risk-classifier** screens any proposed command
  before a "copy/run" affordance appears.
- **Secret/PII redaction at the boundary.** Users will paste device keys, WiFi
  passwords, account IDs, logs. Redact via regex/NER **before** sending to the
  API and **before** persisting to our DB/logs. (Extends the existing
  "never put secrets in code/commits" rule to the LLM boundary.)
- Layered input (jailbreak/off-topic/secret) + output (no system-prompt leak,
  command-policy check) guards.

### 5.5 Cost & ops

- **Streaming** via SSE (async Django view / `StreamingHttpResponse`).
- **Model tiering:** default **Sonnet** worker; **Haiku** for routing / step
  classification / command risk-classification; **Opus** escalation for hard
  multi-step debugging (better at asking for missing info than guessing). Pin and
  re-verify model IDs/prices at build time.
- **Prompt caching** is what makes per-turn cost sane; 1-hour TTL if users pause
  mid-setup. Per-user request/token quotas at the Django layer (reuse existing
  rate-limit patterns). Confirm data-retention/ZDR terms for the account.

---

## 6. Phased delivery

**Phase 1 — Structured content + static walkthrough (no AI, no live checks).**
Port README Steps 0–10 into the content model; render the stepper + copy buttons
+ WiFi/cellular branch; generate `hardware/README.md` from the content to prove
single-source-of-truth. Nav entry + device-list empty-state CTA. *Ships value
immediately and de-risks everything downstream.*

**Phase 2 — Resume + live validation.** `SetupSession`/`SetupStepState`; the
verify endpoints reading existing telemetry; Continue-gating + Skip;
auto-"device online" detection. Mobile/a11y polish (ARIA live, touch targets).

**Phase 3 — AI tutor (guided mode).** `anthropic` dep + `ANTHROPIC_API_KEY`;
Messages API with cached guide + citations; SSE streaming chat panel; current-
step context injection; Sonnet default + "I don't know"/docs-only guardrails.
No tools yet — pure grounded Q&A about the current step.

**Phase 4 — AI debugger (tool use + safety).** Read-only client tools over
telemetry; dual tutor/debugger mode; Haiku risk-classifier; destructive-command
confirmation gate; secret redaction; injection framing of pasted logs. Opus
escalation path.

**Phase 5 — Polish & learn.** Real-novice usability test (cheapest high-impact
QA); per-step analytics ("where do people get stuck"); structured note-taking /
compaction for long debug sessions; optional RAG only if the corpus outgrows
context.

---

## 7. Open questions for review

1. **Scope of v1** — ship Phase 1 (static guided walkthrough) first and treat the
   AI tutor as a fast follow? (Recommended.)
2. **App boundary** — one `apps/setup/` app for both walkthrough + assistant, or
   split `apps/assistant/`? (Lean: start unified, split if it grows.)
3. **README as generated artifact** — OK to make `hardware/README.md` a build
   output of the content model (regenerated, not hand-edited)? This is the
   single-source-of-truth payoff but changes the editing workflow.
4. **Device-scoped vs generic guide** — require selecting/creating a device first
   (so commands are fully pre-filled), or allow a generic browse-only mode?
5. **AI budget** — acceptable per-user/session token quota + which model tier as
   default (Sonnet) given cost?
6. **Where the assistant lives** — docked panel beside the walkthrough only, or
   also a general help button across the dashboard?

---

## 8. Key references

UX/IoT onboarding: NN/G Wizards (nngroup.com/articles/wizards), NN/G Smart-Device
Onboarding (.../smart-device-onboarding), NN/G Status Trackers
(.../status-tracker-progress-update), grandcentrix IoT Onboarding, W3C Mobile
A11y touch targets, Particle device setup + Device Doctor, Home Assistant
on-device onboarding, Azure IoT Hub DPS / ThingsBoard provisioning, docs-as-code.

Claude API: prompt caching, citations, tool use (overview/how/implement),
streaming, Messages API, models overview, building-effective-agents,
effective-context-engineering, reduce-hallucinations, mitigate-jailbreaks — all
under platform.claude.com/docs and anthropic.com/engineering & /research.
(Full URLs captured in the research notes backing this doc.)
