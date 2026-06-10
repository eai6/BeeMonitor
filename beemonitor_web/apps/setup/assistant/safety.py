"""Safety rails for the assistant: secret redaction + destructive-command gating.

Two independent jobs:

1. ``redact_secrets`` scrubs device keys, tokens, passwords and the like out of
   anything a user pastes *before* it is sent to the Claude API or persisted to
   our DB/logs. Extends the project rule "never put secrets in code/commits" to
   the LLM boundary.
2. ``classify_command`` flags shell commands the assistant proposes so the UI
   can require explicit confirmation before offering a one-click run. The
   assistant itself never executes anything — these are advisory rails for the
   human.
"""

import re

REDACTION = "[REDACTED]"

# Ordered (pattern, replacement) — applied to every inbound user message.
_SECRET_PATTERNS = [
    # BeeMonitor device keys.
    (re.compile(r"bmk_device_[A-Za-z0-9_\-]+"), REDACTION),
    # Bearer / Authorization headers.
    (re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)\S+"), r"\1" + REDACTION),
    # AWS access key ids + secret access keys.
    (re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), REDACTION),
    (re.compile(r"(?i)(aws_secret_access_key\s*[:=]\s*)\S+"), r"\1" + REDACTION),
    # KEY/TOKEN/PASSWORD/SECRET assignments in env/conf lines.
    (re.compile(r"(?i)\b([A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD|PWD)[A-Z0-9_]*\s*[:=]\s*)\S+"),
     r"\1" + REDACTION),
    # nmcli wifi password argument.
    (re.compile(r"(?i)(password\s+)\S+"), r"\1" + REDACTION),
    # Long opaque tokens (40+ url-safe chars) — last, to avoid eating normal words.
    (re.compile(r"\b[A-Za-z0-9_\-]{40,}\b"), REDACTION),
]


def redact_secrets(text: str) -> str:
    if not text:
        return text
    out = text
    for pat, repl in _SECRET_PATTERNS:
        out = pat.sub(repl, out)
    return out


# Patterns that make a proposed command destructive (irreversible / dangerous).
_DESTRUCTIVE = [
    re.compile(r"\brm\s+(-[a-zA-Z]*\s+)*-?[a-zA-Z]*[rf]"),  # rm -rf / rm -fr / rm -r
    re.compile(r"\bdd\s+.*\bof=/dev/"),
    re.compile(r"\bmkfs(\.\w+)?\b"),
    re.compile(r"\b(fdisk|parted|wipefs|sgdisk)\b"),
    re.compile(r">\s*/dev/sd"),
    re.compile(r"\bchattr\s+\+i\b.*resolv"),         # info: pins DNS immutable
    re.compile(r"\b(shutdown|reboot|halt|poweroff)\b"),
    re.compile(r"\bchmod\s+-R?\s*777\b"),
    re.compile(r":\(\)\s*\{.*\};:"),                  # fork bomb
    re.compile(r"\bmkfs|\bdd\b"),
]

# Patterns worth a heads-up but not destructive.
_CAUTION = [
    re.compile(r"\bsystemctl\s+(stop|disable|mask)\b"),
    re.compile(r"\bsudo\b"),
    re.compile(r"\bapt(-get)?\s+(remove|purge)\b"),
    re.compile(r"\bkill(all)?\b"),
]


def classify_command(cmd: str) -> str:
    """Return 'destructive' | 'caution' | 'safe' for a shell command string."""
    if not cmd:
        return "safe"
    for pat in _DESTRUCTIVE:
        if pat.search(cmd):
            return "destructive"
    for pat in _CAUTION:
        if pat.search(cmd):
            return "caution"
    return "safe"
