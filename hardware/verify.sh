#!/usr/bin/env bash
#
# BeeMonitor — mechanical verification steps (run on the Pi).
#
# Covers the probeable, copy-error-prone parts of hardware/VERIFY.md:
#   Step 0  — backend health, venv import check, service restart
#
# The human-eye dashboard checks (online badge, activity graph, photo appearing,
# nav, WittyPi) stay in VERIFY.md — this script can't see them.
#
# Usage:
#   cd ~/BeeMonitor && git pull
#   ./hardware/verify.sh                 # all checks
#   ./hardware/verify.sh --no-restart    # skip the systemctl restart
#
# Exit status is non-zero if any hard check (health, venv) fails.

set -uo pipefail

API_BASE="${BEEMONITOR_API_BASE:-https://mqnafc3ejc.us-east-1.awsapprunner.com}"
VENV_PY="${VENV_PY:-$HOME/BeeMonitor/hardware/venv/bin/python}"
SERVICES="cellular beemonitor-recorder beemonitor-telemetry beemonitor-uploader"

do_restart=1
[ "${1:-}" = "--no-restart" ] && do_restart=0

fail=0
pass() { printf '  \033[32mok\033[0m   %s\n' "$1"; }
warn() { printf '  \033[33mwarn\033[0m %s\n' "$1"; }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$1"; fail=1; }
hdr()  { printf '\n\033[1m%s\033[0m\n' "$1"; }

# --- Step 0: backend health -------------------------------------------------
hdr "Step 0 — backend health ($API_BASE)"
code=$(curl -s -o /dev/null -w "%{http_code}" "$API_BASE/api/v1/health/" 2>/dev/null || true)
code=${code:-000}
if [ "$code" = "200" ]; then pass "health returned 200"; else bad "health returned $code (expected 200)"; fi

# --- Step 0: venv imports ---------------------------------------------------
hdr "Step 0 — venv import check ($VENV_PY)"
if [ ! -x "$VENV_PY" ]; then
  bad "venv python not found/executable at $VENV_PY"
elif "$VENV_PY" -c "import picamera2, cv2, requests" 2>/tmp/verify_venv.err; then
  pass "picamera2, cv2, requests import cleanly"
else
  bad "venv import failed:"; sed 's/^/         /' /tmp/verify_venv.err
fi

# --- Step 0: restart services ----------------------------------------------
hdr "Step 0 — services"
if [ "$do_restart" = "1" ]; then
  if sudo systemctl restart $SERVICES; then pass "restarted: $SERVICES"; else bad "restart failed"; fi
else
  warn "skipped restart (--no-restart)"
fi
for svc in $SERVICES; do
  state=$(systemctl is-active "$svc" 2>/dev/null || true)
  if [ "$state" = "active" ]; then pass "$svc active"; else warn "$svc is '$state'"; fi
done

# --- cellular link (the link itself; signal/GPS widgets were removed) -------
hdr "Cellular link"
if ip -o -4 addr show wwan0 >/dev/null 2>&1 && ip -o -4 addr show wwan0 | grep -q inet; then
  pass "wwan0 has an IPv4 address"
else
  warn "wwan0 has no IPv4 (only matters for the cellular link; WiFi units can ignore)"
fi

# --- summary ----------------------------------------------------------------
hdr "Next: human-eye checks in hardware/VERIFY.md (steps 1–6)"
echo "  Watch the telemetry log while you click through the dashboard:"
echo "    journalctl -u beemonitor-telemetry -f"

if [ "$fail" = "0" ]; then
  printf '\n\033[32mMechanical checks passed.\033[0m\n'; exit 0
else
  printf '\n\033[31mSome hard checks FAILED (see above).\033[0m\n'; exit 1
fi
