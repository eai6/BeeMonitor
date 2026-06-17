#!/bin/bash
# Join this unit to your Tailscale tailnet with Tailscale SSH enabled — remote
# shell access through CGNAT with NO inbound port and no public exposure.
# Idempotent: safe to run on every boot (beemonitor-tailscale.service does).
#
# Needs TAILSCALE_AUTHKEY in /etc/beemonitor/uploader.env — a *tagged, reusable,
# ephemeral* pre-auth key from the Tailscale admin console (Settings → Keys). Tag
# it (e.g. tag:beemonitor) + lock it down with an ACL so a leaked key/unit can't
# roam your tailnet.
#
# Network model — same as rpi-connect on these units:
#   * WiFi:     always reachable on the tailnet.
#   * Cellular: gated OFF by the cellular firewall normally (tailscaled isn't in
#     the telemetry allowlist, so no metered data), and reachable only when you
#     DROP THE GATE — `cellular-firewall.sh open` / the dashboard's Cellular-access
#     debug toggle (open mode removes the nft table, so anything can use cellular).
#     tailscaled reconnects over cellular within its retry window while the gate is
#     open, then goes quiet again when it auto-re-gates. No firewall change needed.
#
# Do the FIRST join on WiFi or with the gate open (it must reach the Tailscale
# control plane once); tailscaled then persists the session and reconnects on its
# own on whatever link is permitted.
set -u

ENV_FILE="${BEEMONITOR_ENV_FILE:-/etc/beemonitor/uploader.env}"
# shellcheck disable=SC1090
[ -f "$ENV_FILE" ] && . "$ENV_FILE"
KEY="${TAILSCALE_AUTHKEY:-}"

log() { echo "tailscale-up: $*"; }

# Install tailscale if missing (first boot). Pre-baking it into the golden image
# is preferred so first boot doesn't depend on the installer / a network fetch.
if ! command -v tailscale >/dev/null 2>&1; then
    log "tailscale not installed — installing"
    curl -fsSL https://tailscale.com/install.sh | sh || { log "install failed (no route?)"; exit 0; }
fi

# Already authenticated + running? tailscaled persists + auto-reconnects, so don't
# re-auth (just let it ride whatever link the firewall currently permits).
state="$(tailscale status --json 2>/dev/null | tr -d ' \n' | grep -o '"BackendState":"[^"]*"' | head -1 | cut -d'"' -f4)"
if [ "$state" = "Running" ]; then
    log "already up ($(tailscale ip -4 2>/dev/null | head -1))"
    exit 0
fi

[ -n "$KEY" ] || { log "no TAILSCALE_AUTHKEY set — remote access disabled (skipping)"; exit 0; }

# Bring it up with Tailscale SSH. --accept-dns=false so it doesn't fight the
# pinned, immutable /etc/resolv.conf these units use (see cellular-up.sh).
log "joining tailnet as '$(hostname)' (Tailscale SSH on)"
if tailscale up --ssh --hostname="$(hostname)" --authkey="$KEY" --accept-dns=false; then
    log "up: $(tailscale ip -4 2>/dev/null | head -1)"
else
    log "up failed (control unreachable — need WiFi or the gate open) — will retry next boot"
fi
