#!/bin/bash
# Gate the cellular link (wwan0) so ONLY BeeMonitor telemetry can use mobile data.
#
# Why: when WiFi is down in the field, wwan0 becomes the default route and ANY
# process — apt / unattended-upgrades, snapd, rpi-connect, OS background jobs —
# will happily push bytes over your metered SIM. This installs an nftables egress
# allowlist on wwan0 that permits only:
#   * the beemonitor-telemetry.service cgroup (the heartbeat JSON, the on-demand
#     image, and the command poll — i.e. the only things meant to use cellular),
#   * DNS (53), DHCP renew (67) and NTP (123) — needed for the link + TLS clock,
#   * ICMP — the bring-up/watchdog ping and path-MTU discovery,
#   * established/related return traffic,
# and DROPS everything else leaving wwan0. WiFi (wlan0) and loopback are never
# touched, so normal operation over WiFi is unaffected.
#
# The video uploader is deliberately NOT in the allowlist: even if its WiFi-only
# guard were misconfigured, bulk video still can't escape over cellular.
#
# TWO-PHASE LOAD (the telemetry allow rule matches a cgroup that only exists
# once beemonitor-telemetry.service is running, so it can't be loaded early):
#
#   base       default-drop on wwan0 + link essentials, NO cgroup rule. Has no
#              dependency on a running service, so cellular-firewall.service loads
#              it EARLY (before cellular.service) — this closes the leak window.
#   telemetry  the same table PLUS the telemetry-cgroup allow rule. Applied by
#              beemonitor-telemetry.service's ExecStartPost once its cgroup
#              exists, and re-applied on every (re)start so the cgroup id is
#              re-resolved after a restart. (default mode)
#
# Re-run any time:  sudo bash cellular-firewall.sh base       # phase 1 only
#                   sudo bash cellular-firewall.sh telemetry  # full (needs telemetry up)
set -u

MODE="${1:-telemetry}"
IFACE="${BEEMONITOR_CELL_IFACE:-wwan0}"
TABLE="beemon_cell"
# systemd service whose traffic is allowed on cellular (the telemetry beat).
TELEMETRY_UNIT="${BEEMONITOR_TELEMETRY_UNIT:-beemonitor-telemetry.service}"

if ! command -v nft >/dev/null 2>&1; then
    echo "cellular-firewall: nft not found — install with 'sudo apt install nftables'" >&2
    exit 1
fi

case "$MODE" in
    base)
        # No cgroup rule — telemetry is also dropped, but it isn't running yet
        # at this phase, so nothing legitimate is blocked. Safe to load early.
        TELEMETRY_RULE="# (telemetry allow rule added in phase 2 once its cgroup exists)"
        ;;
    telemetry)
        # nft resolves this cgroup path to an id AT LOAD TIME, so the cgroup must
        # already exist (i.e. beemonitor-telemetry.service has started).
        TELEMETRY_RULE="socket cgroupv2 level 2 \"system.slice/${TELEMETRY_UNIT}\" accept"
        ;;
    *)
        echo "cellular-firewall: unknown mode '$MODE' (use 'base' or 'telemetry')" >&2
        exit 2
        ;;
esac

# Replace any previous version of our table atomically.
nft list table inet "$TABLE" >/dev/null 2>&1 && nft delete table inet "$TABLE"

if ! nft -f - <<EOF
table inet ${TABLE} {
    chain output {
        type filter hook output priority 0; policy accept;

        # Only police the cellular interface; WiFi + loopback are unrestricted.
        oifname != "${IFACE}" accept

        # Return traffic for connections we already allowed out.
        ct state established,related accept

        # Link essentials (all tiny): ICMP (watchdog ping + PMTU), DNS, DHCP
        # renew, NTP clock sync.
        meta l4proto { icmp, ipv6-icmp } accept
        udp dport { 53, 67, 123 } accept
        tcp dport 53 accept

        # The telemetry service — the ONLY app allowed to use mobile data.
        ${TELEMETRY_RULE}

        # Everything else leaving cellular is blocked (apt, snap, rpi-connect, …).
        counter drop comment "cellular-blocked-nontelemetry"
    }
}
EOF
then
    echo "cellular-firewall: failed to load nftables ruleset ($MODE) — cellular is NOT gated" >&2
    exit 1
fi

if [ "$MODE" = "telemetry" ]; then
    echo "cellular-firewall: egress on ${IFACE} gated to ${TELEMETRY_UNIT} (+DNS/DHCP/NTP/ICMP)"
else
    echo "cellular-firewall: base drop on ${IFACE} loaded (+DNS/DHCP/NTP/ICMP); telemetry allow pending phase 2"
fi
