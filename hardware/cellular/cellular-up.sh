#!/bin/bash
# Sixfab 4G/LTE keep-alive (QMI mode) for unattended boot + WittyPi wake cycles.
#
# Brings wwan0 online via libqmi (qmi-network) + udhcpc, then monitors the link
# and re-establishes it if it drops. Runs as a long-lived systemd service
# (cellular.service) so it executes on every cold boot and WittyPi wake — the
# manual `qmi-network start` / `udhcpc` state does NOT survive a reboot.
#
# DNS is intentionally NOT managed here: /etc/resolv.conf is pinned with
# `chattr +i` (immutable) and is the single source of truth. Don't duplicate it.
#
# APN is read from /etc/qmi-network.conf (see qmi-network.conf.sample).
set -u

DEV=/dev/cdc-wdm0          # QMI control device
IFACE=wwan0                # network interface the modem presents
CHECK_HOST=8.8.8.8         # reachability probe target
CHECK_INTERVAL=60          # seconds between health checks
RETRY_WAIT=15              # seconds between bring-up attempts
CELL_METRIC=700            # default-route metric; must be > WiFi's (NM uses 600)
                           # so an available WiFi link is always preferred.

log() { echo "cellular: $*"; }

bring_up() {
    # 1. Wait for the modem to enumerate — cold boot / WittyPi wake is slow.
    for _ in $(seq 1 30); do
        [ -e "$DEV" ] && break
        sleep 2
    done
    [ -e "$DEV" ] || { log "modem $DEV not present"; return 1; }

    # 2. wwan0 is recreated fresh each boot; put it back in raw-IP mode
    #    (required by qmi_wwan for Quectel modems). Must be done link-down.
    #    Harmless if the modem doesn't expose this knob or it's already set.
    if [ -f "/sys/class/net/$IFACE/qmi/raw_ip" ]; then
        ip link set "$IFACE" down
        echo Y > "/sys/class/net/$IFACE/qmi/raw_ip"
    fi
    ip link set "$IFACE" up

    # 3. (Re)start the packet data session. APN comes from /etc/qmi-network.conf.
    qmi-network "$DEV" stop  >/dev/null 2>&1 || true
    qmi-network "$DEV" start || { log "qmi-network start failed"; return 1; }

    # 4. DHCP for IP + default route. -q quit after lease, -n fail if no lease.
    udhcpc -i "$IFACE" -q -n >/dev/null 2>&1 || { log "no DHCP lease"; return 1; }

    # 4b. udhcpc installs the default route at metric 0, which outranks WiFi
    #     (metric 600) and silently sends bulk video over metered cellular.
    #     Re-home it above WiFi so cellular is the fallback, not the default.
    GW=$(ip route show default dev "$IFACE" | awk '/default/ {print $3; exit}')
    if [ -n "$GW" ]; then
        ip route del default dev "$IFACE" 2>/dev/null || true
        ip route replace default via "$GW" dev "$IFACE" metric "$CELL_METRIC" \
            || log "warning: could not set cellular route metric"
    else
        log "warning: no default route on $IFACE after DHCP"
    fi

    # 5. Prove end-to-end reachability before declaring success.
    ping -c 2 -W 5 -I "$IFACE" "$CHECK_HOST" >/dev/null 2>&1 \
        || { log "no connectivity after bring-up"; return 1; }

    log "link up on $IFACE"
    return 0
}

# --- initial bring-up: retry forever (carrier attach can take minutes) -------
until bring_up; do
    log "bring-up failed, retrying in ${RETRY_WAIT}s"
    sleep "$RETRY_WAIT"
done

# --- watchdog: re-establish if the carrier drops the session ----------------
while true; do
    sleep "$CHECK_INTERVAL"
    if ! ping -c 2 -W 5 -I "$IFACE" "$CHECK_HOST" >/dev/null 2>&1; then
        log "link down — re-establishing"
        until bring_up; do
            log "re-establish failed, retrying in ${RETRY_WAIT}s"
            sleep "$RETRY_WAIT"
        done
    fi
done
