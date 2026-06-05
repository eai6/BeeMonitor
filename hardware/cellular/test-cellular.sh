#!/bin/bash
# BeeMonitor cellular (QMI) test — run on the Pi.
#
#   git pull                                   # get this script
#   bash ~/BeeMonitor/hardware/cellular/test-cellular.sh          # safe checks
#   bash ~/BeeMonitor/hardware/cellular/test-cellular.sh --split  # full WiFi-off test
#
# DEFAULT MODE is safe over any connection (incl. SSH-over-WiFi): it tests the
# cellular link by binding pings to wwan0, so it never turns WiFi off.
#
# --split MODE proves the real split (telemetry flows on cellular, video holds):
# it drops WiFi for ~60s then turns it back ON automatically, running detached
# and logging to /tmp/cellular-split-test.log — so even if your WiFi SSH session
# drops, WiFi returns on its own and you reconnect and read the log.

set -u
DEV=/dev/cdc-wdm0
IFACE=wwan0
SPLIT_LOG=/tmp/cellular-split-test.log

pass=0; fail=0
ok(){ echo "  [ OK ] $*"; pass=$((pass+1)); }
no(){ echo "  [FAIL] $*"; fail=$((fail+1)); }
info(){ echo "  [info] $*"; }
hdr(){ echo; echo "== $* =="; }

# ---------------------------------------------------------------------------
# --split: detached WiFi-off → behavior check → WiFi-on (self-healing)
# ---------------------------------------------------------------------------
wifi_off(){ nmcli radio wifi off 2>/dev/null || sudo rfkill block wifi; }
wifi_on(){  nmcli radio wifi on  2>/dev/null || sudo rfkill unblock wifi; }

if [ "${1:-}" = "--split" ]; then
    echo "Running split test DETACHED. WiFi drops for ~60s then auto-returns."
    echo "Reconnect and read:  cat $SPLIT_LOG"
    nohup bash -c '
        LOG="'"$SPLIT_LOG"'"
        {
          echo "=== split test $(date) ==="
          echo "-- WiFi OFF"; nmcli radio wifi off 2>/dev/null || rfkill block wifi
          sleep 6
          echo "-- ping google.com (now only cellular can answer):"
          ping -c4 google.com
          echo "-- uploader (expect: holding video, no uploads):"
          journalctl -u beemonitor-uploader -n 8 --no-pager
          echo "-- telemetry (expect: still beating):"
          journalctl -u beemonitor-telemetry -n 8 --no-pager
          sleep 45
          echo "-- WiFi ON"; nmcli radio wifi on 2>/dev/null || rfkill unblock wifi
          sleep 8
          echo "-- uploader after WiFi back (expect: uploading/drained):"
          journalctl -u beemonitor-uploader -n 8 --no-pager
          echo "=== done $(date) ==="
        } >"$LOG" 2>&1
    ' >/dev/null 2>&1 &
    disown
    exit 0
fi

# ---------------------------------------------------------------------------
# Default: safe link checks (no WiFi toggle)
# ---------------------------------------------------------------------------
hdr "1. Modem present"
lsusb | grep -iqE "quectel|telit|1bc7:1201" && ok "cellular modem on USB" || no "no modem in lsusb"
[ -e "$DEV" ] && ok "$DEV present" || no "$DEV missing"

hdr "2. cellular.service"
if systemctl is-active --quiet cellular.service; then
    ok "cellular.service active"
else
    no "cellular.service not active  (journalctl -u cellular.service -n 30)"
fi

hdr "3. wwan0 interface + IP"
addr=$(ip -o -4 addr show "$IFACE" 2>/dev/null | awk '{print $4}')
[ -n "$addr" ] && ok "$IFACE has IPv4: $addr" || no "$IFACE has no IPv4 address"

hdr "4. Connectivity over cellular (bound to $IFACE — safe with WiFi up)"
ping -c2 -W5 -I "$IFACE" 8.8.8.8 >/dev/null 2>&1 \
    && ok "ping 8.8.8.8 via $IFACE" || no "no ping via $IFACE"
getent hosts google.com >/dev/null 2>&1 \
    && ok "DNS resolves google.com" || no "DNS resolution failed"

hdr "5. BeeMonitor services"
for s in beemonitor-recorder beemonitor-telemetry beemonitor-uploader; do
    systemctl is-active --quiet "$s" && ok "$s active" || no "$s not active"
done

hdr "6. Recent telemetry beat"
if journalctl -u beemonitor-telemetry -n 30 --no-pager 2>/dev/null | grep -q "heartbeat ok"; then
    ok "telemetry posted a heartbeat recently"
else
    info "no 'heartbeat ok' in last 30 log lines (may just be between beats)"
fi

echo
echo "==================  $pass passed, $fail failed  =================="
if [ "$fail" -eq 0 ]; then
    echo "Cellular link looks good. For the full split test (video holds on"
    echo "cellular, drains on WiFi) run:  bash $0 --split"
else
    echo "See [FAIL] lines above. Most common: wrong APN in /etc/qmi-network.conf,"
    echo "ModemManager re-enabled, or cellular.service not installed/enabled."
fi
