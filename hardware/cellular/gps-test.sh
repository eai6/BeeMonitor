#!/bin/bash
# GPS test for the Sixfab/Quectel modem — run on the Pi to confirm the modem
# gives a fix BEFORE we wire GPS into telemetry + the dashboard.
#
#   sudo bash ~/BeeMonitor/hardware/cellular/gps-test.sh
#   sudo bash ~/BeeMonitor/hardware/cellular/gps-test.sh /dev/ttyUSB2   # force a port
#
# Needs: the GNSS/GPS antenna connected, a clear-ish sky view (near a window or
# outside), and patience — a cold first fix can take a few minutes.
#
# It auto-detects the AT port, enables GNSS, and polls for a fix, printing the
# coordinates (and a map link). Tell me the AT port it finds + that it got a fix
# and I'll wire GPS back into the telemetry beat + dashboard.
set -u

PORTS="${1:-/dev/ttyUSB2 /dev/ttyUSB3 /dev/ttyUSB1 /dev/ttyUSB0 /dev/ttyUSB4}"
POLL_SECONDS="${POLL_SECONDS:-180}"

# at <port> <cmd> -> echo ~2s of response
at() {
    local port="$1" cmd="$2"
    stty -F "$port" 115200 -echo raw 2>/dev/null || return 1
    exec 9<>"$port" 2>/dev/null || return 1
    printf '%s\r\n' "$cmd" >&9
    timeout 2 cat <&9
    exec 9>&- 2>/dev/null || true
}

# 1. Find the AT port (responds OK to a bare AT).
ATPORT=""
for p in $PORTS; do
    [ -e "$p" ] || continue
    if at "$p" "AT" | grep -q "OK"; then ATPORT="$p"; break; fi
done
if [ -z "$ATPORT" ]; then
    echo "No AT port responded (tried: $PORTS)."
    echo "List what exists:  ls -l /dev/ttyUSB*   then re-run with that port as the arg."
    exit 1
fi
echo "== AT port: $ATPORT =="

# 2. Enable GNSS (idempotent; 'already on' shows as an error — that's fine).
echo "-- enabling GNSS (AT+QGPS=1):"
at "$ATPORT" "AT+QGPS=1" | tr -d '\r' | sed '/^$/d;s/^/   /'

# 3. Poll for a position fix.
echo "-- polling AT+QGPSLOC=2 for up to ${POLL_SECONDS}s (needs sky view)..."
end=$(( $(date +%s) + POLL_SECONDS ))
while [ "$(date +%s)" -lt "$end" ]; do
    resp=$(at "$ATPORT" "AT+QGPSLOC=2" | tr -d '\r')
    line=$(printf '%s' "$resp" | grep '+QGPSLOC:')
    if [ -n "$line" ]; then
        lat=$(printf '%s' "$line" | sed 's/.*+QGPSLOC: *//' | cut -d, -f2)
        lon=$(printf '%s' "$line" | sed 's/.*+QGPSLOC: *//' | cut -d, -f3)
        echo
        echo "==> FIX  lat=$lat  lon=$lon"
        echo "    https://www.openstreetmap.org/?mlat=$lat&mlon=$lon#map=16/$lat/$lon"
        exit 0
    fi
    note=$(printf '%s' "$resp" | grep -iE 'error|QGPSLOC' | head -1)
    echo "   no fix yet… ${note:-(waiting)}"
    sleep 5
done

echo
echo "No fix within ${POLL_SECONDS}s. Checklist:"
echo "  - GNSS/GPS antenna connected to the modem's GNSS port (not the main LTE port)?"
echo "  - Clear sky view (window/outdoors)? Cold start can take several minutes."
echo "  - Re-run; once it gets a fix the location is cached and later fixes are fast."
exit 1
