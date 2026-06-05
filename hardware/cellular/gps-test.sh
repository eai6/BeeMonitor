#!/bin/bash
# GPS test for the Sixfab Telit LE910C4-NF modem (USB 1bc7:1201) — run on the Pi
# to confirm the modem gives a fix BEFORE we wire GPS into telemetry + dashboard.
#
#   bash ~/BeeMonitor/hardware/cellular/gps-test.sh
#   bash ~/BeeMonitor/hardware/cellular/gps-test.sh /dev/ttyUSB2   # force a port
#
# No sudo needed: the login user is in the 'dialout' group, which can open
# /dev/ttyUSB*. (sudo still works if you prefer.)
#
# This is a TELIT modem, not Quectel — GPS uses AT$GPSP / AT$GPSACP, not the
# Quectel AT+QGPS / AT+QGPSLOC commands. The AT port is /dev/ttyUSB2 (ttyUSB3
# also answers AT). cdc-wdm0/wwan0 carry the QMI data link separately, so this
# test does not disturb the cellular connection.
#
# Needs: the GNSS antenna connected (Sixfab dual-antenna GNSS connector, not the
# main LTE port), a clear-ish sky view (near a window or outside), and patience —
# a cold first fix can take a few minutes.
#
# It auto-detects the AT port, powers GNSS on, and polls for a fix, printing the
# coordinates (and a map link). Tell me the AT port it finds + that it got a fix
# and I'll wire GPS back into the telemetry beat + dashboard.

# Raw-shell serial reads are unreliable on these Telit ports (DTR/termios), so we
# drive the port with pyserial, which the telemetry code already uses.
exec python3 - "$@" <<'PY'
import os, sys, time

try:
    import serial
except ImportError:
    sys.exit("pyserial not installed:  pip3 install pyserial   (or: sudo apt install python3-serial)")

CAND = sys.argv[1:] or ["/dev/ttyUSB2", "/dev/ttyUSB3", "/dev/ttyUSB1",
                        "/dev/ttyUSB0", "/dev/ttyUSB4"]
POLL = int(os.environ.get("POLL_SECONDS", "180"))


def at(ser, cmd, wait=1.5):
    ser.reset_input_buffer()
    ser.write((cmd + "\r\n").encode())
    time.sleep(wait)
    return ser.read(ser.in_waiting or 1).decode(errors="replace")


def dm_to_deg(tok):
    # Telit $GPSACP gives ddmm.mmmm[N/S] / dddmm.mmmm[E/W]. Degrees are all but
    # the last two integer digits; the rest are decimal minutes.
    if not tok or tok[-1] not in "NSEW" or "." not in tok:
        return None
    hemi, num = tok[-1], tok[:-1]
    dot = num.index(".")
    deg = int(num[:dot - 2] or "0")
    minutes = float(num[dot - 2:])
    val = deg + minutes / 60.0
    return -val if hemi in ("S", "W") else val


# 1. Find the AT port (answers OK to a bare AT).
port = ser = None
for p in CAND:
    try:
        t = serial.Serial(p, 115200, timeout=2)
    except Exception:
        continue
    if "OK" in at(t, "AT"):
        port, ser = p, t
        break
    t.close()

if not port:
    sys.exit("No AT port responded (tried: %s).\n"
             "  ls -l /dev/ttyUSB*   then re-run with that port as the arg."
             % " ".join(CAND))
print("== AT port: %s ==" % port)

# 2. Power GNSS on (Telit: $GPSP=1). Idempotent — an ERROR if already on is fine.
print("-- powering GNSS on (AT$GPSP=1):")
print("   " + at(ser, "AT$GPSP=1").strip().replace("\n", "\n   "))

# 3. Poll for a position fix.
#    $GPSACP: <UTC>,<lat>,<lon>,<hdop>,<alt>,<fix>,<cog>,<spkm>,<spkn>,<date>,<nsat>
#    <fix>: 0/1 = no fix, 2 = 2D, 3 = 3D.
print("-- polling AT$GPSACP for up to %ds (needs sky view)..." % POLL)
end = time.time() + POLL
while time.time() < end:
    resp = at(ser, "AT$GPSACP")
    hit = [ln for ln in resp.splitlines() if "$GPSACP:" in ln]
    if hit:
        body = hit[0].split("$GPSACP:", 1)[1].strip()
        f = body.split(",")
        fix = f[5] if len(f) > 5 else ""
        lat = dm_to_deg(f[1]) if len(f) > 1 else None
        lon = dm_to_deg(f[2]) if len(f) > 2 else None
        if fix in ("2", "3") and lat is not None and lon is not None:
            sats = f[10] if len(f) > 10 else "?"
            print("\n==> FIX  lat=%.6f  lon=%.6f  (fix=%sD, sats=%s)"
                  % (lat, lon, fix, sats))
            print("    https://www.openstreetmap.org/?mlat=%.6f&mlon=%.6f#map=16/%.6f/%.6f"
                  % (lat, lon, lat, lon))
            ser.close()
            sys.exit(0)
        print("   no fix yet… $GPSACP: %s" % body)
    else:
        print("   no fix yet… %s" % resp.strip().replace("\n", " ") or "(waiting)")
    time.sleep(5)

ser.close()
print("\nNo fix within %ds. Checklist:" % POLL)
print("  - GNSS antenna on the modem's GNSS port (Sixfab dual-antenna GNSS connector, not main LTE)?")
print("  - Clear sky view (window/outdoors)? Cold start can take several minutes.")
print("  - Re-run; once it gets a fix the location is cached and later fixes are fast.")
sys.exit(1)
PY
