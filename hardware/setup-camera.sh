#!/usr/bin/env bash
#
# BeeMonitor — Arducam 64MP OwlSight (OV64A40) bring-up (run on the Pi).
#
# WHY THIS EXISTS: neither Arducam 64MP module is auto-detectable. `camera_auto_detect`
# only recognises official Pi sensors (ov5647, imx219/477/708...), so a stock
# config.txt leaves an OwlSight completely invisible — `rpicam-hello --list-cameras`
# reports "No cameras available!" and every Picamera2() call site raises. The sensor
# needs an explicit `dtoverlay=ov64a40`.
#
# The OwlSight is the *native* 64MP part: the ov64a40 kernel driver and the
# ov64a40.json libcamera tuning both ship with Raspberry Pi OS. Do NOT run
# Arducam's install_pivariety_pkgs.sh here — that is for the 64MP Hawkeye and it
# replaces system libcamera/rpicam-apps with Arducam's fork, which would break the
# fleet updater (update.sh) and the golden image. Nothing in this script installs
# packages; it only loads a device-tree overlay.
#
# SAFETY: verifies at RUNTIME first (no reboot, nothing persistent) and only edits
# config.txt once a sensor has actually enumerated. config.txt is backed up first,
# and --revert undoes the edit.
#
# Usage:
#   ./hardware/setup-camera.sh               # probe, then persist to config.txt
#   ./hardware/setup-camera.sh --probe-only  # probe only, never touch config.txt
#   ./hardware/setup-camera.sh --force       # persist even if the runtime probe
#                                            #   fails (some sensors only bind when
#                                            #   the overlay is applied at boot)
#   ./hardware/setup-camera.sh --revert      # remove the overlay from config.txt
#
# Exit status is non-zero if the camera could not be brought up.

set -uo pipefail

CFG="${BEEMONITOR_BOOT_CONFIG:-/boot/firmware/config.txt}"
OVERLAY=ov64a40
MARKER="# Arducam 64MP OwlSight (OV64A40) — BeeMonitor"
# Read by provision.sh: a deliberate --revert must not be undone by the next update.
REVERT_MARKER="# BeeMonitor: camera overlay removed by setup-camera.sh --revert"
# 456 MHz is the overlay default; 360 MHz is the documented fallback and the one a
# 2-lane Pi 4 CSI link more often needs. Probe in that order.
LINK_FREQS=("" "link-frequency=360000000")

SUDO=""
[ "$(id -u)" -ne 0 ] && SUDO=sudo

step() { printf '\n\033[1m== %s\033[0m\n' "$1"; }
pass() { printf '  \033[32mok\033[0m   %s\n' "$1"; }
warn() { printf '  \033[33mwarn\033[0m %s\n' "$1"; }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$1"; }

mode=persist
case "${1:-}" in
  --probe-only) mode=probe ;;
  --force)      mode=force ;;
  --revert)     mode=revert ;;
  "")           ;;
  *) echo "unknown option: $1 (see the header for usage)" >&2; exit 2 ;;
esac

camera_present() {
  rpicam-hello --list-cameras 2>&1 | grep -q "Available cameras"
}

# Is the sensor electrically alive? The overlay puts the device on a muxed i2c bus
# at 0x36 (see `ov64a40@36` in the DT). If 0x36 answers, the ribbon and power are
# fine and any remaining failure is software/clocks; if nothing answers, it's
# wiring. Only meaningful while the overlay is loaded — the bus is created by it.
i2c_sensor_responds() {
  local bus addr found=1
  for bus in $(ls /dev/i2c-* 2>/dev/null | sed 's|.*/i2c-||' | sort -n); do
    # Skip the buses that exist without the overlay (arm + HDMI DDC).
    case "$bus" in 1|20|21|22) continue ;; esac
    for addr in "-y" "-y -r"; do
      if i2cdetect $addr "$bus" 2>/dev/null | grep -qE '^30:.* 36'; then
        pass "sensor ACKs at 0x36 on i2c-$bus — ribbon and power are good"
        found=0
      fi
    done
    [ $found -eq 0 ] && break
    warn "no ACK at 0x36 on i2c-$bus"
  done
  return $found
}

# --- revert -----------------------------------------------------------------
if [ "$mode" = revert ]; then
  step "Removing the OwlSight overlay from $CFG"
  if ! grep -qE "^\s*dtoverlay=${OVERLAY}" "$CFG"; then
    pass "nothing to remove"
    exit 0
  fi
  $SUDO cp -a "$CFG" "$CFG.bak-$(date +%Y%m%d-%H%M%S)"
  $SUDO sed -i "\|^${MARKER}\$|d; \|^\s*dtoverlay=${OVERLAY}|d" "$CFG"
  # provision.sh adds this overlay on units whose config.txt says nothing about
  # cameras; leave it a marker so a deliberate revert survives the next update.
  printf '%s\n' "$REVERT_MARKER" | $SUDO tee -a "$CFG" >/dev/null
  pass "overlay line removed (camera_auto_detect left as-is)"
  echo "  reboot to apply"
  exit 0
fi

# --- don't fight the recorder for the camera --------------------------------
step "Stopping the recorder so it can't grab the camera mid-setup"
if systemctl is-active --quiet beemonitor-recorder.service; then
  $SUDO systemctl stop beemonitor-recorder.service && pass "recorder stopped"
else
  # With no camera it sits in activating/auto-restart rather than active — stop it
  # anyway so its ~11s restart loop doesn't race the probe.
  $SUDO systemctl stop beemonitor-recorder.service 2>/dev/null
  pass "recorder was not running"
fi
echo "  (re-enable when you're done:  sudo systemctl start beemonitor-recorder)"

# --- runtime probe ----------------------------------------------------------
working=""
probed=0
wired=0
if camera_present; then
  step "A camera already enumerates — skipping the overlay probe"
  pass "sensor present"
  probed=1
else
  for lf in "${LINK_FREQS[@]}"; do
    desc="${lf:-link-frequency=456000000 (default)}"
    step "Trying: dtoverlay=$OVERLAY $lf"
    if ! $SUDO dtoverlay "$OVERLAY" $lf 2>&1; then
      warn "overlay failed to load"
      continue
    fi
    sleep 3
    if camera_present; then
      pass "sensor enumerated at $desc"
      working="$lf"
      probed=1
      break
    fi
    warn "no camera at $desc"
    # Split wiring faults from software faults while the bus still exists.
    i2c_sensor_responds && wired=1
    dmesg | tail -12 | grep -iE "ov64|csi|unicam|i2c|probe" || true
    $SUDO dtoverlay -r "$OVERLAY" 2>/dev/null
    sleep 1
  done
fi

if [ "$probed" -eq 0 ]; then
  step "Runtime probe did not enumerate the sensor"
  bad "no camera"
  if [ "$wired" -eq 1 ]; then
    cat <<'EOF'
  The sensor ACKed on i2c, so the ribbon, orientation and power are FINE — this is
  a software/clock fault, not wiring. On a Pi 4 the firmware sets up the CSI clocks
  at boot, which a runtime `dtoverlay` cannot do. Persist and reboot:
       ./hardware/setup-camera.sh --force && sudo reboot
  If it still fails after a cold boot, try the overlay variants in order:
       dtoverlay=ov64a40,link-frequency=360000000
       dtoverlay=ov64a40,media-controller=off
EOF
  else
    cat <<'EOF'
  The sensor did not ACK at 0x36, so it is probably not electrically present:
    1. Ribbon not seated, or inserted backwards. On a Pi 4 the contacts face the
       HDMI side; the blue stiffener faces the USB/Ethernet side.
    2. Wrong cable. Pi 5 kits ship a 22-pin cable that does not fit the Pi 4's
       15-pin connector — you need the 15-pin cable.
    3. Check the ribbon at the CAMERA end too — it's easy to leave that one half
       seated inside the enclosure.
  Note an i2c non-response is not 100% conclusive (the sensor's regulator may be
  off until the driver binds), so a cold-boot test is still worth doing:
       ./hardware/setup-camera.sh --force && sudo reboot
EOF
  fi
  [ "$mode" = force ] || exit 1
  warn "--force given: persisting anyway"
fi

# --- sensor modes: these drive the recorder's capture config -----------------
if [ "$probed" -eq 1 ]; then
  step "Sensor modes"
  echo "  At 64MP the full mode is 9152x6944 at ~2-3fps. motion/recorder.py must"
  echo "  pin a binned mode explicitly or picamera2 may pick the full one and"
  echo "  collapse the 25fps capture loop."
  echo
  rpicam-hello --list-cameras 2>&1
fi

[ "$mode" = probe ] && { step "--probe-only: leaving $CFG untouched"; exit 0; }

# --- persist ----------------------------------------------------------------
step "Persisting to $CFG"
# Setting it up again clears any previous --revert marker (see provision.sh).
$SUDO sed -i "\|^${REVERT_MARKER}\$|d" "$CFG"
if grep -qE "^\s*dtoverlay=${OVERLAY}" "$CFG"; then
  pass "dtoverlay=$OVERLAY already present — leaving it alone"
else
  $SUDO cp -a "$CFG" "$CFG.bak-$(date +%Y%m%d-%H%M%S)" || { bad "backup failed"; exit 1; }
  pass "backup: $CFG.bak-*"
  # A third-party sensor needs auto-detect off, or the firmware's own probing
  # fights the explicit overlay.
  $SUDO sed -i 's/^\s*camera_auto_detect=1/camera_auto_detect=0/' "$CFG"
  pass "camera_auto_detect=0"
  # The file's last section is [all], so appending lands in an all-models section.
  printf '\n%s\ndtoverlay=%s%s\n' "$MARKER" "$OVERLAY" "${working:+,${working}}" \
    | $SUDO tee -a "$CFG" >/dev/null
  pass "dtoverlay=$OVERLAY${working:+,${working}}"
fi

step "Next"
cat <<'EOF'
  sudo reboot                      # confirm it comes up from cold
  rpicam-hello --list-cameras      # verify after reboot
  ./hardware/setup-camera.sh --revert   # back out if it misbehaves
EOF
