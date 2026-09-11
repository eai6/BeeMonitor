#!/usr/bin/env bash
# BeeMonitor — switch the camera overlay from the Arducam OwlSight to the OV5647.
#
# WHY: this unit's config.txt still pins dtoverlay=ov64a40 with camera_auto_detect=0.
# With a night-vision module on the ribbon, the ov64a40 driver probes 0x36, reads
# chip id 0x56476c (= 0x5647 plus a trailing byte the OV5647 does not have) and
# refuses to bind. libcamera then enumerates zero cameras, so every Picamera2()
# raises IndexError and beemonitor-recorder crash-loops. The sensor itself is fine —
# it ACKs and returns an ID; only the driver is wrong.
#
# Backs config.txt up first, and refuses to reboot if the edit did not take.
#
# Usage:
#   sudo scripts/switch-to-ov5647.sh              # edit config.txt, then prompt to reboot
#   sudo scripts/switch-to-ov5647.sh --no-reboot  # edit only; reboot yourself later
#   sudo scripts/switch-to-ov5647.sh --revert     # put dtoverlay=ov64a40 back

set -uo pipefail

CFG="${BEEMONITOR_BOOT_CONFIG:-/boot/firmware/config.txt}"
FROM=ov64a40
TO=ov5647

SUDO=""
[ "$(id -u)" -ne 0 ] && SUDO=sudo

step() { printf '\n\033[1m== %s\033[0m\n' "$1"; }
pass() { printf '  \033[32mok\033[0m   %s\n' "$1"; }
warn() { printf '  \033[33mwarn\033[0m %s\n' "$1"; }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$1"; }

reboot_after=1
case "${1:-}" in
  --no-reboot) reboot_after=0 ;;
  --revert)    FROM=ov5647; TO=ov64a40 ;;
  "")          ;;
  *) echo "unknown option: $1 (see the header for usage)" >&2; exit 2 ;;
esac

step "Checking $CFG"
[ -r "$CFG" ] || { bad "$CFG not readable — is this the Pi?"; exit 1; }

if grep -qE "^\s*dtoverlay=${TO}\b" "$CFG"; then
  pass "dtoverlay=$TO is already set — nothing to change"
  grep -nE "^\s*(dtoverlay=(${FROM}|${TO})|camera_auto_detect)" "$CFG"
  exit 0
fi

if ! grep -qE "^\s*dtoverlay=${FROM}\b" "$CFG"; then
  bad "no active 'dtoverlay=${FROM}' line found — not editing blind"
  warn "camera lines currently in $CFG:"
  grep -nE "^\s*(dtoverlay=|camera_auto_detect)" "$CFG" || echo "  (none)"
  exit 1
fi

step "Backing up"
BAK="$CFG.bak-$(date +%Y%m%d-%H%M%S)"
$SUDO cp -a "$CFG" "$BAK" || { bad "backup failed — refusing to edit"; exit 1; }
pass "backup: $BAK"

step "Swapping the overlay"
# Keep any trailing overlay parameters off: ov64a40 options (link-frequency,
# media-controller) are not ov5647 options and would fail to load.
$SUDO sed -i -E "s|^(\s*)dtoverlay=${FROM}\b.*|\1dtoverlay=${TO}|" "$CFG"

if ! grep -qE "^\s*dtoverlay=${TO}\b" "$CFG"; then
  bad "edit did not take — restoring $BAK"
  $SUDO cp -a "$BAK" "$CFG"
  exit 1
fi
pass "dtoverlay=$TO"
grep -nE "^\s*(dtoverlay=${TO}|camera_auto_detect)" "$CFG" | sed 's/^/       /'

step "Next"
cat <<NOTE
  The overlay only binds at boot — a reboot is required. After it comes back:

      rpicam-hello --list-cameras      # expect an $TO entry
      systemctl status beemonitor-recorder   # expect: active (running)

  To undo:  sudo scripts/switch-to-ov5647.sh --revert   (or restore the .bak above)
NOTE

if [ "$reboot_after" -eq 1 ]; then
  printf '\n  Reboot now? [y/N] '
  read -r ans
  case "$ans" in
    [yY]*) step "Rebooting"; $SUDO reboot ;;
    *)     warn "not rebooting — the camera stays down until you do" ;;
  esac
fi
