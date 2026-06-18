#!/usr/bin/env bash
# prepare-card.sh — drop a BeeMonitor enrollment token onto a freshly-flashed SD
# card so it self-registers on first boot. The CLI equivalent of the enrollment
# page's "Choose SD card & write" button, for Safari/Firefox/headless prep.
#
# The card must already be flashed with the BeeMonitor golden image and inserted.
# We only write to the FAT *boot* partition (config.txt lives there), so this needs
# no root and no ext4 tooling — it runs the same on macOS and Linux.
#
# Usage:
#   ./prepare-card.sh --token bmk_enroll_xxx [--api-base https://host] [--boot PATH]
#
# Get the token from the web app: Devices -> Zero-touch enrollment -> Generate token.
set -euo pipefail

API_BASE="https://mqnafc3ejc.us-east-1.awsapprunner.com"
TOKEN=""
BOOT=""
WIFI_SSID=""
WIFI_PASS=""

usage() {
  cat <<'EOF'
prepare-card.sh — drop a BeeMonitor enrollment token onto a flashed SD card so it
self-registers on first boot. Writes beemonitor.conf to the card's FAT boot
partition (works on macOS and Linux, no root needed).

Usage:
  ./prepare-card.sh --token bmk_enroll_xxx [--api-base https://host] \
                    [--wifi-ssid NAME --wifi-pass PASS] [--boot PATH]

  --token      enrollment token from Devices -> Add a device (required)
  --api-base   backend URL (default: the production App Runner host)
  --wifi-ssid  WiFi network the unit joins on first boot (optional)
  --wifi-pass  WiFi password (optional; omit for an open network)
  --boot       path to the card's boot partition (default: auto-detect bootfs/boot)

The card must already be flashed with the BeeMonitor golden image and inserted.
EOF
  exit "${1:-0}"
}

while [ $# -gt 0 ]; do
  case "$1" in
    --token)     TOKEN="${2:-}"; shift 2 ;;
    --api-base)  API_BASE="${2:-}"; shift 2 ;;
    --wifi-ssid) WIFI_SSID="${2:-}"; shift 2 ;;
    --wifi-pass) WIFI_PASS="${2:-}"; shift 2 ;;
    --boot)      BOOT="${2:-}"; shift 2 ;;
    -h|--help)  usage 0 ;;
    *) echo "Unknown argument: $1" >&2; usage 1 ;;
  esac
done

[ -n "$TOKEN" ] || { echo "Error: --token is required." >&2; usage 1; }
case "$TOKEN" in
  bmk_enroll_*) : ;;
  *) echo "Warning: token doesn't start with 'bmk_enroll_' — continuing anyway." >&2 ;;
esac
case "$API_BASE" in
  https://*) : ;;
  *) echo "Warning: --api-base is not https:// — the token would cross the wire in clear text." >&2 ;;
esac

# Find the boot partition if not given. Bookworm names it "bootfs"; older "boot".
if [ -z "$BOOT" ]; then
  for c in /Volumes/bootfs /Volumes/boot \
           "/media/$USER/bootfs" "/media/$USER/boot" \
           "/run/media/$USER/bootfs" "/run/media/$USER/boot"; do
    if [ -d "$c" ]; then BOOT="$c"; break; fi
  done
fi
[ -n "$BOOT" ] && [ -d "$BOOT" ] || {
  echo "Error: couldn't find the card's boot partition. Insert a flashed card," >&2
  echo "       or pass it explicitly: --boot /Volumes/bootfs" >&2
  exit 1
}

# Guard against clobbering the wrong volume — a Pi boot partition has config.txt.
if [ ! -e "$BOOT/config.txt" ] && [ ! -e "$BOOT/cmdline.txt" ]; then
  echo "Error: '$BOOT' doesn't look like a Raspberry Pi boot partition" >&2
  echo "       (no config.txt / cmdline.txt). Refusing to write." >&2
  exit 1
fi

CONF="$BOOT/beemonitor.conf"
printf 'BEEMONITOR_API_BASE=%s\nBEEMONITOR_ENROLL_TOKEN=%s\n' "$API_BASE" "$TOKEN" > "$CONF"
if [ -n "$WIFI_SSID" ]; then
  printf 'BEEMONITOR_WIFI_SSID=%s\n' "$WIFI_SSID" >> "$CONF"
  [ -n "$WIFI_PASS" ] && printf 'BEEMONITOR_WIFI_PASSWORD=%s\n' "$WIFI_PASS" >> "$CONF"
fi
sync

echo "Wrote $CONF"
echo "  API base: $API_BASE"
echo "  token:    bmk_enroll_…${TOKEN: -4}   (last 4 shown)"
[ -n "$WIFI_SSID" ] && echo "  wifi:     $WIFI_SSID"
echo "Eject the card, assemble the hardware, insert it, and power on."
echo "The unit enrolls itself on first boot and appears on your Devices page."
