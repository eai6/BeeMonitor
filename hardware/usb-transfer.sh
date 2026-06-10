#!/bin/bash
# Copy BeeMonitor recordings onto a plugged-in USB drive in the field (no laptop).
#
# Tracks each transfer with a per-clip ``<clip>.mp4.usb`` sidecar — the same
# pattern uploader.py uses with ``.uploaded`` for the cloud — so re-runs only
# copy NEW clips. It also writes a manifest.csv on the USB so the clips can later
# be uploaded to the cloud and attributed to the right device (Videos → Upload,
# pick the device; see the web "upload for a specific device" flow).
#
# Usage:
#   sudo bash usb-transfer.sh /media/pi/STICK      # copy to an already-mounted dir
#   sudo bash usb-transfer.sh /dev/sda1            # mount this partition, copy, unmount
#   sudo bash usb-transfer.sh                      # auto-detect a single removable mount
#   BEEMONITOR_USB_ALL=1 sudo bash usb-transfer.sh …   # re-copy everything (ignore sidecars)
#
# Auto-runs on plug-in via 99-beemonitor-usb.rules + beemonitor-usb-transfer@.service
# (gated to USB drives LABELLED "BEEMONITOR" so a random stick won't trigger it).
set -uo pipefail

RECORD_DIR="${BEEMONITOR_RECORD_DIR:-/home/beemonitor/Desktop/cameraOutput/beeHotel}"
ENV_FILE="${BEEMONITOR_ENV:-/etc/beemonitor/uploader.env}"
COPY_ALL="${BEEMONITOR_USB_ALL:-0}"
HOST="$(hostname)"

log() { echo "usb-transfer: $*"; }

# Device identity for the manifest — the key PREFIX only, never the secret.
DEV_PREFIX=""
[ -f "$ENV_FILE" ] && DEV_PREFIX="$(grep -E '^BEEMONITOR_DEVICE_KEY=' "$ENV_FILE" 2>/dev/null | cut -d= -f2- | cut -c1-16)"

# --- resolve the destination mount ---------------------------------------
TARGET="${1:-}"
MOUNTED_BY_US=""
if [ -z "$TARGET" ]; then
    # auto-detect: first mounted *removable* partition
    TARGET="$(lsblk -rno MOUNTPOINT,RM,TYPE | awk '$2==1 && $3=="part" && $1!="" {print $1; exit}')"
    [ -n "$TARGET" ] || { log "no USB mount found (pass a mountpoint or /dev/sdX1)"; exit 1; }
elif [ -b "$TARGET" ]; then
    mp="$(lsblk -no MOUNTPOINT "$TARGET" 2>/dev/null | head -1)"
    if [ -z "$mp" ]; then
        mp="/run/beemonitor-usb"; mkdir -p "$mp"
        mount "$TARGET" "$mp" || { log "mount $TARGET failed"; exit 1; }
        MOUNTED_BY_US="$mp"
    fi
    TARGET="$mp"
fi
[ -d "$TARGET" ] || { log "destination '$TARGET' is not a directory"; exit 1; }

DEST="$TARGET/BeeMonitor/$HOST"
mkdir -p "$DEST" || { log "cannot write to $DEST (USB read-only or full?)"; exit 1; }
MANIFEST="$DEST/manifest.csv"
[ -f "$MANIFEST" ] || echo "relpath,bytes,mtime_utc,device_host,device_key_prefix" > "$MANIFEST"

# --- copy ----------------------------------------------------------------
copied=0; skipped=0; failed=0; bytes=0
cd "$RECORD_DIR" 2>/dev/null || { log "record dir $RECORD_DIR missing"; exit 1; }
while IFS= read -r -d '' mp4; do
    rel="${mp4#./}"
    sidecar="$mp4.usb"
    if [ "$COPY_ALL" != "1" ] && [ -f "$sidecar" ]; then skipped=$((skipped+1)); continue; fi
    out="$DEST/$rel"
    mkdir -p "$(dirname "$out")"
    if cp -p "$mp4" "$out" 2>/dev/null && [ "$(stat -c%s "$mp4")" = "$(stat -c%s "$out")" ]; then
        sz="$(stat -c%s "$mp4")"; mt="$(date -u -r "$mp4" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null)"
        echo "$rel,$sz,$mt,$HOST,$DEV_PREFIX" >> "$MANIFEST"
        printf 'transferred=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$sidecar"
        copied=$((copied+1)); bytes=$((bytes+sz))
    else
        log "copy failed: $rel"; rm -f "$out" 2>/dev/null; failed=$((failed+1))
    fi
done < <(find . -type f -name '*.mp4' -print0)

sync
human="$(numfmt --to=iec "$bytes" 2>/dev/null || echo "${bytes}B")"
log "done: copied $copied, skipped $skipped (already on USB), failed $failed ($human) -> $DEST"

if [ -n "$MOUNTED_BY_US" ]; then
    umount "$MOUNTED_BY_US" 2>/dev/null && rmdir "$MOUNTED_BY_US" 2>/dev/null
    log "unmounted $MOUNTED_BY_US — safe to remove the USB"
fi
