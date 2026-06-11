#!/bin/bash
# Copy BeeMonitor recordings onto a plugged-in USB drive in the field (no laptop).
#
# Tracks each transfer with a per-clip ``<clip>.mp4.usb`` sidecar — the same
# pattern uploader.py uses with ``.uploaded`` for the cloud — so re-runs only
# copy NEW clips. It also writes a manifest.csv on the USB so the clips can later
# be uploaded to the cloud and attributed to the right device (Videos → Upload,
# pick the device; see the web "upload for a specific device" flow).
#
# A headless Pi has no "allow this USB" prompt like a desktop, so this script
# does the mounting itself and writes a STATUS FILE (usb-status.json) reporting
# whether a drive was found/mounted and how much was copied. telemetry.py ships
# that status in the heartbeat so the dashboard can show "copied N" or
# "no USB detected" — that's how you know the stick was actually visible.
#
# Usage:
#   sudo bash usb-transfer.sh /media/pi/STICK      # copy to an already-mounted dir
#   sudo bash usb-transfer.sh /dev/sda1            # mount this partition, copy, unmount
#   sudo bash usb-transfer.sh                      # auto-detect a single removable drive
#   BEEMONITOR_USB_ALL=1 sudo bash usb-transfer.sh …   # re-copy everything (ignore sidecars)
#
# Auto-runs on plug-in via 99-beemonitor-usb.rules + beemonitor-usb-transfer@.service
# (any USB partition triggers it — no special label needed).
set -uo pipefail

RECORD_DIR="${BEEMONITOR_RECORD_DIR:-/home/beemonitor/Desktop/cameraOutput/beeHotel}"
ENV_FILE="${BEEMONITOR_ENV:-/etc/beemonitor/uploader.env}"
COPY_ALL="${BEEMONITOR_USB_ALL:-0}"
STATE_DIR="${BEEMONITOR_STATE_DIR:-/home/beemonitor/.beemonitor}"
STATUS_FILE="$STATE_DIR/usb-status.json"
HOST="$(hostname)"

log() { echo "usb-transfer: $*"; }

copied=0; skipped=0; failed=0; bytes=0; human=""; STATUS_DEV=""

# Write a small JSON status the dashboard can read (via telemetry). Called on
# every exit path so "no USB found" / "mount failed" surface too, not just success.
write_status() {  # ok(true|false) detail
    mkdir -p "$STATE_DIR" 2>/dev/null
    printf '{"at":"%s","ok":%s,"detail":"%s","device":"%s","copied":%s,"skipped":%s,"failed":%s,"bytes":%s,"human":"%s"}\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" "$STATUS_DEV" \
        "$copied" "$skipped" "$failed" "$bytes" "$human" \
        > "$STATUS_FILE" 2>/dev/null
    chmod 644 "$STATUS_FILE" 2>/dev/null
}
fail() { log "$2"; write_status false "$1"; exit 1; }

# Device identity for the manifest — the key PREFIX only, never the secret.
DEV_PREFIX=""
[ -f "$ENV_FILE" ] && DEV_PREFIX="$(grep -E '^BEEMONITOR_DEVICE_KEY=' "$ENV_FILE" 2>/dev/null | cut -d= -f2- | cut -c1-16)"

# --- resolve the destination mount ---------------------------------------
TARGET="${1:-}"
MOUNTED_BY_US=""
mount_dev() {  # $1=device node -> echoes mountpoint, mounts if needed
    local d="$1" m
    m="$(lsblk -no MOUNTPOINT "$d" 2>/dev/null | head -1)"
    if [ -z "$m" ]; then
        m="/run/beemonitor-usb"; mkdir -p "$m"
        mount "$d" "$m" 2>/dev/null || return 1
        MOUNTED_BY_US="$m"
    fi
    echo "$m"
}

if [ -z "$TARGET" ]; then
    # Auto-detect ANY hot-plugged USB with a filesystem — partition first, then
    # a whole-disk filesystem (some sticks are formatted without a partition
    # table). Requiring a non-empty FSTYPE skips unmountable junk.
    part="$(lsblk -rno NAME,HOTPLUG,TYPE,FSTYPE | awk '$2==1 && $3=="part" && $4!="" {print $1; exit}')"
    [ -n "$part" ] || part="$(lsblk -rno NAME,HOTPLUG,TYPE,FSTYPE | awk '$2==1 && $3=="disk" && $4!="" {print $1; exit}')"
    [ -n "$part" ] || fail "no_usb" "no USB drive found (plug one in, or pass a mountpoint / /dev/sdX1)"
    dev="/dev/$part"; STATUS_DEV="$dev"
    mp="$(mount_dev "$dev")" || fail "mount_failed" "could not mount $dev (unsupported filesystem? try exfat/ntfs support)"
    TARGET="$mp"
    log "using USB $dev -> $mp"
elif [ -b "$TARGET" ]; then
    STATUS_DEV="$TARGET"
    mp="$(mount_dev "$TARGET")" || fail "mount_failed" "could not mount $TARGET"
    TARGET="$mp"
else
    STATUS_DEV="$TARGET"
fi
[ -d "$TARGET" ] || fail "bad_target" "destination '$TARGET' is not a directory"

DEST="$TARGET/BeeMonitor/$HOST"
mkdir -p "$DEST" || fail "not_writable" "cannot write to $DEST (USB read-only or full?)"
MANIFEST="$DEST/manifest.csv"
[ -f "$MANIFEST" ] || echo "relpath,bytes,mtime_utc,device_host,device_key_prefix" > "$MANIFEST"

# --- copy ----------------------------------------------------------------
cd "$RECORD_DIR" 2>/dev/null || fail "no_record_dir" "record dir $RECORD_DIR missing"
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
write_status true "copied $copied, skipped $skipped, failed $failed"

if [ -n "$MOUNTED_BY_US" ]; then
    umount "$MOUNTED_BY_US" 2>/dev/null && rmdir "$MOUNTED_BY_US" 2>/dev/null
    log "unmounted $MOUNTED_BY_US — safe to remove the USB"
fi
