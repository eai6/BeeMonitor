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
total_to_copy=0; done_count=0; STATE="idle"

# Write a small JSON status the dashboard can read (via telemetry). Called on
# every exit path (incl. mid-copy progress) so "no USB found" / "mount failed" /
# "copying 45%" / "done" / "ejected" all surface — that's how the user knows
# when it's safe to remove the stick.
write_status() {  # ok(true|false) detail
    mkdir -p "$STATE_DIR" 2>/dev/null
    local pct=0
    [ "$total_to_copy" -gt 0 ] && pct=$(( done_count * 100 / total_to_copy ))
    [ "$STATE" = "done" ] && pct=100
    printf '{"at":"%s","state":"%s","ok":%s,"detail":"%s","device":"%s","copied":%s,"skipped":%s,"failed":%s,"done":%s,"total":%s,"pct":%s,"bytes":%s,"human":"%s"}\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$STATE" "$1" "$2" "$STATUS_DEV" \
        "$copied" "$skipped" "$failed" "$done_count" "$total_to_copy" "$pct" "$bytes" "$human" \
        > "$STATUS_FILE" 2>/dev/null
    chmod 644 "$STATUS_FILE" 2>/dev/null
}
fail() { STATE=error; log "$2"; write_status false "$1"; exit 1; }

# --eject: safely unmount any plugged-in USB so the user can pull it out. Runs
# instead of a copy (dashboard "Eject" button -> usb_eject command).
do_eject() {
    STATE=ejected
    if mountpoint -q /run/beemonitor-usb 2>/dev/null; then
        sync; umount /run/beemonitor-usb 2>/dev/null && rmdir /run/beemonitor-usb 2>/dev/null
    fi
    root_src="$(findmnt -no SOURCE / 2>/dev/null)"
    root_disk="$(lsblk -no PKNAME "$root_src" 2>/dev/null | head -1)"
    for disk in $(lsblk -rno NAME,TYPE,TRAN | awk '$2=="disk" && $3=="usb" {print $1}'); do
        [ -n "$root_disk" ] && [ "$disk" = "$root_disk" ] && continue
        STATUS_DEV="/dev/$disk"
        for mp in $(lsblk -rno MOUNTPOINT "/dev/$disk" 2>/dev/null | grep -v '^$'); do
            sync; umount "$mp" 2>/dev/null
        done
        udisksctl power-off -b "/dev/$disk" >/dev/null 2>&1 || true
    done
    log "ejected — safe to remove the USB"
    write_status true "ejected — safe to remove the USB"
    exit 0
}
[ "${1:-}" = "--eject" ] && do_eject

# Device identity for the manifest — the key PREFIX only, never the secret.
# DEV_ID is a short, stable hash of the key; it's the FALLBACK folder suffix
# ("<hostname>-<dev-id>") used only when no dashboard location is known, so two
# devices that both default to the hostname "raspberrypi" still get distinct
# folders. The hash can't be reversed to the key. Empty when no key is set.
DEV_KEY=""
[ -f "$ENV_FILE" ] && DEV_KEY="$(grep -E '^BEEMONITOR_DEVICE_KEY=' "$ENV_FILE" 2>/dev/null | cut -d= -f2-)"
DEV_PREFIX="$(printf '%s' "$DEV_KEY" | cut -c1-16)"
DEV_ID="$(printf '%s' "$DEV_KEY" | sha256sum 2>/dev/null | cut -c1-8)"
[ -n "$DEV_KEY" ] || DEV_ID=""

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
    # Auto-detect a plugged-in USB drive by TRANSPORT (tran==usb), not the
    # HOTPLUG flag. On the Pi the flag is inverted: the USB stick reports
    # HOTPLUG=0 while the SD card reports HOTPLUG=1, so the old hotplug
    # heuristic picked the SD card's /boot/firmware partition instead of the
    # stick. Transport is the dependable signal. For each USB disk take its
    # first partition with a filesystem (or a whole-disk fs for unpartitioned
    # sticks); skip the disk backing the OS root in case the Pi boots from USB.
    root_src="$(findmnt -no SOURCE / 2>/dev/null)"
    root_disk="$(lsblk -no PKNAME "$root_src" 2>/dev/null | head -1)"
    dev=""
    for disk in $(lsblk -rno NAME,TYPE,TRAN | awk '$2=="disk" && $3=="usb" {print $1}'); do
        [ -n "$root_disk" ] && [ "$disk" = "$root_disk" ] && continue
        cand="$(lsblk -rno NAME,TYPE,FSTYPE "/dev/$disk" | awk '$2=="part" && $3!="" {print $1; exit}')"
        [ -n "$cand" ] || cand="$(lsblk -rno NAME,TYPE,FSTYPE "/dev/$disk" | awk '$2=="disk" && $3!="" {print $1; exit}')"
        [ -n "$cand" ] && { dev="/dev/$cand"; break; }
    done
    [ -n "$dev" ] || fail "no_usb" "no USB drive found (plug one in, or pass a mountpoint / /dev/sdX1)"
    STATUS_DEV="$dev"
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

# Per-device subfolder named after the device's dashboard LOCATION, so one stick
# can hold several hives in clearly-labelled folders (e.g. "north_hedgerow").
# telemetry.py caches the location to STATE_DIR/location from the heartbeat;
# BEEMONITOR_USB_LABEL overrides it (handy for manual runs). When no location is
# known, fall back to "<hostname>-<dev-id>" — hostname alone isn't unique because
# Pis often all default to "raspberrypi".
fat_safe() {  # FAT-safe folder name: keep [A-Za-z0-9._-], collapse the rest to _
    printf '%s' "$1" | tr -c 'A-Za-z0-9._-' '_' | sed -E 's/_+/_/g; s/^[._-]+//; s/[._-]+$//'
}
LABEL="${BEEMONITOR_USB_LABEL:-}"
[ -n "$LABEL" ] || LABEL="$(cat "$STATE_DIR/location" 2>/dev/null)"
SUBDIR="$(fat_safe "$LABEL")"
[ -n "$SUBDIR" ] || SUBDIR="$HOST${DEV_ID:+-$DEV_ID}"
DEST="$TARGET/BeeMonitor/$SUBDIR"
mkdir -p "$DEST" || fail "not_writable" "cannot write to $DEST (USB read-only or full?)"
MANIFEST="$DEST/manifest.csv"
[ -f "$MANIFEST" ] || echo "relpath,bytes,mtime_utc,device_host,device_key_prefix" > "$MANIFEST"

# --- copy ----------------------------------------------------------------
cd "$RECORD_DIR" 2>/dev/null || fail "no_record_dir" "record dir $RECORD_DIR missing"

# Count how many clips we'll actually copy (those not already on this USB), so
# the dashboard can show a real percentage. Quick extra scan.
while IFS= read -r -d '' f; do
    if [ "$COPY_ALL" = "1" ] || [ ! -f "$f.usb" ]; then total_to_copy=$((total_to_copy+1)); fi
done < <(find . -type f -name '*.mp4' -print0)
STATE=running
write_status true "starting"
# Re-write progress every ~2% (and at least every file when there are few).
step=$(( total_to_copy / 50 )); [ "$step" -lt 1 ] && step=1

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
    done_count=$((done_count+1))
    if [ $(( done_count % step )) -eq 0 ]; then
        bytes_h="$(numfmt --to=iec "$bytes" 2>/dev/null || echo "${bytes}B")"
        human="$bytes_h"; write_status true "copying"
    fi
done < <(find . -type f -name '*.mp4' -print0)

sync
human="$(numfmt --to=iec "$bytes" 2>/dev/null || echo "${bytes}B")"
STATE=done
log "done: copied $copied, skipped $skipped (already on USB), failed $failed ($human) -> $DEST"
write_status true "copied $copied, skipped $skipped, failed $failed"

if [ -n "$MOUNTED_BY_US" ]; then
    umount "$MOUNTED_BY_US" 2>/dev/null && rmdir "$MOUNTED_BY_US" 2>/dev/null
    log "unmounted $MOUNTED_BY_US — safe to remove the USB"
fi
