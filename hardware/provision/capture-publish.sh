#!/bin/bash
# Capture a generalized BeeMonitor card into a publishable golden image — run on a
# LINUX box (macOS can't shrink ext4). Automates provision/README.md §1.3–1.4:
#   dd the whole card  →  pishrink (-aZ) to a compressed .img.xz  →  (optional) S3
#
# The dd step is irreversible if pointed at the wrong disk, so this REFUSES to run
# unless the target is a whole removable disk that is NOT the system disk, shows you
# exactly what it is, and makes you confirm. Read-only on the card itself.
#
# Usage:
#   sudo bash capture-publish.sh --device /dev/sdX
#   sudo bash capture-publish.sh --device /dev/sdX --bucket my-bucket
#   sudo bash capture-publish.sh --device /dev/sdX --bucket my-bucket \
#        --name beemonitor-golden --key-prefix images/ --out /data/img --yes
#
# Flags:
#   --device PATH     whole-card block device to read (e.g. /dev/sdb, /dev/mmcblk0)
#   --bucket NAME     S3 bucket to publish the .img.xz to (omit = local file only)
#   --key-prefix P    S3 key prefix (default: images/)
#   --name NAME       artifact base name (default: beemonitor-golden)
#   --out DIR         where to write artifacts (default: current dir)
#   --no-shrink       skip pishrink (uploads the raw, full-size .img — discouraged)
#   --keep-raw        keep the uncompressed .img after shrinking
#   --yes             don't prompt (CI/headless) — be certain of --device
set -euo pipefail

DEVICE=""; BUCKET=""; KEY_PREFIX="images/"; NAME="beemonitor-golden"
OUT="."; DO_SHRINK=1; KEEP_RAW=0; ASSUME_YES=0

die() { echo "capture-publish: $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --device) DEVICE="${2:-}"; shift 2 ;;
        --bucket) BUCKET="${2:-}"; shift 2 ;;
        --key-prefix) KEY_PREFIX="${2:-}"; shift 2 ;;
        --name) NAME="${2:-}"; shift 2 ;;
        --out) OUT="${2:-}"; shift 2 ;;
        --no-shrink) DO_SHRINK=0; shift ;;
        --keep-raw) KEEP_RAW=1; shift ;;
        --yes|-y) ASSUME_YES=1; shift ;;
        -h|--help) sed -n '2,28p' "$0"; exit 0 ;;
        *) die "unknown arg: $1" ;;
    esac
done

[ "$(id -u)" -eq 0 ] || die "must run as root (dd + pishrink need it) — use sudo."
[ -n "$DEVICE" ] || die "--device is required (the whole card, e.g. /dev/sdb)."
[ -b "$DEVICE" ] || die "$DEVICE is not a block device."
command -v lsblk >/dev/null || die "lsblk not found (util-linux)."

# --- Safety gate 1: must be a WHOLE disk, not a partition --------------------
TYPE="$(lsblk -dno TYPE "$DEVICE" 2>/dev/null || true)"
[ "$TYPE" = "disk" ] || die "$DEVICE is type '$TYPE', not a whole disk. Pass the card itself (e.g. /dev/sdb), not a partition (e.g. /dev/sdb1)."

# --- Safety gate 2: must NOT be the system / root disk ----------------------
root_src="$(findmnt -no SOURCE / 2>/dev/null || true)"
if [ -n "$root_src" ]; then
    root_pk="$(lsblk -no PKNAME "$root_src" 2>/dev/null | head -1 || true)"
    root_disk="/dev/${root_pk:-__none__}"
    [ "$DEVICE" != "$root_disk" ] || die "REFUSING: $DEVICE is the system/root disk. That would destroy this machine."
fi

# --- Safety gate 3: sanity on size + removability ----------------------------
DEV_BYTES="$(blockdev --getsize64 "$DEVICE" 2>/dev/null || echo 0)"
DEV_GB=$(( DEV_BYTES / 1000000000 ))
RM="$(lsblk -dno RM "$DEVICE" 2>/dev/null || echo 0)"
if [ "$RM" != "1" ]; then
    echo "capture-publish: ! WARNING — $DEVICE is not flagged removable. SD cards via a USB"
    echo "                 reader usually are. Double-check this is the card, not a fixed disk."
fi
if [ "$DEV_GB" -gt 512 ]; then
    die "REFUSING: $DEVICE is ${DEV_GB} GB (> 512 GB) — that's almost certainly a real disk, not an SD card. Override only by editing the script if you truly mean it."
fi

mkdir -p "$OUT"
RAW="${OUT%/}/${NAME}.img"
XZ="${RAW}.xz"

# --- Show the target and confirm --------------------------------------------
echo
echo "About to CAPTURE this device (read-only on the card):"
lsblk -o NAME,SIZE,TYPE,RM,MODEL,MOUNTPOINTS "$DEVICE" 2>/dev/null || lsblk "$DEVICE"
echo
echo "  source device : $DEVICE  (${DEV_GB} GB)"
echo "  raw image     : $RAW"
[ "$DO_SHRINK" -eq 1 ] && echo "  published     : $XZ" || echo "  published     : $RAW (--no-shrink)"
[ -n "$BUCKET" ] && echo "  upload to     : s3://${BUCKET}/${KEY_PREFIX%/}/$(basename "${XZ}")"
echo

# Disk-space check: dd writes a full card-sized .img before shrinking.
avail="$(df -B1 --output=avail "$OUT" 2>/dev/null | tail -1 | tr -d ' ' || echo 0)"
if [ "$avail" -gt 0 ] && [ "$avail" -lt "$DEV_BYTES" ]; then
    die "not enough free space in $OUT ($(( avail/1000000000 )) GB) for a ${DEV_GB} GB raw image."
fi

if [ "$ASSUME_YES" -ne 1 ]; then
    read -r -p "Type the device path again to confirm capture: " confirm
    [ "$confirm" = "$DEVICE" ] || die "confirmation '$confirm' != '$DEVICE' — aborted."
fi

# --- Unmount the card's partitions for a consistent read --------------------
for part in $(lsblk -lno NAME "$DEVICE" | tail -n +2); do
    mp="$(lsblk -no MOUNTPOINTS "/dev/$part" 2>/dev/null | tr -d ' ' || true)"
    if [ -n "$mp" ]; then
        echo "capture-publish: unmounting /dev/$part ($mp)"
        umount "/dev/$part" 2>/dev/null || die "could not unmount /dev/$part — close anything using the card and retry."
    fi
done

# --- 1. Capture --------------------------------------------------------------
echo "==> dd capture → $RAW"
dd if="$DEVICE" of="$RAW" bs=4M status=progress conv=fsync
sync

# --- 2. Shrink + compress ----------------------------------------------------
ARTIFACT="$RAW"
if [ "$DO_SHRINK" -eq 1 ]; then
    command -v pishrink.sh >/dev/null || die "pishrink.sh not found. Install it: https://github.com/Drewsif/PiShrink (then re-run, or pass --no-shrink)."
    echo "==> pishrink -aZ (shrink rootfs + first-boot resize + xz)"
    pishrink.sh -aZ "$RAW"          # -> ${RAW}.xz
    [ -f "$XZ" ] || die "expected $XZ after pishrink, but it's missing."
    ARTIFACT="$XZ"
    if [ "$KEEP_RAW" -eq 0 ]; then rm -f "$RAW"; fi
else
    echo "capture-publish: ! --no-shrink — publishing the full ${DEV_GB} GB raw image (large download)."
fi

SIZE_H="$(du -h "$ARTIFACT" | cut -f1)"
echo "==> artifact ready: $ARTIFACT ($SIZE_H)"

# --- 3. Publish (optional) ---------------------------------------------------
if [ -n "$BUCKET" ]; then
    command -v aws >/dev/null || die "aws CLI not found — install it or upload $ARTIFACT manually."
    KEY="${KEY_PREFIX%/}/$(basename "$ARTIFACT")"
    echo "==> aws s3 cp → s3://${BUCKET}/${KEY}"
    aws s3 cp "$ARTIFACT" "s3://${BUCKET}/${KEY}"
    echo
    echo "Published. Now make it downloadable and point the app at it:"
    echo "  • Serve it public-read (or via CloudFront / a long-lived presigned URL) —"
    echo "    it's a non-secret artifact (no keys baked in)."
    echo "  • Set in App Runner prod env (and local .env):"
    echo "      BEEMONITOR_GOLDEN_IMAGE_URL=https://<bucket-or-cdn>/${KEY}"
    echo "    The enrollment page's Download button appears once this is set."
else
    echo
    echo "Local artifact only (no --bucket). To publish:"
    echo "  aws s3 cp \"$ARTIFACT\" s3://<bucket>/${KEY_PREFIX%/}/$(basename "$ARTIFACT")"
    echo "  then set BEEMONITOR_GOLDEN_IMAGE_URL to its public URL."
fi
echo "Done."
