#!/bin/bash
# Generalize a reference BeeMonitor Pi into a GOLDEN IMAGE — run on the Pi itself,
# as the last step before you capture the card (provision/README.md §1.2).
#
# It strips everything per-unit so clones don't share identity or leak a credential:
#   * removes BEEMONITOR_DEVICE_KEY / BEEMONITOR_ENROLL_TOKEN from uploader.env
#   * deletes any dropped boot-partition token (beemonitor.conf)
#   * blanks machine-id and SSH host keys (regenerated per clone on first boot)
#   * vacuums logs, clears test recordings + shell history
#   * VERIFIES no key/token survives, then (optionally) powers off
#
# The image keeps only BEEMONITOR_API_BASE — the per-card token is dropped in later
# (Part 2). The stable hw_id is the Pi CPU serial, so identical clones still enroll
# as distinct devices, which is why blanking machine-id is safe.
#
# Usage (on the reference Pi):
#   sudo bash generalize.sh            # generalize, confirm, then power off
#   sudo bash generalize.sh --yes      # no prompt
#   sudo bash generalize.sh --no-poweroff
set -u

ENV_FILE="${BEEMONITOR_ENV_FILE:-/etc/beemonitor/uploader.env}"
BOOT_CONFS=(/boot/firmware/beemonitor.conf /boot/beemonitor.conf)
# Test clips to clear. README default + whatever RECORD_DIR points at, if set.
RECORD_DIRS=(/home/beemonitor/Desktop/cameraOutput)

ASSUME_YES=0
DO_POWEROFF=1
for arg in "$@"; do
    case "$arg" in
        --yes|-y) ASSUME_YES=1 ;;
        --no-poweroff) DO_POWEROFF=0 ;;
        -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

if [ "$(id -u)" -ne 0 ]; then
    echo "generalize: must run as root (use sudo)." >&2
    exit 1
fi

# Sanity: this really is a BeeMonitor install, so we don't nuke a random Pi.
if [ ! -f "$ENV_FILE" ]; then
    echo "generalize: $ENV_FILE not found — is this a BeeMonitor reference unit?" >&2
    echo "            (set BEEMONITOR_ENV_FILE if your env lives elsewhere.)" >&2
    exit 1
fi

echo "generalize: preparing a GOLDEN IMAGE from this unit."
echo "  env file : $ENV_FILE"
if [ "$ASSUME_YES" -ne 1 ]; then
    echo
    read -r -p "This strips credentials + per-unit identity and (by default) powers off. Continue? [y/N] " ans
    case "$ans" in y|Y|yes|YES) ;; *) echo "aborted."; exit 0 ;; esac
fi

echo
echo "==> 1/5 Removing credentials from $ENV_FILE"
# Keep a one-time backup so a mistake is recoverable before capture.
cp -a "$ENV_FILE" "${ENV_FILE}.pregeneralize.bak" 2>/dev/null || true
sed -i '/^[[:space:]]*BEEMONITOR_DEVICE_KEY=/d;/^[[:space:]]*BEEMONITOR_ENROLL_TOKEN=/d' "$ENV_FILE"
for f in "${BOOT_CONFS[@]}"; do
    [ -e "$f" ] && { rm -f "$f" && echo "    removed boot token: $f"; }
done

echo "==> 2/5 Regenerating machine-id + SSH host keys per clone"
truncate -s 0 /etc/machine-id
rm -f /var/lib/dbus/machine-id 2>/dev/null || true
ln -sf /etc/machine-id /var/lib/dbus/machine-id 2>/dev/null || true
rm -f /etc/ssh/ssh_host_*
# Make sure host keys regenerate on next boot (service name varies by distro).
systemctl enable regenerate-ssh-host-keys 2>/dev/null \
    || systemctl enable ssh-host-keys-regeneration 2>/dev/null || true

echo "==> 3/5 Clearing logs, test recordings, shell history"
journalctl --rotate >/dev/null 2>&1 || true
journalctl --vacuum-time=1s >/dev/null 2>&1 || true
# Pull any extra RECORD_DIR from the env so we don't leave clips behind.
extra_rec="$(grep -E '^[[:space:]]*BEEMONITOR_RECORD_DIR=' "$ENV_FILE" 2>/dev/null | tail -1 | cut -d= -f2- | tr -d '"' )"
[ -n "${extra_rec:-}" ] && RECORD_DIRS+=("$extra_rec")
for d in "${RECORD_DIRS[@]}"; do
    [ -d "$d" ] && { rm -rf "${d:?}/"* 2>/dev/null || true; echo "    cleared recordings: $d"; }
done
rm -f /home/beemonitor/.bash_history /root/.bash_history 2>/dev/null || true
# Artifact-native images (feature 18): drop the git clone that
# migrate-to-releases.sh sets aside, so the image ships NO git remote / .git /
# cloud-web-src code — only the hardware/-only release tree. No-op on git-layout
# images (nothing to remove).
if [ -e /home/beemonitor/BeeMonitor.git-clone.bak ]; then
    rm -rf /home/beemonitor/BeeMonitor.git-clone.bak
    echo "    removed git clone backup (artifact-native image — no git/cloud code)"
fi

echo "==> 4/5 Verifying the image carries NO credential"
fail=0
if grep -Eq '^[[:space:]]*(BEEMONITOR_DEVICE_KEY|BEEMONITOR_ENROLL_TOKEN)=' "$ENV_FILE"; then
    echo "    ✗ a key/token line still present in $ENV_FILE" >&2; fail=1
fi
for f in "${BOOT_CONFS[@]}"; do
    [ -e "$f" ] && { echo "    ✗ boot token still present: $f" >&2; fail=1; }
done
if ! grep -Eq '^[[:space:]]*BEEMONITOR_API_BASE=' "$ENV_FILE"; then
    echo "    ! warning: BEEMONITOR_API_BASE is not set — enrollment needs it (or it comes from beemonitor.conf)." >&2
fi
if [ "$fail" -ne 0 ]; then
    echo "generalize: ABORTING before capture — a credential would ship in the image." >&2
    echo "            Fix the above and re-run. (Did NOT power off.)" >&2
    exit 1
fi
echo "    ✓ clean: no device key / enrollment token in env or boot partition"
# Drop the backup now that we've verified — don't capture it into the image.
rm -f "${ENV_FILE}.pregeneralize.bak" 2>/dev/null || true

echo "==> 5/5 Done."
echo
echo "Next: shut down, move the card to a LINUX box, and capture + shrink:"
echo "    sudo dd if=/dev/sdX of=beemonitor-golden.img bs=4M status=progress conv=fsync"
echo "    sudo pishrink.sh -aZ beemonitor-golden.img      # -> beemonitor-golden.img.xz"
echo "  then publish + set BEEMONITOR_GOLDEN_IMAGE_URL (provision/README.md §1.3–1.4)."
echo

if [ "$DO_POWEROFF" -eq 1 ]; then
    echo "Powering off in 5s (Ctrl-C to cancel)…"; sleep 5; poweroff
else
    echo "(--no-poweroff) Remember to 'sudo poweroff' before pulling the card."
fi
