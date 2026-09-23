#!/usr/bin/env bash
# Idempotent system provisioning for BeeMonitor field units.
#
# WHY: a deployed unit on cellular has NO shell access, so root-owned config that
# lives OUTSIDE the git repo — sudoers rules in /etc/sudoers.d, systemd unit files
# in /etc/systemd/system — can't be changed by hand. This makes that config
# version-controlled: it syncs the repo's desired state onto the system. update.sh
# runs it (as root) in phase B, so "commit a config change + Update to latest"
# lands it on a unit you can't reach (e.g. adding the cellular-firewall sudoers
# rule to a field unit).
#
# SAFE BY CONSTRUCTION:
#   * sudoers files are validated with `visudo -c` BEFORE install — a file that
#     would break sudo is never written;
#   * everything is content-compared, so it's idempotent (no-op when correct);
#   * a single failure is logged but never aborts (the update continues);
#   * only ALREADY-INSTALLED systemd units are refreshed — provisioning never
#     enables, masks, or adds units, so it can't silently start or stop anything.
#     The one exception is beemonitor-camera-detect (see below), a oneshot that
#     does nothing on a unit whose camera already works.
#
# Run any time by hand too:  sudo hardware/provision.sh
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUDOERS_SRC="$REPO_DIR/hardware/provision/sudoers.d"
UNITS_SRC="$REPO_DIR/hardware/systemd"
SUDOERS_DST="/etc/sudoers.d"
UNITS_DST="/etc/systemd/system"

log() { echo "$(date -u +%FT%TZ) provision: $*"; }

if [ "$(id -u)" != "0" ]; then
    log "must run as root — skipping (no changes made)"
    exit 0
fi

# --- minisign (signed-artifact update verify) -------------------------------
# The artifact update path verifies the bundle with `minisign` BEFORE unpacking;
# legacy git-layout units never had it installed (only the golden image / the
# migrate-to-releases.sh script do). Installing it here — root, on every update's
# apply phase — means a plain GIT update is enough to make a unit artifact-ready,
# with no per-unit SSH. The verify KEY already ships in the repo (update.sh falls
# back to hardware/provision/minisign.pub), so only the binary is needed.
ensure_minisign() {
    command -v minisign >/dev/null 2>&1 && return 0
    # Prefer the static binary vendored in the repo for this arch: it arrives via
    # the (cellular-allowlisted) git update, so installing it needs NO apt/WiFi.
    local vb="$REPO_DIR/hardware/provision/minisign-$(uname -m)"
    if [ -x "$vb" ]; then
        if install -m 0755 "$vb" /usr/local/bin/minisign; then
            log "minisign: installed the vendored static binary -> /usr/local/bin"
            return 0
        fi
        log "WARN: could not install vendored minisign"
    fi
    # Last resort: apt (needs WiFi — the cellular firewall blocks the apt mirrors).
    log "minisign missing + no vendored binary for $(uname -m) — trying apt (WiFi only)"
    apt-get install -y minisign >/dev/null 2>&1 && log "minisign installed (apt)" \
        || log "WARN: minisign install failed (build+commit the $(uname -m) binary via the build-minisign workflow)"
}

# --- sudoers rules ----------------------------------------------------------
sync_sudoers() {
    [ -d "$SUDOERS_SRC" ] || return 0
    shopt -s nullglob
    for src in "$SUDOERS_SRC"/*; do
        local name dst
        name="$(basename "$src")"
        # sudoers.d ignores filenames with a dot or ending in ~; don't ship those.
        case "$name" in *.* | *~) continue ;; esac
        dst="$SUDOERS_DST/$name"
        # Validate FIRST — never install a sudoers file that would break sudo.
        if ! visudo -cf "$src" >/dev/null 2>&1; then
            log "SKIP sudoers/$name — failed visudo validation (NOT installed)"
            continue
        fi
        if [ -f "$dst" ] && cmp -s "$src" "$dst"; then
            continue  # already correct
        fi
        if install -m 0440 -o root -g root "$src" "$dst"; then
            log "sudoers: installed/updated $dst"
        else
            log "sudoers: FAILED to install $dst"
        fi
    done
}

# --- systemd units (refresh EXISTING only) ----------------------------------
sync_units() {
    [ -d "$UNITS_SRC" ] || return 0
    local changed=false src name dst
    shopt -s nullglob
    for src in "$UNITS_SRC"/*.service "$UNITS_SRC"/*.timer; do
        name="$(basename "$src")"
        dst="$UNITS_DST/$name"
        [ -f "$dst" ] || continue   # only refresh units already installed on this host
        cmp -s "$src" "$dst" && continue
        if install -m 0644 -o root -g root "$src" "$dst"; then
            log "unit: updated $dst"; changed=true
        else
            log "unit: FAILED to update $dst"
        fi
    done
    if [ "$changed" = true ]; then
        systemctl daemon-reload && log "systemctl daemon-reload" || log "daemon-reload FAILED"
    fi
}

# --- camera detection -------------------------------------------------------
# Both camera modules are in service, so a unit must come up with whatever is on
# its ribbon rather than with one sensor baked into config.txt. That is
# camera-autodetect.sh's job (installed as beemonitor-camera-detect.service and
# pulled in by the recorder): it covers the Arducam OwlSight, which
# camera_auto_detect cannot see, and unpins a stale sensor when one is blocking
# detection after a swap.
#
# All this needs from config.txt is that detection is not switched off. It is
# deliberately the smallest possible edit:
#
#   * a unit with an explicit dtoverlay=<sensor> is left completely alone. That
#     pin is working today, and camera_auto_detect=1 alongside an explicit
#     overlay means two overlays for one sensor. When the pin later becomes the
#     WRONG sensor, no camera enumerates and camera-autodetect.sh unpins it and
#     sets camera_auto_detect=1 together, which is consistent — so a swap still
#     self-heals without this ever forcing a reboot at provision time.
#   * a unit with no pin gets camera_auto_detect=1, so every official Pi sensor
#     is found at boot and the OwlSight is picked up at runtime.
BOOT_CONFIG="${BEEMONITOR_BOOT_CONFIG:-/boot/firmware/config.txt}"
CAMERA_PIN_RE='^[[:space:]]*dtoverlay=(ov[0-9a-z]+|imx[0-9]+|arducam)'

ensure_camera_autodetect() {
    [ -f "$BOOT_CONFIG" ] || { log "camera: no $BOOT_CONFIG — skipping"; return 0; }

    if grep -qE "$CAMERA_PIN_RE" "$BOOT_CONFIG"; then
        # Pinned and presumably working. camera-autodetect.sh takes over if that
        # stops being true.
        return 0
    fi
    if grep -qE "^[[:space:]]*camera_auto_detect=1" "$BOOT_CONFIG"; then
        return 0  # already right: the normal case, no logging noise
    fi

    cp -a "$BOOT_CONFIG" "$BOOT_CONFIG.bak-$(date +%Y%m%d-%H%M%S)" || {
        log "camera: FAILED to back up $BOOT_CONFIG — not editing it"; return 0; }

    if grep -qE "^[[:space:]]*camera_auto_detect=" "$BOOT_CONFIG"; then
        sed -i -E "s|^([[:space:]]*)camera_auto_detect=.*|\\1camera_auto_detect=1|" "$BOOT_CONFIG" || {
            log "camera: FAILED to set camera_auto_detect=1"; return 0; }
        log "camera: camera_auto_detect switched on in $BOOT_CONFIG — takes effect on the next reboot"
    else
        {
            printf '\n# BeeMonitor: detect whichever sensor is fitted (camera-autodetect.sh)\n'
            printf 'camera_auto_detect=1\n'
        } >> "$BOOT_CONFIG" || { log "camera: FAILED to append camera_auto_detect"; return 0; }
        log "camera: added camera_auto_detect=1 to $BOOT_CONFIG — takes effect on the next reboot"
    fi
}

# --- camera detect unit (the one unit provisioning ADDS) ---------------------
# The exception to "never adds units", and a narrow one. beemonitor-recorder
# already Wants= this unit, but Wants= on a unit that is not installed is
# silently nothing — and sync_units only refreshes what is there — so units
# flashed from a golden image that predates it never got it, and a 64MP
# OwlSight on one of them stays invisible and the recorder crash-loops. Adding
# it here means one update brings any unit, old image or new, up to date
# without rebuilding the image.
#
# Safe to add: it is a oneshot that exits 0 with nothing to do whenever a
# camera already enumerates, which is every working unit.
#
# It is also STARTED here, because update.sh restarts the recorder and judges
# the update on its health right after this — waiting for the next boot would
# roll back the very update that fixes the camera. Run with auto-reboot off: a
# reboot in the middle of the apply phase would cut off the restart, health
# check and status report. If only a boot-time bind will do, config.txt is left
# ready and the enabled unit finishes the job on the next boot.
CAMERA_UNIT=beemonitor-camera-detect.service

ensure_camera_detect_unit() {
    local src="$UNITS_SRC/$CAMERA_UNIT" dst="$UNITS_DST/$CAMERA_UNIT"
    [ -f "$src" ] || return 0
    if [ ! -f "$dst" ]; then
        if install -m 0644 -o root -g root "$src" "$dst"; then
            log "unit: installed $dst"
            systemctl daemon-reload || log "daemon-reload FAILED"
        else
            log "unit: FAILED to install $dst"; return 0
        fi
    fi
    if ! systemctl is-enabled --quiet "$CAMERA_UNIT" 2>/dev/null; then
        systemctl enable "$CAMERA_UNIT" >/dev/null 2>&1 \
            && log "unit: enabled $CAMERA_UNIT" \
            || log "unit: FAILED to enable $CAMERA_UNIT"
    fi
    BEEMONITOR_CAMERA_AUTOREBOOT=0 bash "$REPO_DIR/hardware/camera-autodetect.sh" 2>&1 \
        | sed 's/\x1b\[[0-9;]*m//g' | while IFS= read -r line; do
            [ -n "$line" ] && log "camera-detect: $line"
        done
}

ensure_minisign
sync_sudoers
sync_units
ensure_camera_autodetect
ensure_camera_detect_unit
log "done"
exit 0
