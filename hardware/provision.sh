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

sync_sudoers
sync_units
log "done"
exit 0
