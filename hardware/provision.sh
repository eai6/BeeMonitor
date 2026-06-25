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

ensure_minisign
sync_sudoers
sync_units
log "done"
exit 0
