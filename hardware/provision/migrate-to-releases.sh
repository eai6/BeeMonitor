#!/bin/bash
# Convert a git-clone BeeMonitor unit to the release/symlink layout (feature 18).
# This is BOTH the on-Pi test setup for the artifact update path (Phase 1) AND the
# Phase-2 cutover step. Run as root on the Pi. Idempotent: no-ops if already done.
#
# After this the unit updates via SIGNED ARTIFACTS instead of git:
#   ~/BeeMonitor            -> releases/<v0>/        (symlink; all hardcoded paths
#                                                     still resolve through it)
#   releases/<v0>/hardware/                          the code (hardware/ only)
#   releases/<v0>/hardware/venv -> ~/beemonitor-venv (symlink to the STABLE venv)
#   releases/<v0>/models        -> ~/models          (symlink to the STABLE models)
#   ~/beemonitor-venv, ~/models                      stable, shared across releases
#   ~/minisign.pub                                   the signature verify key
# update.sh recreates those two symlinks for every new release, so the repo's
# fixed paths (systemd ExecStart .../hardware/venv, config.py parents[2]/models)
# keep working with no unit-file edits.
set -u

HOME_DIR="${BEEMONITOR_HOME:-/home/beemonitor}"
SYMLINK="$HOME_DIR/BeeMonitor"
RELEASES="$HOME_DIR/releases"
VENV="$HOME_DIR/beemonitor-venv"
MODELS="$HOME_DIR/models"
PUBKEY_DST="$HOME_DIR/minisign.pub"
UNITS=(beemonitor-recorder beemonitor-uploader beemonitor-telemetry beemonitor-calibrate)

log() { echo "migrate: $*"; }
[ "$(id -u)" = 0 ] || { log "run as root (sudo)"; exit 1; }

if [ -L "$SYMLINK" ]; then
    log "already release-layout: $SYMLINK -> $(readlink "$SYMLINK") — nothing to do"
    exit 0
fi
[ -d "$SYMLINK/.git" ] || { log "expected a git clone at $SYMLINK (none found)"; exit 1; }

SRC="$SYMLINK"
USER_NAME="$(stat -c %U "$SRC")"
VERSION="v0-$(git -C "$SRC" rev-parse --short HEAD 2>/dev/null || date -u +%Y%m%d)"
REL="$RELEASES/$VERSION"
log "migrating $SRC -> $REL (version $VERSION, owner $USER_NAME)"

# Stop services so nothing is running from the clone during the swap.
systemctl stop "${UNITS[@]}" 2>/dev/null || true

# 1. Stable venv + models (move the in-repo copies out, once).
[ -d "$SRC/hardware/venv" ] && [ ! -e "$VENV" ] && { log "venv -> $VENV"; mv "$SRC/hardware/venv" "$VENV"; }
[ -d "$SRC/models" ] && [ ! -e "$MODELS" ] && { log "models -> $MODELS"; mv "$SRC/models" "$MODELS"; }

# 2. Build the v0 release = hardware/ only + the two stable symlinks + a baseline
#    manifest (so update.sh knows the active version).
mkdir -p "$REL"
cp -a "$SRC/hardware" "$REL/hardware"
rm -rf "$REL/hardware/venv"
ln -sfn "$VENV" "$REL/hardware/venv"
ln -sfn "$MODELS" "$REL/models"
REQS_HASH="$(sha256sum "$REL/hardware/requirements.txt" 2>/dev/null | cut -d' ' -f1)"
printf '{"version": "%s", "sha256": "", "sig": "", "reqs_hash": "%s"}\n' "$VERSION" "$REQS_HASH" \
    > "$REL/.edge-manifest.json"

# 3. Verify key at the stable path + the minisign tool.
if [ -f "$REL/hardware/provision/minisign.pub" ]; then
    cp "$REL/hardware/provision/minisign.pub" "$PUBKEY_DST"
else
    log "WARN: minisign.pub not in the tree — artifact verify will fail until it's at $PUBKEY_DST"
fi
command -v minisign >/dev/null 2>&1 || { log "installing minisign"; apt-get install -y minisign >/dev/null 2>&1 || log "WARN: install minisign manually"; }

chown -h -R "$USER_NAME":"$USER_NAME" "$RELEASES" "$VENV" "$MODELS" "$PUBKEY_DST" 2>/dev/null || true

# 4. Flip: move the clone aside, point ~/BeeMonitor at the release.
mv "$SRC" "${SRC}.git-clone.bak"
ln -sfn "$REL" "$SYMLINK"

systemctl start "${UNITS[@]}" 2>/dev/null || true

log "done. $SYMLINK -> $(readlink "$SYMLINK")"
log "old clone kept at ${SRC}.git-clone.bak (remove once verified)"
log "now trigger 'Update via signed artifact' from the dashboard to test the path."
