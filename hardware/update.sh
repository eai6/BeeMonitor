#!/usr/bin/env bash
# BeeMonitor remote software updater — two-phase, cellular-safe.
#
# WHY TWO PHASES: the cellular firewall only lets the beemonitor-telemetry.service
# cgroup egress on wwan0, and an update must restart telemetry itself (to pick up
# new telemetry.py) — but a process can't both keep network access (telemetry
# cgroup) and restart its own cgroup without being killed. So:
#
#   fetch / fetch-artifact (phase A) — spawned BY telemetry, runs in its cgroup, so
#                      it can reach GitHub/PyPI/S3 over cellular. Does all NETWORK
#                      work, then hands off to phase B and exits.
#   apply  (phase B) — run as beemonitor-update.service (its own unit, root, no
#                      network). Restarts services, health-checks, and ROLLS BACK
#                      if they don't come up.
#
# TWO TRANSPORTS (backward-compatible — feature 18, Approach A):
#   * git       — `fetch <ref>`: git fetch + reset on a git-clone layout (the
#                 original path; UNCHANGED). Rollback = git reset to the prev sha.
#   * artifact  — `fetch-artifact`: download a signed hardware-only bundle, verify
#                 sha256 + minisign, unpack to releases/<version>/ on a symlink
#                 layout. Rollback = flip the ~/BeeMonitor symlink back.
# Phase B picks the mechanism from the request file, so a unit can run either.
#
# Status for the dashboard is written to <state>/update-status.json and reported
# in the telemetry beat. Triggered by the cloud "update" command.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_USER="$(stat -c %U "$REPO_DIR")"
VENV_PIP="$REPO_DIR/hardware/venv/bin/pip"
REQ_REL="hardware/requirements.txt"
STATE_DIR="${BEEMONITOR_STATE_DIR:-/home/beemonitor/.beemonitor}"
REQUEST_FILE="$STATE_DIR/update-request.json"
STATUS_FILE="$STATE_DIR/update-status.json"
DESCRIPTOR_FILE="$STATE_DIR/update-descriptor.json"   # telemetry writes the artifact descriptor here
UPDATE_UNIT="beemonitor-update.service"
RESTART_UNITS=(beemonitor-recorder.service beemonitor-uploader.service beemonitor-telemetry.service)
HEALTH_UNITS=(beemonitor-recorder.service beemonitor-uploader.service beemonitor-telemetry.service)
HEALTH_WAIT="${BEEMONITOR_UPDATE_HEALTH_WAIT:-15}"

# --- Artifact (symlink-release) layout ---------------------------------------
BEEMON_HOME="${BEEMONITOR_HOME:-/home/beemonitor}"
SYMLINK="${BEEMONITOR_SYMLINK:-$BEEMON_HOME/BeeMonitor}"      # the path systemd/sudoers reference
RELEASES_DIR="${BEEMONITOR_RELEASES_DIR:-$BEEMON_HOME/releases}"
STABLE_VENV="${BEEMONITOR_VENV:-$BEEMON_HOME/beemonitor-venv}"
STABLE_VENV_PIP="$STABLE_VENV/bin/pip"
STABLE_MODELS="${BEEMONITOR_MODELS:-$BEEMON_HOME/models}"
KEEP_RELEASES="${BEEMONITOR_KEEP_RELEASES:-3}"
ACTIVE_MANIFEST="$REPO_DIR/.edge-manifest.json"              # written into each release dir on install
# Public verify key — must live OUTSIDE the updatable bundle (baked into the
# golden image). Falls back to the in-repo copy for git-layout / bench testing.
PUBKEY="${BEEMONITOR_PUBKEY:-$BEEMON_HOME/minisign.pub}"
[ -f "$PUBKEY" ] || PUBKEY="$REPO_DIR/hardware/provision/minisign.pub"

log() { echo "$(date -u +%FT%TZ) update[$1]: ${*:2}"; }

# Run git/pip as the repo owner: avoids root "dubious ownership" in phase B and
# keeps venv/repo files owned by beemonitor. In phase A we already ARE that user.
as_owner() {
    if [ "$(id -un)" = "$REPO_USER" ]; then "$@"; else runuser -u "$REPO_USER" -- "$@"; fi
}
git_r() { as_owner git -C "$REPO_DIR" "$@"; }

# Tiny JSON readers (no jq on the Pi). json_get <file> <key> -> value or "".
json_get() { python3 -c 'import json,sys
try: print(json.load(open(sys.argv[1])).get(sys.argv[2],"") or "")
except Exception: print("")' "$1" "$2" 2>/dev/null; }
sha256_of() { sha256sum "$1" 2>/dev/null | cut -d' ' -f1; }

# version string for status: artifact manifest version if present, else git sha.
_version_str() {
    local v; v="$(json_get "$ACTIVE_MANIFEST" version)"
    [ -n "$v" ] && { echo "$v"; return; }
    git_r rev-parse --short HEAD 2>/dev/null || echo unknown
}

# write_status <state> [result] [detail]
write_status() {
    mkdir -p "$STATE_DIR" 2>/dev/null || true
    local commit; commit="$(_version_str)"
    python3 - "$STATUS_FILE" "$1" "$commit" "${2:-}" "${3:-}" <<'PY' 2>/dev/null || true
import json, os, sys, datetime
path, state, commit, result, detail = sys.argv[1:6]
d = {"state": state, "commit": commit,
     "ts": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
if result: d["result"] = result
if detail: d["detail"] = detail
open(path + ".tmp", "w").write(json.dumps(d))
os.replace(path + ".tmp", path)
PY
    chmod 644 "$STATUS_FILE" 2>/dev/null || true
}

# write_request <mechanism> <prev> <target> [req_changed] [target_dir]
write_request() {
    mkdir -p "$STATE_DIR" 2>/dev/null || true
    python3 - "$REQUEST_FILE" "$@" <<'PY' 2>/dev/null || true
import json, os, sys
path, mech, prev, target = sys.argv[1:5]
req = (len(sys.argv) > 5 and sys.argv[5] == "true")
tdir = sys.argv[6] if len(sys.argv) > 6 else ""
d = {"mechanism": mech, "prev": prev, "target": target, "req_changed": req}
if tdir: d["target_dir"] = tdir
open(path + ".tmp", "w").write(json.dumps(d))
os.replace(path + ".tmp", path)
PY
}

handoff_to_phase_b() {
    log A "handoff -> $UPDATE_UNIT"
    if [ "$(id -un)" = "root" ]; then
        systemctl start --no-block "$UPDATE_UNIT"
    else
        sudo -n systemctl start --no-block "$UPDATE_UNIT" \
            || { write_status error handoff_failed "cannot start $UPDATE_UNIT (sudoers rule missing?)"; exit 1; }
    fi
}

# ----------------------------------------------------------------------------
cmd_fetch() {  # phase A (git) — in telemetry's cgroup (network allowed). UNCHANGED.
    local ref="${1:-origin/main}"
    write_status fetching "" "ref=$ref"
    local prev; prev="$(git_r rev-parse HEAD 2>/dev/null)" \
        || { write_status error fetch_failed "not a git repo at $REPO_DIR"; exit 1; }
    log A "fetch ref=$ref prev=${prev:0:7}"

    if ! git_r fetch --prune origin; then
        write_status error fetch_failed "git fetch failed (cellular down / no route?)"; exit 1
    fi
    local target; target="$(git_r rev-parse "$ref" 2>/dev/null)"
    [ -n "${target:-}" ] || { write_status error fetch_failed "cannot resolve ref '$ref'"; exit 1; }

    if [ "$target" = "$prev" ]; then
        log A "already up to date at ${prev:0:7}"; write_status idle uptodate "already at ${prev:0:7}"; exit 0
    fi

    local req_changed=false
    git_r diff --quiet "$prev" "$target" -- "$REQ_REL" || req_changed=true
    log A "checkout ${target:0:7} (req_changed=$req_changed)"
    if ! git_r reset --hard "$target"; then
        write_status error apply_failed "git reset failed"; git_r reset --hard "$prev"; exit 1
    fi

    if [ "$req_changed" = true ]; then
        log A "pip install -r $REQ_REL (only-if-needed; torch untouched)"
        if ! as_owner "$VENV_PIP" install --upgrade-strategy only-if-needed -r "$REPO_DIR/$REQ_REL"; then
            log A "pip failed — reverting code to ${prev:0:7}"
            git_r reset --hard "$prev"
            write_status error pip_failed "pip install failed; reverted to ${prev:0:7}"; exit 1
        fi
    fi

    write_request git "$prev" "$target" "$req_changed"
    write_status applying "" "restarting services for ${target:0:7}"
    handoff_to_phase_b
}

# ----------------------------------------------------------------------------
cmd_fetch_artifact() {  # phase A (artifact) — download + verify + unpack a release
    [ -f "$DESCRIPTOR_FILE" ] || { write_status error fetch_failed "no update descriptor"; exit 1; }
    local version url sha sig_present active
    version="$(json_get "$DESCRIPTOR_FILE" version)"
    url="$(json_get "$DESCRIPTOR_FILE" url)"
    sha="$(json_get "$DESCRIPTOR_FILE" sha256)"
    active="$(json_get "$ACTIVE_MANIFEST" version)"
    write_status fetching "" "version=$version"
    [ -n "$version" ] && [ -n "$url" ] && [ -n "$sha" ] \
        || { write_status error fetch_failed "incomplete descriptor"; exit 1; }
    if [ "$version" = "$active" ]; then
        log A "already up to date at $version"; write_status idle uptodate "already at $version"; exit 0
    fi
    command -v minisign >/dev/null 2>&1 || { write_status error verify_failed "minisign not installed — run a git update (installs it via provision.sh) or migrate-to-releases.sh, then retry"; exit 1; }
    [ -f "$PUBKEY" ] || { write_status error verify_failed "verify key missing at $PUBKEY"; exit 1; }

    local tmp; tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' RETURN
    log A "download $version"
    if ! curl -fsSL --retry 2 "$url" -o "$tmp/b.tar.gz"; then
        write_status error fetch_failed "download failed (cellular down / URL expired?)"; return 1
    fi
    # 1) sha256, then 2) signature — NEVER unpack an unverified blob.
    local got; got="$(sha256_of "$tmp/b.tar.gz")"
    [ "$got" = "$sha" ] || { write_status error verify_failed "sha256 mismatch"; return 1; }
    python3 -c 'import json,sys; open(sys.argv[2],"w").write(json.load(open(sys.argv[1])).get("sig","") or "")' \
        "$DESCRIPTOR_FILE" "$tmp/b.tar.gz.minisig"
    if ! minisign -Vm "$tmp/b.tar.gz" -p "$PUBKEY" >/dev/null 2>&1; then
        write_status error verify_failed "signature rejected (key $(basename "$PUBKEY"))"; return 1
    fi
    log A "verified sha256 + signature for $version"

    # Guardrail mirror: the bundle must be hardware/-only.
    if tar -tzf "$tmp/b.tar.gz" | grep -qvE '^hardware/'; then
        write_status error verify_failed "bundle has paths outside hardware/"; return 1
    fi

    local release="$RELEASES_DIR/$version"
    rm -rf "$release"; mkdir -p "$release"
    if ! tar -C "$release" -xzf "$tmp/b.tar.gz"; then
        write_status error apply_failed "unpack failed"; rm -rf "$release"; return 1
    fi
    # The bundle is hardware/-only, so point the repo's hardcoded venv/models paths
    # at the stable shared dirs — systemd ExecStart's .../hardware/venv and
    # config.py's parents[2]/models then resolve on a release layout with no
    # per-release copy and no unit-file edits.
    ln -sfn "$STABLE_VENV" "$release/hardware/venv"
    ln -sfn "$STABLE_MODELS" "$release/models"
    cp "$DESCRIPTOR_FILE" "$release/.edge-manifest.json"
    chown -h -R "$REPO_USER":"$REPO_USER" "$release" 2>/dev/null || true

    # pip into the STABLE shared venv only when requirements changed.
    local reqs_now reqs_active
    reqs_now="$(json_get "$DESCRIPTOR_FILE" reqs_hash)"
    reqs_active="$(json_get "$ACTIVE_MANIFEST" reqs_hash)"
    if [ -n "$reqs_now" ] && [ "$reqs_now" != "$reqs_active" ] && [ -f "$release/hardware/requirements.txt" ]; then
        log A "pip install (only-if-needed) into stable venv"
        if ! as_owner "$STABLE_VENV_PIP" install --upgrade-strategy only-if-needed -r "$release/hardware/requirements.txt"; then
            write_status error pip_failed "pip install failed"; rm -rf "$release"; return 1
        fi
    fi

    write_request artifact "${active:-none}" "$version" "false" "$release"
    write_status applying "" "swapping to $version"
    handoff_to_phase_b
}

# ----------------------------------------------------------------------------
cmd_apply() {  # phase B — beemonitor-update.service (root, offline)
    [ -f "$REQUEST_FILE" ] || { log B "no pending request; nothing to apply"; exit 0; }
    local mech prev target target_dir
    mech="$(json_get "$REQUEST_FILE" mechanism)"; [ -n "$mech" ] || mech=git   # legacy = git
    prev="$(json_get "$REQUEST_FILE" prev)"
    target="$(json_get "$REQUEST_FILE" target)"
    target_dir="$(json_get "$REQUEST_FILE" target_dir)"
    rm -f "$REQUEST_FILE"
    [ -n "${target:-}" ] || { log B "bad request file"; write_status error apply_failed "bad request"; exit 1; }

    # Sync root-owned system config (sudoers, units) from the repo BEFORE restart.
    if [ -f "$REPO_DIR/hardware/provision.sh" ]; then
        log B "provisioning system config from repo"
        bash "$REPO_DIR/hardware/provision.sh" || log B "provision reported issues (continuing)"
    fi

    if [ "$mech" = artifact ]; then
        _apply_artifact "$prev" "$target" "$target_dir"
    else
        _apply_git "$prev" "$target"
    fi
}

_restart_and_check() {  # -> echoes the list of unhealthy units (empty = healthy)
    systemctl restart "${RESTART_UNITS[@]}" 2>/dev/null || true
    sleep "$HEALTH_WAIT"
    local bad=""
    for u in "${HEALTH_UNITS[@]}"; do
        systemctl is-active --quiet "$u" || bad="$bad ${u%.service}"
    done
    echo "$bad"
}

_apply_git() {
    local prev="$1" target="$2"
    log B "restart for ${target:0:7} (rollback point ${prev:0:7})"
    local bad; bad="$(_restart_and_check)"
    if [ -z "$bad" ]; then
        log B "healthy at ${target:0:7}"; write_status ok updated "updated to ${target:0:7}"; exit 0
    fi
    log B "UNHEALTHY ($bad) — rolling back to ${prev:0:7}"
    git_r reset --hard "$prev" || log B "WARNING: git rollback failed"
    _restart_and_check >/dev/null
    write_status rolledback rollback "update ${target:0:7} failed ($bad ); reverted to ${prev:0:7}"
}

_apply_artifact() {
    local prev="$1" target="$2" target_dir="$3"
    [ -d "$target_dir" ] || { write_status error apply_failed "release dir $target_dir missing"; exit 1; }
    # remember where the symlink pointed, for rollback.
    local prev_dir; prev_dir="$(readlink -f "$SYMLINK" 2>/dev/null || true)"
    log B "symlink-swap -> $target (was ${prev:-none})"
    ln -sfn "$target_dir" "$SYMLINK"

    local bad; bad="$(_restart_and_check)"
    if [ -z "$bad" ]; then
        log B "healthy at $target"; write_status ok updated "updated to $target"
        _prune_releases "$target_dir" "$prev_dir"
        exit 0
    fi
    log B "UNHEALTHY ($bad) — rolling symlink back to ${prev:-$prev_dir}"
    if [ -n "$prev_dir" ] && [ -d "$prev_dir" ]; then
        ln -sfn "$prev_dir" "$SYMLINK"
    fi
    _restart_and_check >/dev/null
    rm -rf "$target_dir"   # drop the bad release
    write_status rolledback rollback "update $target failed ($bad ); reverted to ${prev:-prev}"
}

_prune_releases() {  # keep the active + previous + last KEEP_RELEASES, drop the rest
    local keep_active="$1" keep_prev="$2"
    [ -d "$RELEASES_DIR" ] || return 0
    local d
    ls -1dt "$RELEASES_DIR"/*/ 2>/dev/null | tail -n +"$((KEEP_RELEASES + 1))" | while read -r d; do
        d="${d%/}"
        [ "$d" = "$keep_active" ] && continue
        [ "$d" = "$keep_prev" ] && continue
        log B "pruning old release $(basename "$d")"
        rm -rf "$d"
    done
}

# ----------------------------------------------------------------------------
case "${1:-}" in
    fetch) shift; cmd_fetch "${1:-origin/main}" ;;
    fetch-artifact) cmd_fetch_artifact ;;
    apply) cmd_apply ;;
    *) echo "usage: $0 {fetch <ref> | fetch-artifact | apply}" >&2; exit 2 ;;
esac
