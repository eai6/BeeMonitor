#!/usr/bin/env bash
# BeeMonitor remote software updater — two-phase, cellular-safe.
#
# WHY TWO PHASES: the cellular firewall only lets the beemonitor-telemetry.service
# cgroup egress on wwan0, and an update must restart telemetry itself (to pick up
# new telemetry.py) — but a process can't both keep network access (telemetry
# cgroup) and restart its own cgroup without being killed. So:
#
#   fetch  (phase A) — spawned BY telemetry, so it runs in telemetry's cgroup and
#                      can reach GitHub/PyPI over cellular. Does all NETWORK work:
#                      git fetch + reset + (pip if hardware/requirements.txt
#                      changed). Then hands off to phase B and exits.
#   apply  (phase B) — run as beemonitor-update.service (its own unit, root, no
#                      network). Restarts services, health-checks, and ROLLS BACK
#                      the code to the previous commit if they don't come up.
#
# Status for the dashboard is written to <state>/update-status.json and reported
# in the telemetry beat. Triggered by the cloud "update" command (params.ref,
# default origin/main). torch/models are out of scope (do those over WiFi).
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_USER="$(stat -c %U "$REPO_DIR")"
VENV_PIP="$REPO_DIR/hardware/venv/bin/pip"
REQ_REL="hardware/requirements.txt"
STATE_DIR="${BEEMONITOR_STATE_DIR:-/home/beemonitor/.beemonitor}"
REQUEST_FILE="$STATE_DIR/update-request.json"
STATUS_FILE="$STATE_DIR/update-status.json"
UPDATE_UNIT="beemonitor-update.service"
RESTART_UNITS=(beemonitor-recorder.service beemonitor-uploader.service beemonitor-telemetry.service)
HEALTH_UNITS=(beemonitor-recorder.service beemonitor-uploader.service beemonitor-telemetry.service)
HEALTH_WAIT="${BEEMONITOR_UPDATE_HEALTH_WAIT:-15}"

log() { echo "$(date -u +%FT%TZ) update[$1]: ${*:2}"; }

# Run git/pip as the repo owner: avoids root "dubious ownership" in phase B and
# keeps venv/repo files owned by beemonitor. In phase A we already ARE that user.
as_owner() {
    if [ "$(id -un)" = "$REPO_USER" ]; then "$@"; else runuser -u "$REPO_USER" -- "$@"; fi
}
git_r() { as_owner git -C "$REPO_DIR" "$@"; }

# write_status <state> [result] [detail]
write_status() {
    mkdir -p "$STATE_DIR" 2>/dev/null || true
    local commit; commit="$(git_r rev-parse --short HEAD 2>/dev/null || echo unknown)"
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

# ----------------------------------------------------------------------------
cmd_fetch() {  # phase A — in telemetry's cgroup (network allowed)
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

    # Hand off to phase B (separate unit) for restart + health-check + rollback.
    mkdir -p "$STATE_DIR" 2>/dev/null || true
    python3 - "$REQUEST_FILE" "$prev" "$target" "$req_changed" <<'PY' 2>/dev/null || true
import json, os, sys
path, prev, target, req = sys.argv[1:5]
open(path + ".tmp", "w").write(json.dumps({"prev": prev, "target": target, "req_changed": req == "true"}))
os.replace(path + ".tmp", path)
PY
    write_status applying "" "restarting services for ${target:0:7}"
    log A "handoff -> $UPDATE_UNIT"
    if [ "$(id -un)" = "root" ]; then
        systemctl start --no-block "$UPDATE_UNIT"
    else
        sudo -n systemctl start --no-block "$UPDATE_UNIT" \
            || { write_status error handoff_failed "cannot start $UPDATE_UNIT (sudoers rule missing?)"; exit 1; }
    fi
}

# ----------------------------------------------------------------------------
cmd_apply() {  # phase B — beemonitor-update.service (root, offline)
    [ -f "$REQUEST_FILE" ] || { log B "no pending request; nothing to apply"; exit 0; }
    local prev target
    prev="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["prev"])' "$REQUEST_FILE" 2>/dev/null)"
    target="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["target"])' "$REQUEST_FILE" 2>/dev/null)"
    rm -f "$REQUEST_FILE"
    [ -n "${prev:-}" ] && [ -n "${target:-}" ] || { log B "bad request file"; write_status error apply_failed "bad request"; exit 1; }

    log B "restart for ${target:0:7} (rollback point ${prev:0:7})"
    systemctl restart "${RESTART_UNITS[@]}" 2>/dev/null || true
    sleep "$HEALTH_WAIT"

    local bad=""
    for u in "${HEALTH_UNITS[@]}"; do
        systemctl is-active --quiet "$u" || bad="$bad ${u%.service}"
    done

    if [ -z "$bad" ]; then
        log B "healthy at ${target:0:7}"; write_status ok updated "updated to ${target:0:7}"; exit 0
    fi

    log B "UNHEALTHY ($bad) — rolling back to ${prev:0:7}"
    git_r reset --hard "$prev" || log B "WARNING: git rollback failed"
    systemctl restart "${RESTART_UNITS[@]}" 2>/dev/null || true
    sleep "$HEALTH_WAIT"
    write_status rolledback rollback "update ${target:0:7} failed ($bad ); reverted to ${prev:0:7}"
}

# ----------------------------------------------------------------------------
case "${1:-}" in
    fetch) shift; cmd_fetch "${1:-origin/main}" ;;
    apply) cmd_apply ;;
    *) echo "usage: $0 {fetch <ref>|apply}" >&2; exit 2 ;;
esac
