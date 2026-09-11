#!/usr/bin/env bash
#
# BeeMonitor — bring up whichever camera is on the ribbon (runs at every boot).
#
# WHY THIS EXISTS: units used to pin one sensor in config.txt
# (camera_auto_detect=0 + dtoverlay=<model>), so swapping the module made the
# camera vanish — the pinned driver will not bind to a different sensor,
# libcamera enumerates nothing, every Picamera2() raises and
# beemonitor-recorder crash-loops until somebody edits config.txt on site.
# Both camera modules are in service, so a swap has to be a swap, not a visit.
#
# The split this relies on:
#   * camera_auto_detect=1 probes the CSI connector at boot and loads the right
#     overlay for any OFFICIAL Pi sensor — ov5647, imx219, imx477, imx708.
#   * The Arducam 64MP OwlSight (OV64A40) is invisible to it (hardware/
#     setup-camera.sh) and needs an explicit dtoverlay.
#
# So the firmware handles what it is good at and this covers the one case it
# cannot, preferring the cheapest repair that could work:
#
#   1. a camera already enumerated          -> nothing to do, the usual case
#   2. runtime `dtoverlay ov64a40`          -> no reboot, nothing persistent
#   3. config.txt is pinning the wrong      -> unpin, camera_auto_detect=1,
#      sensor and blocking every probe         reboot once
#   4. auto-detect found nothing and the    -> pin ov64a40, reboot once
#      runtime load did not bind either        (some boards only bind at boot)
#
# Steps 3 and 4 alternate, so whichever module is fitted the unit converges on
# a configuration that works, at a cost of one reboot per swap. A boot-applied
# overlay cannot be unloaded at runtime, which is exactly why step 3 exists and
# why the pin has to go before anything else gets a look.
#
# NEVER loops: the repairs that cost a reboot (3 and 4) are counted in the state
# file and stop at MAX_REPAIRS, and the counter resets the moment a camera is
# found. Step 2 is outside that budget — it reboots nothing and reverts itself,
# so it keeps trying on every boot even after the ladder has given up. Set
# BEEMONITOR_CAMERA_AUTOREBOOT=0 to make steps 3 and 4 edit config.txt but
# leave the reboot to you.
#
# Steps 3 and 4 only alternate if the step-3 edit actually lands, so it is
# verified by re-reading config.txt and no reboot is spent unless the pin is
# really gone.
#
# Usage:
#   sudo hardware/camera-autodetect.sh               # detect + repair (boot path)
#   sudo hardware/camera-autodetect.sh --probe-only  # report only, change nothing
#   sudo hardware/camera-autodetect.sh --status      # what the last run decided
#   sudo hardware/camera-autodetect.sh --reset       # clear the repair counter

set -uo pipefail

CFG="${BEEMONITOR_BOOT_CONFIG:-/boot/firmware/config.txt}"
OVERLAY_DIR="${BEEMONITOR_OVERLAY_DIR:-/boot/firmware/overlays}"
STATE_DIR="${BEEMONITOR_STATE_DIR:-/var/lib/beemonitor}"
STATE="$STATE_DIR/camera-autodetect.state"
AUTOREBOOT="${BEEMONITOR_CAMERA_AUTOREBOOT:-1}"
MAX_REPAIRS=4          # two full unpin/pin cycles; more than that is a hardware fault
SETTLE=2               # seconds for the driver to probe after dtoverlay

# Overlay + parameter variants for the sensors auto-detect cannot see. The
# OwlSight binds bare on some boards and only with an explicit link frequency
# or without the media controller on others; setup-camera.sh walks the same list.
RUNTIME_CANDIDATES=(
  "ov64a40|"
  "ov64a40|link-frequency=360000000"
  "ov64a40|media-controller=off"
)
PIN_CANDIDATE=ov64a40
PIN_RE='^\s*dtoverlay=(ov[0-9a-z]+|imx[0-9]+|arducam)'

mode=detect
case "${1:-}" in
  --probe-only) mode=probe ;;
  --status)     mode=status ;;
  --reset)      mode=reset ;;
  "")           ;;
  *) echo "unknown option: $1 (see the header for usage)" >&2; exit 2 ;;
esac

step() { printf '\n\033[1m== %s\033[0m\n' "$1"; }
pass() { printf '  \033[32mok\033[0m   %s\n' "$1"; }
warn() { printf '  \033[33mwarn\033[0m %s\n' "$1"; }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$1"; }

state_get() { [ -f "$STATE" ] && sed -n "s/^$1=//p" "$STATE" | tail -1; }
# Best-effort: the state file only paces the repair ladder, so a read-only or
# unwritable /var/lib must never turn into noise on the boot path or a failure.
# --probe-only promises to change nothing, and that includes this.
state_set() {
  [ "$mode" = probe ] && return 0
  mkdir -p "$STATE_DIR" 2>/dev/null || return 0
  local tmp="$STATE.part"
  : > "$tmp" 2>/dev/null || return 0
  { [ -f "$STATE" ] && grep -vE "^($1)=" "$STATE"; printf '%s=%s\n' "$1" "$2"; } >> "$tmp" 2>/dev/null
  mv -f "$tmp" "$STATE" 2>/dev/null || return 0
}

# libcamera's own answer, and the only one that counts: a sensor that ACKs on
# i2c but never binds is still no camera.
camera_model() {
  rpicam-hello --list-cameras 2>/dev/null \
    | sed -n 's/^[0-9]\+ : \([a-z0-9_]\+\) .*/\1/p' | head -1
}

backup_cfg() {
  local bak="$CFG.bak-$(date +%Y%m%d-%H%M%S)"
  cp -a "$CFG" "$bak" 2>/dev/null || { bad "backup of $CFG failed — not editing"; return 1; }
  pass "backup: $bak"
}

request_reboot() {
  local why="$1" n
  n=$(( $(state_get repairs || echo 0) + 1 ))
  state_set repairs "$n"
  state_set last_action "$why"
  if [ "$AUTOREBOOT" != "1" ]; then
    warn "BEEMONITOR_CAMERA_AUTOREBOOT=0 — config changed, reboot yourself to apply"
    return 0
  fi
  warn "rebooting to apply: $why (repair $n/$MAX_REPAIRS)"
  sync; systemctl reboot
}

case "$mode" in
  status)
    step "Camera auto-detect state"
    [ -f "$STATE" ] && sed 's/^/  /' "$STATE" || echo "  (no state yet)"
    now=$(camera_model)
    echo "  current: ${now:-none}"
    exit 0 ;;
  reset)
    state_set repairs 0
    pass "repair counter reset"
    exit 0 ;;
esac

# --- 1. is a camera already up? ---------------------------------------------
step "Looking for a camera"
found=$(camera_model)
if [ -n "$found" ]; then
  pass "$found enumerated — nothing to do"
  state_set last_model "$found"
  state_set repairs 0            # whatever we last did, it worked
  exit 0
fi
warn "no camera enumerated"

if [ "$mode" = probe ]; then
  grep -nE "^\s*(camera_auto_detect|dtoverlay=(ov|imx|arducam))" "$CFG" 2>/dev/null | sed 's/^/       /'
  warn "--probe-only: changing nothing"
  exit 0
fi

# --- 2. runtime overlay: cheapest repair, nothing persistent ----------------
# Deliberately ahead of the repair budget below: this needs no reboot and undoes
# itself when it does not bind, so there is nothing to ration. Behind the gate, a
# unit parked at MAX_REPAIRS stopped attempting the one repair that costs
# nothing — the budget, spent on rebooting steps, silenced the cheap one too.
step "Trying the sensors auto-detect cannot see (runtime, no reboot)"
for entry in "${RUNTIME_CANDIDATES[@]}"; do
  ov="${entry%%|*}"; params="${entry#*|}"
  [ -f "$OVERLAY_DIR/$ov.dtbo" ] || { warn "$ov.dtbo not installed — skipping"; continue; }
  printf '  trying %s ... ' "$ov${params:+ $params}"
  # shellcheck disable=SC2086 -- params is deliberately word-split
  if ! dtoverlay "$ov" ${params:+$params} 2>/dev/null; then
    echo "overlay would not apply"; continue
  fi
  sleep "$SETTLE"
  found=$(camera_model)
  if [ -n "$found" ]; then
    echo "bound: $found"
    pass "camera up via runtime dtoverlay=$ov${params:+ $params} — no reboot, nothing persisted"
    state_set last_model "$found"
    state_set last_action "runtime-overlay:$ov"
    state_set repairs 0
    exit 0
  fi
  echo "no camera"
  dtoverlay -r "$ov" 2>/dev/null
done

# --- repair budget: rations only the steps below, which each cost a reboot ---
repairs=$(state_get repairs); repairs=${repairs:-0}
if [ "$repairs" -ge "$MAX_REPAIRS" ]; then
  bad "no camera after $repairs reboot-repairs — stopping so this cannot loop"
  warn "the runtime overlay above still runs every boot; the rest needs a hand:"
  warn "  hardware/setup-camera.sh --probe-only    # is the sensor even alive?"
  warn "  hardware/camera-autodetect.sh --reset    # to try the ladder again"
  exit 0
fi

# --- 3/4. persistent repairs, alternating so we converge --------------------
[ -w "$CFG" ] || { bad "$CFG not writable — run as root"; exit 0; }
pinned=$(grep -E "$PIN_RE" "$CFG" 2>/dev/null)

if [ -n "$pinned" ]; then
  # A pin is loaded at boot and cannot be unloaded, so it blocks every probe
  # above. If it were the fitted sensor a camera would have enumerated, so it
  # is the wrong one: hand the boot back to the firmware.
  step "Unpinning the wrong sensor and handing the boot to auto-detect"
  echo "$pinned" | sed 's/^/       /'
  backup_cfg || exit 0
  # '@' delimiter, not '|': the pattern needs '|' for its own alternation, and
  # a '|' delimiter ends the pattern at the first one, so sed rejects the whole
  # expression. This script runs without `set -e` on purpose, so that rejection
  # was silent — the pin survived, this branch was re-taken every boot, and the
  # ladder never got as far as the pin step that would have worked.
  sed -i -E "s@^(\s*)(dtoverlay=(ov[0-9a-z]+|imx[0-9]+|arducam).*)@\1# \2  # BeeMonitor: unpinned by camera-autodetect.sh@" "$CFG" \
    || { bad "unpin edit failed — $CFG left as it was"; exit 0; }
  if grep -qE "^\s*camera_auto_detect=" "$CFG"; then
    sed -i -E "s@^(\s*)camera_auto_detect=.*@\1camera_auto_detect=1@" "$CFG" \
      || { bad "camera_auto_detect edit failed"; exit 0; }
  else
    printf '\n# BeeMonitor: detect whichever sensor is fitted (camera-autodetect.sh)\ncamera_auto_detect=1\n' >> "$CFG"
  fi
  # A reboot is only worth spending on an edit that actually landed. Re-read the
  # file rather than trusting the exit status: this is the step whose silent
  # no-op cost four boots and the whole repair budget.
  if grep -qE "$PIN_RE" "$CFG"; then
    bad "pin still present after the edit — not rebooting"
    grep -nE "$PIN_RE" "$CFG" | sed 's/^/       /'
    exit 0
  fi
  pass "camera_auto_detect=1, previous pin commented out"
  request_reboot "unpinned a sensor that was blocking detection"
  exit 0
fi

# Already on auto-detect, firmware found nothing and the runtime load did not
# bind — the remaining possibility is a sensor that only binds at boot.
step "Pinning $PIN_CANDIDATE for a boot-time bind"
backup_cfg || exit 0
printf '\n# BeeMonitor: camera-autodetect.sh — runtime bind failed, trying at boot\ndtoverlay=%s\n' "$PIN_CANDIDATE" >> "$CFG"
pass "added dtoverlay=$PIN_CANDIDATE"
request_reboot "pinned $PIN_CANDIDATE after the runtime load failed"
exit 0
