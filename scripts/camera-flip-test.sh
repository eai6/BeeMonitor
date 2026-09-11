#!/usr/bin/env bash
# BeeMonitor — does the ISP's 180 actually reach the picture?
#
# WHY: motion/camera.py asks for the 180 as libcamera Transform(hflip, vflip)
# and recorder.py:122 passes it to create_video_configuration(). Measured on
# this unit, that transform reaches the picture in a standalone Picamera2
# script (a 180 scores ncc +0.995 against rot180 of the unflipped frame) but
# is silently dropped inside the recorder: clips recorded with flip=hflip+vflip
# and with flip=none came out the SAME way up (+0.996), and both matched the
# UNFLIPPED orientation. The sensor did toggle either way — the raw Bayer order
# moved GRBG <-> GBRG across the two runs — so the flip is applied at the
# sensor and then undone before the encoder.
#
# So the difference is in the configuration, not the pipeline. This bisects it:
# it captures an unflipped and a flipped still under the recorder's exact
# config and under three cut-down variants, and prints which ones actually
# rotate. The first row that reads DROPPED names the ingredient at fault.
#
# The recorder holds the camera exclusively, so this stops it for ~40s and
# starts it again on the way out — including on failure or Ctrl-C.
#
# Usage:
#   sudo scripts/camera-flip-test.sh              # test, restore the recorder
#   sudo scripts/camera-flip-test.sh --out DIR    # write the stills elsewhere
#   sudo scripts/camera-flip-test.sh --keep-stopped   # leave the recorder down

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO/hardware/venv/bin/python"
OUT="${BEEMONITOR_FLIPTEST_OUT:-/tmp/beemonitor-fliptest}"
UNIT=beemonitor-recorder
keep_stopped=0

step() { printf '\n\033[1m== %s\033[0m\n' "$1"; }
pass() { printf '  \033[32mok\033[0m   %s\n' "$1"; }
warn() { printf '  \033[33mwarn\033[0m %s\n' "$1"; }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$1"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --out)          OUT="${2:?--out needs a directory}"; shift 2 ;;
    --keep-stopped) keep_stopped=1; shift ;;
    *) echo "unknown option: $1 (see the header for usage)" >&2; exit 2 ;;
  esac
done

[ -x "$PY" ] || { bad "no venv python at $PY — is this the Pi?"; exit 1; }
mkdir -p "$OUT" || { bad "cannot write $OUT"; exit 1; }

# The camera must go back to the recorder whatever happens below.
was_active=0
systemctl is-active --quiet "$UNIT" && was_active=1
restore() {
  if [ "$was_active" -eq 1 ] && [ "$keep_stopped" -eq 0 ]; then
    step "Restoring the recorder"
    systemctl start "$UNIT"
    sleep 3
    if systemctl is-active --quiet "$UNIT"; then
      pass "$UNIT is back up"
    else
      bad "$UNIT did NOT come back — start it by hand"
    fi
  elif [ "$keep_stopped" -eq 1 ]; then
    warn "leaving $UNIT stopped (--keep-stopped); the camera records nothing until you start it"
  fi
}
trap restore EXIT INT TERM

if [ "$was_active" -eq 1 ]; then
  step "Freeing the camera"
  systemctl stop "$UNIT"
  sleep 2
  pass "$UNIT stopped"
else
  warn "$UNIT was not running — leaving it that way"
fi

step "Bisecting: which configuration honours the transform?"
BEEMONITOR_FLIPTEST_OUT="$OUT" "$PY" - <<'PYEOF' 2>&1 | grep -vE '^\[[0-9]+:|INFO (Camera|RPI|IPAProxy)'
import os, time
import cv2, numpy as np
from picamera2 import Picamera2
import libcamera

out = os.environ["BEEMONITOR_FLIPTEST_OUT"]

# recorder.py:122 builds the first of these. The others strip one thing each,
# so whichever row stops flipping tells us what is eating the transform.
VARIANTS = {
    "recorder-exact": dict(main={"size": (1920, 1080)},
                           lores={"size": (640, 480), "format": "YUV420"},
                           controls={"FrameRate": 25}),
    "no-controls":    dict(main={"size": (1920, 1080)},
                           lores={"size": (640, 480), "format": "YUV420"}),
    "no-lores":       dict(main={"size": (1920, 1080)},
                           controls={"FrameRate": 25}),
    "main-only":      dict(main={"size": (1920, 1080)}),
}

def grab(kw, tf, path):
    cam = Picamera2()
    cfg = cam.create_video_configuration(transform=tf, **kw)
    cam.configure(cfg)
    applied = cfg.get("transform")
    cam.start(); time.sleep(2)
    cam.capture_file(path)
    cam.stop(); cam.close(); time.sleep(1)
    return applied

def prep(x):
    x = cv2.resize(x, (480, 270)).astype(np.float32)
    return (x - x.mean()) / (x.std() + 1e-6)

print("  %-16s%-24s%-18s%s" % ("variant", "applied transform", "rotates?", "ncc(180, rot180(none))"))
print("  " + "-" * 78)
for name, kw in VARIANTS.items():
    a = "%s/bisect_%s_none.jpg" % (out, name)
    b = "%s/bisect_%s_180.jpg" % (out, name)
    grab(kw, libcamera.Transform(), a)
    applied = grab(kw, libcamera.Transform(hflip=1, vflip=1), b)
    ia, ib = cv2.imread(a, 0), cv2.imread(b, 0)
    if ia is None or ib is None:
        print("  %-16s%s" % (name, "(capture failed)"))
        continue
    A = prep(ia)
    rot  = float((prep(np.rot90(ib, 2)) * A).mean())
    same = float((prep(ib) * A).mean())
    verdict = "YES" if rot > same else "no  <-- DROPPED"
    print("  %-16s%-24s%-18s%+.4f  (unrotated %+.4f)" % (name, str(applied), verdict, rot, same))
PYEOF

step "Stills"
ls -l "$OUT"/bisect_*.jpg 2>/dev/null | sed 's/^/  /'
