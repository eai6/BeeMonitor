#!/bin/bash
# Build the hardware-only edge update artifact (BeeMonitor feature 18, Phase 0).
# CI runs this on a release tag; also runnable locally to validate the bundle.
#
# Outputs into $OUT (default <repo>/dist):
#   beemonitor-edge-<version>.tar.gz          the hardware/ tree ONLY
#   beemonitor-edge-<version>.tar.gz.minisig  detached minisign signature (when signing)
#   beemonitor-edge-<version>.json            manifest {version, src_commit, sha256, sig, reqs_hash}
#   latest.json                               {version} the cloud reads to resolve the channel
#
# GUARDRAIL (the IP guarantee made enforceable): fails if the bundle contains any
# path outside hardware/ — so cloud/web/src code can never leak into an edge update.
#
# Signing: set MINISIGN_SECRET_KEY_FILE to a no-password minisign key (CI secret) to
# attach a detached signature; without it the bundle is built unsigned (sha256 only).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${OUT:-$REPO_ROOT/dist}"
VERSION="${EDGE_VERSION:-$(date -u +%Y.%m.%d)-$(git -C "$REPO_ROOT" rev-parse --short HEAD)}"
SRC_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
NAME="beemonitor-edge-${VERSION}"

_sha256() {  # portable: sha256sum (Linux/CI) or shasum (macOS)
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
    else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

mkdir -p "$OUT"
TAR="$OUT/${NAME}.tar.gz"

# 1. Build the bundle — hardware/ ONLY (explicit allowlist; deterministic-ish).
tar -C "$REPO_ROOT" \
    --exclude='hardware/**/__pycache__' --exclude='hardware/**/*.pyc' \
    --exclude='hardware/venv' --exclude='hardware/dist' \
    -czf "$TAR" hardware

# 2. GUARDRAIL — refuse if anything outside hardware/ slipped in.
bad="$(tar -tzf "$TAR" | grep -vE '^hardware/' || true)"
if [ -n "$bad" ]; then
    echo "edge-artifact: REFUSING — bundle contains paths outside hardware/:" >&2
    echo "$bad" | head >&2
    exit 1
fi

# 3. Hashes.
BUNDLE_SHA="$(_sha256 "$TAR")"
REQS_HASH="$(_sha256 "$REPO_ROOT/hardware/requirements.txt")"

# 4. Sign (minisign) when a key is provided (CI). Detached signature.
SIG=""
if [ -n "${MINISIGN_SECRET_KEY_FILE:-}" ]; then
    command -v minisign >/dev/null 2>&1 || { echo "edge-artifact: minisign not installed" >&2; exit 1; }
    minisign -S -s "$MINISIGN_SECRET_KEY_FILE" -m "$TAR" -x "$TAR.minisig" >/dev/null
    SIG="$(grep -vE '^untrusted comment:' "$TAR.minisig" | head -1)"  # the base64 signature line
fi

# 5. Manifest + the channel pointer the cloud reads.
cat > "$OUT/${NAME}.json" <<JSON
{
  "version": "${VERSION}",
  "src_commit": "${SRC_COMMIT}",
  "sha256": "${BUNDLE_SHA}",
  "sig": "${SIG}",
  "reqs_hash": "${REQS_HASH}"
}
JSON
printf '{"version": "%s"}\n' "$VERSION" > "$OUT/latest.json"

echo "edge-artifact: built ${NAME}"
echo "  bundle : $TAR ($(du -h "$TAR" | cut -f1))"
echo "  sha256 : $BUNDLE_SHA"
echo "  signed : $([ -n "$SIG" ] && echo yes || echo 'no (set MINISIGN_SECRET_KEY_FILE to sign)')"
