#!/usr/bin/env bash
# Sync Claude Code's per-project memory notes between the live store and the repo.
#
# Live store (auto-loaded by Claude, local to the machine):
#   ~/.claude/projects/<encoded-repo-path>/memory/
# Repo backup (committed, travels with the repo):
#   <repo>/claude-memory/
#
# Usage:
#   scripts/sync-claude-memory.sh            # backup:  live  -> repo  (default)
#   scripts/sync-claude-memory.sh --restore  # restore: repo  -> live
#   scripts/sync-claude-memory.sh --dry-run  # show what would change, copy nothing
#
# Override the live path if your setup differs:
#   CLAUDE_MEMORY_DIR=/path/to/memory scripts/sync-claude-memory.sh
set -euo pipefail

# Repo root from this script's location (works regardless of cwd).
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$REPO/claude-memory"

# Claude encodes the project path by replacing '/' with '-' (so /home/x/Foo ->
# -home-x-Foo). Allow an explicit override via CLAUDE_MEMORY_DIR.
LIVE_DIR="${CLAUDE_MEMORY_DIR:-$HOME/.claude/projects/${REPO//\//-}/memory}"

MODE="backup"; DRY=""
for arg in "$@"; do
    case "$arg" in
        --restore) MODE="restore" ;;
        --dry-run) DRY="--dry-run" ;;
        -h|--help) sed -n '2,18p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

if [ "$MODE" = "backup" ]; then
    SRC="$LIVE_DIR"; DST="$REPO_DIR"
else
    SRC="$REPO_DIR"; DST="$LIVE_DIR"
fi

if [ ! -d "$SRC" ]; then
    echo "source dir not found: $SRC" >&2
    [ "$MODE" = "backup" ] && echo "(no live memory for this repo yet — nothing to back up)" >&2
    exit 1
fi
mkdir -p "$DST"

echo "mode : $MODE"
echo "from : $SRC"
echo "to   : $DST"
echo

# Prefer rsync (shows a clean change list); fall back to cp. Only *.md files.
# --delete keeps the destination's notes an exact mirror of the source. Filter
# rules are FIRST-MATCH-WINS, so exclude README.md *before* the '*.md' include —
# it's a repo-only doc and must never be copied to the live store or deleted from
# the repo backup.
if command -v rsync >/dev/null 2>&1; then
    rsync -a --itemize-changes --delete ${DRY} \
        --exclude 'README.md' --include '*.md' --exclude '*' \
        "$SRC"/ "$DST"/
else
    [ -n "$DRY" ] && { echo "(dry-run needs rsync; showing target list only)"; ls -1 "$SRC"/*.md; exit 0; }
    find "$SRC" -maxdepth 1 -name '*.md' ! -name 'README.md' -exec cp {} "$DST"/ \;
    echo "copied $(find "$SRC" -maxdepth 1 -name '*.md' ! -name README.md | wc -l) file(s) (cp fallback; no delete)"
fi

echo
echo "done."
if [ "$MODE" = "backup" ]; then
    echo "Review and commit:  git -C \"$REPO\" add claude-memory && git -C \"$REPO\" commit -m 'sync claude memory'"
fi
