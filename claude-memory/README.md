# Claude memory (backup)

A snapshot of the per-project memory notes that Claude Code accumulated while
working on this repo (operational gotchas, hardware quirks, and design decisions
that aren't obvious from the code alone). The live copy lives **outside** the repo
at `~/.claude/projects/-home-beemonitor-BeeMonitor/memory/` and is local to one
machine; this folder is a committed backup so the knowledge travels with the repo
and survives a re-clone.

> Not loaded automatically by Claude — the live `~/.claude/.../memory/` is. Treat
> this as documentation / a seed.

Keep this backup in sync with [`scripts/sync-claude-memory.sh`](../scripts/sync-claude-memory.sh):

```bash
scripts/sync-claude-memory.sh            # backup:  live store -> this folder (then commit)
scripts/sync-claude-memory.sh --restore  # restore: this folder -> live store (new machine)
scripts/sync-claude-memory.sh --dry-run  # preview changes, copy nothing
```
This `README.md` is repo-only and is never copied to / deleted from the live store.

`MEMORY.md` is the index; each other file is one note. Current notes:

| Note | Gist |
|------|------|
| `cellular-modem-is-telit.md` | Modem is Telit LE910C4-NF (not Quectel); AT port ttyUSB2; GPS via AT$GPSP/AT$GPSACP |
| `pi-torch-must-be-cpu-wheel.md` | On the Pi, torch/torchvision must be +cpu wheels or YOLO SIGILLs (status=4) |
| `cellular-firewall-two-phase.md` | Firewall loads base drop early + telemetry cgroup allow via telemetry's ExecStartPost (cgroup boot race) |
| `cellular-needs-ip-type-4.md` | IoT SIMs are IPv4-only; need `IP_TYPE=4` or QMI err 14 CallFailed / ipv4-only-allowed |
| `cellular-dns-resolv-symlink.md` | Must `rm` the managed resolv.conf symlink before pinning DNS, else stuck on 127.0.0.53 stub |
| `dashboard-wifi-needs-nmcli-sudoers.md` | Dashboard WiFi control needs `/etc/sudoers.d/beemonitor-nmcli` (NOPASSWD nmcli) |
| `motion-gate-shadow-detection.md` | MOG2 gate uses `detectShadows=True` + drops shadow pixels; don't revert |

Most of this is also reflected in the user-facing docs (`hardware/README.md`,
`hardware/cellular/DIAGNOSTICS.md`) — these notes are the terser, decision-oriented
form.
