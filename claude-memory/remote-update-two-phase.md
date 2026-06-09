---
name: remote-update-two-phase
description: Remote software update over cellular is two-phase (telemetry-cgroup fetch + separate apply unit) with auto-rollback. Device side done; dashboard pending.
metadata: 
  node_type: memory
  type: project
  originSessionId: 416c68fc-5615-46ab-98ac-9e38009bdaf2
---

Field Pis update their code remotely over cellular via an `update` command (params.ref, default `origin/main`). Built device-first in commits 2d0f5a1 + d28606c; fully tested on the dev Pi (happy path + auto-rollback). **Dashboard/cloud trigger UI still TODO.**

**Why two phases** (the core design constraint): the cellular firewall only lets the `beemonitor-telemetry.service` cgroup egress on wwan0, AND an update must restart telemetry itself — but one process can't keep network access (telemetry cgroup) while restarting its own cgroup. So:
- **fetch** (`hardware/update.sh fetch <ref>`): spawned BY telemetry (`subprocess.Popen(..., start_new_session=True)`), so it stays in telemetry's firewall-allowed cgroup and child git/pip can reach GitHub/PyPI over cellular. Does ALL network: git fetch + reset + `pip install -r hardware/requirements.txt` if it changed. Then writes a handoff file and triggers phase B.
- **apply** (`beemonitor-update.service`, oneshot, root, offline): restart services, health-check, and `git reset` back to the previous commit if they don't come up. Separate unit so restarting telemetry can't kill it.

**Key implementation facts:**
- Trigger from phase A→B: `sudo -n systemctl start --no-block beemonitor-update.service` needs `/etc/sudoers.d/beemonitor-update` (NOPASSWD, scoped to that exact command). The unit is installed but NOT enabled (on-demand only).
- Phase B (root) runs git/pip via `runuser -u <repo-owner>` to avoid git "dubious ownership" on the beemonitor-owned repo and keep venv files owned right.
- Scope is code + `hardware/requirements.txt` deps only. **torch/torchvision and model weights are deliberately excluded** (CPU-wheel/large — WiFi/manual; see [[pi-torch-must-be-cpu-wheel]]). hardware/requirements.txt is the Pi's tiny dep set (requests, ultralytics), NOT the repo-root requirements.txt.
- The beat reports `code_commit` (deployed short SHA) + `update` (last status: state ok|rolledback|idle|error, commit, detail). Status file: `~/.beemonitor/update-status.json` (STATE_DIR, default /home/beemonitor/.beemonitor).
- Health units gating success: recorder, uploader, telemetry (active ~15s after restart).

Relies on [[cellular-firewall-two-phase]] (the cgroup allowlist that fetch rides) and [[dashboard-wifi-needs-nmcli-sudoers]] (same command-channel pattern).
