---
name: dashboard-wifi-needs-nmcli-sudoers
description: "The dashboard's WiFi on/off/connect buttons need the /etc/sudoers.d/beemonitor-nmcli NOPASSWD rule, or telemetry's `sudo -n nmcli` silently fails."
metadata: 
  node_type: memory
  type: project
  originSessionId: 416c68fc-5615-46ab-98ac-9e38009bdaf2
---

The web dashboard's **WiFi on/off/connect** controls are executed by `beemonitor-telemetry.service`: it polls for commands and runs `nmcli` (via `_nmcli()` in `hardware/telemetry.py`, which calls `sudo -n nmcli ...` because the service runs as the unprivileged `beemonitor` user and radio/connect state changes need root).

This requires a sudoers drop-in granting passwordless nmcli:
```
echo 'beemonitor ALL=(root) NOPASSWD: /usr/bin/nmcli' | sudo tee /etc/sudoers.d/beemonitor-nmcli
sudo chmod 440 /etc/sudoers.d/beemonitor-nmcli
sudo visudo -cf /etc/sudoers.d/beemonitor-nmcli   # must print "parsed OK"
```

**Symptom if missing:** the dashboard WiFi toggle does nothing — `sudo -n nmcli` fails ("a password is required"). No restart needed after adding the rule; sudo re-reads it per call. Check `journalctl -u beemonitor-telemetry | grep -i wifi`.

**The gap that caused this:** the rule was only in the README **Quick Install** block, missing from the step-by-step **Steps 1-7** path — so step-by-step installers hit it. Added to detailed Step 6 + a troubleshooting row in commit f288bbd.

Related cellular/install gotchas: [[cellular-needs-ip-type-4]], [[cellular-dns-resolv-symlink]], [[cellular-firewall-two-phase]].
