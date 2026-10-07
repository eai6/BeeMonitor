---
name: cellular-firewall-two-phase
description: "The cellular egress firewall loads in two phases (base drop early, telemetry allow via telemetry's ExecStartPost) to avoid a cgroup boot race."
metadata: 
  node_type: memory
  type: project
  originSessionId: 416c68fc-5615-46ab-98ac-9e38009bdaf2
---

The BeeMonitor cellular egress firewall (`hardware/cellular/cellular-firewall.sh`, nftables table `inet beemon_cell` on `wwan0`) loads in **two phases by design** — do not collapse it back into a single early load.

**Why:** the telemetry allow rule is `socket cgroupv2 level 2 "system.slice/beemonitor-telemetry.service"`, and nftables resolves that cgroup path to an id *at load time*. The cgroup doesn't exist until `beemonitor-telemetry.service` starts. The firewall used to load once in the pre-network slot (before telemetry), so it failed every boot (`cgroupv2 path fails: No such file or directory`) and left cellular ungated.

**How it works now:**
- `cellular-firewall.sh base` — default-drop on `wwan0` + link essentials (DNS/DHCP/NTP/ICMP), **no cgroup rule**. Loaded early by `cellular-firewall.service` (`Before=cellular.service`), closing the leak window with no service dependency.
- `cellular-firewall.sh telemetry` — same table **plus** the cgroup allow rule. Applied by `beemonitor-telemetry.service`'s `ExecStartPost=+-...cellular-firewall.sh telemetry`. `+` runs as root, `-` makes it never block telemetry / no-op on a WiFi/bench unit. Runs on **every** (re)start incl. `Restart=on-failure` auto-restart, so the rule is re-resolved after the cgroup id changes on restart.

**How to apply:** chose telemetry `ExecStartPost` over a separate phase-2 unit specifically because `PartOf=` does NOT propagate auto-restarts, which would silently break the allow rule when telemetry crash-restarts.

The `socket cgroupv2 …` line only appears in `sudo nft list table inet beemon_cell` **after** telemetry is up — that's expected, not a bug. Fixed in commit f578c53. Related: [[cellular-modem-is-telit]].
