---
name: cellular-needs-ip-type-4
description: Sixfab/IoT cellular needs IP_TYPE=4 in /etc/qmi-network.conf or the QMI data call fails (error 14 CallFailed / ipv4-only-allowed) and wwan0 gets no IP.
metadata: 
  node_type: memory
  type: project
  originSessionId: 416c68fc-5615-46ab-98ac-9e38009bdaf2
---

The cellular data session needs **`IP_TYPE=4`** (IPv4-only) in `/etc/qmi-network.conf`. Most IoT carriers — including the **Sixfab SIM** (Twilio Super SIM reseller, APN `super`) — grant **IPv4 only**. With no `IP_TYPE` set, `qmi-network` requests dual-stack IPv4v6, the network rejects the call with reason **`ipv4-only-allowed`** → **QMI error 14 `CallFailed`**, and `wwan0` never gets an IP (manual bring-up shows 100% packet loss).

**Fix:** add `IP_TYPE=4` to `/etc/qmi-network.conf`, `sudo qmi-network /dev/cdc-wdm0 stop`, then re-run `hardware/cellular/cellular-up.sh`. The shipped `hardware/cellular/qmi-network.conf.sample` now includes `IP_TYPE=4` by default (fixed in commit da1d921) — the bug was that the working unit's conf had it but the sample didn't, so fresh installs failed at README Step 10.5.

**Diagnostic discriminator** (when `wwan0` has no IP / 100% loss):
- `CallFailed` reason `ipv4-only-allowed` → IP type issue, this fix.
- `CallFailed` reason `service-option-not-subscribed` → the SIM isn't activated; activate it (with a data plan) in the Sixfab Connect dashboard (connect.sixfab.com).
- `Registration: 'not-registered'` → antenna on MAIN u.FL port / coverage / SIM seating.

Related: [[cellular-modem-is-telit]], [[cellular-firewall-two-phase]].
