# Cellular + DNS Bring-up Diagnostics

A runbook for when a unit's cellular link or DNS isn't working (README
[Step 10](../README.md#step-10-cellular-connectivity-sixfab-4g-lte)). Run the
stages **in order** — each isolates one layer, so the first one that fails points
at the cause. All commands run **on the affected Pi** (its SSH session / console).

> **Bench tip:** with WiFi on, its default route wins (lower metric) so plain
> `ping`/DNS leave via WiFi, masking cellular problems. To test cellular for real:
> `sudo nmcli radio wifi off` (or `sudo rfkill block wifi`), and `… on` to restore.

---

## One-shot health check

Paste this whole block; it summarizes every layer at once.

```bash
echo '=== 1. modem present (Telit 1bc7:1201 + /dev/cdc-wdm0) ==='
lsusb | grep -i 1bc7:1201; ls -l /dev/cdc-wdm0 2>&1
echo '=== 2. APN + IP type config ==='
cat /etc/qmi-network.conf 2>&1
echo '=== 3. registration (want: registered) ==='
sudo qmicli -d /dev/cdc-wdm0 --nas-get-serving-system 2>&1 | grep -iE 'registration|network' | head
echo '=== 4. signal ==='
sudo qmicli -d /dev/cdc-wdm0 --nas-get-signal-strength 2>&1 | grep -iE 'rssi|dbm' | head
echo '=== 5. cellular.service ==='
systemctl is-active cellular.service 2>&1; systemctl is-enabled cellular.service 2>&1
echo '=== 6. wwan0 IP ==='
ip addr show wwan0 2>&1 | grep -E 'inet|state'
echo '=== 7. routes ==='
ip route | grep -E 'default|wwan0'
echo '=== 8. resolv.conf (want: regular file, 8.8.8.8 / 1.1.1.1, +i) ==='
ls -l /etc/resolv.conf; lsattr /etc/resolv.conf 2>/dev/null; cat /etc/resolv.conf
echo '=== 9. connectivity over cellular ==='
ping -c2 -W3 -I wwan0 8.8.8.8 >/dev/null 2>&1 && echo 'wwan0 -> 8.8.8.8 OK' || echo 'wwan0 -> 8.8.8.8 FAIL'
echo '=== 10. DNS ==='
getent hosts google.com >/dev/null 2>&1 && echo 'DNS OK' || echo 'DNS FAIL'
echo '=== 11. firewall (optional, after 10.7) ==='
systemctl is-active cellular-firewall.service 2>&1
```

---

## Stage-by-stage (run if the one-shot shows a failure)

### A. Modem not enumerated
```bash
lsusb | grep -i 1bc7:1201; ls /dev/cdc-wdm0
```
- **Missing** → modem not in QMI mode or not seated. Do the one-time USB-mode
  switch (Telit: `AT#USBCFG=<n>`), reseat the HAT/USB jumper, power-cycle.
  See README [10.1](../README.md#101-insert-the-sim-and-assemble-the-hat).

### B. SIM / registration
```bash
sudo qmicli -d /dev/cdc-wdm0 --uim-get-card-status 2>&1 | grep -iE 'card state|application state'
sudo qmicli -d /dev/cdc-wdm0 --nas-get-serving-system 2>&1 | head -20
sudo qmicli -d /dev/cdc-wdm0 --nas-get-signal-strength 2>&1 | head
```
- **Card not `present` / app not `ready`** → reseat the SIM (orientation, fully in).
- **`Registration: 'not-registered'`** → antenna on the **MAIN** u.FL port, coverage,
  or the SIM isn't activated.
- **Low/no signal** → antenna / placement.

### C. Data call fails — `QMI error 14 'CallFailed'`
Check `/etc/qmi-network.conf` has **`IP_TYPE=4`**:
```bash
grep -q '^IP_TYPE=' /etc/qmi-network.conf || echo 'IP_TYPE=4' | sudo tee -a /etc/qmi-network.conf
sudo qmi-network /dev/cdc-wdm0 stop 2>/dev/null
sudo ~/BeeMonitor/hardware/cellular/cellular-up.sh & sleep 20
ip addr show wwan0 | grep inet
```
- Reason **`ipv4-only-allowed`** → carrier is IPv4-only; `IP_TYPE=4` fixes it
  (the shipped sample sets it).
- Reason **`service-option-not-subscribed`** → SIM not activated / no data plan →
  activate it in the **Sixfab Connect** dashboard (connect.sixfab.com).

### D. `wwan0` up but "Destination unreachable: No route"
The link is down or has no route — usually the manual run from
[10.5](../README.md#105-bring-it-up-manually-verify-before-automating) was
`kill`ed and `cellular.service` isn't installed/started yet.
```bash
systemctl is-active cellular.service                 # "not found" => not installed (do 10.6)
sudo systemctl restart cellular.service && sleep 20  # if installed
ip addr show wwan0 | grep inet; ip route | grep wwan0
```
- **`Unit cellular.service not found`** → you haven't done
  [10.6](../README.md#106-install-the-cellular-service-survives-reboot--wittypi-wakes) yet — install it.
- After restart, `wwan0` should have an IP and a `default … dev wwan0` route.

### E. DNS — `ping 8.8.8.8` works but `ping google.com` fails
Two causes; the `resolv.conf` check tells you which:
```bash
ls -l /etc/resolv.conf; cat /etc/resolv.conf
```
1. **Symlink, or content shows `127.0.0.53`** → it's still the managed stub.
   Re-pin as a real static file:
   ```bash
   sudo chattr -i /etc/resolv.conf 2>/dev/null || true
   sudo rm -f /etc/resolv.conf
   printf 'nameserver 8.8.8.8\nnameserver 1.1.1.1\n' | sudo tee /etc/resolv.conf
   sudo chattr +i /etc/resolv.conf
   ```
2. **`resolv.conf` is correct** → the query is leaving over WiFi. Force cellular:
   ```bash
   sudo nmcli radio wifi off
   ping -c 3 google.com          # resolves now? cellular DNS is fine; it was the WiFi path
   ```

### F. Firewall (after [10.7](../README.md#107-gate-cellular-to-beemonitor-only-dont-let-anything-else-eat-your-sim))
```bash
systemctl is-active cellular-firewall.service        # active (exited) is correct (oneshot)
sudo nft list table inet beemon_cell                 # cgroupv2 telemetry line + a drop counter
```
- The `socket cgroupv2 … beemonitor-telemetry.service` line only appears **after**
  telemetry has started (two-phase load — telemetry's `ExecStartPost` adds it).
- ICMP/DNS/DHCP/NTP are allowed, so the firewall never blocks the checks above.

---

## Known gotchas (all fixed in the shipped config)
| Symptom | Cause | Fix |
|---|---|---|
| `Unit ModemManager.service does not exist` | ModemManager not installed | Harmless — ignore (nothing fights QMI) |
| `CallFailed` / `ipv4-only-allowed` | dual-stack requested on IPv4-only carrier | `IP_TYPE=4` in `/etc/qmi-network.conf` |
| `8.8.8.8` ok, `google.com` fails | `resolv.conf` symlink → `127.0.0.53` stub | `rm -f` the symlink, write static, `chattr +i` |
| "No route" after 10.5 | manual run killed, service not installed | do 10.6 (install `cellular.service`) |
| firewall `failed` at boot | cgroup didn't exist when it loaded | two-phase load (base early + telemetry `ExecStartPost`) |
| recorder "permission denied" saving video | output tree owned by root | `chown -R beemonitor:beemonitor …/cameraOutput` |
