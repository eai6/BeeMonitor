---
name: cellular-dns-resolv-symlink
description: "Pinning DNS for cellular must rm the managed /etc/resolv.conf symlink first, else DNS stays on the 127.0.0.53 stub and names don't resolve over cellular (8.8.8.8 pings, google.com fails)."
metadata: 
  node_type: memory
  type: project
  originSessionId: 416c68fc-5615-46ab-98ac-9e38009bdaf2
---

When pinning DNS for the cellular link, you must **remove the managed `/etc/resolv.conf` symlink before writing the static file**. On fresh Pi OS images `/etc/resolv.conf` is a symlink to a NetworkManager/systemd-resolved file (usually the `127.0.0.53` stub). Writing with `tee` follows the symlink and the system keeps resolving via the stub, which has **no upstream over cellular** — so `ping 8.8.8.8` works but `ping google.com` fails.

**Symptom:** IP connectivity over `wwan0` is fine (`ping 8.8.8.8` ok) but name resolution fails (`ping google.com` fails). Pure DNS.

**Fix / correct pin sequence:**
```
sudo chattr -i /etc/resolv.conf 2>/dev/null || true
sudo rm -f /etc/resolv.conf            # <-- the key step: drop the managed symlink
printf 'nameserver 8.8.8.8\nnameserver 1.1.1.1\n' | sudo tee /etc/resolv.conf
sudo chattr +i /etc/resolv.conf        # immutable so NM/resolved can't clobber it
```
Verify with `ls -l /etc/resolv.conf` — must be a regular file (`-`), not a symlink (`l ... ->`). README Step 10.4 + Quick Install step 5 hardened with the `rm -f` in commit 41d412c.

Part of the cellular-bringup gotcha family: [[cellular-needs-ip-type-4]], [[cellular-firewall-two-phase]], [[cellular-modem-is-telit]].
