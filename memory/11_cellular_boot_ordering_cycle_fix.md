# Cellular Service Boot Failure — systemd Ordering Cycle (Root Cause + Fix)

**Date diagnosed:** 2026-06-05
**Affected unit:** `cellular.service` (and, as collateral, `beemonitor-uploader.service`)
**Status:** Fixed in repo (`hardware/systemd/cellular.service`) and on the live Pi.

---

## Symptom

After a boot/WittyPi wake, the cellular link never came up on its own:

```
$ systemctl status cellular.service
○ cellular.service - Sixfab cellular link (QMI) — bring up and keep alive
     Loaded: loaded (...; enabled; preset: enabled)
     Active: inactive (dead)

$ journalctl -u cellular.service -b
-- No entries --
```

`enabled` + `inactive (dead)` + **empty journal**. A service that *ran and failed*
leaves logs; one that was *never triggered* leaves nothing. The empty journal is
the tell: systemd never even attempted to start it.

## Root cause — a dependency cycle

The unit had **both**:

```ini
After=multi-user.target      # "start me after this target is reached"
...
[Install]
WantedBy=multi-user.target   # "this target wants me" → reaching it pulls me in
```

That is circular: `multi-user.target` pulls in `cellular.service`, but
`cellular.service` is ordered *after* `multi-user.target`. systemd detects the
ordering cycle at boot and breaks it by **deleting one of the jobs** — and it
deleted `cellular.service`. Confirmed in the boot log:

```
multi-user.target: Found ordering cycle on beemonitor-uploader.service/start
multi-user.target: Found dependency on cellular.service/start
Job beemonitor-uploader.service/start deleted to break ordering cycle ...
beemonitor-telemetry.service: Found ordering cycle on cellular.service/start
Job cellular.service/start deleted to break ordering cycle ...
```

The cycle also took out `beemonitor-uploader.service` on one resolution path, so
the bug silently broke **two** services, not just one.

## Why the obvious "fixes" don't work

- `systemctl enable` / `reenable` does nothing — the `[Install]` symlink was
  already correct. The problem is ordering, not enablement.
- `systemctl daemon-reload` after editing the unit does **not** retroactively
  start the service or undo the deleted boot job. The fix only takes effect on a
  fresh boot transaction (next reboot) or via an explicit `systemctl start`.

## The fix

Remove the single bad edge — `After=multi-user.target` — from `cellular.service`.
A `WantedBy=multi-user.target` service already starts as part of reaching that
target; it must not also be ordered *after* it. This clears the cycle for all
three units (`cellular`, `uploader`, `telemetry`). The only remaining ordering,
`Before=beemonitor-uploader.service`, is correct and was kept.

```bash
sudo sed -i '/^After=multi-user.target/d' /etc/systemd/system/cellular.service
sudo systemctl daemon-reload
sudo systemctl start cellular.service     # this boot; reboot proves autostart
```

After the fix the link came up cleanly: `Network started successfully` →
`udhcpc wwan0 bound IP=...` → `cellular: link up on wwan0`, `0 restarts` (the
watchdog holds steady, not crash-looping).

## Related: clock / NTP (no RTC)

The Pi has no RTC. On every cold boot / WittyPi wake it restores a stale clock
from `fake-hwclock`, and `systemd-timesyncd` can't sync until a network route
exists — which only happens *after* `cellular.service` is up. A wrong clock fails
TLS cert validation, so S3 uploads + telemetry error until time is corrected.
Two safeguards now exist:

1. Install enables it: `sudo timedatectl set-ntp true` (documented in
   `hardware/README.md` §10.6).
2. `cellular-up.sh` kicks a resync (`timedatectl set-ntp true` +
   `systemctl restart systemd-timesyncd`) right after proving connectivity, so
   the clock is correct before the uploader's first request.

## Takeaways

- **`enabled` + `inactive` + empty journal == systemd never started it.** Suspect
  an ordering cycle or a failed condition, not a crashing service. Check the main
  journal for `Found ordering cycle` / `deleted to break ordering cycle`.
- **Never combine `After=X.target` with `WantedBy=X.target`** for the same
  target. `WantedBy` alone both enables and schedules the unit.
- A unit that *creates* the network (cellular) should have **no** network/target
  `After=` ordering — only `Before=` whatever consumes the network.
- `daemon-reload` ≠ re-running the boot transaction. Validate ordering fixes with
  an actual reboot.

## Files changed

- `hardware/systemd/cellular.service` — removed `After=multi-user.target`; added a
  comment warning against re-adding it.
- `hardware/cellular/cellular-up.sh` — added NTP resync after link-up.
- `hardware/README.md` §10.6 — documented the no-`After` rule and NTP enablement.
