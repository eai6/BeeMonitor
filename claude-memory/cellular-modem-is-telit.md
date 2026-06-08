---
name: cellular-modem-is-telit
description: "The Sixfab cellular modem is a Telit LE910C4-NF (not Quectel) — AT port, GPS commands"
metadata: 
  node_type: memory
  type: project
  originSessionId: 87980b96-0146-4a50-ac80-9e5c978a75ec
---

The Sixfab cellular HAT modem is a **Telit LE910C4-NF** (USB `1bc7:1201`), NOT a
Quectel — earlier docs/scripts wrongly assumed Quectel.

- **AT port = `/dev/ttyUSB2`** (`/dev/ttyUSB3` also answers AT). Data link is QMI
  on `/dev/cdc-wdm0` → `wwan0`, managed by `cellular-up.sh` (cellular.service).
- **GPS uses Telit AT commands**, not Quectel: `AT$GPSP=1` to power GNSS on,
  `AT$GPSACP` to read position (`$GPSACP: <UTC>,<lat>,<lon>,<hdop>,<alt>,<fix>,...`,
  `<fix>` 2/3 = 2D/3D, 0/1 = no fix; lat/lon are `ddmm.mmmm[N/S]`). NOT
  `AT+QGPS` / `AT+QGPSLOC`.
- **Use pyserial** to talk to these ports — raw-shell `stty`/`cat` reads are
  unreliable on the Telit ports (DTR/termios). `hardware/cellular/gps-test.sh`
  delegates to a pyserial heredoc. No sudo needed (user is in `dialout`).

Verified on-Pi 2026-06-05: port detect + GPS power + `$GPSACP` parse all work; a
real coordinate fix still needs the GNSS antenna + outdoor sky view.

Related: [[gps-telemetry-wiring-plan]].
