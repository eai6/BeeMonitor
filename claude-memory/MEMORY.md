# Memory Index

- [Cellular modem is Telit](cellular-modem-is-telit.md) — Sixfab modem is Telit LE910C4-NF (not Quectel); AT port ttyUSB2; GPS via AT$GPSP/AT$GPSACP; use pyserial.
- [Pi torch must be CPU wheel](pi-torch-must-be-cpu-wheel.md) — on the Pi, torch/torchvision must be +cpu wheels or YOLO inference SIGILLs (status=4/ILL); install from download.pytorch.org/whl/cpu.
- [Cellular firewall is two-phase](cellular-firewall-two-phase.md) — base drop loads early in cellular-firewall.service; telemetry cgroup allow rule is added by beemonitor-telemetry's ExecStartPost (avoids cgroup boot race). Don't collapse it.
- [Cellular needs IP_TYPE=4](cellular-needs-ip-type-4.md) — Sixfab/IoT SIMs are IPv4-only; without IP_TYPE=4 in /etc/qmi-network.conf the data call fails (QMI err 14 CallFailed / ipv4-only-allowed) and wwan0 gets no IP.
- [Cellular DNS resolv.conf symlink](cellular-dns-resolv-symlink.md) — must rm the managed /etc/resolv.conf symlink before pinning static nameservers, else DNS stays on the 127.0.0.53 stub (8.8.8.8 pings but google.com fails over cellular).
- [Dashboard WiFi needs nmcli sudoers](dashboard-wifi-needs-nmcli-sudoers.md) — dashboard WiFi on/off/connect needs /etc/sudoers.d/beemonitor-nmcli (NOPASSWD nmcli); without it telemetry's `sudo -n nmcli` silently fails.
- [Motion gate uses shadow detection](motion-gate-shadow-detection.md) — MOG2 gate runs detectShadows=True + thresholds out shadow pixels to reject shadows/light-changes; don't revert to False. Tune BEEMONITOR_SHADOW_THRESHOLD; test via motion_replay.py.
