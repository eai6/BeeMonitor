"""Structured hardware-setup content — the single source of truth.

Both the on-dashboard guided walkthrough and the generated markdown guide
(``manage.py export_setup_guide``) render from the data here, so they can never
drift. It is a faithful, condensed port of ``hardware/README.md`` Steps 0-10.

Each step carries everything the wizard needs to render one screen:

    id              stable slug (also the SetupStepState key — never renumber)
    phase           which Phase it belongs to
    title           short imperative label
    concept         one or two sentences: *why* this step / what it does
    command         shell block with {{placeholders}} (see PLACEHOLDERS); may be ""
    expected        "what you should see" — the success signal (incl. physical
                    signals like LEDs), or "" if none
    verify          id of a live check in checks.py, or None (self-attested step)
    common_errors   list of {symptom, fix} shown inline as callouts
    minutes         rough time estimate
    difficulty      "easy" | "medium" | "advanced"
    optional        True if a unit can ship without it
    applies_to      "both" | "wifi" | "cellular"  (drives the unit-type branch)

Placeholders are substituted per-device at render time (see render_command):
    {{device_key}}  the raw bmk_device_ key (only during the one-shot window)
    {{api_base}}    settings.BEEMONITOR_DEVICE_API_BASE
    {{record_dir}}  on-Pi recordings dir
    {{hostname}}    the device's hostname guess (its name, slugified)
"""

from __future__ import annotations

# Placeholder default values; per-device overrides come from the view.
DEFAULTS = {
    "device_key": "bmk_device_REPLACE_ME",
    "api_base": "https://mqnafc3ejc.us-east-1.awsapprunner.com",
    "record_dir": "/home/beemonitor/Desktop/cameraOutput/beeHotel",
    "hostname": "beemonitor",
}

# Top-level phases, in order. 3-6 is the research-backed sweet spot.
PHASES = [
    {"id": "flash", "title": "Flash & first boot",
     "subtitle": "Get Raspberry Pi OS onto the card and reachable over SSH."},
    {"id": "software", "title": "Install software",
     "subtitle": "Code, system packages, the Python venv, and folders."},
    {"id": "configure", "title": "Camera & credentials",
     "subtitle": "Focus the lens and point the unit at your account."},
    {"id": "services", "title": "Install & start services",
     "subtitle": "The recorder, telemetry, and uploader as systemd units."},
    {"id": "power", "title": "Power & remote access",
     "subtitle": "WittyPi scheduling and Raspberry Pi Connect (field units)."},
    {"id": "connectivity", "title": "Cellular connectivity",
     "subtitle": "Sixfab 4G LTE — only for cellular field units."},
]

STEPS: list[dict] = [
    # ---------------------------------------------------------------- flash
    {
        "id": "flash_os", "phase": "flash", "title": "Flash Raspberry Pi OS",
        "concept": "Write Raspberry Pi OS (Bookworm, 64-bit) with Raspberry Pi "
                   "Imager. In the Imager's settings, set the username to "
                   "exactly 'beemonitor', enable SSH, and enter your bench WiFi "
                   "— every path in this guide assumes that user.",
        "command": "",
        "expected": "The card writes and verifies. 64-bit is required (the "
                    "calibration PyTorch wheels are aarch64-only).",
        "verify": None,
        "common_errors": [
            {"symptom": "Username isn't 'beemonitor'",
             "fix": "Re-flash — the systemd units hardcode /home/beemonitor "
                    "paths and run as User=beemonitor."},
        ],
        "minutes": 10, "difficulty": "easy", "optional": False, "applies_to": "both",
    },
    {
        "id": "first_boot", "phase": "flash", "title": "Boot & SSH in",
        "concept": "Connect the HQ Camera ribbon BEFORE first boot (it's only "
                   "probed at boot), power on, then SSH in and update the OS.",
        "command": "ssh beemonitor@{{hostname}}.local\n"
                   "sudo apt update && sudo apt full-upgrade -y\n"
                   "getconf LONG_BIT   # must print 64\n"
                   "whoami             # must print beemonitor",
        "expected": "SSH connects, LONG_BIT is 64, and whoami is beemonitor.",
        "verify": None,
        "common_errors": [
            {"symptom": "can't resolve <host>.local",
             "fix": "Use the Pi's IP from your router instead of mDNS: "
                    "ssh beemonitor@<pi-ip>."},
        ],
        "minutes": 10, "difficulty": "easy", "optional": False, "applies_to": "both",
    },
    # ------------------------------------------------------------- software
    {
        "id": "clone", "phase": "software", "title": "Download the code",
        "concept": "Clone the BeeMonitor repo into the home directory.",
        "command": "cd ~ && git clone https://github.com/eai6/BeeMonitor.git\n"
                   "cd ~/BeeMonitor/hardware",
        "expected": "A ~/BeeMonitor/hardware directory with the scripts and "
                    "systemd/ unit files.",
        "verify": None,
        "common_errors": [],
        "minutes": 2, "difficulty": "easy", "optional": False, "applies_to": "both",
    },
    {
        "id": "deps", "phase": "software", "title": "System deps + virtualenv",
        "concept": "picamera2/cv2 come from apt; the rest live in a venv built "
                   "with --system-site-packages so it can import them. Install "
                   "the CPU torch wheels FIRST or YOLO SIGILLs on the Pi's ARM "
                   "CPU.",
        "command": "sudo apt install -y python3-picamera2 python3-opencv ffmpeg "
                   "python3-venv libqmi-utils udhcpc nftables\n"
                   "python3 -m venv --system-site-packages ~/BeeMonitor/hardware/venv\n"
                   "~/BeeMonitor/hardware/venv/bin/pip install --upgrade pip requests\n"
                   "~/BeeMonitor/hardware/venv/bin/pip install torch torchvision "
                   "--index-url https://download.pytorch.org/whl/cpu\n"
                   "~/BeeMonitor/hardware/venv/bin/pip install ultralytics",
        "expected": "All installs succeed. (libqmi-utils/udhcpc/nftables are the "
                    "cellular stack — harmless to install on a WiFi unit.)",
        "verify": None,
        "common_errors": [
            {"symptom": "YOLO exits with 'illegal instruction' / status=4",
             "fix": "torch was installed after ultralytics. Reinstall the CPU "
                    "torch wheels first, then ultralytics."},
        ],
        "minutes": 12, "difficulty": "medium", "optional": False, "applies_to": "both",
    },
    {
        "id": "dirs", "phase": "software", "title": "Output dirs + seed calibration",
        "concept": "Create the recording tree and seed a tight bee-sized motion "
                   "window so the recorder doesn't start wide-open. Run WITHOUT "
                   "sudo — the recorder runs as 'beemonitor' and can't write a "
                   "root-owned tree.",
        "command": "~/BeeMonitor/hardware/venv/bin/python makeDirectories.py\n"
                   "cp calibration.sample.json {{record_dir}}/../calibration.json",
        "expected": "The cameraOutput tree exists and calibration.json is in "
                    "place ([62, 554], not the wide-open [20, 5000]).",
        "verify": None,
        "common_errors": [
            {"symptom": "'permission denied' saving video later",
             "fix": "You ran makeDirectories.py as root. Fix ownership: sudo "
                    "chown -R beemonitor:beemonitor "
                    "/home/beemonitor/Desktop/cameraOutput"},
        ],
        "minutes": 2, "difficulty": "easy", "optional": False, "applies_to": "both",
    },
    # ------------------------------------------------------------ configure
    {
        "id": "camera", "phase": "configure", "title": "Focus the camera",
        "concept": "Run the focus helper and turn the lens until the preview is "
                   "sharp on the bee hotel. A blurry lens wrecks both motion "
                   "detection and calibration.",
        "command": "~/BeeMonitor/hardware/venv/bin/python runFocus.py",
        "expected": "A live preview; the hotel entrances look crisp. (Verify the "
                    "camera is detected first: libcamera-hello --list-cameras.)",
        "verify": None,
        "common_errors": [
            {"symptom": "no cameras available",
             "fix": "Ribbon seated after boot — reseat it and reboot; the camera "
                    "is only probed at boot."},
        ],
        "minutes": 5, "difficulty": "medium", "optional": False, "applies_to": "both",
    },
    {
        "id": "credentials", "phase": "configure", "title": "Credentials & tuning",
        "concept": "Point the unit at YOUR account. The device key below is "
                   "filled in for this device — paste the whole block into "
                   "/etc/beemonitor/uploader.env.",
        "command": "sudo mkdir -p /etc/beemonitor\n"
                   "sudo tee /etc/beemonitor/uploader.env >/dev/null <<'EOF'\n"
                   "BEEMONITOR_API_BASE={{api_base}}\n"
                   "BEEMONITOR_DEVICE_KEY={{device_key}}\n"
                   "BEEMONITOR_RECORD_DIR={{record_dir}}\n"
                   "EOF\n"
                   "sudo chmod 600 /etc/beemonitor/uploader.env",
        "expected": "uploader.env exists, mode 600, with your real device key.",
        "verify": None,
        "common_errors": [
            {"symptom": "key shows bmk_device_REPLACE_ME",
             "fix": "The one-time key window has closed. Re-issue a key for this "
                    "device (Devices → this device → re-issue) and paste it in."},
        ],
        "minutes": 5, "difficulty": "easy", "optional": False, "applies_to": "both",
    },
    # ------------------------------------------------------------- services
    {
        "id": "install_units", "phase": "services", "title": "Install the services",
        "concept": "Copy the systemd unit files and grant the telemetry user "
                   "passwordless nmcli + USB-transfer so the dashboard can "
                   "control WiFi and copy to USB.",
        "command": "cd ~/BeeMonitor/hardware\n"
                   "sudo cp systemd/beemonitor-recorder.service systemd/"
                   "beemonitor-telemetry.service systemd/beemonitor-uploader.service "
                   "systemd/beemonitor-calibrate.service systemd/"
                   "beemonitor-calibrate.timer /etc/systemd/system/\n"
                   "echo 'beemonitor ALL=(root) NOPASSWD: /usr/bin/nmcli' | "
                   "sudo tee /etc/sudoers.d/beemonitor-nmcli >/dev/null\n"
                   "sudo chmod 440 /etc/sudoers.d/beemonitor-nmcli\n"
                   "sudo systemctl daemon-reload",
        "expected": "daemon-reload returns cleanly; visudo -cf reports no syntax "
                    "errors on the drop-in.",
        "verify": None,
        "common_errors": [],
        "minutes": 5, "difficulty": "medium", "optional": False, "applies_to": "both",
    },
    {
        "id": "start_services", "phase": "services", "title": "Start & enable on boot",
        "concept": "Enable and start the app layer. After this the unit should "
                   "check in — watch the dashboard flip to ONLINE here.",
        "command": "sudo systemctl enable --now beemonitor-recorder.service "
                   "beemonitor-telemetry.service beemonitor-uploader.service\n"
                   "sudo systemctl enable --now beemonitor-calibrate.timer",
        "expected": "The dashboard shows this device as ONLINE within a minute "
                    "(the first heartbeat arrived), recorder + uploader active.",
        "verify": "device_online",
        "common_errors": [
            {"symptom": "stays offline",
             "fix": "journalctl -u beemonitor-telemetry -n 30 — look for "
                    "'heartbeat POST failed' (network) or a 401 (wrong key)."},
        ],
        "minutes": 3, "difficulty": "easy", "optional": False, "applies_to": "both",
    },
    {
        "id": "verify_services", "phase": "services", "title": "Verify recorder & uploader",
        "concept": "Confirm the recorder is capturing and the uploader is "
                   "running. Wave a hand in front of the hotel to trigger a clip.",
        "command": "systemctl is-active beemonitor-recorder beemonitor-uploader\n"
                   "journalctl -u beemonitor-uploader -f   # watch a clip upload",
        "expected": "Both report 'active'; a hand-wave produces a snippet that "
                    "uploads to the cloud and appears under this device's videos.",
        "verify": "services_running",
        "common_errors": [],
        "minutes": 5, "difficulty": "easy", "optional": False, "applies_to": "both",
    },
    # ---------------------------------------------------------------- power
    {
        "id": "wittypi", "phase": "power", "title": "Set up WittyPi scheduling",
        "concept": "WittyPi powers the Pi on a schedule (e.g. dawn-to-dusk) to "
                   "save battery in the field. Install it, set auto power-on, and "
                   "load your daily window.",
        "command": "# Follow Step 8 in hardware/README.md for the WittyPi installer\n"
                   "# and the production on/off schedule for your site.",
        "expected": "A scheduled power cycle the unit follows on its own.",
        "verify": None,
        "common_errors": [],
        "minutes": 30, "difficulty": "advanced", "optional": True, "applies_to": "both",
    },
    {
        "id": "pi_connect", "phase": "power", "title": "Raspberry Pi Connect",
        "concept": "Remote shell/screen to the unit from anywhere — invaluable "
                   "for a field device you can't physically reach.",
        "command": "sudo apt install -y rpi-connect\n"
                   "rpi-connect on\n"
                   "rpi-connect signin   # follow the URL to link your account",
        "expected": "The device appears at connect.raspberrypi.com.",
        "verify": None,
        "common_errors": [],
        "minutes": 5, "difficulty": "easy", "optional": True, "applies_to": "both",
    },
    # --------------------------------------------------------- connectivity
    {
        "id": "cell_modem", "phase": "connectivity", "title": "Modem in QMI mode",
        "concept": "Confirm the Sixfab/Telit modem enumerated in QMI mode and "
                   "stop ModemManager (it fights QMI). One-time USB-mode switch "
                   "may be needed first — see Step 10.1.",
        "command": "lsusb | grep -i 1bc7:1201 && ls /dev/cdc-wdm0\n"
                   "sudo systemctl disable --now ModemManager.service",
        "expected": "The Telit modem and /dev/cdc-wdm0 control node are present.",
        "verify": None,
        "common_errors": [
            {"symptom": "no /dev/cdc-wdm0",
             "fix": "Do the one-time USB-mode switch (Telit: AT#USBCFG) in Step "
                    "10.1, then reboot."},
        ],
        "minutes": 10, "difficulty": "advanced", "optional": False, "applies_to": "cellular",
    },
    {
        "id": "cell_apn_dns", "phase": "connectivity", "title": "APN + pinned DNS",
        "concept": "Set the carrier APN (the sample is 'super' for the Sixfab "
                   "SIM) and pin DNS to a real static file — fresh Pi OS images "
                   "symlink resolv.conf at 127.0.0.53, which breaks name "
                   "resolution over cellular.",
        "command": "sudo cp cellular/qmi-network.conf.sample /etc/qmi-network.conf\n"
                   "sudo chattr -i /etc/resolv.conf 2>/dev/null || true\n"
                   "sudo rm -f /etc/resolv.conf\n"
                   "printf 'nameserver 8.8.8.8\\nnameserver 1.1.1.1\\n' | "
                   "sudo tee /etc/resolv.conf\n"
                   "sudo chattr +i /etc/resolv.conf",
        "expected": "After the link is up, google.com resolves (not just 8.8.8.8).",
        "verify": None,
        "common_errors": [
            {"symptom": "8.8.8.8 pings but google.com doesn't",
             "fix": "resolv.conf is still the managed symlink — rm it and write "
                    "the static file as above."},
        ],
        "minutes": 8, "difficulty": "advanced", "optional": False, "applies_to": "cellular",
    },
    {
        "id": "cell_up", "phase": "connectivity", "title": "Bring up the link + firewall",
        "concept": "Install the cellular + firewall units. The firewall gates "
                   "mobile data to ONLY the telemetry service so nothing else "
                   "eats your SIM; bulk video stays WiFi-gated.",
        "command": "chmod +x cellular/cellular-up.sh cellular/cellular-firewall.sh\n"
                   "sudo cp systemd/cellular-firewall.service systemd/cellular.service "
                   "/etc/systemd/system/\n"
                   "sudo systemctl daemon-reload\n"
                   "sudo systemctl enable --now cellular-firewall.service cellular.service",
        "expected": "ping -c3 8.8.8.8 succeeds over the modem and the device "
                    "keeps checking in with cellular_active true on the dashboard.",
        "verify": "cellular_up",
        "common_errors": [
            {"symptom": "cellular.service enabled but never runs, empty journal",
             "fix": "Dependency cycle — never add After=multi-user.target to it "
                    "(see Step 10.6)."},
            {"symptom": "uploads fail with TLS/cert errors after a cold boot",
             "fix": "Clock is stale — wait for cellular.service then NTP sync."},
        ],
        "minutes": 12, "difficulty": "advanced", "optional": False, "applies_to": "cellular",
    },
]

# ---------------------------------------------------------------------------
# Helpers used by both the wizard views and the markdown generator.
# ---------------------------------------------------------------------------

def phase_index() -> dict:
    return {p["id"]: i for i, p in enumerate(PHASES)}


def steps_for(unit_type: str) -> list[dict]:
    """Steps that apply to a unit_type ('wifi' | 'cellular'), in order."""
    return [s for s in STEPS if s["applies_to"] == "both" or s["applies_to"] == unit_type]


def step_by_id(step_id: str) -> dict | None:
    for s in STEPS:
        if s["id"] == step_id:
            return s
    return None


def render_command(template: str, values: dict | None = None) -> str:
    """Substitute {{placeholders}} with per-device values (or DEFAULTS)."""
    if not template:
        return ""
    vals = {**DEFAULTS, **(values or {})}
    out = template
    for key, val in vals.items():
        out = out.replace("{{%s}}" % key, str(val))
    return out


def as_markdown(values: dict | None = None) -> str:
    """Render the whole guide to markdown (the generated artifact)."""
    lines = ["# BeeMonitor — Device Setup Guide (generated)",
             "",
             "> Generated from `apps/setup/content.py` — do not edit by hand.",
             "> The interactive version lives on the dashboard under **Set up a device**.",
             ""]
    pidx = phase_index()
    for phase in PHASES:
        lines.append(f"## {phase['title']}")
        lines.append("")
        lines.append(f"*{phase['subtitle']}*")
        lines.append("")
        for step in STEPS:
            if step["phase"] != phase["id"]:
                continue
            badge = []
            if step["optional"]:
                badge.append("optional")
            if step["applies_to"] != "both":
                badge.append(f"{step['applies_to']}-only")
            badge.append(f"~{step['minutes']} min")
            lines.append(f"### {step['title']}  ({', '.join(badge)})")
            lines.append("")
            lines.append(step["concept"])
            lines.append("")
            cmd = render_command(step["command"], values)
            if cmd:
                lines.append("```bash")
                lines.append(cmd)
                lines.append("```")
                lines.append("")
            if step["expected"]:
                lines.append(f"**What you should see:** {step['expected']}")
                lines.append("")
            for ce in step["common_errors"]:
                lines.append(f"> ⚠️ **{ce['symptom']}** — {ce['fix']}")
                lines.append("")
    _ = pidx  # reserved for future cross-refs
    return "\n".join(lines).rstrip() + "\n"
