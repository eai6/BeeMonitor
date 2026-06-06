# BeeMonitor Hardware Guide

**Complete assembly and deployment instructions for the BeeMonitor video recording system**

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Bill of Materials](#bill-of-materials)
4. [3D Printed Enclosure](#3d-printed-enclosure)
5. [Assembly Instructions](#assembly-instructions)
6. [Software Installation](#software-installation)
7. [How Motion-Gated Recording Works](#how-motion-gated-recording-works)
8. [Configuration Reference](#configuration-reference)
9. [WittyPi Setup](#step-8-set-up-wittypi)
10. [Raspberry Pi Connect Setup](#step-9-set-up-raspberry-pi-connect-remote-access)
11. [Cellular Connectivity (Sixfab 4G-LTE)](#step-10-cellular-connectivity-sixfab-4g-lte)
12. [Testing & Verification](#testing--verification)
13. [Field Deployment](#field-deployment)
14. [Maintenance](#maintenance)
15. [Troubleshooting](#troubleshooting)

## Overview

The BeeMonitor hardware system consists of two independent modules:

1. **Video Recording Module** (~$350 USD) — Raspberry Pi-based system that records **only video snippets where there is activity** and uploads them automatically

![alt text](recording_module.png)

2. **Energy Module** (~$245 USD) — Solar panel and battery for off-grid deployment

![alt text](energy_module.png)

The recording module can operate standalone with grid power, or combined with the energy module for remote field sites.

> **Motion-gated recording.** Instead of saving fixed 10-minute chunks, the Pi
> runs the BeeMonitor motion detector (MOG2 background subtraction) on a cheap
> low-resolution stream and only writes the seconds where something is moving —
> with a few seconds of pre-roll and post-roll around each event. This cuts the
> data volume by 10–100× so the snippets can be pushed over a **cellular link**
> rather than hand-carried on an SD card. The blob-size thresholds are
> **auto-calibrated from the bees in your own recordings** using YOLO (see
> [How Motion-Gated Recording Works](#how-motion-gated-recording-works)).

**Total System Cost: $595 USD**



## System Architecture

![alt text](hardware_architecture.png)


## Bill of Materials

### Video Recording Module

| Component | Specification | Est. Price (USD) | Supplier |
|-----------|---------------|------------------|----------|
| Raspberry Pi 4 Model B | 4GB RAM | $70 | [Amazon](https://www.amazon.com/gp/product/B07TC2BK1X/ref=ox_sc_act_title_21) |
| Witty Pi 4 | Power management & RTC | $40 | [Adafruit](https://www.adafruit.com/product/5704?gad_source=1&gclid=CjwKCAjw_LOwBhBFEiwAmSEQASJOcA2QVEtrBkJMBaF8xFwxfno8XAqbypi9hw6s3qwa3Ln1Njmb4hoCvVMQAvD_BwE) |
| Raspberry Pi HQ Camera | IMX477, 12.3MP | $75 | [Amazon](https://www.amazon.com/Raspberry-Pi-Camera-Sensitivity-Alternative/dp/B08LHJR3K4/ref=sr_1_3?crid=11BWYCYQTDCZW&dib=eyJ2IjoiMSJ9.QwyDKWYwZ_FkTH3vkvv00UFR0v1NSGQ-pLOf3Oo_8oEYvU79_8s7gFAT8hPF3Tdk3DgH9Z096msrGQCmM9Yedf1P2aUMkT19e1EH7eccBb-9dZSQ6FGcy6r4G7xXRJyBi2rEZ9HeLG5K7SaUGtNdFjXY8icSmy2Hbqm2W3EazCY_xTj3E0BA98ETOHHaYlney0e2VpfLulsqUNccs7pEUEpOAkyfKldocsFUHK08SFI.MYIJShfswLYUqeiVWdK6rJZj86uv0XxzZyB31rQBQOI&dib_tag=se&keywords=HD+camera+for+raspberry+pi&qid=1710100897&sprefix=hd+camera+for+raspberry+pi%2Caps%2C98&sr=8-3) |
| CS-Mount Lens | 6mm focal length | $25 | [Amazon](https://www.amazon.com/Arducam-Raspberry-CS-Mount-Adjustable-Aperture/dp/B088GWZPL1/ref=pd_bxgy_d_sccl_1/135-4369406-5324613?pd_rd_w=uapNQ&content-id=amzn1.sym.2b132e63-5dcd-4ba1-be9f-9e044543d59f&pf_rd_p=2b132e63-5dcd-4ba1-be9f-9e044543d59f&pf_rd_r=H4QG7YB4PB6QYMBH9S70&pd_rd_wg=ZLzQu&pd_rd_r=25aeacc9-5e85-4f91-810f-4e735d169f89&pd_rd_i=B088GWZPL1&psc=1) |
| MicroSD Card | 256GB | $30 | [Amazon](https://www.amazon.com/SanDisk-256GB-Extreme-microSD-Adapter/dp/B07FCR3316/ref=sr_1_18?crid=1JZQLL34SSAJZ&dib=eyJ2IjoiMSJ9.qg2wNyziPjEqStb07QLG45zx-FVFHBC8GME7mvX9HR3OiCIaSXAw07xwu08FEb_nNUAq5lbFbkd1zs_1S1XNJr72pKPaq5Xum029mgRs3YPolb8HZfLrmZfchea0KMfUFImpeKvg6KzsKS-RdR4QjDMjA2QoctnX8BzlWJcIfHTcAB9Qfr_WWwaYDRB-Jjodqi3DcoqSKcVF4Ag7yyEvEGSo64OxqsODW6sUTNES8rYH77rMK6OQW9QTx_fYTn-g2HrPGQ1olTtIW5zolFfA5aXfRysJOFH19iWHsYyu7jM.gvQojHrIjCQfOKQhtke7Bx0zBZsk-Js78Lfbd6GfWrU&dib_tag=se&keywords=sd%2Bcard&qid=1714412070&s=electronics&sprefix=SD%2Celectronics%2C92&sr=1-18&th=1) |
| DC-DC Buck Converter | 12V to 5V, 3A min | $10 | [Amazon](https://www.amazon.com/dp/B09DGDQ48H/ref=sspa_dk_detail_2?pd_rd_i=B09DGFR24W&pd_rd_w=T6wo9&content-id=amzn1.sym.386c274b-4bfe-4421-9052-a1a56db557ab&pf_rd_p=386c274b-4bfe-4421-9052-a1a56db557ab&pf_rd_r=Z5H018PQ379NKB85MGSM&pd_rd_wg=uPn0L&pd_rd_r=d4455b8a-da5d-47b0-a462-78c67b0ded54&s=electronics&sp_csd=d2lkZ2V0TmFtZT1zcF9kZXRhaWxfdGhlbWF0aWM&th=1) |
| Waterproof USB Connector | Panel mount USB-C | $10 | [Amazon](https://www.amazon.com/dp/B091TMHVSS/ref=sspa_dk_detail_1?pd_rd_i=B091TMHVSS&pd_rd_w=yF3Od&content-id=amzn1.sym.386c274b-4bfe-4421-9052-a1a56db557ab&pf_rd_p=386c274b-4bfe-4421-9052-a1a56db557ab&pf_rd_r=GAWNCGR78GAEGBN2P8AG&pd_rd_wg=fMwbk&pd_rd_r=c61b544e-fa6f-4fc0-a4f6-baa11e678a21&s=electronics&sp_csd=d2lkZ2V0TmFtZT1zcF9kZXRhaWxfdGhlbWF0aWM&th=1) |
| Male A to Male A | USB | $10 | [Amazon](https://www.amazon.com/Your-Cable-Store-USB-Cables/dp/B07BZ2M3WM/ref=sr_1_4?crid=34TPGV497YJZB&dib=eyJ2IjoiMSJ9.0uqtYuGtdSIKXM0N_5DeiR_RavNVA9MToj1VRBd6exHlvNqojLA2eTELLtyAZuCCQNbdwg4fdkCIk5C2KPmmhuE74yCbs6WrOJVRjE1YlryB92ksQt7PKXpEB2cWmqfRvBXFuGniyQpI54h8UIhqcRs2k2nVoWiRvo7eEet0HeBaqRl-IjYiq6Nd6BoCUqTdJbRo3gfdyhA-L2zmBA97uB4AmPm64U00WXhFG5l1V0uGqekhfIUDhoBHRcJb0HBEqyO8fJ2I1ffFywEUKzg0V4Pv290iyLOyee-mJqX3rHM.k0NOeN_5xxS_wkdmG4Wz1eVhe994sC9QT1OPo8KMhcU&dib_tag=se&keywords=usb%2Bto%2Busb%2Bshort&qid=1708566655&s=electronics&sprefix=usb%2Bto%2Busb%2Bshort%2Celectronics%2C112&sr=1-4&th=1) |
| 3D Printed Enclosure | PETG recommended | $50 | Self-print or service |
| Camera Tripod | 50-inch portable | $20 | [Amazon](https://www.amazon.com/dp/B00XI87KV8?ref_=cm_sw_r_apan_dp_KDSDE4XYJGSKXYBDA4A4&language=en-US&th=1) |
| Mounting Hardware | M2.5 standoffs, screws, cable glands | $10 | [Amazon](https://www.amazon.com/iUniker-Raspberry-Standoffs-Spacers-Standoff/dp/B0F6MNBQVL/ref=sr_1_1_sspa?dib=eyJ2IjoiMSJ9.R1Y_pSmTsEnF_05yeQt1b0Cosr-xaJQUdix8ximCQWt15Ups-IOyLmjXx8enOAQkz698By1tNK9ZqzE0YB3fp57vkrhey_2U-jLtZyz6e8vRmTNLPpb_hk7bsNcRwBRoHd5pcqROMpt9pk2OUFHUYAPeuT3Wp7sjASDGwtVTa11R8NfpazntKmgVModWjjgd0qkBlER9Ogrx5nDPTWq8ScDG4XfLq1ZAmpzOiH94ok4.Nyid2rA0qLSC6bsZwS6xu2o6kh6tvJSfb6KlWw8MKUo&dib_tag=se&keywords=M2.5+standoffs&qid=1769380197&sr=8-1-spons&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&psc=1) |

**Subtotal: $350 USD**

### Energy Module (Off-Grid)

| Component | Specification | Est. Price (USD) | Supplier |
|-----------|---------------|------------------|----------|
| Solar Panel | Renogy 100W 12V Solar Panel Starter Kit | $120 | [Amazon](https://www.amazon.com/Renogy-Monocrystalline-Negative-Controller-Connectors/dp/B00BFCNFRM/ref=dp_prsubs_d_sccl_1/144-8060600-4556963?pd_rd_w=LNQsf&content-id=amzn1.sym.8a163a7b-6a2a-45ae-8510-8d5419efb828&pf_rd_p=8a163a7b-6a2a-45ae-8510-8d5419efb828&pf_rd_r=QTQB0ZAXC4F2T5D6ECZV&pd_rd_wg=CGslW&pd_rd_r=dcc3fe90-f2d6-47d5-a551-b932c0671273&pd_rd_i=B00BFCNFRM&th=1) |
| Battery | 12V 30Ah LiFePO4 | $80 | [Amazon](https://www.amazon.com/dp/B09N9BBS68?ref=emc_p_m_5_i_atc&th=1) |
| Battery Enclosure | Waterproof toolbox | $30 | Hardware store |
| Wires | 16 Gauge Wire  | $15 | [Amazon](https://www.amazon.com/TYUMEN-Electrical-Extension-Flexible-Lighting/dp/B07SG23DT1/ref=sr_1_7?crid=10Q62QUB3LV1N&dib=eyJ2IjoiMSJ9.2iXpWNmEaXXfEOKOyz1qrFlHNJPZiPifVazRkDqYa9Vn6N5IUlzXqqmqfyOPPo4GFxNu9KianaH90wHJ-Htq-QmldRT9wn8FVIMzsPPA7WpXP46fbmlhNX7TGpLheil-lBsrJcNDWcbBWulf-FX0d-kzHTLT9Yp7JgryuVrfBUMgk9KMo661i7HSCBdA198hOtk6UyPNK58A3lz7U72_-iAve4rZCmMVNGJ2sqIChqJwjBf6BC8QEEQe5VOL4DiZPc6SUyAfira_gP_XTzAKnfn9FGJs8-jcbCplaZ8VPqY.PJfGj73KdDWQ8B7rrVf6q7b17V9BjJ2ocOpDrv-Grpw&dib_tag=se&keywords=12v%2Bcables&qid=1712062868&sprefix=12v%2Bcables%2Caps%2C69&sr=8-7&th=1) |

**Subtotal: $245 USD**

## 3D Printed Enclosure

### STL Files

Located in `/hardware/enclosure/`:

- `enclosure_body.stl` — Main housing (fits Pi + Witty Pi + converter)
- `enclosure_lid.stl` — Removable lid with camera mount hole
- `enclosure_tripod_connector.stl` — Adjustable bracket for HQ camera
- `power_cable_connector.stl` — Adapter for waterproof cable entry

### Print Settings

| Setting | Recommendation |
|---------|----------------|
| **Material** | PETG (UV resistant) or PLA |
| **Layer Height** | 0.2mm |
| **Infill** | 20% |
| **Walls** | 3 perimeters |
| **Supports** | Yes |
| **Bed Adhesion** | Brim recommended |

**Print Time:** ~8–12 hours total

### Weatherproofing

1. Apply silicone sealant around all cable entry points
2. Apply conformal coating to exposed PCB edges (optional)
3. Ensure cable glands are properly tightened

## Assembly Instructions

### Electronics Assembly

1. **Install Raspberry Pi** to enclosure 
2. **Stack Witty Pi 4** on GPIO header (align carefully)
3. **Connect HQ Camera** ribbon cable to CSI port
4. **Mount camera** in lid bracket
5. **Install DC-DC converter** in the enclosure 
6. **Wire DC-DC output** (5V) to Witty Pi power input
7. **Route power cable** through cable gland (ensure weatherproof)
8. **Test fit lid** (don't seal yet until software is configured)

### Energy Module Assembly

**Connection Sequence (IMPORTANT):**
1. Connect battery to controller FIRST (BAT terminals)
2. Connect solar panel to controller (PV terminals)
3. Connect load output to recording module DC-DC input

**Warning:** Always connect battery BEFORE solar panel to prevent controller damage.

## Software Installation

The field system runs **three independent systemd services** (kept separate so a
failure in one never stops the others — a network outage can't stop recording,
and a recorder crash can't stop uploads):

| Unit | Script | Job |
|------|--------|-----|
| `beemonitor-recorder.service` | `hardware/main_motion.py` | Records **only activity snippets** (MOG2 motion gate); captures stills **on demand** (picture / live view) |
| `beemonitor-telemetry.service` | `hardware/telemetry.py` | JSON health beat every 60s to the cloud, **over cellular** (no image) |
| `beemonitor-uploader.service` | `hardware/uploader.py` | Streams snippets to S3 — **WiFi-gated** (video waits for WiFi) |
| `beemonitor-calibrate.timer` | `hardware/main_motion.py --calibrate` | Daily: learns the bee blob-size window from recorded snippets with YOLO |

All read configuration from `/etc/beemonitor/uploader.env`.

> **Split transport (cost control).** A tiny JSON telemetry beat goes over **cellular**
> hourly (tiny — see the dashboard to know the unit is alive); bulk **video is
> WiFi-gated** and held on disk until WiFi is available. See
> [How Motion-Gated Recording Works](#how-motion-gated-recording-works) and
> [Device monitoring](#device-monitoring--telemetry).

### Quick Install (cellular field unit)

For a fast (re)deploy on a Pi that already has the camera focused and WittyPi
set up. Run the blocks below top-to-bottom; the only manual steps are editing the
env file and bringing up the modem. The detailed walkthrough follows in Steps 1–7.

```bash
# 1. Code
cd ~ && git clone https://github.com/eai6/BeeMonitor.git
cd ~/BeeMonitor/hardware

# 2. System deps + virtualenv (picamera2/cv2 from apt; pip deps in the venv).
#    --system-site-packages lets the venv import the apt-installed picamera2/cv2.
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-picamera2 python3-opencv ffmpeg python3-venv libqmi-utils
python3 -m venv --system-site-packages ~/BeeMonitor/hardware/venv
~/BeeMonitor/hardware/venv/bin/pip install --upgrade pip
~/BeeMonitor/hardware/venv/bin/pip install requests ultralytics   # ultralytics = calibration only

# 3. Output directories
~/BeeMonitor/hardware/venv/bin/python makeDirectories.py

# 4. Credentials + tuning  (EDIT: paste your bmk_device_ key, then save)
sudo mkdir -p /etc/beemonitor
sudo tee /etc/beemonitor/uploader.env >/dev/null <<'EOF'
BEEMONITOR_API_BASE=https://mqnafc3ejc.us-east-1.awsapprunner.com
BEEMONITOR_DEVICE_KEY=bmk_device_REPLACE_ME
BEEMONITOR_RECORD_DIR=/home/beemonitor/Desktop/cameraOutput/beeHotel
EOF
sudo nano /etc/beemonitor/uploader.env      # <-- paste the real device key
sudo chmod 600 /etc/beemonitor/uploader.env

# 5. Install + enable the three services
sudo cp systemd/beemonitor-recorder.service  /etc/systemd/system/
sudo cp systemd/beemonitor-telemetry.service /etc/systemd/system/
sudo cp systemd/beemonitor-uploader.service  /etc/systemd/system/
sudo cp systemd/beemonitor-calibrate.service /etc/systemd/system/
sudo cp systemd/beemonitor-calibrate.timer   /etc/systemd/system/

# 5b. Let the telemetry service control WiFi from the dashboard (on/off/connect).
#     The telemetry service runs as 'beemonitor'; nmcli state changes need root,
#     so grant passwordless nmcli to that user only.
echo 'beemonitor ALL=(root) NOPASSWD: /usr/bin/nmcli' | sudo tee /etc/sudoers.d/beemonitor-nmcli >/dev/null
sudo chmod 440 /etc/sudoers.d/beemonitor-nmcli
sudo visudo -cf /etc/sudoers.d/beemonitor-nmcli   # syntax-check the drop-in

sudo systemctl daemon-reload
sudo systemctl enable --now beemonitor-recorder.service beemonitor-telemetry.service beemonitor-uploader.service
sudo systemctl enable --now beemonitor-calibrate.timer

# 6. Bring up cellular (see Step 10 for your exact Sixfab kit), then verify:
ping -c 3 8.8.8.8
journalctl -u beemonitor-uploader.service -f   # watch snippets upload to S3
```

> First boot runs on permissive motion thresholds until the calibrate timer
> finds bees in the recorded snippets. To calibrate immediately once a few clips
> exist: `~/BeeMonitor/hardware/venv/bin/python ~/BeeMonitor/hardware/main_motion.py --calibrate --force`

### Step 1: Download Source Code

```bash
cd ~
git clone https://github.com/eai6/BeeMonitor.git
cd BeeMonitor/hardware
```

> The systemd unit files expect the repo at **`/home/beemonitor/BeeMonitor`**. If you
> clone elsewhere, edit the `ExecStart=` paths in `hardware/systemd/*.service`.

### Step 2: Install Dependencies (system packages + virtualenv)

`picamera2` and OpenCV must come from **apt** (the camera stack is built against
the system libraries and doesn't install cleanly via pip). Everything pip-based
(`requests`, `ultralytics`) goes in a **virtualenv created with
`--system-site-packages`** so it can still import the apt-installed
`picamera2`/`cv2`. On Raspberry Pi OS Bookworm a plain `pip install` is blocked
(PEP 668 "externally managed"), so the venv is required, not optional.

```bash
sudo apt update && sudo apt upgrade -y

# System packages: camera, computer vision, video muxing, venv, cellular tools.
sudo apt install -y python3-picamera2 python3-opencv ffmpeg \
                    python3-venv libqmi-utils

# Create the project virtualenv (must see the apt-installed picamera2/cv2).
python3 -m venv --system-site-packages ~/BeeMonitor/hardware/venv

# Install the pip-only deps into the venv.
~/BeeMonitor/hardware/venv/bin/pip install --upgrade pip
~/BeeMonitor/hardware/venv/bin/pip install requests          # uploader + telemetry
# Calibration ONLY (heavy — pulls in PyTorch). Needed by the calibrate timer.
# Skip if you'll calibrate on another machine instead.
~/BeeMonitor/hardware/venv/bin/pip install ultralytics
```

> The systemd units run `/home/beemonitor/BeeMonitor/hardware/venv/bin/python`. Keep the venv
> at that path (or edit the `ExecStart=` lines in `hardware/systemd/*.service`).
> Sanity check: `~/BeeMonitor/hardware/venv/bin/python -c "import picamera2, cv2, requests; print('ok')"`.

### Step 3: Create Output Directories

```bash
~/BeeMonitor/hardware/venv/bin/python makeDirectories.py
```

(The recorder also auto-creates its working directories on first run.)

### Step 4: Focus the Camera

```bash
~/BeeMonitor/hardware/venv/bin/python runFocus.py
```

**Note:** The camera must be connected when the Pi boots. If it wasn't, reboot:

```bash
sudo reboot
```

### Step 5: Configure Credentials and Tuning

Issue a **device key** from the BeeMonitor web app (Devices page → "Issue device
key" — it starts with `bmk_device_`), then create the shared env file:

```bash
sudo mkdir -p /etc/beemonitor
sudo nano /etc/beemonitor/uploader.env
```

Paste, replacing the device key with yours:

```ini
BEEMONITOR_API_BASE=https://mqnafc3ejc.us-east-1.awsapprunner.com
BEEMONITOR_DEVICE_KEY=bmk_device_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
BEEMONITOR_RECORD_DIR=/home/beemonitor/Desktop/cameraOutput/beeHotel

# --- models: NONE of these are needed ---
# The recorder uses the BeeMonitor weights committed in the repo's models/
# automatically (nest_detection.pt for the hotel ROI, bee_tracking.pt for
# calibration). Only set these to override the auto-resolved repo paths:
# BEEMONITOR_MODELS_DIR=/home/beemonitor/BeeMonitor/models
# BEEMONITOR_HOTEL_ROI_DETECT=true        # set false to record on the whole frame

# --- optional tuning (sensible defaults if omitted; see Configuration Reference) ---
# BEEMONITOR_PRE_ROLL=3
# BEEMONITOR_POST_ROLL=4
```

The file holds the device key, so lock it down:

```bash
sudo chmod 600 /etc/beemonitor/uploader.env
```

### Step 6: Install the Services

Copy the unit files shipped in the repo and reload systemd:

```bash
cd ~/BeeMonitor/hardware
sudo cp systemd/beemonitor-recorder.service  /etc/systemd/system/
sudo cp systemd/beemonitor-telemetry.service /etc/systemd/system/
sudo cp systemd/beemonitor-uploader.service  /etc/systemd/system/
sudo cp systemd/beemonitor-calibrate.service /etc/systemd/system/
sudo cp systemd/beemonitor-calibrate.timer   /etc/systemd/system/
sudo systemctl daemon-reload
```

### Step 7: Start and Enable on Boot

```bash
# Record + telemetry + upload (start now, and on every boot)
sudo systemctl enable --now beemonitor-recorder.service
sudo systemctl enable --now beemonitor-telemetry.service
sudo systemctl enable --now beemonitor-uploader.service

# Daily auto-calibration
sudo systemctl enable --now beemonitor-calibrate.timer
```

Verify and watch logs:

```bash
systemctl status beemonitor-recorder.service
journalctl -u beemonitor-recorder.service -f     # clip START / STOP lines
journalctl -u beemonitor-uploader.service -f     # upload progress
```

> **First run:** until the recorder has captured a few activity clips, the
> calibrate job has nothing to learn from and the recorder runs on *permissive
> defaults* (it over-triggers slightly — safe, just less selective). Once the
> daily calibrate job finds bees in the snippets it writes `calibration.json`,
> and the recorder **hot-reloads** it within a few minutes (no restart needed).
> You can force the first calibration once clips exist:
> `~/BeeMonitor/hardware/venv/bin/python main_motion.py --calibrate --force`

---

## How Motion-Gated Recording Works

```
                 ┌─────────────────── Raspberry Pi ───────────────────┐
 camera ─┬─ main (1080p) ─► H.264 encoder ─► ring buffer ─► snippet.mp4 ─┐
         │                                        ▲                      │
         └─ lores (640x480) ─► MOG2 motion gate ──┘                      │
                                                                         ▼
                            calibration.json ◄── calibrate (YOLO,    uploader ─► S3
                                  ▲                daily, from        (API, cellular)
                                  └── learns bee blob sizes ── snippets on disk)
```

1. **Recorder** (`main_motion.py`) runs two camera streams at once: a full-res
   stream that is always being H.264-encoded into a short **ring buffer**, and a
   small `lores` stream fed to the MOG2 motion detector. When motion appears it
   flushes the ring buffer (capturing **pre-roll**) and keeps writing until
   motion stops plus a **post-roll**, then remuxes the clip to `.mp4` under
   `RECORD_DIR/YYYY-MM-DD/`. 
2. **Calibrate timer** (`main_motion.py --calibrate`) runs once a day. It does
   *not* need the camera or a live bee — it reads snippets the recorder already
   saved, runs YOLO over them, measures the motion-blob size of every confirmed
   bee, and writes the 5th/95th-percentile size window to `calibration.json`.
   This is the on-device version of cloud BeeMonitor's threshold learning.
3. **Recorder hot-reloads** `calibration.json` and tightens its motion gate to
   bee-sized blobs — no restart.
4. **Uploader** (`uploader.py`) ships each finished `.mp4` to S3 through the
   authenticated API and marks it with a `.uploaded` sidecar so it is never sent
   twice.

YOLO runs **only** during the daily calibration pass (a Pi 4 can't run it in
real time); the recording hot path is pure MOG2 and stays light.

## Configuration Reference

All knobs are environment variables in `/etc/beemonitor/uploader.env` (read by
all three services). Only `API_BASE`, `DEVICE_KEY`, and `RECORD_DIR` are
required; everything else has a sensible default.

| Variable | Default | Purpose |
|----------|---------|---------|
| `BEEMONITOR_API_BASE` | — | API base URL for uploads |
| `BEEMONITOR_DEVICE_KEY` | — | `bmk_device_…` key from the web app |
| `BEEMONITOR_RECORD_DIR` | `…/cameraOutput/beeHotel` | Where snippets are written / uploaded from |
| `BEEMONITOR_PRE_ROLL` | `3` | Seconds kept *before* motion starts |
| `BEEMONITOR_POST_ROLL` | `4` | Seconds kept *after* motion stops |
| `BEEMONITOR_MAX_SEGMENT` | `120` | Force-rotate a clip after this many seconds |
| `BEEMONITOR_LORES_W` / `_H` | `640` / `480` | Resolution of the detection stream |
| `BEEMONITOR_FPS` | `25` | Capture frame rate |
| `BEEMONITOR_ROI` | (auto-detect) | Manual override `x1,y1,x2,y2` in lores px. Empty ⇒ auto-detect the hotel at startup (below) |
| `BEEMONITOR_MODELS_DIR` | `<repo>/models` | Where the BeeMonitor weights live. Auto-resolves to the repo's `models/` — no need to set |
| `BEEMONITOR_NEST_MODEL` | `<models>/nest_detection.pt` | Hotel/nest detector used to set the recording ROI (class 0 = hotel, 1 = nest hole) |
| `BEEMONITOR_HOTEL_ROI_DETECT` | `true` | Detect the hotel before recording and confine detection to it. `false` ⇒ whole frame |
| `BEEMONITOR_NEST_CONF` | `0.25` | Confidence threshold for the hotel/nest detector |
| `BEEMONITOR_HOTEL_PAD_X` / `_Y` | `100` / `50` | Padding (base px @ 1920×1080, scaled) around the detected hotel |
| `BEEMONITOR_BG_RESET_INTERVAL` | `600` | Rebuild the MOG2 background model this often (s) so the gate tracks sun/shadow drift in real time. Also fresh at startup. `0` = off |
| `BEEMONITOR_WIFI_IFACE` | `wlan0` | WiFi interface used by the dashboard's WiFi on/off/connect controls |
| `BEEMONITOR_YOLO_MODEL` | `<models>/bee_tracking.pt` | BeeMonitor's bee/wasp detector, used by the calibrate job |
| `BEEMONITOR_CALIB_MAX_AGE_DAYS` | `7` | Skip recalibration if `calibration.json` is younger than this |
| `BEEMONITOR_POLL_SECONDS` | `30` | How often the uploader scans for new snippets |
| **`BEEMONITOR_TELEMETRY_INTERVAL`** | `60` | Telemetry beat cadence (s). JSON-only, so it's cheap — 60s gives ~1-minute offline detection |
| **`BEEMONITOR_ACTIVITY_PERIOD`** | `3600` | Trailing window for the snippets/period activity proxy (s), decoupled from the beat |
| `BEEMONITOR_TELEMETRY_IMAGE_INTERVAL` | `0` | Periodic still capture (s). `0` = off; stills are on-demand only |
| `BEEMONITOR_TELEMETRY_IMAGE_HEIGHT` | `720` | Downscale height for on-demand stills |
| `BEEMONITOR_SCHEDULE_WINDOW` | (none) | WittyPi on/off window string, shown on the dashboard |
| `BEEMONITOR_WIFI_ONLY_VIDEO` | `true` | Hold video off cellular — upload only when WiFi is up |

Tuning is rarely needed — start with defaults and adjust pre/post-roll or `ROI`
only if you see clips clipped short or too much background motion triggering.

> **Detection mirrors cloud BeeMonitor.** The recorder uses the same committed
> weights in `models/` (no `yolo11n` anymore), resolved automatically:
>
> 1. **Hotel detection (once, at startup).** Before recording, the recorder runs
>    `nest_detection.pt` on one settled frame to find the **hotel** (class 0; or
>    the bounding box of the detected nest holes, class 1), pads it, and uses that
>    as the detection **ROI** — exactly like the cloud pipeline confines bee
>    detection to the hotel. If detection fails (no bees/model/ultralytics, empty
>    result), it **falls back to the whole frame** and keeps recording.
> 2. **Motion gate (live).** MOG2 runs inside that ROI to gate snippet recording.
> 3. **Calibration (scheduled, offline).** `bee_tracking.pt` — BeeMonitor's own
>    bee/wasp detector — measures real bee blob sizes in saved snippets to tune
>    the MOG2 area window. Every detection is a bee, so no COCO class filter.
>
> The models ship in the repo, so a `git pull` on the Pi is all that's needed —
> the recorder picks them up from `<repo>/models/` with no env to set.

> **Telemetry is JSON-only and cheap**, so the 60s beat (fast offline detection)
> costs almost nothing on cellular. The **activity window** is separate
> (`BEEMONITOR_ACTIVITY_PERIOD`, 1h) so the dashboard's snippets/period is
> meaningful even with a 60s beat. Watch `journalctl -u beemonitor-telemetry -f`.

---

## Device monitoring & telemetry

Because video is too expensive to push over cellular, each unit sends a small
**JSON health beat every 60s** over cellular — no image, so it's tiny and gives
~1-minute offline detection. Bulk video stays WiFi-gated; images are **on demand
only** (picture / live view).

**What a beat carries** (`hardware/telemetry.py`): storage %, uptime, CPU temp,
service health (recorder / uploader / cellular), cellular signal, GPS (when the
modem has a fix), the WittyPi schedule window, and **`snippets recorded this
period`** — since a snippet only exists when motion fired, that count is a direct
**activity proxy**. The activity window is `BEEMONITOR_ACTIVITY_PERIOD` (1h),
decoupled from the 60s beat so it stays meaningful.

**Where it shows up:** the web app's **Devices** page lists each unit with an
Online/Offline badge, storage %, and activity; clicking a device opens a
**dashboard** — health cards (incl. Activity), GPS/map, an activity-over-time
graph, on-demand picture/live-view, and the videos that device has uploaded.
On-demand images are pulled straight from S3 via short-lived signed URLs.

**Online/offline** is derived, not stored: a unit shows Offline if no beat has
arrived within `DEVICE_ONLINE_GRACE_SECONDS` (default 180 = ~3 missed beats). The
device reports its schedule window so "off as planned" is distinguishable from
"died".

---

## Step 8: Set Up WittyPi

### 8.1 Download and Install WittyPi
```bash
cd ~/Desktop
wget http://uugear.com/repo/WittyPi4/install.sh
sudo sh install.sh
```

### 8.2 Recording Launches Automatically

You do **not** need to launch recording from WittyPi's `afterStartup.sh` — the
`beemonitor-recorder.service` you enabled in Step 7 starts automatically on every
boot (and the uploader and calibrate timer with it). WittyPi only manages the
**power on/off schedule** below. If you previously added a `driver.py` line to
`afterStartup.sh`, remove it to avoid a second recorder process competing for the
camera.

### 8.3 Enable I2C
```bash
sudo raspi-config
```
Navigate to: **Interface Options → I5 I2C → YES → OK**

### 8.4 Auto power-on when power is applied (no button press)

So a field unit boots itself whenever power appears (solar/battery cycling,
WittyPi wakes) — you never touch the button:

```bash
cd ~/wittypi          # or ~/Desktop/wittypi, wherever it's installed
sudo ./wittyPi.sh
```
1. Choose **"Other settings…"** (the configuration submenu, ~option **9**).
2. Find **"Default state when powered"** (default ON/OFF) and set it to **ON**
   (default-on / `1`).
3. Quit to save.

Now: power applied → WittyPi powers the Pi → it boots → systemd starts
`cellular`, `beemonitor-recorder`, `beemonitor-telemetry`, `beemonitor-uploader`
automatically. Fully hands-off. (Menu wording/number varies slightly by WittyPi
firmware; it's under the advanced/other-settings submenu as the default state.)
This works alongside the scheduled on/off window below.

### 8.5 Test WittyPi Through 1 Cycle

Open terminal:
```bash
cd ~/Desktop/wittypi/
sudo ./wittyPi.sh
```

**Configuration steps:**

1. Select **(1)** to write system time to RTC on the WittyPi
   - (Assumes the time on the Raspberry Pi is accurate and the time on the RTC is not)
   - If it is the other way around, select **(2)** instead

2. **Schedule next startup (4):** Set to something relatively soon, like `?? ??:01:00` (one minute past the next hour — change the minute to something reasonable)

3. **Schedule next shutdown (5):** Set to 60 seconds before the startup, like `?? ??:00:00` (at the next hour)

4. Wait for Pi to shutdown and restart 60 seconds later to verify the cycle works

### 8.6 Create Production Schedule

Create the scheduler file:
```bash
sudo nano /home/beemonitor/Desktop/wittypi/schedules/beeHotelScheduler_2024.wpi
```

Paste the following:
```
BEGIN 2024-03-00 07:50:00
END   2024-09-01 00:00:00
ON    H10 M15 # will start recording from 7:50am to 6:05pm
OFF   H13 M45 # will be off until the next day
```

### 8.7 Apply the Schedule

```bash
cd ~/Desktop/wittypi/
sudo ./wittyPi.sh
```

1. Choose **schedule script (6)**
2. Pick the `beeHotelScheduler_2024.wpi` script
3. Verify that the next power on/off times make sense

## Step 9: Set Up Raspberry Pi Connect (Remote Access)

[Raspberry Pi Connect](https://www.raspberrypi.com/software/connect/) gives you
remote shell and screen access through a browser from anywhere — no VPN, port
forwarding, or static IP required, which is ideal once the unit is on a cellular
link in the field.

### 9.1 Install Raspberry Pi Connect
```bash
sudo apt update
sudo apt install -y rpi-connect
```

### 9.2 Enable for the current user
```bash
# Headless / Lite (shell access only):
rpi-connect on

# Or enable the service to start on boot:
systemctl --user enable --now rpi-connect
loginctl enable-linger beemonitor      # keep the user service alive without a login
```

### 9.3 Sign in and link the device
```bash
rpi-connect signin
```
Open the printed URL, sign in with your Raspberry Pi ID, and the device appears
at **<https://connect.raspberrypi.com>**. Check status any time with:
```bash
rpi-connect status
```

> Remote **screen** sharing needs Raspberry Pi OS Desktop; on Lite you get
> remote **shell** access, which is all that's needed to manage the services.

## Step 10: Cellular Connectivity (Sixfab 4G-LTE)

This is what keeps a remote, off-grid unit reachable. On cellular it carries the
**60s JSON telemetry beat** (`beemonitor-telemetry.service`) so you can
monitor the unit; bulk **video is WiFi-gated** and held on disk until WiFi
appears (see [Device monitoring & telemetry](#device-monitoring--telemetry)).

> **The transport is already built.** Both the telemetry beat and (over WiFi)
> the video upload talk to AWS through the BeeMonitor API — network-agnostic,
> so once the modem gives the Pi internet, telemetry "just works" with **no
> software changes**. The split keeps the cellular bill tiny.

We use **QMI mode** (libqmi) — the modem presents a `wwan0` interface that we
bring up with `qmi-network` + `udhcpc`, and a small systemd service
(`cellular.service`) re-establishes it on every boot/WittyPi wake and watches it
for drops. The BeeMonitor software needs **no changes** — once `wwan0` has a
route, the uploader uses it.

> **Why a service is non-negotiable here.** `qmi-network start` / `udhcpc` create
> *runtime* state that does **not** survive a reboot, and WittyPi power-cycles the
> Pi on every wake. Without `cellular.service`, the Pi wakes up with no internet.

> **Heads-up — two default routes.** When WiFi is also on (e.g. at your desk),
> both WiFi and `wwan0` have default routes and traffic may exit either one. To
> *prove* an upload actually went over cellular, turn WiFi off first
> (`nmcli radio wifi off` or `sudo rfkill block wifi`). In the field there is no
> WiFi, so cellular is the only route.

Reference: [Sixfab — Hardware Assemble](https://docs.sixfab.com/docs/raspberry-pi-4g-lte-cellular-modem-kit-getting-started-with-ecm-mode)
and [Setting up a data connection over QMI](https://docs.sixfab.com/page/setting-up-a-data-connection-over-qmi-interface-using-libqmi).

### 10.1 Insert the SIM and Assemble the HAT

1. Attach the mini PCIe module to the HAT and insert an **activated** micro SIM.
2. Connect the antennas — the main LTE antenna to the **MAIN** port (and the
   diversity/GNSS antenna to its matching port).
3. Stack the HAT on the Pi's GPIO header (mind clearance with the WittyPi) and
   connect the modem's **USB** jumper to the Pi.
4. Power the Pi and confirm the modem enumerates (you should see the Telit
   modem — USB `1bc7:1201`, LE910C4-NF — and a `/dev/cdc-wdm0` control node):
   ```bash
   lsusb
   ls /dev/cdc-wdm0
   ```

### 10.2 Install Tools and Disable ModemManager

```bash
sudo apt install -y libqmi-utils udhcpc
# ModemManager fights manual QMI control — turn it off:
sudo systemctl disable --now ModemManager.service
```

### 10.3 Set the APN

`qmi-network` reads the APN from `/etc/qmi-network.conf`. Install the sample and
edit it for your SIM:

```bash
sudo cp ~/BeeMonitor/hardware/cellular/qmi-network.conf.sample /etc/qmi-network.conf
sudo nano /etc/qmi-network.conf     # set APN= (e.g. super / hologram / soracom.io)
```

> If the modem isn't already in QMI/RmNet mode you'll set it once via AT command,
> then power-cycle. This is a **Telit** modem, so use `AT#USBCFG=<n>` (query the
> available compositions with `AT#USBCFG=?`); on a Quectel it would be
> `AT+QCFG="usbnet",0`. Most Sixfab kits ship in QMI mode already — check with
> `lsusb` showing the Telit modem (`1bc7:1201`) and `/dev/cdc-wdm0` present.

### 10.4 Pin DNS (immutable resolv.conf)

QMI/`udhcpc` won't reliably set DNS, so set it once and lock it so nothing
overwrites it:

```bash
printf 'nameserver 8.8.8.8\nnameserver 1.1.1.1\n' | sudo tee /etc/resolv.conf
sudo chattr +i /etc/resolv.conf      # immutable — survives reboots
```

> This makes `/etc/resolv.conf` the **single source of truth** for DNS. The
> bring-up script deliberately does not touch DNS. To change nameservers later,
> unlock first: `sudo chattr -i /etc/resolv.conf`.

### 10.5 Bring It Up Manually (verify before automating)

```bash
sudo ~/BeeMonitor/hardware/cellular/cellular-up.sh &
sleep 20
ip addr show wwan0            # should have an IP
ping -c 3 -I wwan0 8.8.8.8    # connectivity over cellular
ping -c 3 google.com         # DNS + connectivity
kill %1                       # stop the manual run before installing the service
```

✅ When `google.com` resolves and pings with 0% loss (WiFi off), the link works —
now make it automatic.

### 10.6 Install the Cellular Service (survives reboot + WittyPi wakes)

The repo ships the bring-up script and a keep-alive systemd unit. The script
brings `wwan0` up with retry, then watchdogs it and re-establishes if the carrier
drops the session:

```bash
cd ~/BeeMonitor/hardware
chmod +x cellular/cellular-up.sh          # the unit runs it in place from the repo
sudo cp systemd/cellular.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now cellular.service
systemctl status cellular.service
journalctl -u cellular.service -f      # "link up on wwan0"
```

`cellular.service` is ordered **before** `beemonitor-uploader.service`, so on
every boot the link comes up before uploads are attempted (and the uploader's
backoff covers any remaining lag).

> **Note:** the unit is `WantedBy=multi-user.target` and deliberately has **no**
> `After=multi-user.target` — adding that creates a dependency cycle that makes
> systemd silently delete the start job (the service shows `enabled` but never
> runs, with an empty journal). If you ever see that symptom, this is the cause.

### 10.7 Gate cellular to BeeMonitor only (don't let anything else eat your SIM)

When WiFi is down, `wwan0` becomes the default route and **any** process — `apt`
/ `unattended-upgrades`, `snapd`, `rpi-connect`, OS background jobs — will push
bytes over your metered SIM. The repo ships an nftables egress allowlist that
permits **only** the telemetry beat on cellular and drops everything else.
Allowed on `wwan0`: the `beemonitor-telemetry.service` cgroup (heartbeat JSON +
on-demand image + command poll), DNS, DHCP renew, NTP, and ICMP (the watchdog
ping). WiFi (`wlan0`) is untouched, so normal operation is unaffected. The video
uploader is intentionally *not* allowed on cellular — bulk video stays WiFi-only
even at the network layer.

```bash
cd ~/BeeMonitor/hardware
sudo apt install -y nftables
chmod +x cellular/cellular-firewall.sh
sudo cp systemd/cellular-firewall.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now cellular-firewall.service
```

Verify (with WiFi **off** so cellular is the only link):

```bash
sudo nft list table inet beemon_cell        # shows the rules + a "drop" counter
journalctl -u beemonitor-telemetry -f       # beats still succeed
sudo apt update                              # should hang/fail — blocked, as intended
sudo nft list table inet beemon_cell        # the drop counter has climbed
```

> Optional belt-and-suspenders: stop the worst offenders from even trying, so the
> firewall has less to drop:
> ```bash
> sudo systemctl mask apt-daily.timer apt-daily-upgrade.timer
> sudo systemctl disable --now unattended-upgrades 2>/dev/null || true
> ```
> `rpi-connect` keeps working over WiFi; it's just blocked over cellular. To allow
> remote access over cellular too, add its cgroup to `cellular-firewall.sh`.

**Enable network time (NTP) — required for uploads.** The Pi has no RTC, so on
every cold boot / WittyPi wake it starts with a stale clock restored from
`fake-hwclock`. A wrong clock fails TLS certificate validation, so S3 uploads and
telemetry will error until the time is corrected. `systemd-timesyncd` fixes this
automatically — but only once a network exists, which on a field unit means
*after* `cellular.service` is up. Make sure it's enabled (idempotent; on
Raspberry Pi OS it usually already is):

```bash
sudo timedatectl set-ntp true
timedatectl       # "NTP service: active"; "System clock synchronized: yes" once cellular is up
```

> Time sync can't happen until the modem has a route, so right after boot
> `timedatectl` may briefly show `synchronized: no` — it corrects within seconds
> of `cellular: link up on wwan0`. The bring-up script also kicks a resync at
> that moment, so the clock is right before the uploader makes its first request.

### 10.7 Confirm It Survives a Reboot

This is the test that proves field-readiness — no manual commands after boot:

```bash
sudo reboot
# after it returns, with WiFi off to prove traffic goes over cellular:
nmcli radio wifi off 2>/dev/null || sudo rfkill block wifi
ip addr show wwan0           # has an IP, hands-free
ping -c 4 google.com         # resolves + 0% loss
journalctl -u cellular.service -b   # what the service did this boot
```

### 10.8 Confirm Telemetry Flows (over cellular)

With the modem up, the telemetry service sends its 60s JSON beat over
cellular (this is what flows on cellular — video waits for WiFi):

```bash
journalctl -u beemonitor-telemetry.service -f
# look for: "beat: storage=… snippets/…=…" then "heartbeat ok: {...}"
```

The unit should now show **Online** on the web app's Devices page within a
minute (the beat is JSON-only — no image — so 60s is cheap on cellular). Stills
appear only when you request one on-demand (picture / live view).

### 10.9 Keep Cellular Data Under Control

- **Motion-gating** is the main lever — only activity snippets are sent.
- Tune `BEEMONITOR_PRE_ROLL` / `BEEMONITOR_POST_ROLL` down to shrink clips.
- Restrict detection to the hotel face with `BEEMONITOR_ROI` to avoid triggering
  on background motion (waving plants, passers-by).
- The uploader retries with backoff, so brief signal drops self-heal — clips
  accumulate on disk and upload when signal returns.

## Testing & Verification

Run these stages **in order** before relying on the unit in the field. Each one
is independently verifiable, so a failure points at a single component instead of
the whole chain. Do the early stages on the bench over WiFi; only the last
stages need cellular.

> Most-likely first-run issues (all easy to fix once isolated): motion not
> triggering at 640×480 (Stage 1), YOLO finding 0 bees during calibration —
> usually model path or color order (Stage 3), and the AT port not being
> `/dev/ttyUSB2` (Stage 6).

> Commands below run the scripts with the **project venv** Python
> (`~/BeeMonitor/hardware/venv/bin/python`). You can instead
> `source ~/BeeMonitor/hardware/venv/bin/activate` once and then just use `python`.

### Stage 0 — Code sanity (anywhere)
```bash
cd ~/BeeMonitor/hardware
~/BeeMonitor/hardware/venv/bin/python -m py_compile main_motion.py && echo OK
```

### Stage 1 — Recorder bench test (on the Pi, foreground)

Run it by hand first so you see the logs live. Use a throwaway directory and a
short warmup so the gate starts looking for motion quickly:

```bash
# Make sure the service isn't already holding the camera:
sudo systemctl stop beemonitor-recorder.service 2>/dev/null

BEEMONITOR_RECORD_DIR=/tmp/bm_test \
BEEMONITOR_WARMUP=3 \
~/BeeMonitor/hardware/venv/bin/python main_motion.py
```

**Wave your hand** in front of the lens. You should see:
```
recorder up: main=1920x1080 lores=640x480 @ 25fps ...
clip START (motion) -> 2026-06-04_14_03_11.mp4
clip STOP (idle) len=6.2s -> remux 2026-06-04_14_03_11.mp4
snippet ready: 2026-06-04_14_03_11.mp4 (1.4 MB)
```
Stop with **Ctrl-C**. ✅ **Pass:** snippets exist — `ls -R /tmp/bm_test`. If no
clips appear when you wave, the detection thresholds need tuning (see
[Configuration Reference](#configuration-reference)).

### Stage 2 — Inspect snippet content
```bash
ffprobe -v error -show_entries format=duration -of csv=p=0 /tmp/bm_test/*/*.mp4
```
✅ **Pass:** the clip plays and its length ≈ pre-roll (3s) + motion + post-roll
(4s), and the pre-roll shows the moment *before* your hand entered frame.

### Stage 3 — Calibration on existing footage

No live bees needed — point it at an **old 1080p clip that contains bees**:

```bash
BEEMONITOR_RECORD_DIR=/tmp/bm_test \
~/BeeMonitor/hardware/venv/bin/python main_motion.py --calibrate-from /path/to/old_bee_clip.mp4 \
  --model ~/BeeMonitor/models/your_bee.pt
cat /tmp/calibration.json
```
✅ **Pass:** the log reports `bee-frames` > 0 and it writes `min_area`/`max_area`.
❌ If `bee-frames` is 0, fix the model path / detection before going further —
calibration depends on it. (`calibration.json` defaults to one level above
`RECORD_DIR`, i.e. `/tmp/calibration.json` here.)

### Stage 4 — Hot-reload

With the recorder running, write a calibration from another shell and confirm it
is picked up without a restart (reload interval lowered for the test):

```bash
# Terminal A — recorder:
BEEMONITOR_RECORD_DIR=/tmp/bm_test BEEMONITOR_CALIB_RELOAD_SECONDS=30 ~/BeeMonitor/hardware/venv/bin/python main_motion.py

# Terminal B — force a fresh calibration:
BEEMONITOR_RECORD_DIR=/tmp/bm_test ~/BeeMonitor/hardware/venv/bin/python main_motion.py \
  --calibrate-from /path/to/old_bee_clip.mp4 --model ~/BeeMonitor/models/your_bee.pt --force
```
✅ **Pass:** within ~30s Terminal A logs `reloaded calibration: area=[...]`.

### Stage 5 — Uploader (device key + any internet)

Test over WiFi first to isolate upload logic from cellular:

```bash
sudo BEEMONITOR_API_BASE=https://mqnafc3ejc.us-east-1.awsapprunner.com \
     BEEMONITOR_DEVICE_KEY=bmk_device_yourkey \
     BEEMONITOR_RECORD_DIR=/tmp/bm_test \
     ~/BeeMonitor/hardware/venv/bin/python uploader.py
```
✅ **Pass:** `uploading <file>` → `uploaded video_id=…`, a `.uploaded` sidecar
appears next to the mp4, and the clip shows up in the BeeMonitor web app.

### Stage 6 — Cellular link (after Step 10)
```bash
lsusb                         # Telit modem present (1bc7:1201)
systemctl status cellular.service
ip addr show wwan0            # has an IP
ping -c 3 -I wwan0 8.8.8.8    # connectivity over cellular
ping -c 3 google.com         # DNS works (immutable resolv.conf)
```
✅ **Pass:** ping replies on `wwan0` and `google.com` resolves. If `wwan0` is
missing, check `journalctl -u cellular.service`, that ModemManager is disabled,
and that `/etc/qmi-network.conf` has the right APN.

### Stage 7 — Full end-to-end (telemetry over cellular, video over WiFi)
```bash
nmcli radio wifi off 2>/dev/null || sudo rfkill block wifi   # force telemetry onto cellular
sudo systemctl start beemonitor-recorder beemonitor-telemetry beemonitor-uploader
journalctl -u beemonitor-telemetry.service -f
```
Wave at the camera, then check the device dashboard. ✅ **Pass:** a snippet
records to disk, the unit shows **Online** with a fresh image + activity count
on the web app (telemetry over cellular). Re-enable WiFi (`nmcli radio wifi on`)
and ✅ the uploader drains the recorded snippet(s) to S3 and they appear under
the device's videos. (With WiFi off, video correctly stays queued on disk —
proving the WiFi gate.)

### Stage 8 — Services + reboot persistence (the field-readiness test)
```bash
sudo systemctl enable --now cellular.service beemonitor-recorder \
     beemonitor-telemetry beemonitor-uploader beemonitor-calibrate.timer
sudo reboot
# after it returns — with NO manual commands:
systemctl status cellular.service beemonitor-recorder beemonitor-telemetry beemonitor-uploader
ip addr show wwan0
systemctl list-timers beemonitor-calibrate.timer
```
✅ **Pass:** `cellular.service` brought `wwan0` up on its own, recorder +
telemetry + uploader are `active`, and the calibrate timer is scheduled. This is
the test that proves the unit will survive WittyPi wake cycles unattended.

> **Clean up after bench testing:** remove the throwaway data so it isn't later
> mistaken for real recordings — `rm -rf /tmp/bm_test /tmp/calibration.json`.

## Field Deployment

### Site Selection

- **Distance:** Camera 0.5–1.0m from bee hotel
- **Angle:** Perpendicular to hotel face (minimize skew)
- **Lighting:** Avoid direct sun on lens (causes glare)
- **Stability:** Secure mounting to prevent vibration


### Data Transfer

**Snippets upload themselves.** In normal operation you don't transfer files
manually — `beemonitor-uploader.service` streams every activity snippet to AWS S3
over the cellular link as soon as it is finished writing, then marks it with a
`.uploaded` sidecar. Watch it happen with:

```bash
journalctl -u beemonitor-uploader.service -f
# "uploading <file>" → "uploaded video_id=…"
```

Uploaded clips appear in the BeeMonitor web app for analysis. If the link is
down, clips queue on disk and drain automatically when signal returns.

#### Manual retrieval (fallback)

If you need a raw file off the Pi (debugging, or before the link is up), open a
remote shell via [Raspberry Pi Connect](#step-9-set-up-raspberry-pi-connect-remote-access)
or SSH on the local network and copy from the recordings directory:

```bash
# Files live at: /home/beemonitor/Desktop/cameraOutput/beeHotel/YYYY-MM-DD/HH_MM_SS.mp4
scp beemonitor@<pi-address>:/home/beemonitor/Desktop/cameraOutput/beeHotel/2026-06-04/*.mp4 .
```

A USB flash drive copied via the desktop file manager also works if you have a
monitor or Raspberry Pi Connect screen sharing.

## Maintenance

### Daily Checks (remote, via Raspberry Pi Connect)

- [ ] Recorder running (`systemctl status beemonitor-recorder.service`)
- [ ] Uploads draining (`journalctl -u beemonitor-uploader.service --since today`)
- [ ] Storage not filling (`df -h`) — uploaded clips can be pruned
- [ ] Battery voltage healthy

### Weekly Checks

- [ ] Confirm `calibration.json` exists and is recent (calibrate timer working)
- [ ] Inspect enclosure seals and cable connections
- [ ] Clean camera lens and solar panel
- [ ] Verify RTC accuracy
- [ ] Check cellular data usage against plan

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Camera not detected | Check ribbon cable, reboot if camera wasn't connected at boot |
| Recorder not starting | `systemctl status beemonitor-recorder.service`; check journal for errors |
| No uploads | Confirm cellular is up (`ping 8.8.8.8`); check `beemonitor-uploader` journal; verify `BEEMONITOR_DEVICE_KEY` |
| Too many / too few clips | Force a recalibration (`main_motion.py --calibrate --force`); tune `BEEMONITOR_ROI` |
| Calibration never written | Need bee-containing snippets first; check `beemonitor-calibrate.service` journal and that `ultralytics` is installed |
| WittyPi not detected | Run `i2cdetect -y 1`, should show device at 0x08 |
| Storage full | Prune already-uploaded clips (those with a `.uploaded` sidecar) |
| Wrong time on recordings | Sync RTC time via `wittyPi.sh` option (1) or (2) |

### Diagnostic Commands

```bash
# Check camera
libcamera-hello --list-cameras

# Check storage
df -h

# Check I2C (WittyPi)
i2cdetect -y 1

# Cellular link up?
ip addr && ping -c 3 8.8.8.8

# Service status + logs
systemctl status beemonitor-recorder.service beemonitor-uploader.service
journalctl -u beemonitor-recorder.service -f
journalctl -u beemonitor-uploader.service -f
systemctl list-timers beemonitor-calibrate.timer
```

## Support

- **Author:** Edward Amoah
- **Email:** eai6@psu.edu
- **Lab:** [Grozinger Lab](https://www.grozingerlab.com/), INSECT-NET, Penn State University