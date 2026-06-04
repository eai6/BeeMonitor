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
12. [Field Deployment](#field-deployment)
13. [Maintenance](#maintenance)
14. [Troubleshooting](#troubleshooting)

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
| `beemonitor-recorder.service` | `hardware/main_motion.py` | Records **only activity snippets** (MOG2 motion gate) |
| `beemonitor-uploader.service` | `hardware/uploader.py` | Streams finished snippets to S3 via the API |
| `beemonitor-calibrate.timer` | `hardware/main_motion.py --calibrate` | Daily: learns the bee blob-size window from recorded snippets with YOLO |

All three read configuration from `/etc/beemonitor/uploader.env`.

### Quick Install (cellular field unit)

For a fast (re)deploy on a Pi that already has the camera focused and WittyPi
set up. Run the blocks below top-to-bottom; the only manual steps are editing the
env file and bringing up the modem. The detailed walkthrough follows in Steps 1–7.

```bash
# 1. Code
cd ~ && git clone https://github.com/eai6/BeeMonitor.git
cd ~/BeeMonitor/hardware

# 2. Dependencies (ultralytics is heavy — needed only for auto-calibration)
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-picamera2 python3-opencv ffmpeg python3-requests
pip3 install ultralytics

# 3. Output directories
python3 makeDirectories.py

# 4. Credentials + tuning  (EDIT: paste your bmk_device_ key, then save)
sudo mkdir -p /etc/beemonitor
sudo tee /etc/beemonitor/uploader.env >/dev/null <<'EOF'
BEEMONITOR_API_BASE=https://mqnafc3ejc.us-east-1.awsapprunner.com
BEEMONITOR_DEVICE_KEY=bmk_device_REPLACE_ME
BEEMONITOR_RECORD_DIR=/home/apis/Desktop/cameraOutput/beeHotel
EOF
sudo nano /etc/beemonitor/uploader.env      # <-- paste the real device key
sudo chmod 600 /etc/beemonitor/uploader.env

# 5. Install + enable the three services
sudo cp systemd/beemonitor-recorder.service  /etc/systemd/system/
sudo cp systemd/beemonitor-uploader.service  /etc/systemd/system/
sudo cp systemd/beemonitor-calibrate.service /etc/systemd/system/
sudo cp systemd/beemonitor-calibrate.timer   /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now beemonitor-recorder.service beemonitor-uploader.service
sudo systemctl enable --now beemonitor-calibrate.timer

# 6. Bring up cellular (see Step 10 for your exact Sixfab kit), then verify:
ping -c 3 8.8.8.8
journalctl -u beemonitor-uploader.service -f   # watch snippets upload to S3
```

> First boot runs on permissive motion thresholds until the calibrate timer
> finds bees in the recorded snippets. To calibrate immediately once a few clips
> exist: `python3 ~/BeeMonitor/hardware/main_motion.py --calibrate --force`

### Step 1: Download Source Code

```bash
cd ~
git clone https://github.com/eai6/BeeMonitor.git
cd BeeMonitor/hardware
```

> The systemd unit files expect the repo at **`/home/apis/BeeMonitor`**. If you
> clone elsewhere, edit the `ExecStart=` paths in `hardware/systemd/*.service`.

### Step 2: Install Dependencies

```bash
sudo apt update && sudo apt upgrade -y

# Camera, computer vision, and video muxing (ffmpeg remuxes snippets to .mp4)
sudo apt install -y python3-picamera2 python3-opencv ffmpeg

# Uploader HTTP client
sudo apt install -y python3-requests

# Calibration ONLY (heavy — pulls in PyTorch). Needed by the calibrate timer.
# Skip on the Pi if you intend to calibrate on another machine instead.
pip3 install ultralytics
```

### Step 3: Create Output Directories

```bash
python3 makeDirectories.py
```

(The recorder also auto-creates its working directories on first run.)

### Step 4: Focus the Camera

```bash
python3 runFocus.py
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
BEEMONITOR_RECORD_DIR=/home/apis/Desktop/cameraOutput/beeHotel

# --- optional: YOLO model for auto-calibration ---
# Defaults to stock "yolo11n.pt" (downloaded on first calibrate). Point this at
# your trained bee model for best results.
# BEEMONITOR_YOLO_MODEL=/home/apis/BeeMonitor/models/bee_yolo.pt

# --- optional tuning (sensible defaults if omitted; see Configuration Reference) ---
# BEEMONITOR_PRE_ROLL=3
# BEEMONITOR_POST_ROLL=4
# BEEMONITOR_HEARTBEAT_INTERVAL=3600
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
sudo cp systemd/beemonitor-uploader.service  /etc/systemd/system/
sudo cp systemd/beemonitor-calibrate.service /etc/systemd/system/
sudo cp systemd/beemonitor-calibrate.timer   /etc/systemd/system/
sudo systemctl daemon-reload
```

### Step 7: Start and Enable on Boot

```bash
# Record + upload (start now, and on every boot)
sudo systemctl enable --now beemonitor-recorder.service
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
> `python3 main_motion.py --calibrate --force`

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
   `RECORD_DIR/YYYY-MM-DD/`. A short **heartbeat clip** is recorded periodically
   even with no motion so you can audit detection remotely.
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
| `BEEMONITOR_HEARTBEAT_INTERVAL` | `3600` | Seconds between audit clips (`0` disables) |
| `BEEMONITOR_HEARTBEAT_SECONDS` | `10` | Length of each heartbeat clip |
| `BEEMONITOR_LORES_W` / `_H` | `640` / `480` | Resolution of the detection stream |
| `BEEMONITOR_FPS` | `25` | Capture frame rate |
| `BEEMONITOR_ROI` | (full frame) | `x1,y1,x2,y2` in lores px to restrict detection to the hotel face |
| `BEEMONITOR_YOLO_MODEL` | `yolo11n.pt` | YOLO weights used by the calibrate job |
| `BEEMONITOR_CALIB_MAX_AGE_DAYS` | `7` | Skip recalibration if `calibration.json` is younger than this |
| `BEEMONITOR_POLL_SECONDS` | `30` | How often the uploader scans for new snippets |

Tuning is rarely needed — start with defaults and adjust pre/post-roll or `ROI`
only if you see clips clipped short or too much background motion triggering.

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

### 8.4 Test WittyPi Through 1 Cycle

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

### 8.5 Create Production Schedule

Create the scheduler file:
```bash
sudo nano /home/apis/Desktop/wittypi/schedules/beeHotelScheduler_2024.wpi
```

Paste the following:
```
BEGIN 2024-03-00 07:50:00
END   2024-09-01 00:00:00
ON    H10 M15 # will start recording from 7:50am to 6:05pm
OFF   H13 M45 # will be off until the next day
```

### 8.6 Apply the Schedule

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
loginctl enable-linger apis      # keep the user service alive without a login
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

This is what lets a remote, off-grid unit upload its snippets — the
[uploader service](#software-installation) pushes them to AWS S3 over whatever
internet connection the Pi has, and the Sixfab 4G/LTE kit provides that
connection where there is no WiFi.

> **The transport is already built.** `beemonitor-uploader.service` does the
> AWS transfer (Pi → presigned S3 PUT via the BeeMonitor API). It is
> network-agnostic — once the modem gives the Pi internet, uploads "just work"
> with **no software changes**. Motion-gating keeps the data volume small enough
> for a cellular plan; see [How Motion-Gated Recording Works](#how-motion-gated-recording-works).

We use **ECM mode** — the modem presents itself as a USB Ethernet device
(`usb0`) that the Linux kernel supports natively (no drivers), and
NetworkManager brings it up with DHCP automatically. This is a **one-time setup
on the Pi**; the BeeMonitor software needs no changes.

Reference: [Sixfab — Hardware Assemble](https://docs.sixfab.com/docs/raspberry-pi-4g-lte-cellular-modem-kit-getting-started-with-ecm-mode)
and [Cellular Internet Connection in ECM Mode](https://docs.sixfab.com/page/cellular-internet-connection-in-ecm-mode).

### 10.1 Insert the SIM and Assemble the HAT

1. Attach the mini PCIe module to the HAT and insert an **activated** micro SIM.
2. Connect the antennas — the main LTE antenna to the **MAIN** port (and the
   diversity/GNSS antenna to its matching port).
3. Stack the HAT on the Pi's GPIO header (mind clearance with the WittyPi) and
   connect the modem's **USB** jumper to the Pi.
4. Power the Pi and confirm the modem enumerates:
   ```bash
   lsusb        # should list a Quectel device
   ```

### 10.2 Disable ModemManager

On Raspberry Pi OS (Bookworm+) ModemManager grabs the modem ports and interferes
with ECM. Turn it off:

```bash
sudo systemctl stop ModemManager.service
sudo systemctl disable ModemManager.service
```

### 10.3 Set the APN and Enable ECM Mode (one-time, via AT commands)

Install a serial terminal and connect to the modem's AT port (usually
`/dev/ttyUSB2` or `/dev/ttyUSB3`):

```bash
sudo apt install -y minicom
sudo minicom -D /dev/ttyUSB2 -b 115200
```

At the prompt, set your carrier's APN, then switch the modem to ECM. Replace
`super` with your SIM's APN (e.g. Twilio Super SIM = `super`, Hologram = `hologram`):

```
AT+CGDCONT=1,"IPV4V6","super"
AT+QCFG="usbnet",1
AT+CFUN=1,1
```

`AT+QCFG="usbnet",1` switches to ECM and `AT+CFUN=1,1` reboots the modem
(~30 s). Exit minicom with **Ctrl-A** then **X**. After it reboots you can
confirm ECM is active (should return `+QCFG: "usbnet",1`):

```
AT+QCFG="usbnet"
AT+CPIN?      # +CPIN: READY
AT+CREG?      # +CREG: 0,1  (or 0,5 = roaming) means registered
```

### 10.4 Bring Up and Verify the Interface

NetworkManager should auto-DHCP the new `usb0`. Verify:

```bash
ip addr show usb0           # should have an IP
ping -c 3 -I usb0 8.8.8.8   # connectivity over cellular
```

### 10.5 Make It Persistent (NetworkManager autoconnect profile)

To guarantee `usb0` comes up on every boot, create a dedicated, auto-connecting
NetworkManager profile bound to that interface:

```bash
sudo nmcli connection add type ethernet ifname usb0 con-name cellular \
  ipv4.method auto ipv6.method auto \
  connection.autoconnect yes \
  connection.autoconnect-retries 0 \
  connection.autoconnect-priority 10
sudo nmcli connection up cellular
```

- `autoconnect yes` + `autoconnect-retries 0` = always bring it up, retry forever
  (important for a field unit that may boot before the modem has registered).
- `autoconnect-priority 10` makes cellular the preferred route. If this unit also
  sometimes has WiFi you'd rather use, give the WiFi profile a **higher** number,
  or raise the cellular **route metric** so WiFi wins when present:
  ```bash
  sudo nmcli connection modify cellular ipv4.route-metric 700
  ```

Confirm it survives a reboot:

```bash
sudo reboot
# after it comes back:
nmcli connection show --active     # "cellular" should be listed on usb0
ip addr show usb0                  # should have an IP
ping -c 3 -I usb0 8.8.8.8
```

### 10.6 Confirm Uploads Flow

With the modem up, the already-running uploader will drain any pending snippets:

```bash
journalctl -u beemonitor-uploader.service -f
# look for: "uploading <file>" then "uploaded video_id=…"
```

### 10.7 Keep Cellular Data Under Control

- **Motion-gating** is the main lever — only activity snippets are sent.
- Tune `BEEMONITOR_PRE_ROLL` / `BEEMONITOR_POST_ROLL` down to shrink clips.
- Lower `BEEMONITOR_HEARTBEAT_INTERVAL` frequency (or set `0`) if audit clips are
  eating into your plan.
- Restrict detection to the hotel face with `BEEMONITOR_ROI` to avoid triggering
  on background motion (waving plants, passers-by).
- The uploader retries with backoff, so brief signal drops self-heal — clips
  accumulate on disk and upload when signal returns.

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
# Files live at: /home/apis/Desktop/cameraOutput/beeHotel/YYYY-MM-DD/HH_MM_SS.mp4
scp apis@<pi-address>:/home/apis/Desktop/cameraOutput/beeHotel/2026-06-04/*.mp4 .
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