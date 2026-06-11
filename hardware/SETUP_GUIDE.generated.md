# BeeMonitor — Device Setup Guide (generated)

> Generated from `apps/setup/content.py` — do not edit by hand.
> The interactive version lives on the dashboard under **Set up a device**.

## Flash & first boot

*Get Raspberry Pi OS onto the card and reachable over SSH.*

### Flash & first boot  (~15 min)

Write Raspberry Pi OS (Bookworm, 64-bit) with Raspberry Pi Imager; in its settings set the username to exactly 'beemonitor' and your WiFi (every path here assumes that user). Connect the HQ Camera ribbon BEFORE first boot, then boot with a monitor, keyboard, and mouse — setup is done on-screen, no SSH needed. Open a Terminal and update:

```bash
sudo apt update && sudo apt full-upgrade -y
getconf LONG_BIT   # must print 64
whoami             # must print beemonitor
```

**What you should see:** The desktop loads; LONG_BIT is 64 and whoami is beemonitor. 64-bit is required (the calibration PyTorch wheels are aarch64-only).

> ⚠️ **Username isn't 'beemonitor'** — Re-flash — the systemd units hardcode /home/beemonitor paths and run as User=beemonitor.

> ⚠️ **no display / black screen** — Use the micro-HDMI port nearest the USB-C power, and connect the monitor before powering on.

## Install software

*Code, system packages, the Python venv, and folders.*

### Download the code  (~2 min)

Clone the BeeMonitor repo into the home directory.

```bash
cd ~ && git clone https://github.com/eai6/BeeMonitor.git
cd ~/BeeMonitor/hardware
```

**What you should see:** A ~/BeeMonitor/hardware directory with the scripts and systemd/ unit files.

### System deps + virtualenv  (~12 min)

picamera2/cv2 come from apt; the rest live in a venv built with --system-site-packages so it can import them. Install the CPU torch wheels FIRST or YOLO SIGILLs on the Pi's ARM CPU.

```bash
sudo apt install -y python3-picamera2 python3-opencv ffmpeg python3-venv libqmi-utils udhcpc nftables
python3 -m venv --system-site-packages ~/BeeMonitor/hardware/venv
~/BeeMonitor/hardware/venv/bin/pip install --upgrade pip requests
~/BeeMonitor/hardware/venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
~/BeeMonitor/hardware/venv/bin/pip install ultralytics
```

**What you should see:** All installs succeed. (libqmi-utils/udhcpc/nftables are the cellular stack — harmless to install on a WiFi unit.)

> ⚠️ **YOLO exits with 'illegal instruction' / status=4** — torch was installed after ultralytics. Reinstall the CPU torch wheels first, then ultralytics.

### Output dirs + seed calibration  (~2 min)

Create the recording tree and seed a tight bee-sized motion window so the recorder doesn't start wide-open. Run WITHOUT sudo — the recorder runs as 'beemonitor' and can't write a root-owned tree.

```bash
~/BeeMonitor/hardware/venv/bin/python makeDirectories.py
cp calibration.sample.json /home/beemonitor/Desktop/cameraOutput/beeHotel/../calibration.json
```

**What you should see:** The cameraOutput tree exists and calibration.json is in place ([62, 554], not the wide-open [20, 5000]).

> ⚠️ **'permission denied' saving video later** — You ran makeDirectories.py as root. Fix ownership: sudo chown -R beemonitor:beemonitor /home/beemonitor/Desktop/cameraOutput

## Camera & credentials

*Focus the lens and point the unit at your account.*

### Focus the camera  (~5 min)

Run the focus helper and turn the lens until the preview is sharp on the bee hotel. A blurry lens wrecks both motion detection and calibration.

```bash
~/BeeMonitor/hardware/venv/bin/python runFocus.py
```

**What you should see:** A live preview; the hotel entrances look crisp. (Verify the camera is detected first: libcamera-hello --list-cameras.)

> ⚠️ **no cameras available** — Ribbon seated after boot — reseat it and reboot; the camera is only probed at boot.

### Credentials & tuning  (~5 min)

Point the unit at YOUR account. The device key below is filled in for this device — paste the whole block into /etc/beemonitor/uploader.env.

```bash
sudo mkdir -p /etc/beemonitor
sudo tee /etc/beemonitor/uploader.env >/dev/null <<'EOF'
BEEMONITOR_API_BASE=https://mqnafc3ejc.us-east-1.awsapprunner.com
BEEMONITOR_DEVICE_KEY=bmk_device_REPLACE_ME
BEEMONITOR_RECORD_DIR=/home/beemonitor/Desktop/cameraOutput/beeHotel
EOF
sudo chmod 600 /etc/beemonitor/uploader.env
```

**What you should see:** uploader.env exists, mode 600, with your real device key.

> ⚠️ **key shows bmk_device_REPLACE_ME** — The one-time key window has closed. Re-issue a key for this device (Devices → this device → re-issue) and paste it in.

## Install & start services

*The recorder, telemetry, and uploader as systemd units.*

### Install the services  (~5 min)

Copy the systemd units (incl. the remote-update + USB units) and grant the non-root telemetry user passwordless rights to exactly what the dashboard drives: WiFi (nmcli), remote update, and USB copy.

```bash
cd ~/BeeMonitor/hardware
sudo cp systemd/beemonitor-recorder.service systemd/beemonitor-telemetry.service systemd/beemonitor-uploader.service systemd/beemonitor-calibrate.service systemd/beemonitor-calibrate.timer systemd/beemonitor-update.service systemd/beemonitor-usb-transfer@.service /etc/systemd/system/
sudo cp cellular/99-beemonitor-usb.rules /etc/udev/rules.d/
chmod +x usb-transfer.sh
printf 'beemonitor ALL=(root) NOPASSWD: /usr/bin/nmcli\nbeemonitor ALL=(root) NOPASSWD: /usr/bin/systemctl start --no-block beemonitor-update.service\nbeemonitor ALL=(root) NOPASSWD: /home/beemonitor/BeeMonitor/hardware/usb-transfer.sh\n' | sudo tee /etc/sudoers.d/beemonitor >/dev/null
sudo chmod 440 /etc/sudoers.d/beemonitor
sudo visudo -cf /etc/sudoers.d/beemonitor
sudo udevadm control --reload && sudo systemctl daemon-reload
```

**What you should see:** daemon-reload returns cleanly and visudo -cf prints '/etc/sudoers.d/beemonitor: parsed OK'.

> ⚠️ **remote update fails: 'cannot start beemonitor-update.service (sudoers rule missing?)'** — This step's sudoers drop-in wasn't applied. Re-run the printf…/etc/sudoers.d/beemonitor block above, then retry Software → Update.

### Start & enable on boot  (~3 min)

Enable and start the app layer. After this the unit should check in — watch the dashboard flip to ONLINE here.

```bash
sudo systemctl enable --now beemonitor-recorder.service beemonitor-telemetry.service beemonitor-uploader.service
sudo systemctl enable --now beemonitor-calibrate.timer
```

**What you should see:** The dashboard shows this device as ONLINE within a minute (the first heartbeat arrived), recorder + uploader active.

> ⚠️ **stays offline** — journalctl -u beemonitor-telemetry -n 30 — look for 'heartbeat POST failed' (network) or a 401 (wrong key).

### Verify recorder & uploader  (~5 min)

Confirm the recorder is capturing and the uploader is running. Wave a hand in front of the hotel to trigger a clip.

```bash
systemctl is-active beemonitor-recorder beemonitor-uploader
journalctl -u beemonitor-uploader -f   # watch a clip upload
```

**What you should see:** Both report 'active'; a hand-wave produces a snippet that uploads to the cloud and appears under this device's videos.

## Power & remote access

*WittyPi scheduling and Raspberry Pi Connect (field units).*

### Set up WittyPi scheduling  (optional, ~30 min)

WittyPi powers the Pi on a schedule (e.g. dawn-to-dusk) to save battery in the field. Install it, set auto power-on, and load your daily window.

```bash
# Follow Step 8 in hardware/README.md for the WittyPi installer
# and the production on/off schedule for your site.
```

**What you should see:** A scheduled power cycle the unit follows on its own.

### Raspberry Pi Connect  (optional, ~5 min)

Remote shell/screen to the unit from anywhere — invaluable for a field device you can't physically reach.

```bash
sudo apt install -y rpi-connect
rpi-connect on
rpi-connect signin   # follow the URL to link your account
```

**What you should see:** The device appears at connect.raspberrypi.com.

## Cellular connectivity

*Sixfab 4G LTE — only for cellular field units.*

### Modem in QMI mode  (cellular-only, ~10 min)

Confirm the Sixfab/Telit modem enumerated in QMI mode and stop ModemManager (it fights QMI). One-time USB-mode switch may be needed first — see Step 10.1.

```bash
lsusb | grep -i 1bc7:1201 && ls /dev/cdc-wdm0
sudo systemctl disable --now ModemManager.service
```

**What you should see:** The Telit modem and /dev/cdc-wdm0 control node are present.

> ⚠️ **no /dev/cdc-wdm0** — Do the one-time USB-mode switch (Telit: AT#USBCFG) in Step 10.1, then reboot.

### APN + pinned DNS  (cellular-only, ~8 min)

Set the carrier APN (the sample is 'super' for the Sixfab SIM) and pin DNS to a real static file — fresh Pi OS images symlink resolv.conf at 127.0.0.53, which breaks name resolution over cellular.

```bash
sudo cp cellular/qmi-network.conf.sample /etc/qmi-network.conf
sudo chattr -i /etc/resolv.conf 2>/dev/null || true
sudo rm -f /etc/resolv.conf
printf 'nameserver 8.8.8.8\nnameserver 1.1.1.1\n' | sudo tee /etc/resolv.conf
sudo chattr +i /etc/resolv.conf
```

**What you should see:** After the link is up, google.com resolves (not just 8.8.8.8).

> ⚠️ **8.8.8.8 pings but google.com doesn't** — resolv.conf is still the managed symlink — rm it and write the static file as above.

### Bring up the link + firewall  (cellular-only, ~12 min)

Install the cellular + firewall units. The firewall gates mobile data to ONLY the telemetry service so nothing else eats your SIM; bulk video stays WiFi-gated.

```bash
chmod +x cellular/cellular-up.sh cellular/cellular-firewall.sh
sudo cp systemd/cellular-firewall.service systemd/cellular.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now cellular-firewall.service cellular.service
```

**What you should see:** ping -c3 8.8.8.8 succeeds over the modem and the device keeps checking in with cellular_active true on the dashboard.

> ⚠️ **cellular.service enabled but never runs, empty journal** — Dependency cycle — never add After=multi-user.target to it (see Step 10.6).

> ⚠️ **uploads fail with TLS/cert errors after a cold boot** — Clock is stale — wait for cellular.service then NTP sync.
