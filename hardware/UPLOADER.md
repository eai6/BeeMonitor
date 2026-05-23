# Pi uploader — install on a Raspberry Pi

`hardware/uploader.py` watches the recording directory and uploads finished
`.mp4` files to BeeMonitor via the authenticated `/api/v1/uploads/*` API. It
runs as a separate systemd service from `main.py` so a network blip cannot
stop recording (and a recorder crash cannot stop uploads).

## 1. Issue a device key from the web admin

In a browser, log into the BeeMonitor admin
(`https://mqnafc3ejc.us-east-1.awsapprunner.com/admin/`), go to
**Devices → Add device**, fill in a name + location, and Save. A green
success banner shows the raw `bmk_device_…` key **once** — copy it now,
it is not recoverable.

## 2. Lay out the Pi

```bash
sudo apt install python3-pip
sudo pip3 install requests           # the only Python dep the uploader needs
sudo mkdir -p /etc/beemonitor
sudo chmod 750 /etc/beemonitor
```

Create `/etc/beemonitor/uploader.env` (root-only readable):

```bash
sudo tee /etc/beemonitor/uploader.env >/dev/null <<'EOF'
BEEMONITOR_API_BASE=https://mqnafc3ejc.us-east-1.awsapprunner.com
BEEMONITOR_DEVICE_KEY=bmk_device_PASTE_THE_KEY_HERE
BEEMONITOR_RECORD_DIR=/home/apis/Desktop/cameraOutput/beeHotel
BEEMONITOR_POLL_SECONDS=30
EOF
sudo chmod 600 /etc/beemonitor/uploader.env
sudo chown root:root /etc/beemonitor/uploader.env
```

## 3. Install the systemd unit

Assumes the repo is checked out at `/home/apis/BeeMonitor`.

```bash
sudo cp /home/apis/BeeMonitor/hardware/systemd/beemonitor-uploader.service \
        /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now beemonitor-uploader.service
```

## 4. Verify

```bash
sudo systemctl status beemonitor-uploader      # should be 'active (running)'
journalctl -u beemonitor-uploader -f           # live logs
```

A successful upload looks like:

```
uploader started — API=https://… , record_dir=/home/apis/…
found 1 pending file(s)
uploading natalies_2026-05-23_10_00_00.mp4 (47.3 MB, recorded_at=…)
uploaded video_id=42 key=users/7/devices/3/2026/05/23/…mp4
```

Once a file is uploaded the uploader writes a `<filename>.uploaded` sidecar
so it never re-uploads it.

## Failure modes

- **No network**: the uploader retries with exponential backoff (5 s → 5 min cap).
  Recording continues regardless.
- **Bad device key**: the API returns 401; the uploader logs and backs off.
  Reissue the key from the admin if it was revoked.
- **Partial PUT** (Pi reboots mid-upload): no `.uploaded` sidecar is written,
  so the next poll picks the file back up and starts from scratch. Multipart
  / resumable uploads are a planned follow-up.
