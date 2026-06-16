# Fleet provisioning — golden image + browser enrollment

This is the **Option A** workflow: build one BeeMonitor SD image *once*, publish it
for download, and from then on every unit is **flash → drop a token in the browser
→ power on → it appears.** No per-device install, no copy-paste keys.

There are two audiences here:

- **You (admin), once:** [build & publish the golden image](#part-1-build-the-golden-image-admin-one-time).
- **Whoever preps a card:** [the per-card flow](#part-2-prepare-a-card-per-unit) — download, flash, browser-enroll.

How a flashed card self-registers on boot is in [`hardware/enroll.py`](../enroll.py)
and `beemonitor-enroll.service`; this doc is only about getting the token onto the
card.

---

## Part 1: Build the golden image (admin, one-time)

The golden image is a **generic, fully-installed** BeeMonitor card with **no device
key and no enrollment token** — the token is dropped in per-card afterward (Part 2),
so one image serves every user.

### 1.1 Build a reference unit

On a spare Pi 4 + card, do a **normal full install** following
[`hardware/README.md`](../README.md) Steps 0–10:

- Stock **Raspberry Pi OS (Bookworm, 64-bit)**, user **`beemonitor`**.
- Source, venv, models, camera, and **all** systemd units installed and
  `enable`d (recorder, telemetry, uploader, calibrate, cellular, **and
  `beemonitor-enroll.service`** — that's the one that self-registers on boot).
- Set only `BEEMONITOR_API_BASE` in `/etc/beemonitor/uploader.env`. **Do not** add
  a `BEEMONITOR_DEVICE_KEY` or `BEEMONITOR_ENROLL_TOKEN` — those must stay out of
  the image.

Boot it once and confirm the services come up (it'll just sit in "no token —
skipping" for enroll, which is correct). Then **shut down**.

### 1.2 Generalize before capture

Clones must not share per-unit identity/state, and **no credential may ship in the
image**. Run the provided script on the reference Pi — it does every step below,
**verifies** that no device key / enrollment token survives, and refuses to power
off if one does:

```bash
sudo bash hardware/provision/generalize.sh      # confirm, then powers off
# flags: --yes (no prompt)   --no-poweroff
```

It removes the key/token from `uploader.env`, deletes any dropped boot-partition
token, blanks `machine-id` + SSH host keys (regenerated per clone on first boot),
vacuums logs, clears test recordings and shell history, then asserts the card is
credential-free before shutdown.

<details><summary>What it runs (manual equivalent, for reference)</summary>

```bash
# 1. No credentials in the image
sudo sed -i '/^BEEMONITOR_DEVICE_KEY=/d;/^BEEMONITOR_ENROLL_TOKEN=/d' /etc/beemonitor/uploader.env
sudo rm -f /boot/firmware/beemonitor.conf            # in case one was dropped while testing

# 2. Regenerate machine-id + SSH host keys per clone (don't ship one identity)
sudo truncate -s 0 /etc/machine-id
sudo rm -f /etc/ssh/ssh_host_*                        # recreated on first boot
sudo systemctl enable regenerate-ssh-host-keys 2>/dev/null || true

# 3. Clear logs / history / per-unit recordings so the image is clean + small
sudo journalctl --rotate && sudo journalctl --vacuum-time=1s
sudo rm -rf /home/beemonitor/Desktop/cameraOutput/*  # no test clips
rm -f /home/beemonitor/.bash_history
sudo poweroff
```

</details>

> The device's stable `hw_id` is the **Pi CPU serial** (unique per board), so clones
> self-enroll as distinct devices even though the image is identical — that's why
> blanking `machine-id` is safe.

### 1.3 Capture + shrink + compress

Move the card to a **Linux** machine (a cheap EC2, a second Pi, or a Linux VM —
the shrink step needs ext4 tools that macOS doesn't have). Identify the card with
`lsblk`, then:

```bash
# Raw read of the whole card (replace sdX with your card; NOT a partition)
sudo dd if=/dev/sdX of=beemonitor-golden.img bs=4M status=progress conv=fsync

# Shrink the rootfs to its minimum, re-expand on first boot, and xz-compress.
# PiShrink: https://github.com/Drewsif/PiShrink
sudo pishrink.sh -aZ beemonitor-golden.img           # -> beemonitor-golden.img.xz
```

`-a` adds a first-boot resize so the rootfs grows back to fill whatever card it's
flashed to; `-Z` xz-compresses. A fresh install typically lands around **1.5–3 GB**
compressed.

> **On macOS only?** `dd` can *read* the card (`/dev/rdiskN`) but can't shrink ext4.
> Either capture+shrink on a Linux box as above, or capture raw on the Mac and
> accept a full-card-sized download (not recommended).

### 1.4 Publish + wire up the download link

```bash
aws s3 cp beemonitor-golden.img.xz s3://<your-bucket>/images/beemonitor-golden.img.xz
```

Then point the web app at it (the enrollment page shows a **Download** button when
this is set):

```bash
# App Runner env (and local .env)
BEEMONITOR_GOLDEN_IMAGE_URL=https://<your-bucket-or-cdn>/images/beemonitor-golden.img.xz
```

Use a CDN/public-read object or a long-lived presigned URL — it's a public,
non-secret artifact (it has no keys in it).

### 1.5 When to rebuild

Rarely. Field units `git pull` and self-update on boot (see `hardware/update.sh`),
so the image only has to **boot and self-update**. Rebuild the golden image when the
*base* changes in a way a `git pull` can't fix — a new apt/pip system dependency, a
new model file too big for the repo, or an OS bump.

---

## Part 2: Prepare a card (per unit)

This is all in the browser + Raspberry Pi Imager — no terminal.

1. **Download** the image from the enrollment page (or the published URL).
2. **Flash** it with [Raspberry Pi Imager](https://www.raspberrypi.com/software/)
   → *Choose OS* → **Use custom image** → pick the `.img.xz`. Set the hostname,
   WiFi, and locale in Imager's gear menu as usual. Write, keep the card inserted.
3. **Enroll** in the browser: **Devices → Zero-touch enrollment → Generate token →
   "Choose SD card & write."** Pick the card's `bootfs` drive; it writes
   `beemonitor.conf` (the API base + token) onto the boot partition.
   - *Chrome or Edge only* (the File System Access API). On Safari/Firefox, use the
     [CLI fallback](#cli-fallback) or the manual snippet on that page.
4. **Assemble the hardware, insert the card, power on.** On first boot the unit
   reads the token, registers itself, writes its own device key, clears the token
   from the boot partition, and starts recording. It appears on your Devices page.

Re-flashing the same Pi later rebinds it to the **same** device (matched on the Pi
serial) and rotates its key.

### CLI fallback

For Safari/Firefox/headless prep, [`prepare-card.sh`](prepare-card.sh) writes the
same `beemonitor.conf` onto a flashed card's boot partition from macOS or Linux:

```bash
./prepare-card.sh --token bmk_enroll_xxx --api-base https://<host>
```

It auto-detects the mounted `bootfs`/`boot` volume (override with `--boot PATH`) and
refuses to write unless the volume looks like a Pi boot partition. See
[`beemonitor.conf.example`](beemonitor.conf.example) for the file it produces.
