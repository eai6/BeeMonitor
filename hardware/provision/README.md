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

### 1.1b Make it artifact-native (recommended — feature 18)

Convert the reference unit from the git-clone layout to the **release/symlink
layout** so clones update via **signed artifacts** instead of `git pull` — no git
remote, no `.git`, no cloud/web/src code on a field device, signed + rollback-safe
updates. Run on the reference Pi:

```bash
sudo bash hardware/provision/migrate-to-releases.sh
```

This sets up `~/BeeMonitor -> releases/<v0>/` (hardware/ only), the stable
`~/beemonitor-venv` + `~/models` (symlinked into the release), installs `minisign`,
and copies the verify key to `~/minisign.pub`. Confirm services still come up
(`systemctl is-active beemonitor-recorder beemonitor-telemetry`), then shut down.
`generalize.sh` (next) drops the leftover git clone so the image carries no git or
cloud code. After flashing, a unit self-updates from the dashboard's **"Update via
signed artifact"** button.

> Skip this step to keep the older **git-update** image (still works — units `git
> pull` from the repo). The artifact path needs the repo's `minisign.pub`
> (committed) + a published, signed bundle (the `edge-artifact` CI on a `v*` tag).

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

### 1.3 Capture + shrink (on a Linux box)

The shrink needs ext4 tools macOS doesn't have, so do this on a Linux box. We used
a second Pi with the golden card in a **USB SD reader** (the reader is `/dev/sda`;
the Pi's own card stays `/dev/mmcblk0` — never touch it).

Install the tools once:

```bash
sudo apt-get install -y parted e2fsprogs xz-utils
wget https://raw.githubusercontent.com/Drewsif/PiShrink/master/pishrink.sh
chmod +x pishrink.sh && sudo mv pishrink.sh /usr/local/bin/pishrink.sh
```

Identify the card with `lsblk`, then shrink it in place and capture only the used
part (a raw `dd` of the whole card would be the card's full size — pishrink's `-a`
auto-expands on flash, so the source card size doesn't matter):

```bash
DEV=/dev/sda; P2=/dev/sda2; mkdir -p ~/golden     # confirm DEV with lsblk first

# unmount the card
for p in ${DEV}*[0-9]; do sudo umount "$p" 2>/dev/null; done

# shrink the rootfs filesystem to its minimum (data preserved)
sudo e2fsck -fy "$P2"
sudo resize2fs -M "$P2"

# shrink the partition to fs size + 300 MB slack (parted -s answers the shrink
# warning "No", so force a "Yes" through a pseudo-tty)
BS=$(sudo dumpe2fs -h "$P2" 2>/dev/null | awk -F: '/Block size/{gsub(/ /,"",$2);print $2}')
BC=$(sudo dumpe2fs -h "$P2" 2>/dev/null | awk -F: '/Block count/{gsub(/ /,"",$2);print $2}')
START=$(sudo parted -ms "$DEV" unit B print | awk -F: '$1==2{gsub(/B/,"",$2);print $2}')
NEWEND=$(( START + BS*BC + 314572800 ))
echo Yes | sudo parted ---pretend-input-tty "$DEV" unit B resizepart 2 ${NEWEND}B

# capture up to the new partition end (not the whole card)
END=$(sudo parted -ms "$DEV" unit B print | awk -F: '$1==2{gsub(/B/,"",$3);print $3}')
sudo dd if="$DEV" of=~/golden/beemonitor-golden.img bs=1M count=$(( END/1048576 + 16 )) status=progress conv=fsync

# shrink + compress, adding the first-boot auto-expand (xz is slow on a Pi, ~45 min)
sudo pishrink.sh -aZ ~/golden/beemonitor-golden.img   # -> ~/golden/beemonitor-golden.img.xz
```

A 238 GB card shrank to ~12 GB captured → **2.4 GB** `beemonitor-golden.img.xz`.

### 1.4 Copy it to your laptop (+ optionally publish)

```bash
# on the Mac:
scp beemonitor@<pi-ip>:~/golden/beemonitor-golden.img.xz ~/Downloads/
```

To publish so the enrollment page shows a **Download** button (optional — you can
flash the local file directly):

```bash
aws s3 cp ~/Downloads/beemonitor-golden.img.xz s3://<your-bucket>/images/beemonitor-golden.img.xz
# then set in App Runner env (+ local .env):
BEEMONITOR_GOLDEN_IMAGE_URL=https://<bucket-or-cdn>/images/beemonitor-golden.img.xz
```

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
3. **Enroll** in the browser: **Devices → Add a device → Generate token →
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

---

## Remote access — Tailscale SSH (replaces Pi Connect, no device cap)

Remote-shell a unit through CGNAT with **no inbound port and no public exposure**,
and without Pi Connect's free-tier device limit (Tailscale free ≈ 100 devices). Same
gate model as the rest of the unit: reachable over **WiFi always**, and over
**cellular only when you drop the gate** (the dashboard's Cellular-access toggle /
`cellular-firewall.sh open`) — `tailscaled` isn't in the telemetry allowlist, so the
gated firewall keeps it off metered data until you open it, then it reconnects.

1. In the Tailscale admin console → **Settings → Keys**, make a **reusable,
   ephemeral, tagged** pre-auth key (tag e.g. `tag:beemonitor`) and add an ACL that
   scopes that tag, so a leaked key/unit can't roam your tailnet.
2. Put it in the unit's `/etc/beemonitor/uploader.env`: `TAILSCALE_AUTHKEY=tskey-auth-…`
3. Pre-install tailscale in the golden image (`curl -fsSL https://tailscale.com/install.sh | sh`)
   and `systemctl enable beemonitor-tailscale.service` during the build, so every
   clone joins on first boot. (`provision.sh` only refreshes units already present,
   so a brand-new unit type must be enabled in the image, not pushed via an update.)
4. **First join must reach the Tailscale control plane** — do it on WiFi or with the
   cellular gate open. After that `tailscaled` persists the session and reconnects on
   its own on whatever link the firewall currently permits.

Then `tailscale ssh beemonitor@<hostname>` (or the unit's tailnet IP). Tailscale SSH
authenticates via your tailnet identity, so the system `sshd` never has to be opened.
[`tailscale-up.sh`](tailscale-up.sh) does the join (idempotent, fails soft).
