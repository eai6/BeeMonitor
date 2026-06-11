# BeeMonitor — Golden-Image Fleet Provisioning Design

**Status:** Phase 1 implemented (commit `bcfb914`, branch
`devices/golden-image-provisioning`); golden image not yet built/published.
**Author:** Drafted with Claude Code, 2026-06-11
**Goal:** Make standing up a field unit "flash → drop a token in the browser →
assemble hardware → insert card → power on → it appears" — so a non-expert only
assembles hardware and inserts a card. No per-device install, no copy-paste keys.

---

## 1. Motivation

Today a new unit means working through `hardware/README.md` Steps 0–10 by hand on
the Pi (venv, picamera2, models, services, cellular) and pasting a device key into
`uploader.env`. That's fine for the maintainer, intractable for a collaborator who
just wants a working hive. We want the heavy install done **once**, captured into a
reusable image, and the only per-unit step to be a token drop a browser can do.

Builds on the zero-touch enrollment already shipped (commit `2f1bf3d`): a unit with
an enrollment token but no device key self-registers on first boot via
`hardware/enroll.py` + `beemonitor-enroll.service`. This work is about getting that
token onto the card with zero friction.

---

## 2. Key decision — Option A (generic golden image), not per-user baking

Three architectures were considered:

1. **Drop-in token file** — browser/CLI writes the token; card flashed from a
   golden image. (This is what we built.)
2. **Server-baked per-user image** — backend injects each user's token into a copy
   of the golden image and streams a ready-to-flash `.img.xz`.
3. **Stock Pi OS + first-boot self-install** — flash plain Pi OS; a firstrun script
   installs BeeMonitor on first boot.

**Chosen: Option A / #1.** Rationale:

- **Browsers can't raw-write an SD card** (no block access; SD cards aren't
  WebUSB/Serial devices). So burning the OS always needs a native flasher (Pi
  Imager). The browser's achievable job is writing small files onto the **FAT boot
  partition** via the File System Access API — i.e. the token.
- Because the token is dropped **after** flashing, **one generic image serves every
  user** — no privileged image-baker worker, no multi-GB per-user builds, no S3
  bake pipeline (the cost of #2).
- #3 keeps "flash stock Pi OS" but makes **first boot slow and fragile** (heavy
  apt/pip/PyTorch install in the field). Rejected for field reliability.

The boot partition (FAT32) was chosen over `uploader.env` (ext4) deliberately: FAT
is writable from macOS/Windows/Linux and from the browser with no special tooling;
ext4 is not (esp. on macOS).

---

## 3. Architecture / data flow

```
Admin (once):  build reference Pi → generalize → capture+shrink → publish .img.xz
                                                                       │
User (per card):                                                       ▼
  1. download .img.xz (browser, just an HTTPS file)
  2. flash with Pi Imager → "Use custom image"
  3. enrollment page → Generate token → "Choose SD card & write"
        browser writes /bootfs/beemonitor.conf  { API_BASE, ENROLL_TOKEN }
  4. insert card, power on
        beemonitor-enroll.service → enroll.py:
          reads token from boot partition (or uploader.env; boot wins)
          POST /api/v1/devices/enroll {token, hw_id=Pi serial, hostname, tz}
          ← { device_key }  → writes uploader.env, clears token from boot file
        recorder/telemetry/uploader start → 📡 device appears on dashboard
```

Re-flashing the same Pi rebinds to the **same** Device (idempotent by
`(owner, hw_id)`) and rotates its key.

---

## 4. Phase 1 — what was built (commit `bcfb914`)

- **`hardware/enroll.py`** — reads `BEEMONITOR_ENROLL_TOKEN` / `BEEMONITOR_API_BASE`
  from the FAT boot partition (`/boot/firmware/beemonitor.conf`, candidates incl.
  `/boot/...` and `$BEEMONITOR_BOOT_CONF`) as well as `uploader.env` (boot wins);
  **clears the token line** from the boot file after a key is issued, so an enrolled
  card carries no live credential.
- **`apps/devices/templates/devices/enrollment.html`** — "Choose SD card & write"
  button (File System Access API) writes `beemonitor.conf` onto the card's `bootfs`,
  verifying `config.txt` first; manual `<details>` fallback auto-opens on
  Safari/Firefox. Plus a "How to set up a card" steps block with a Download link.
- **`config/settings/base.py`** — `BEEMONITOR_GOLDEN_IMAGE_URL` (empty = no download
  link). Surfaced as `image_url` in `DeviceEnrollmentView`.
- **`hardware/provision/`** — `README.md` (golden-image build + publish guide +
  per-card flow), `prepare-card.sh` (CLI fallback, macOS/Linux), and
  `beemonitor.conf.example`.

Constraint to remember: the browser writer is **Chromium-only** (Chrome/Edge/Brave);
Safari/Firefox use the CLI or manual snippet.

---

## 5. Execution plan — remaining steps (Phase 1 completion)

1. **Build the golden image** on a reference Pi: full install per
   `hardware/README.md` 0–10, all services `enable`d (incl. `beemonitor-enroll`),
   only `BEEMONITOR_API_BASE` in `uploader.env` (no key, no token).
2. **Generalize:** strip device key/token, blank `/etc/machine-id`, remove SSH host
   keys (regen on boot), clear logs/history/test clips.
3. **Capture + shrink** on a Linux box: `dd` the card → `pishrink.sh -aZ` →
   `beemonitor-golden.img.xz` (~1.5–3 GB). (macOS can `dd`-read but not shrink ext4.)
4. **Publish:** `aws s3 cp` to the bucket; set `BEEMONITOR_GOLDEN_IMAGE_URL` in App
   Runner prod env (+ local `.env`). It's a non-secret, public artifact.
5. **Hardware verify:** flash a clone, browser-enroll, confirm it appears and the
   token is cleared from `bootfs` after first boot.
6. **Merge** `devices/golden-image-provisioning`.

Detailed commands live in `hardware/provision/README.md` (don't duplicate here).

**Rebuild cadence:** rarely. Field units `git pull` + self-update on boot
(`update.sh`), so the image only needs to boot and self-update. Rebuild on a new
apt/pip system dep, a new large model, or an OS bump.

---

## 6. Deferred / future phases

- **Phase 2 — per-user server-baked images (Option #2):** a privileged Linux worker
  loop-mounts the golden image, injects the token into the boot partition, compresses,
  uploads; "Prepare a card" yields a personalized download. Adds turnkey flashing at
  the cost of a privileged worker + per-build CPU/storage. The Phase 1 boot-partition
  reader + `.conf` format are forward-compatible, so this layers on without rework.
- **Image/source security (deferred this round):** threat model discussed =
  "determined attacker," assets = credentials, recorder/CV logic, model weights, whole
  image. Honest conclusion: a Pi (no real TPM) can't fully protect on-device code from
  a determined holder. Best realistic stack = **encrypted rootfs + network-bound
  unlock** (key served by backend per `hw_id`, revocable — leaked/stolen card is
  ciphertext) + **no baked secrets** (already true: only the revocable token) + keep
  genuinely proprietary inference **server-side** (the only robust protection for the
  CV logic/weights, but it fights the cellular-cost model since on-device gating is
  what keeps upload bytes down). Decision pending; not started.

---

## 7. Risks / open questions

- **Chromium-only browser writer** — acceptable given the CLI + manual fallbacks, but
  worth a UI note steering field users to Chrome/Edge.
- **Token at rest on FAT** — the enrollment token sits on an easily-read partition
  until first boot clears it. Mitigated: it's per-user, revocable, and yields only one
  rotatable device key. Acceptable for Phase 1.
- **Golden-image drift** — relies on `update.sh` self-update; if a unit can't reach
  the network on first boots it runs older code until it can. Acceptable.
- **macOS shrink gap** — capture+shrink needs Linux; Edward is on macOS, so step 3
  needs a Linux box (EC2/second Pi/VM). Called out in the provision README.
