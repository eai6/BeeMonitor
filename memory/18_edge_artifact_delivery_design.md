# BeeMonitor — Edge Artifact Delivery (keep cloud code off field devices)

**Status:** **Phase 0 LIVE + verified end-to-end (2026-06-17, tag `v0.1.0`)** — CI
built + minisigned + published the bundle/manifest/`latest.json` to
`s3://beemonitor-dev-models-…/edge/`; signature verifies against the committed
`hardware/provision/minisign.pub`, sha256 matches. Keypair + GH secret + IAM
(`pulumi up --target` the RolePolicy only — left the unrelated App Runner
`SAGEMAKER_ENDPOINT_NAME=""` drift alone) all done. Earlier Phase 0 build — `hardware/provision/build-edge-artifact.sh`
(hardware-only tar + no-path-outside-hardware/ guardrail + sha256/manifest + optional
minisign), `.github/workflows/edge-artifact.yml` (build+guardrail on push; sign+S3
publish on `v*` tags), and a CI S3 `edge-publish` IAM policy in `infra/aws/__main__.py`.
**To activate:** generate the minisign keypair (`minisign -G -W`; private → GH secret
`MINISIGN_SECRET_KEY`, public → `hardware/provision/minisign.pub`), `pulumi up` the IAM
change, then tag a release. **Phase 1 (A) plumbing IMPLEMENTED on `main` (backward-compatible, 2026-06-17)** —
`update.sh` `fetch-artifact` (download→sha256→minisign→unpack→symlink-swap; dual-mode
phase B rollback; git path byte-identical), telemetry routing (`_start_update`
git-ref vs descriptor) + version reporting from the manifest, cloud
`DeviceUpdateView` `mode=artifact` (`_resolve_edge_descriptor` reads
edge/latest.json + manifest, presigns the bundle), and a dashboard "Update via
signed artifact" button. Verify path tested locally vs the real v0.1.0 bundle.
**NOT yet validated on a Pi** (symlink swap + rollback + tamper-reject over
cellular) — needs a unit in the release-layout. **Phase 2 mechanism IMPLEMENTED
(2026-06-17):** `migrate-to-releases.sh` converts a git-clone unit to the
release/symlink layout (stable venv/models symlinked into each release; update.sh
recreates the symlinks per release, so unit files / config.py paths are unchanged)
+ installs minisign + pubkey; `generalize.sh` drops the leftover git clone so the
image is git/cloud-free; README §1.1b documents the artifact-native build. **All
code for Phases 0–2 is on `main`; only hands-on remains: run migrate + the
artifact update on a test Pi, then rebuild the golden image with §1.1b.**
**Author:** Drafted with Claude Code, 2026-06-14

> **Locked decisions (2026-06-17):**
> - **Go straight to artifact delivery — no interim git-creds bridge.** The
>   generalized golden image has no git auth and the repo is private, so current
>   field units can't `git fetch` to self-update; the accepted answer is to
>   **re-flash to push changes** until the artifact update path (Phase 1) ships,
>   rather than baking a deploy key. (Confirmed: nothing under `hardware/` imports
>   cloud/web/src; only cross-tree dep is the 2 `.pt` models.)
> - **Sign the artifact with minisign.** Private key in CI secrets; public verify
>   key committed at `hardware/provision/minisign.pub` and baked into the golden
>   image. Device rejects any unsigned/tampered bundle (resolves §2 ⚙).
> - **Delivery channel = presigned S3 in the heartbeat update command** (resolves
>   §2 ⚙), reusing `presigned_get(container="models")` + the `DeviceKeyAuth` path.
> - **Stable paths out of the release tree:** venv → `/home/beemonitor/beemonitor-venv/`
>   (update the 4 `ExecStart` interpreter paths in
>   `beemonitor-{telemetry,recorder,uploader,calibrate}.service`); models →
>   `/home/beemonitor/models/` via `BEEMONITOR_MODELS_DIR` (the bundle is
>   `hardware/`-only, so `config.py`'s `parents[2]/models` won't find them in a
>   release dir). `REPO_DIR` resolution in both `update.sh` and `telemetry.py` is
>   symlink-transparent — verified — so all other hardcoded paths stay valid.
> - **Cloud seam:** `DeviceUpdateView` (`apps/devices/views.py`) emits
>   `command_params={version,url,sha256,sig}` instead of `{"ref": …}`; `code_commit`
>   metric → `version`.
> - **Sequencing:** Part A (Zero 2 W lite profile, [[20_pi_zero2w_lite_profile]])
>   and **Phase 0** (CI build+sign+publish, zero device risk) first; **Phase 1**
>   (the on-device lifeline rewrite) only with a Pi to test over real cellular,
>   then Phase 2/3.
**Goal:** Stop shipping the **whole monorepo** to field devices. Keep one repo for
development, but deliver only a **signed, hardware-only artifact** to the Pi — so a
field unit never holds `cloud/`, `beemonitor_web/`, `src/`, `sagemaker_backend/`,
`infra/`, a git remote, or any history. This protects the proprietary cloud/web
code from leaking off a device without going through the right license process.

Related: [[14_golden_image_provisioning_design]] (the artifact is the golden
image's payload), [[10_cellular_telemetry_design]] (the cellular-safe update
path + heartbeat command this rewrites), [[16_remote_scheduling_design]] /
[[12_device_dashboard_telemetry_v2]] (same command/heartbeat plumbing),
[[17_bee_confirmation_design]] (new edge code that ships in this artifact), and
the on-Pi reality in `claude-memory/pi-torch-must-be-cpu-wheel.md`.

---

## 1. Problem & framing

Today every field Pi **clones the whole repo** to `~/BeeMonitor` and
`hardware/update.sh` runs `git fetch origin` + `git reset --hard` against the full
GitHub remote. So each device physically holds the entire cloud/web/source tree
**and** a credentialed remote that can pull all history. A stolen, serviced, or
collaborator-handled unit exposes everything — the exact license/IP risk we want
gone.

Survey result that makes this cheap to fix: **`hardware/` is already
import-independent.** Nothing under `hardware/` imports `cloud/`,
`beemonitor_web/`, `src/`, `sagemaker_backend/`, or `infra/` (verified). Its only
cross-repo dependency is **`models/`**, and only **two** files —
`nest_detection.pt` (40 MB) + `bee_tracking.pt` (5.4 MB); `event_classifier_model.pkl`
is cloud-only. So a hardware-only payload is a clean cut.

**Why artifact delivery, not a second repo or sparse-checkout.** A `.gitignore`
or git sparse-checkout does **not** help — sparse-checkout still downloads every
object into `.git`, and the remote stays. Only removing git-from-the-device
(artifact delivery) or a separate repo truly keeps cloud code off the Pi. Artifact
delivery was chosen because it keeps **one development repo** (Edward's
preference) while giving the **strongest** boundary — no source, no `.git`, no
remote, no history on the device — plus a security + rollback upgrade (below).

---

## 2. Locked decisions (recommended; Edward to confirm the ⚙ ones)

- **One monorepo for dev; the device gets a built artifact only.** No git, no
  remote, no `.git`, no cloud code on a field unit.
- **The artifact is `hardware/`-only**, built in **CI** (never locally —
  [[feedback_docker_in_ci_only]] convention), published via the existing
  GitHub-Actions **OIDC→AWS** path.
- **⚙ Delivery channel → presigned S3 via the existing heartbeat + `DeviceKeyAuth`.**
  No new secret on the device, reuses infra already running. (GitHub Releases is
  the fallback, but it needs a token or a public repo on the device.)
- **⚙ Sign the artifact** (e.g. minisign/cosign); bake the **public verify key into
  the golden image**. A compromised URL/bucket then can't push code to the fleet —
  a real upgrade over plain `git` today. (Checksum-only is the lighter fallback.)
- **Atomic releases via a symlink** (`~/BeeMonitor` → `releases/<version>/`), last
  N kept for instant rollback. Keeps every hardcoded device path identical.
- **Models stay a separate, WiFi-pulled artifact** (or golden-image-baked) — the
  code bundle stays tiny and cellular-cheap, matching today's "torch/models out of
  scope for cellular updates" stance.

---

## 3. How it fits the existing code (seams)

- **`hardware/update.sh`** — already a careful **two-phase, cellular-safe,
  auto-rollback** script: phase A (`cmd_fetch`) runs in telemetry's cgroup (network
  allowed) and does all network work; phase B (`cmd_apply`,
  `beemonitor-update.service`, root, offline) restarts + health-checks + rolls back.
  We keep this exact shape and swap only the *transport* (git→download) and the
  *swap mechanism* (`git reset`→symlink flip).
- **`hardware/telemetry.py`** — owns `REPO_DIR`, `UPDATE_SCRIPT`, and the heartbeat
  **"update" command** (today `params.ref` = git ref). It becomes an artifact
  descriptor `{version, url, sha256, sig}`. `git rev-parse HEAD` version reporting
  becomes "read `version` from the active manifest".
- **systemd units** (`hardware/systemd/*`) + **sudoers** (`provision/sudoers.d/*`)
  hardcode `/home/beemonitor/BeeMonitor/hardware/...`. The symlink keeps these
  **byte-for-byte valid** (one venv-path change — see §6).
- **Golden image** ([[14_golden_image_provisioning_design]]) — the baked payload
  *becomes* artifact v0 + the verify key + the new `releases/`/symlink layout.
- **Cloud** — reuses `DeviceKeyAuth`, S3, and the OIDC build path already in
  `infra/`. A new tiny "latest edge manifest" the cloud serves to devices.

---

## 4. Architecture / data flow

```
DEV (monorepo, private)                         FIELD Pi (no git, no cloud code)
───────────────────────                         ────────────────────────────────
push tag vX ─▶ GitHub Actions (OIDC→AWS)         heartbeat ─▶ cloud
  build bundle = hardware/ ONLY                    cloud replies: update cmd
  + manifest{version, src_commit,                    {version,url(presigned),sha256,sig}
     sha256, sig, reqs_hash}                       │
  sign (key in CI secrets)                         ▼  update.sh phase A (telemetry cgroup, net)
  assert: no path outside hardware/  ──S3──▶  ◀──── download bundle+manifest
  publish bundle + manifest + latest.json           verify sha256 + signature (key from golden img)
                                                     unpack ─▶ releases/<version>/
                                                     pip install iff reqs_hash changed
                                                   │  handoff
                                                   ▼  update.sh phase B (own unit, offline)
                                                     flip symlink ~/BeeMonitor ─▶ releases/<version>
                                                     restart + health-check
                                                     unhealthy ─▶ flip back (instant rollback)
```

---

## 5. The artifact

- **Code bundle** `beemonitor-edge-<version>.tar.gz` = the `hardware/` tree only
  (a few hundred KB → cheap over cellular).
- **Manifest** `beemonitor-edge-<version>.json`:
  ```json
  {
    "version": "2026.06.14-a1b2c3",
    "src_commit": "<monorepo git sha>",      // provenance: artifact → source
    "sha256": "<bundle hash>",
    "sig": "<detached signature of the bundle>",
    "reqs_hash": "<sha256 of hardware/requirements.txt>",
    "models": {"min_version": "...", "url": "<wifi-only>"}  // optional
  }
  ```
- **`latest.json`** the cloud reads to decide what a device should run (channel →
  version). The cloud puts the resolved descriptor into the heartbeat update cmd.
- **Models artifact** `beemonitor-models-<mver>.tar.gz` (the 2 `.pt` files), pulled
  **over WiFi** or baked into the golden image; versioned independently of code.

---

## 6. On-device layout & the symlink swap

```
/home/beemonitor/
  BeeMonitor            -> releases/2026.06.14-a1b2c3      (atomic-swap symlink)
  releases/
    2026.06.14-a1b2c3/  hardware/ ...  models/ (2 .pt)
    2026.06.10-9f8e7d/  ...                              (kept N deep for rollback)
  beemonitor-venv/      (STABLE, shared across releases — NOT inside a release)
  .beemonitor/          (state: update-status.json, update-request.json — unchanged)
```

- Every existing absolute path — systemd `ExecStart=.../BeeMonitor/hardware/...`,
  sudoers, `config.py` `parents[2]/"models"`, `telemetry.REPO_DIR` — resolves
  through the symlink **unchanged**.
- **One real change: the venv moves to a stable path** outside release dirs
  (torch ≈ GB; a per-release venv copy is too expensive). systemd `ExecStart`'s
  *interpreter* becomes the fixed `…/beemonitor-venv/bin/python`, while the
  *script* path stays `…/BeeMonitor/hardware/…` (symlinked). pip deltas apply to
  the shared venv on `reqs_hash` change. (Caveat, same as today's git flow: a
  rollback *after* a requirements change leaves the newer packages installed —
  acceptable with `--upgrade-strategy only-if-needed`.)
- **Rollback = re-point the symlink** to the previous release dir + restart. Atomic,
  instant, no half-state — strictly better than `git reset --hard`.

---

## 7. Update flow (rewritten `update.sh`, same two-phase skeleton)

**Phase A — `cmd_fetch`** (spawned by telemetry, in its cgroup, network OK):
1. Read descriptor `{version,url,sha256,sig}` from the request.
2. If `version` == active manifest version → `idle/uptodate`, exit (unchanged short-circuit).
3. Download bundle + manifest to a temp dir.
4. **Verify** sha256, then **signature** against the baked public key. Fail →
   `error/verify_failed`, discard, exit (never unpack an unverified blob).
5. Unpack into `releases/<version>/`.
6. If `reqs_hash` != active → `pip install --upgrade-strategy only-if-needed` into
   the shared venv; failure → drop the new release dir, exit (no symlink change).
7. Write `update-request.json {prev_version, target_version}`, hand off to phase B.

**Phase B — `cmd_apply`** (`beemonitor-update.service`, root, offline):
1. Flip `~/BeeMonitor` symlink → `releases/<target>`.
2. Restart `RESTART_UNITS`; health-check `HEALTH_UNITS` for `HEALTH_WAIT`.
3. Unhealthy → flip symlink back to `prev`, restart, `error/rolled_back`.
4. Healthy → `idle/ok`; prune `releases/` beyond the last N.

`write_status`/`update-status.json`/the dashboard beat are unchanged — only the
`commit` field becomes `version`.

---

## 8. CI build / publish / sign

- Trigger: tag/release on the monorepo. GitHub Actions, **OIDC→AWS** (existing
  pattern; no static AWS keys — [[reference_aws_profile]]).
- Steps: `tar` **an explicit allowlist** (`hardware/` only) → compute sha256 →
  sign with the CI-held private key → write manifest → upload bundle + manifest +
  update `latest.json` to S3.
- **Guardrail (the IP guarantee made enforceable):** a CI assertion that the
  bundle contains **no path outside `hardware/`** — build fails otherwise. This
  turns "trust the build" into a tested check.
- Secrets: signing private key + bucket in CI/AWS only, never in the repo
  ([[feedback_no_secrets_in_commits]], [[feedback_no_account_ids_in_code]]).

---

## 9. Cloud side (small)

- Store/serve `latest.json` per release channel (stable/beta); resolve the
  device's target and embed `{version,url(presigned),sha256,sig}` in the heartbeat
  **update command** (reuses the existing command path + `DeviceKeyAuth`).
- Presign S3 GETs for the bundle/models (short TTL), same as existing image
  handling.
- Dashboard: show device `version` + update status (already plumbed; rename
  `commit`→`version`).

---

## 10. Migration plan (phased, each independently shippable)

- **Phase 0 — build pipeline, no device change.** CI builds + signs + publishes the
  `hardware/`-only bundle/manifest to S3 on tag. Add the "no path outside
  `hardware/`" guardrail. *Validate:* artifact downloads, verifies, unpacks on a
  laptop; signature checks; provenance (`src_commit`) resolves.
- **Phase 1 — device update path.** Rewrite `update.sh` (download+verify+symlink
  swap+rollback), move venv to the stable path, point systemd interpreter at it,
  add the verify key. Cloud emits the artifact descriptor in the update cmd.
  *Validate on one Pi over real cellular:* update, rollback-on-failure, signature
  rejection of a tampered bundle.
  > **⚠ Don't push the rewritten `update.sh` to `main` while the live fleet still
  > updates by `git pull` from `main`** — the next pull would swap in an update path
  > that expects an artifact descriptor (not a git ref) on a git-clone layout (not
  > releases/symlink), breaking self-update fleet-wide. Do Phase 1 EITHER (a)
  > backward-compatible — new `update.sh` detects git-ref vs descriptor AND
  > git-clone vs symlink layout and handles both, so it can land on `main` safely;
  > OR (b) on a branch, tested on one Pi, merged only at the Phase-2 golden-image
  > cutover. Decide before starting Phase 1.
- **Phase 2 — golden image cutover.** Bake artifact v0 + verify key + `releases/`/
  symlink layout into the golden image ([[14_golden_image_provisioning_design]]).
  New devices have **no git, no remote, no cloud code** from first boot.
- **Phase 3 — decommission git-on-device.** Remove the old git remote/clone from
  the provisioning path; models become the separate WiFi artifact. Fleet now runs
  artifact-only.

---

## 11. Risks / open questions

- **Critical-path rewrite.** `update.sh` is the field lifeline; the two-phase +
  rollback logic is preserved but the transport/swap is new → must be tested on a
  real Pi over real cellular (incl. mid-download power loss, partial unpack) before
  fleet rollout. The symlink swap makes failures recoverable (old release intact).
- **Build correctness = the IP guarantee.** A glob bug could include cloud files;
  the CI allowlist + "no path outside `hardware/`" assertion is the enforced
  mitigation (and is itself testable).
- **Signing key management.** Where the private key lives (CI secret / KMS), how
  it's rotated, and key-pinning in the golden image. ⚙ Edward to set the key
  custody policy.
- **venv rollback skew** after a requirements change (§6) — bounded by
  `only-if-needed`; acceptable and identical to today.
- **First-boot / offline-new-device** must ship a working v0 in the golden image
  (no network assumed at first boot) — already the image's job.
- **Channels** (stable/beta/pinned-per-device) — start with one `stable` channel;
  per-device pinning can come later via the cloud descriptor.
- **Models cadence** — large; keep on WiFi/golden-image, versioned separately, so a
  retrain doesn't bloat the cellular code path.

---

## 12. Relationship to other work

Rewrites the update path from [[10_cellular_telemetry_design]] and becomes the
payload mechanism for [[14_golden_image_provisioning_design]]; ships the edge code
including [[17_bee_confirmation_design]] and the `motion/` package refactor; reuses
the device-auth + S3 + OIDC patterns already in the repo. The single biggest
non-code dependency, `models/`, is handled as its own WiFi-pulled artifact so the
cellular code path stays tiny.
