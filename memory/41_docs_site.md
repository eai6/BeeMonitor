# 41 · Public documentation site (like Insect Detect)

Status: **v1 built** (2026-10-04) — https://eai6.github.io/BeeMonitor/
Design: https://claude.ai/artifact/Bvu8HPUGBk2ibN5c1mY5x7
Reference: https://maxsitt.github.io/insect-detect-docs/ (MkDocs Material,
GitHub Pages, source in its own repo; CC BY-SA 4.0 docs, GPL/AGPL code).

## Goal

One place where someone can (1) understand BeeMonitor, (2) build a unit
themselves — parts, enclosure, assembly, software, field deployment — and
(3) use the platform: devices, recording, photos, analysis pipelines,
annotation and training.

## Tool and hosting

- **MkDocs Material** (same as Insect Detect): Markdown pages, search, dark
  mode, tabs, admonitions, image lightbox, mermaid diagrams. No server.
- Source: `docs/` + `mkdocs.yml` in this repo (it is public, AGPLv3), so a
  hardware or UI change and its docs land in the same commit.
- Deploy: a GitHub Actions workflow builds on push to `main` (docs paths only)
  and publishes to **GitHub Pages** — `https://eai6.github.io/BeeMonitor/`, or a
  custom domain (e.g. `docs.beemonitor.edwardamoah.com`, one CNAME record).
- Licence: docs CC BY-SA 4.0 (proposed, matching Insect Detect); code stays
  AGPLv3.

## Sources already in the repo

- `hardware/README.md` (1,476 lines): BOM, enclosure, assembly, software
  install, services, motion-gated recording, configuration reference, WittyPi,
  Raspberry Pi Connect, Sixfab cellular, telemetry.
- `hardware/SETUP_GUIDE.generated.md`, `UPLOADER.md`, `VERIFY.md`,
  `hardware/oak/README.md`, `hardware/provision/README.md`.
- `hardware/enclosure/*.stl` (body, lid, tripod connector, power-cable
  connector); `hardware/*.png` (architecture, energy and recording modules);
  `hardware/Bill_of_Parts.xlsx`.
- `README.md`: algorithm details, performance, citation, licence.

These are written for us; the site rewrites them for an outside builder (task
order, photos, "you should now see…" checks), linking the long references.

**Not published:** `memory/` (internal plans), AWS account ids, bucket names,
device keys, enrollment tokens, internal hostnames. A build check fails on
known secret patterns.

## Site map

1. **Home** — what it is (hero photo of a unit on a hotel), how it works
   (device → cloud → analysis), key numbers, cards: Build one · Use the
   platform · Analyse your data · Cite.
2. **Introduction** — background (solitary bees, nest hotels), what a unit
   records, data you get (clips, photos, visits, foraging trips, species).
3. **Hardware**
   - Overview & choosing a configuration (camera: Pi Camera 3 / Arducam 64 MP
     OwlSight / Luxonis OAK-1-AF; power: mains / battery + solar with WittyPi;
     connectivity: WiFi / Sixfab 4G). Pi 4 2 GB+ recommended; 1 GB limits.
   - Components (BOM with links, prices, alternatives) + downloadable sheet.
   - Enclosure (STL downloads, print settings, hardware).
   - Assembly (numbered steps, one photo each).
   - Field deployment (mounting, aiming at the hotel, weatherproofing, power budget).
4. **Software (device)**
   - Flash the image & enrol the device (enrollment token from the dashboard).
   - WiFi / cellular, updates, Raspberry Pi Connect.
   - Verify a unit (VERIFY.md), services, troubleshooting, config reference.
5. **Using the platform**
   - Account & devices (add, share, the device page).
   - Camera: picture, ROI & reference objects, Autofocus, recording settings,
     full-resolution photos, freeing device space.
   - Videos & photos: Processing hub, a clip's page.
   - Pipelines: builder, running on clips, schedules, results & exports.
   - Annotation: projects, adding clips, sample & label (GPU), review,
     assigning frames, export.
   - Training custom models; Browse / publishing datasets.
6. **Methods** — motion gating, detection & tracking, nest entry/exit events,
   foraging trips, species identification, accuracy (README's numbers).
7. **FAQ & troubleshooting** · **Contributing** · **Licence & citation**.

## Photos and screenshots

The site needs: unit photos (assembled, open, mounted), one photo per assembly
step, and dashboard screenshots. The STLs can be rendered to images; the
dashboard screenshots I can't capture (no browser here) — the user supplies
them, or we capture with a headless browser against a demo account later.
Pages ship with clearly marked placeholders until then.

## Build order

1. Scaffold: mkdocs.yml (Material, green theme to match the platform), the
   site map with stub pages, the Pages workflow, secret check. Deploy.
2. Hardware + Software (device) pages from hardware/*.md (largest value for
   builders).
3. Using the platform pages (needs screenshots).
4. Home, Introduction, Methods, FAQ, licence/citation polish.

## Decisions (2026-10-04)

- GitHub Pages for now (`eai6.github.io/BeeMonitor`); a custom domain later.
- Licence: AGPLv3 for code, hardware and docs (Ultralytics YOLO is AGPL).
- Audience: people using the hosted platform (not self-hosting the cloud).
- Photos/screenshots: the user generates them as needed; pages carry
  `<!-- PHOTO: ... -->` / `<!-- SCREENSHOT: ... -->` markers where they go.
- Keep it concise, like Insect Detect: 14 short pages.

## Open questions (answered above)

1. Address: `eai6.github.io/BeeMonitor` or a custom domain?
2. Docs licence CC BY-SA 4.0?
3. Audience for the cloud: people who use *your* hosted platform (the plan
   above), or also people who self-host the whole stack on their own AWS
   (much bigger: Pulumi, SageMaker, costs)?
4. Photos: do you have unit and assembly photos to use?

## v2 (2026-10-04): general architecture, not just hotels

The site now presents BeeMonitor as source → detect → track → reference → two
tables (events, interactions). Per the user: no per-application pages —
foraging trips, time on a flower, assay zones are *reads* the scientist does
over events/interactions, shown as one table + pandas on concepts/results.md.

- New: Concepts (architecture, pipelines & steps, events & interactions,
  glossary); Platform: Getting video in (upload, S3/GCS/Drive), Sharing &
  publishing, API (pipeline API only — webhooks have a model/CRUD but nothing
  dispatches them, so they are not documented); Open source (repo map,
  contributing). Self-hosting: architecture only, no guide (decision).
- In-app /docs/ now redirects to the site; its stale template is deleted.
- Nav links Sources and API (developer page); Lessons were removed (2026-10-04). The Colab export
  (/pipelines/<id>/colab/) still has no button.
- Pollen assay: no protocol yet — mentioned only as a reference-zones example.

## v3 (2026-10-04): fewer pages, one menu

User found the site "a pain to navigate" (top tabs + left nav + right TOC,
23 short pages). Wanted: TWO menus — top tabs + right TOC, no left sidebar
(hidden on desktop via docs/stylesheets/extra.css; kept as the phone drawer).
Each section tab opens an overview page with cards to its pages. 13 pages — Home, Get started, Build a unit (Build · Set up & deploy ·
Troubleshooting), Use the platform (Devices · Videos · Pipelines · Results ·
Annotation & sharing · API), About (open source, contributing, methods,
glossary, license). Keep it this flat; add sections to a page before adding
a page.
