# 54 — Cloud storage (data sources), back, with "copy into BeeMonitor"

**Status:** steps 1–4 built 2026-10-22 (S3 only; GCS/Drive not offered). Decided: S3 only for now; copies follow the 3-month rule falling back to the bucket; no credits for copying. Design: canvas board "Cloud storage" (https://claude.ai/artifact/9kjvmu9SWhZQ47CzB2eAB1).

> Write under any **Feedback:** line. *Proposed* = a starting point.

## Why
Users with footage already in their own S3 (or Google) storage should connect it, pick clips, and either analyse them from there or copy them into their BeeMonitor storage.

## What exists (audit 2026-10-22)
- **Hidden, not removed** (`55ae780`, 2026-10-04: nav links and docs pulled; `cca206b` earlier). `apps/sources` (DataSource, add/list/browse/sync/delete pages), `/sources/` and `/api/v1/sources/` are still live; 66 tests pass.
- Credentials encrypted (Fernet, `apps/sources/crypto.py`, key in Secrets Manager).
- **Import** (`SourceSyncView`) makes pointer rows: `Video.storage_key = "s3://<key>"`, `source` FK, metadata `remote_bucket/remote_key`. The first analysis (or opening the annotation editor) copies the file into our `raw-videos` bucket (`analysis/views.py:60 _ingest_external_s3_to_storage`) — **inside a web request**, through local disk, then rewrites `storage_key`. Thumbnails, frames, streaming, retention skip `s3://` rows.
- **Only S3 works.** GCS and Drive connectors read the wrong fields, there's no listing or real connection test for them, and their libraries aren't installed.
- **Broken / unsafe today:** no Celery in production (the `ingest_from_source` task never runs); copies in requests can hit the 120 s limit and run twice at once (the editor starts a new copy on every page load); a source without keys falls back to **our** AWS role (the API skips field checks); secrets are written back into the form on a re-render and raw errors are shown; `test_connection` needs ListAllMyBuckets; duplicates are matched by key only (not bucket); `storage_key` drops the bucket; copied clips get no duration/thumbnail, so credit estimates assume 10 minutes; external bytes count as our storage.

## The plan
### 1. What a user does
- **Videos → Cloud storage** (nav link back, next to Sites and Upload): connect an S3 bucket (bucket, optional folder, read-only access key, region). The connection is tested with a real read of that bucket.
- **Browse** the bucket (folders, video files, size, already-imported marks), select clips (or a whole folder), choose a site (optional, so a layout applies), and pick:
  - **Keep in my bucket** — the clip appears in BeeMonitor; when it's analysed we copy it temporarily, and delete our copy after the run (results stay). No storage on our side.
  - **Copy into BeeMonitor** — a full copy in the user's BeeMonitor storage: plays, previews and re-runs without their bucket. Same rules as uploads (Standard: kept 3 months, then the copy goes; see §4).
- A clip can be switched later: "Copy into BeeMonitor" on its page.
- Copy progress per clip on the Cloud storage page and the clip page ("Copying… 1.2 of 3.4 GB", "Copied", "Failed: access denied").

### 2. Copying, safely
- **In the background**, from the reconcile loop (the platform's only clock; no Celery): a `SourceCopy` queue, a few at a time, claimed atomically so one copy per clip.
- **Streamed bucket-to-bucket** (S3 GetObject → multipart upload), no local disk, any size; resumable parts.
- After copying: duration, frame size and thumbnail are measured (ffmpeg over a presigned URL), so credits and frames work.
- **Never our AWS role:** a connector always uses the user's keys; missing keys = an error, not a fallback.

### 3. Fixes that come with it
- Key and bucket required per source type (web and API); `storage_key` keeps the bucket (`s3://bucket/key`); duplicates matched by bucket + key.
- Connection test = HeadBucket + a ListObjects of the folder (works with a read-only key).
- Secrets never re-rendered; errors shown as plain messages ("Access denied to that bucket"), details only in logs.
- The annotation editor stops starting a copy on every load (it queues one, once).
- Platform storage counts only bytes in our buckets; linked bytes shown separately.
- Content type from the file extension (.mov, .avi, .mkv).

### 4. Copies and the 3-month rule (Standard)
- *Proposed:* copied footage follows the same rule as uploads, **but nothing is lost**: when our copy is deleted, the clip goes back to **Keep in my bucket** (it still plays from their bucket at analysis time) as long as the source is connected. The notice emails say so.

### 5. Google Cloud Storage and Google Drive
- *Proposed:* **S3 first**; GCS next (service-account JSON, same flow); Drive later (needs OAuth sign-in to do properly). Until then they're not offered.

## Build order
1. Source fixes (§3) + the nav link + docs page back.
2. Browse + import with the two choices; site on import.
3. Background streamed copy + measuring + status; temporary copy for "keep in my bucket" runs.
4. Retention fallback to the bucket (§4); storage accounting.
5. GCS.

## Built
- `apps/sources/s3.py`: every bucket access on the user's keys only (missing keys = SourceError, never the instance role), friendly errors, HeadBucket + ListObjects test, folder browsing, presign, streamed multipart copy (no local disk; parts in-thread so progress writes use one DB connection).
- `SourceCopy` (keep / temp / measure) + `apps/sources/copying.py`, ticked by the reconcile loop: 2 at a time, claimed atomically, stale (>1 h) requeued, temp copies removed after a day. `measure` = ffprobe + thumbnail over a URL (`videos/frames.py` `readable_url`, `probe`).
- Import stores `s3://bucket/key` + metadata `remote_bucket/remote_key`; dedupe by bucket+key; keys outside the source folder ignored; max 2000 per import; optional site.
- GPU spawn / pre-annotate: `_ingest_external_s3_to_storage(video, tag)` = temporary copy, clip stays linked. Annotation editor / Transfer button queue one KEEP copy (no more copy per page load).
- Linked clips play and give frames from the user's bucket (presigned on their key). Clip page: "Stored in your bucket · Copy into BeeMonitor".
- Retention: a copied clip whose source still exists goes back to `s3://…` instead of `raw_deleted_at`.
- Platform storage excludes `s3://` rows. API: S3 only, keys required, `test_connection` via `s3.test`.
- Pages: Cloud storage list (copies + progress), Connect (secret never re-rendered), Browse & import. Link on Processing. Docs: platform/videos.md "From your own cloud storage".

**Feedback:**
