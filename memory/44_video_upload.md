# 44 · Video upload as a first-class feature

Status: **plan** (2026-10-06) — awaiting decisions below, then a design canvas.

## Ask
"We should have a real video upload feature on the platform" — clips from any
camera, not only BeeMonitor units (today the docs can only say "through the API").

## Audit — what exists
- `videos/upload/` (one file) and `videos/batch-upload/` (many; drag-and-drop,
  per-file progress). Browser → presigned PUT straight to S3 → `web-uploads/
  complete` creates the Video (`apps/api/web_uploads.py`); bytes never touch
  Django. `.mp4/.mov/.mkv/.h264`, **5 GiB single-PUT cap**.
- Optional **device** (clip shows on the device page, inherits its location
  and layout) or free-text **site name**.
- Recording time: only parsed from a `site_YYYY-MM-DD_HH_MM_SS` filename, else
  unknown (`Video.resolve_recorded_at`) — event timestamps fall back to the
  upload time.
- Only entry point: the empty-state "Upload a video" on Processing. No nav,
  no button once a user has clips (removed earlier).

## Gaps for a "real" feature
1. **Findable**: an Upload button in Processing's header (and the empty state).
2. **When was it recorded**: set a start time per batch / per file, or read it
   from the file (MP4/MOV `creation_time`, via ffprobe after upload). Event
   timestamps, hour-of-day filters and BioCLIP's month prior all depend on it.
3. **Where**: a *site* with a location (lat/lon, picked on a map or typed),
   reusable across uploads. Today a site is just a name, so uploaded clips get
   no location → BioCLIP falls back to the whole Tree of Life.
4. **Big files**: multipart upload (S3 parts, resumable) instead of one PUT,
   lifting the 5 GiB cap and surviving a dropped connection mid-file.
5. **More formats**: `.avi` is common on trail/camera-trap cameras; transcode
   on arrival when the GPU worker can't read a container.
6. **Then what**: after upload, "Run a pipeline on these" (pre-selects the new
   clips in Processing), and a per-batch label to find them again.
7. **Duplicates**: warn when the same file (name + size, or hash) was uploaded.

## Proposed scope (v1)
1–3, 6 and 7. 4 (multipart) if files over 5 GiB are expected; 5 later.

## Open questions
1. Files over 5 GiB — do you expect them (long continuous recordings)?
2. Recording time — set per batch, read from the file, or both (file first)?
3. Sites with a map location, reusable — yes, or keep free-text names?
