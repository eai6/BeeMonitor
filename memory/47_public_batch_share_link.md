# 47 · Public share link for a batch's results

Status: **built** (2026-10-08) — canvas https://claude.ai/artifact/WXRt3RR9msQL69HhCtuchr

## Ask
"Publicly share a batch run's results with collaborators online, so they can
review without logging in."

## Audit
- Every HTML view was `login_required`; no view served data anonymously. The
  existing "public" things (annotation projects, models) still need an account.
- The batch page carried owner-only and private material: re-run forms (video
  pks, the viewer's pipelines), GPU time, raw error text (`g.sample`, row
  titles), device site/location in the header, and site/location columns in
  every combined CSV (`aggregate.PROVENANCE_FIELDS`).
- Media were already lazy endpoints that 302 to a presigned S3 URL (24 h
  default); the overlay endpoint builds and stores its payload on first use.

## Design
- `BatchShare(batch_id, token, created_by, revoked_at, show_videos,
  show_locations, view_count, last_viewed_at)`; one live share per batch
  (partial unique constraint). Token `secrets.token_urlsafe(16)`, kept in the
  clear so the owner can copy it again. Off = `revoked_at`; on again = a NEW
  token, so a passed-around link stays dead. No expiry: on until turned off.
- Only the launcher (every run in the batch is theirs) sees the Share panel
  and may POST `pipelines:batch_share` (on / save options / off).
- Public routes under `/s/<token>/` (`apps/pipelines/public_urls.py`): page,
  `data/<kind>.csv`, `clip/<video>/video`, `clip/<video>/still`,
  `tracks/<job>.json`. Each resolves the token, then checks the clip/job is one
  of that batch's runs (launched by the sharer) — nothing else is reachable.
- Media presigned for 1 h (`sharing.PUBLIC_MEDIA_HOURS`); responses carry
  `X-Robots-Tag: noindex` and `Referrer-Policy: no-referrer` (token in URL).
- Page (`batch_public.html` on `templates/public/base.html`, no app nav):
  summary, contact time per reference (from the interaction analyzers' saved
  `per_reference`, no S3 reads), CSV downloads, clip list with the same inline
  player + track overlay (`_clip_viewer.html`, `static/js/clip_viewer.js`,
  shared with the signed-in batch page). Failed clips read "didn't finish".
- Defaults: videos on, locations off. Locations off also drops `site_name`
  and `location` from public CSVs.
- Refactors: `videos.views.stream_redirect/thumbnail_redirect`,
  `analysis.views.overlay_response`, `sharing.batch_csv` (signed-in batch CSV
  view uses it too).

## Not in v1
- The per-clip "follow a track" page with crops. Photo batches show the list
  and the photos CSV, without the photo viewer.
