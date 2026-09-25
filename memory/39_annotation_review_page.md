# 39 · Annotation project page: review first

Status: **plan, awaiting design approval** (2026-09-25)
Design canvas: https://claude.ai/artifact/QL6XaWmJzaF5cAzVwFfPfY
Source request: Google Doc "Annotation Review Page" (screenshots of Class 597).

## Why

Sampling and pre-labelling are now one GPU pass (memory/38). What people do
on a project is **review pre-labelled frames**, not annotate clips. The page
still reads as a clip pipeline: two rows of overlapping metrics, a stage strip,
Sample + Auto-label + Assign-clips toolbar, and a clip table. Assignment is per
clip, but reviewers work on frames.

## What changes (per the doc)

| Today | After |
|---|---|
| 4 cards (Videos, Annotated frames, Total boxes, Class breakdown) + stage strip (Clips, Sampled, Labelled, Reviewed, Review N, Export) | **3 metrics: Clips · Labelled frames · Reviewed frames** (boxes count as a sub-line) |
| Header: Add Videos · People · Settings · **Back** | Add Videos · People · Settings · **Export ⤓**; "← All projects" above the title (device-page pattern, `devices/detail.html:5-7`) |
| Body = clip table | Body = **Frames to review** tab: grid of pre-labelled frames (default); **Clips** tab keeps the clip table |
| Sample all · Auto-label all (GPU) · Assign all · Annotate | Clips tab: one **Sample & label** button. Auto-label is removed. |
| Assign clips to people | **Assign N frames** (100/250/500/1,000/custom) to a reviewer, taken from the current frame filter |
| Annotator sees "Your work: N clips" | Reviewer sees **Your review queue: 380 left · 120 of 500** + Continue reviewing |

## Audit findings that shape the build

- `ProjectDetailView` (views.py:232-594) already builds both the detail page and
  the `/review/` page (`review=True` → `review.html`). The frame grid you liked
  is `review.html:83-117`, **not** `/annotations/browse/` (published datasets) and
  not `videos/_review_grid.html` (clip cards).
- A "frame" is an `Annotation` row (project, video, frame_number, boxes,
  reviewed, review_source, reviewed_at, sampled_only). There is no `reviewed_by`
  and no index on `reviewed`.
- Scale problems at 7k–100k frames, to fix while moving the grid:
  - `review.html`'s view loops every matching annotation in Python for class
    filter + box counts (views.py:521);
  - the class breakdown loops every annotation's JSON (views.py:493-504);
  - page size is 500 and thumbnails are presigned per card;
  - editor prev/next loads every frame key and uses `list.index` (views.py:1317-1337).
- Assignment is `ClipAssignment` (models.py:215, unique project+video) with
  `assignments.py` services, `AssignClipsView`, `ClaimClipsView`,
  `may_annotate_video` (403 in `SaveAnnotationView`), `_editor_landing`,
  People page workloads, the "Your work" banner. No API, email or notification
  touches it.
- Auto-label: `PreAnnotateAllView`, `preannotate_pick.py`, detail.html:522-604,
  "Retry these N" (detail.html:226). Keep `PreAnnotateView`/`PreAnnotationTask`
  and the reconciler's `poll_preannotation_tasks`.
- Export is `ExportProjectView` (GET, viewers allowed); just a link move.
- Existing bug: review.html "Clear" links go to detail (review.html:72, :124).

## Data model

Per-frame assignment lives on the frame (one reviewer per frame):

- `Annotation.assigned_to` FK user (null), `assigned_by` FK user (null),
  `assigned_at`; `reviewed_by` FK user (null).
- Indexes: (project, reviewed, assigned_to), (project, assigned_to).
- Migration `0012`: add fields; data migration expands existing
  `ClipAssignment` rows to their clips' unreviewed labelled frames (Class 597
  has none today). `ClipAssignment` stays read-only for one release, then is dropped.

Chosen over a separate `FrameAssignment` table: one row per frame already
exists, the grid filters on it, and a bulk `UPDATE … WHERE id IN (…)` assigns
500 frames in one query.

## Assign N frames

`AssignFramesView` (manager+), POST: `reviewer`, `count`, `order`
(spread|clips), and the grid's current filter (device, class, date).

- Pool: labelled, not reviewed, `assigned_to IS NULL`, matching the filter.
- **Spread** (default): round-robin across clips ordered by recorded time, so
  500 frames come from many clips, devices and days (a few per clip).
- **Whole clips**: oldest clip first, all its pool frames, until N.
- One `UPDATE` with a conditional on `assigned_to IS NULL` so two managers can't
  double-assign; the response says how many were actually assigned.
- "Take frames back": unassign that person's **unreviewed** frames.
- Removing someone from the project (ShareUpdateView) frees their unreviewed frames.
- Annotators can "Take 100" themselves from the unassigned pool (replaces ClaimClipsView).

## Review flow

- Grid: DB-level filters (status, assigned, device, class, date), 60 per page,
  keyset or Paginator on (video.recorded_at, frame_number); box count from
  `jsonb_array_length` (Postgres) with a Python fallback for SQLite tests;
  class filter via `boxes__contains` on Postgres, Python fallback for SQLite.
- "Continue reviewing →" / "Review these →" open the editor on the first
  unreviewed frame of the queue (mine, or the current filter). Next/prev walk
  that queue by query, not by loading every key.
- Saving marks `reviewed=True, reviewed_by=user`. The 403 rule becomes: you may
  save a frame if it is assigned to you, unassigned, or you are reviewer+.

## Removed

- Auto-label: `PreAnnotateAllView`, its URL, `preannotate_pick.py`, the panel,
  the Retry form, their tests. With `SAMPLING_BACKEND=local` (dev only) frames
  come back unlabelled; Sample & label stays the only bulk path.
- Stage strip, 4 cards, "Your work" clip banner, assigned-to chips on the clip
  table, per-clip Assign/Take.

## Build order (each step deployable)

1. Migration + model fields + `reviewed_by` on save. Tests.
2. Frame grid query (DB-level, 60/page) as an include; `/review/` uses it
   first (fixes its scale and the Clear bug).
3. Detail page: header, 3 metrics, tabs (Frames default, Clips), queue strip;
   remove stage strip, cards, Auto-label.
4. Assign N frames + take back + self-take; People page workloads in frames.
5. Reviewer view: queue hero, Mine filters, editor walks the queue.
6. Data migration of old clip assignments; retire ClipAssignment code paths.
7. Update tests: test_assignments, test_access, test_view_access,
   test_workflow_ui, test_page_split, test_batch_autolabel (delete).

## Open questions

1. Assign pool: current filter + spread (designed) — confirm.
2. Should `/review/` stay as its own URL, or redirect to the Frames tab?
   Proposal: redirect; one page.
3. Reviewed frames with zero boxes (true negatives): do they count as
   "reviewed frames" in the metric? Proposal: yes; labelled counts only frames with boxes.
