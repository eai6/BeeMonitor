# 64 — Name a batch; add clips to it

**Status:** built (2026-10-31), with the planned answers to all three questions. Design canvas: https://claude.ai/artifact/V1WZ7rdxz2L8UZLYXdDxW9 (boards "Batch page", "Processing hub in add mode", "Runs list"). Requested from batch `a1789cee` ("Foraging Trips", 11 clips of mendels):
- every batch is named after its pipeline, so three batches of one pipeline read the same in the Runs list;
- there's no way to grow a batch with more days of clips.

> Write under any **Feedback:** line. *Proposed* means a starting point, not a decision.

## What exists (audit, 2026-10-31)

- **A batch is only a `batch_id`.** There's no `Batch` row: a batch is the `PipelineRun`s that share a UUID.
  - `BatchShare` and `BatchMember` key on the same UUID.
  - Who launched it: `sharing.owns_batch` (every run in it is theirs).
- **Launch:**
  - `engine.launch_batch(pipeline, videos, user)` makes a fresh `batch_id`, checks credits for the whole set, and starts one run per clip from the pipeline's **current** steps (`steps_with_video`).
  - It's called from the Processing hub (`pipelines:run_on_videos`, with a clip picker, filters and "run on all filtered"), from the batch page's "Run these clips through a pipeline" (a new batch), and from the device scheduler.
  - GPU steps reuse `StepResult` when a clip was already tracked the same way.
- **Name shown:**
  - the batch page `<h1>` is `pipeline.title`, with "batch a1789cee" beside it;
  - the Runs list shows `b.pipeline.title` for your batches and for batches shared with you;
  - the public share page also uses `pipeline.title`.
- **Everything reads by `batch_id`:** the batch table, downloads, review metrics, period review (`?batch=`), the share link and members' access (`member_job_ids`). Runs added under the same id join all of it with no other change.
- **Missing:** a place to store a name, and a way to start runs under an existing `batch_id`.

## The plan

### 1. Name a batch
- **Data:** a new `Batch` model:
  - `batch_id` (UUID, unique);
  - `name` (up to 120 characters, may be blank);
  - `created_by`, `updated_at`.

  A row is made when someone names a batch. A batch with no row, or a blank name, shows the pipeline's title, as now.
- **Rename:** on the batch page the title gets a pencil, **Rename**, for the launcher only. It edits in place: Enter or **Save** saves, Esc cancels.
- **Subtitle:** the line under the title becomes "Foraging Trips pipeline · batch a1789cee · mendels", so the pipeline is still named.
- **At launch:** an optional **Batch name** field in the Processing hub's run form and in the batch page's re-run form. *Proposed default:* blank, which means the pipeline's name.
- **Where the name shows:**
  - the batch page title and browser tab;
  - the Runs list, yours and shared with you, with the pipeline's name in grey after it;
  - the public share page;
  - the period review header when it's opened from a batch ("Foraging Trips · May").

### 2. Add clips to a batch
- **Entry:** an **Add clips** button on the batch page, for the launcher only, next to People and Share. It opens the Processing hub in *add mode* (`?add_to=<batch>`):
  - a banner: "Adding clips to **Foraging Trips · May** (11 clips). Same pipeline and settings as the rest of the batch.";
  - the pipeline picker is replaced by the batch's pipeline, which can't be changed;
  - clips already in the batch are marked **in this batch** and can't be picked;
  - the run button reads **Add N clips**, and also works with "all filtered" (for example all of mendels 24–30 May).
- **Which settings:** *Proposed:* the batch's own frozen steps (its first run's `steps`, with each new clip's id swapped in), not the pipeline's current version, so every clip in a batch is counted the same way. If the pipeline has changed since, the banner says so: "The pipeline was edited after this batch ran; added clips use the batch's version."
- **Engine:** `launch_batch(..., batch_id=<existing>, steps=<batch steps>)` adds to an existing batch:
  - credits are checked for the new clips only;
  - clips already in the batch are skipped and counted in the message;
  - the GPU cache applies as usual.
- **After:** you land back on the batch page with "Adding 14 clips. They show as running below." Downloads, metrics, period review, the share link and members all include the new clips with no other change. The header's "run 30 Oct, 01:15" becomes "run 30 Oct – 2 Nov" once clips were added later.
- **Not planned now:** removing a clip from a batch; a schedule that keeps adding a unit's new clips to a batch.

## Built (2026-10-31)
- **Names:**
  - the `Batch` model (migration 0017) and `apps/pipelines/batches.py` (`names`, `title`, `rename`);
  - `pipelines:batch_rename` (launcher only) with an inline **Rename** on the batch page;
  - the name shows on the batch page and its tab, the Runs list (yours and shared), the public page and the period header;
  - an optional "Batch name" at launch in the Processing hub and the batch re-run form.
- **Add clips:**
  - `engine.batch_template(batch_id)` returns the first run's frozen steps with the clip unbound, plus its window;
  - `launch_batch(..., batch_id=, steps=)`;
  - `run_on_videos` takes `add_to` (launcher only, skips clips already in the batch, uses the batch's window);
  - the hub's `?add_to=` mode (`_add_batch`): banner, locked pipeline, "in this batch" chips, "Add N to the batch", then back to the batch;
  - the batch header shows a date range once clips were added on later days.
- **Tests:** `apps/pipelines/tests/test_batch_names.py`.

## Build order
1. `Batch` model and migration; `batch_name(batch_id)` helper plus a bulk version for the Runs list; the rename endpoint (launcher only) with an inline editor; the name in the Runs list, share page and period header. Tests.
2. Optional name at launch (Processing hub, batch re-run form).
3. `launch_batch` into an existing batch with frozen steps; Processing hub add mode (banner, locked pipeline, "in this batch" marks, Add N clips); the batch page button. Tests for skipped clips, credits for new clips only, frozen steps, and members seeing added clips.

## Questions
1. Should added clips use the **batch's version** of the pipeline (planned), or the pipeline as it is now?
2. Should a clip already in the batch be **skipped** (planned), or re-run in place?
3. Should a batch name be **optional** at launch (planned), or asked for every time?

**Feedback:**
