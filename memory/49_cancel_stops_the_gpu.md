# 49 · Cancelling actually stops the GPU

Status: **built** (2026-10-08). Needs the GPU image with the worker change
and `pulumi up` before cancels stop GPU work; the web side works without it.
Canvas: https://claude.ai/artifact/RTC6sH1yxdw5aZQ6zaBY4P

## Ask
"Can I cancel a batch run? Does it actually stop the GPU run?"
It didn't stop anything. SageMaker async inference has no abort API, so
`JobCancelView` / `JobCancelAllView` only marked jobs cancelled: queued clips
still ran, running clips ran to the end, and both billed. There was no cancel
on the batch page.

## Design
- **Marker:** cancelling a job writes an empty object to the "processed"
  bucket at `cancel/<modal_job_id>` (`apps/analysis/cancelling.py`). The
  worker can already read that bucket, so no IAM change is needed. A chunked
  job's pieces (`<id>-cN`) share the marker. If the write fails, the job is
  still cancelled in the web app; it just runs to the end on the GPU.
- **Worker** (`sagemaker_backend/inference.py`, `src/beemonitor/cancellation.py`):
  - Before any download it looks once for the marker. If found, it returns
    `{"status": "cancelled"}` in about a second.
  - While the clip runs, a daemon `Watcher` thread checks every
    `BEEMONITOR_CANCEL_CHECK_SECONDS` (15). It sets a per-thread flag.
  - The frame loop in `bee_tracking.process_video` reads that flag every
    100 frames. When set, it closes the reader and the capture and raises
    `JobCancelled`, and the clip returns "cancelled".
  - Partial results are discarded.
  - Overhead is about 44 S3 HEAD calls per 11-min clip (~0.03% time).
  - A failing lookup never cancels.
  - The module sits at the package top level, so the cloud tests import it
    without cv2 or torch.
- **Web:**
  - `cancel_jobs()` is shared by the Processing page's Cancel / Cancel all and
    the batch cancel.
  - `_apply_result_to_job` treats a worker "cancelled" like a failure but
    sets the status to cancelled.
  - `pipelines:batch_cancel` is owner-only. It cancels the batch's active
    jobs, then calls `engine.cancel_run`, which fails every step not yet
    done with "Cancelled by user.".
  - Batch page:
    - a **Cancel batch** button with a confirmation (shown only while running);
    - the auto-reload is held while the dialog is open (`window.bmHoldRefresh`);
    - cancelled clips are counted apart from failures (`outcome.cancelled`)
      and get a grey "cancelled" pill;
    - the failure cause "Cancelled by you" uses the in-place **Re-run these
      clips** button.

## Not covered
- The SAM 3 endpoint runs different worker code (`sagemaker_backend/sam3/`).
  Its jobs are cancelled in the web app only and still run to the end.
- Post-tracking work (crops, species; ~30 s) is not interrupted.
